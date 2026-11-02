"""Jetson GPU fault localization. Python 3.6 compatible; diagnostic timings only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def memory():
    result = {'process_pid': os.getpid()}
    for filename, keys in [('/proc/meminfo', ('MemAvailable', 'MemFree', 'SwapFree', 'SwapTotal')),
                           ('/proc/self/status', ('VmRSS', 'VmHWM'))]:
        try:
            for line in Path(filename).read_text().splitlines():
                key = line.split(':')[0]
                if key in keys:
                    result[key + '_kib'] = int(line.split()[1])
        except OSError:
            pass
    return result


def worker(args):
    os.environ.setdefault('OPENBLAS_CORETYPE', 'ARMV8')
    trace = open(str(Path(args.output) / 'stages.jsonl'), 'x', buffering=1)
    torch = None

    def event(stage, **extra):
        record = dict(time_s=time.time(), stage=stage, memory=memory(), **extra)
        if torch is not None and torch.cuda.is_initialized():
            record['cuda'] = dict(allocated=torch.cuda.memory_allocated(),
                                  reserved=torch.cuda.memory_reserved(),
                                  peak_allocated=torch.cuda.max_memory_allocated(),
                                  peak_reserved=torch.cuda.max_memory_reserved())
        trace.write(json.dumps(record) + '\n')
        trace.flush()
        print(stage, flush=True)

    try:
        event('import_torch_begin')
        import torch as torch_module
        torch = torch_module
        import run
        from models import Model
        from preprocess import load_input
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        run.configure_cudnn(torch, args.cudnn_mode)
        event('import_torch_done', torch_version=torch.__version__, cuda_version=torch.version.cuda,
              cudnn_version=torch.backends.cudnn.version(), cudnn_mode=args.cudnn_mode)
        event('cuda_init_begin')
        torch.cuda.init()
        event('tiny_cuda_begin')
        a = torch.ones(4, device='cuda')
        assert float((a + a).sum().cpu()) == 8.0
        del a
        torch.cuda.synchronize()
        event('tiny_cuda_done', gpu=torch.cuda.get_device_name(0))
        event('model_cpu_load_begin')
        model = Model(args.model, device='cpu', backend='float')
        event('model_cpu_load_done', parameter_bytes=sum(p.numel()*p.element_size() for p in model.network.parameters()))
        event('model_to_cuda_begin')
        model.network.to('cuda')
        model.device = torch.device('cuda')
        torch.cuda.synchronize()
        event('model_to_cuda_done')
        manifest = json.loads((Path(__file__).parent/'dummy_data'/'manifest.json').read_text())
        path = Path(__file__).parent/manifest['inputs'][args.model+'_float']['file']
        event('input_load_begin')
        x, meta, expected = load_input(str(path), model)
        event('input_load_done', input_shape=list(x.shape), synthetic=meta['synthetic'])
        handles = []
        for name, module in model.network.named_modules():
            if list(module.children()):
                continue
            def pre(mod, inputs, label=name):
                torch.cuda.synchronize()
                event('layer_begin', layer=label, layer_type=type(mod).__name__)
            def post(mod, inputs, output, label=name):
                torch.cuda.synchronize()
                event('layer_done', layer=label)
            handles.extend([module.register_forward_pre_hook(pre), module.register_forward_hook(post)])
        for repeat in range(2):
            model.reset()
            event('forward_begin', repeat=repeat+1)
            out = model.forward(x[:1])
            torch.cuda.synchronize()
            if not bool(torch.isfinite(out).all()):
                raise ValueError('nonfinite output')
            event('forward_done', repeat=repeat+1, output_shape=list(out.shape))
            del out
        for handle in handles:
            handle.remove()
        event('success', note='Diagnostic synchronization/hooks invalidate benchmark timing; task accuracy unvalidated')
    except BaseException as exc:
        event('failed', error=repr(exc))
        traceback.print_exc()
        raise
    finally:
        trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['pd', 'bus', 'comp', 'gru'], default='pd')
    parser.add_argument('--cudnn-mode', choices=['deterministic', 'heuristic', 'off'], default='off')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    args.output = str(folder)
    cmd = [sys.executable, '-u', '-X', 'faulthandler', str(Path(__file__).resolve()),
           '--worker', '--model', args.model, '--cudnn-mode', args.cudnn_mode, '--output', str(folder)]
    start = time.monotonic()
    reason = None
    child = None
    monitor = None
    with open(str(folder/'worker.log'), 'x') as log, open(str(folder/'system_memory.jsonl'), 'x', buffering=1) as mem:
        try:
            try:
                monitor = subprocess.Popen(['tegrastats', '--interval', '500', '--logfile', str(folder/'tegrastats.log')],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                mem.write(json.dumps(dict(time_s=time.time(), **memory()))+'\n')
                elapsed = time.monotonic()-start
                last = 'starting'
                try:
                    stage = json.loads((folder/'stages.jsonl').read_text().splitlines()[-1])
                    last = stage['stage'] + ' ' + stage.get('layer', '')
                except (OSError, ValueError, IndexError):
                    pass
                print('[GPU PROBE] %.0fs | PID %d | %s' % (elapsed, child.pid, last), flush=True)
                if elapsed >= args.timeout:
                    reason = 'timeout'
                    break
                time.sleep(2)
        finally:
            for process in (child, monitor):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    try:
        kernel = subprocess.run(['dmesg', '-T'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        (folder/'kernel.log').write_text(kernel.stdout)
    except OSError as exc:
        (folder/'kernel.log').write_text(str(exc))
    code = child.returncode if child is not None else None
    result = dict(model=args.model, cudnn_mode=args.cudnn_mode, returncode=code, stop_reason=reason,
                  success=code == 0 and reason is None,
                  note='SIGKILL alone does not prove OOM; correlate kernel timestamps. CUDA allocated/reserved omit some driver/library memory.')
    (folder/'diagnosis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not result['success']:
        sys.exit(1)


if __name__ == '__main__':
    main()
