"""FP32 DNN benchmark: one process, one persistent CUDA stream/thread per model.

Weights and prepared inputs are identical to trial. Only execution architecture
changes. Stream synchronization measures completion without a device-wide barrier.
Python dispatch/GPU contention remain in latency; this is not kernel-only timing.
"""
import contextlib
import gc
import json
import math
from pathlib import Path
import platform
import queue
import sys
import threading
import time
import traceback

from power import PowerRecorder, attach_recorded_power


def add_parser(sub):
    p=sub.add_parser('gpu-shared',help='FP32, shared CUDA context, three model streams')
    p.add_argument('--comp-model',choices=['comp','gru'],default='comp')
    p.add_argument('--duration',type=float,default=60.)
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--record-power',action='store_true')
    p.add_argument('--power-interval',type=float,default=.1)
    p.add_argument('--tegrastats',action='store_true')
    p.add_argument('--startup-timeout',type=float,default=300.)
    p.add_argument('--output',required=True)
    p.set_defaults(func=run)


def measure(model,x,meta,ops,duration,sync):
    """Bounded online statistics; one full frame per forward, batch size 1."""
    model.reset(); sync()
    wall=time.time(); begin=time.perf_counter()
    frames=rows=0; total=mean=m2=hi=0.; lo=float('inf')
    while frames==0 or time.perf_counter()-begin<duration:
        i=frames%len(x)
        if i==0: model.reset()
        a=time.perf_counter()
        out=model.forward(x[i:i+1]); sync()
        elapsed=time.perf_counter()-a
        total+=elapsed; frames+=1; rows+=meta['valid_rows'][i]
        delta=elapsed-mean; mean+=delta/frames; m2+=delta*(elapsed-mean)
        lo=min(lo,elapsed); hi=max(hi,elapsed)
    end=time.perf_counter(); wall_end=time.time()
    record=dict(start_unix_s=wall,end_unix_s=wall_end,wall_seconds=end-begin,
        inference_seconds=total,frames=frames,valid_rows=rows,mean_latency_ms=mean*1000,
        min_latency_ms=lo*1000,max_latency_ms=hi*1000,latency_std_ms=(m2/frames)**.5*1000,
        frames_per_second=frames/total,valid_rows_per_second=rows/total,
        gops=ops['operations_per_frame']*frames/total/1e9,
        operations_per_second=ops['operations_per_frame']*frames/total)
    return record,out


def worker(name,path,args,inbox,outbox,gate,finished):
    try:
        import torch
        from models import Model
        from preprocess import load_input
        from run import file_sha_light
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            model=Model(name,'cuda','float')
            x,meta,expected=load_input(path,model)
            if expected is not None:
                raise ValueError('gpu-shared currently accepts bundled synthetic inputs without references only')
            ops=model.operations()
            info=dict(schema=1,kind='dnn',model=name,backend='float',device='cuda',
                implementation=model.description,cudnn_mode='off',threads=1,
                execution='single process; dedicated Python thread and CUDA stream per model',
                torch=torch.__version__,python=sys.version,machine=platform.machine(),
                weight_sha256=model.sha,input_sha256=file_sha_light(path),input_shape=list(x.shape),
                input=meta,operations=ops,validation=dict(expected_provided=False,passed=None),
                scope='preloaded CUDA tensor through DNN output and stream synchronization; includes dispatch/contention; excludes preprocessing/loading/transfers',
                runs=[])
            # Float networks own their parameters. These CPU-only source copies
            # are no longer needed after input validation and static OPS counting.
            model.checkpoint=None; model.contract=None
            gc.collect()
            for _ in range(args.warmup):
                model.reset(); output=model.forward(x[:1]); stream.synchronize()
            if not bool(torch.isfinite(output).all()): raise ValueError('nonfinite warmup output')
            reference=output.detach().clone()
            stream.synchronize()
            outbox.put(('ready',name,info))
            while True:
                repeat=inbox.get()
                if repeat is None: return
                gate.wait(timeout=args.startup_timeout)
                record,output=measure(model,x,meta,ops,args.duration,stream.synchronize)
                finished.wait(timeout=args.startup_timeout)
                if not bool(torch.isfinite(output).all()): raise ValueError('nonfinite output')
                # Outside the timed window: repeat the initial frame/state and
                # check that concurrent execution did not alter model behavior.
                model.reset(); check=model.forward(x[:1]); stream.synchronize()
                if not torch.allclose(check,reference,rtol=1e-5,atol=1e-6):
                    raise ValueError('output changed from isolated warmup baseline')
                record['warmup_output_consistency_passed']=True
                stream.synchronize()
                record['repeat']=repeat
                outbox.put(('result',name,record))
    except BaseException:
        outbox.put(('error',name,traceback.format_exc()))
        gate.abort()
        finished.abort()


def run(args):
    from run import ROOT, write_json, telemetry, configure_cudnn, file_sha_light
    if not math.isfinite(args.duration) or args.duration<=0 or args.repeat<1 or args.warmup<1:
        raise ValueError('positive duration/repeat and at least one warmup required')
    if not math.isfinite(args.startup_timeout) or args.startup_timeout<=0:
        raise ValueError('startup timeout must be positive')
    folder=Path(args.output).resolve(); folder.mkdir(parents=True,exist_ok=False)
    names=['pd','bus',args.comp_model]
    manifest=json.loads((ROOT/'dummy_data'/'manifest.json').read_text(encoding='utf-8'))
    inputs={n:ROOT/manifest['inputs'][n+'_float']['file'] for n in names}
    for name,path in inputs.items():
        if file_sha_light(path)!=manifest['inputs'][name+'_float']['sha256']:
            raise ValueError('bundled dummy input hash mismatch: '+str(path))
    write_json(folder/'config.json',dict(arguments={k:v for k,v in vars(args).items() if k!='func'},models=names,
        source_sha256={n:file_sha_light(ROOT/n) for n in ('gpu_shared.py','models.py','preprocess.py','power.py')}))
    print('[SHARED] Import PyTorch; one process / three streams / FP32 / cuDNN off',flush=True)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(0)
    configure_cudnn(torch,'off')
    if hasattr(torch.backends,'cuda') and hasattr(torch.backends.cuda,'matmul'):
        torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    outbox=queue.Queue(); inboxes={}; threads=[]; infos={}
    gate=threading.Barrier(4)
    finished=threading.Barrier(3)
    recorder=PowerRecorder(folder/'power.csv',args.power_interval) if args.record_power else None
    def receive(timeout):
        item=outbox.get(timeout=timeout)
        if item[0]=='error': raise RuntimeError(item[1]+': '+item[2])
        return item
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(telemetry(str(folder/'tegrastats.log') if args.tegrastats else None))
            if recorder: stack.enter_context(recorder)
            for name in names:
                inboxes[name]=queue.Queue()
                t=threading.Thread(target=worker,args=(name,inputs[name],args,inboxes[name],outbox,gate,finished),daemon=True)
                threads.append(t); t.start(); started=time.monotonic()
                while True:
                    try:
                        kind,got,info=receive(5.)
                        if kind!='ready' or got!=name: raise RuntimeError('unexpected worker readiness')
                        infos[name]=info; break
                    except queue.Empty:
                        elapsed=time.monotonic()-started
                        print('[SHARED] Loading/warmup %s | %.0fs'%(name.upper(),elapsed),flush=True)
                        if elapsed>args.startup_timeout: raise TimeoutError('startup '+name)
                print('[SHARED] READY %s (%d/3)'%(name.upper(),len(infos)),flush=True)
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            for repeat in range(1,args.repeat+1):
                torch.cuda.reset_peak_memory_stats()
                repdir=folder/('together_%02d'%repeat); repdir.mkdir()
                for name in names: inboxes[name].put(repeat)
                gate.wait(timeout=args.startup_timeout)
                started=time.monotonic(); records={}
                while len(records)<3:
                    try:
                        kind,name,record=receive(5.)
                        if kind!='result' or name in records: raise RuntimeError('unexpected worker result')
                        records[name]=record
                    except queue.Empty:
                        elapsed=time.monotonic()-started
                        print('[SHARED %d/%d] %.1f/%.1fs | %d/3 finished'%(repeat,args.repeat,elapsed,args.duration,len(records)),flush=True)
                        if elapsed>args.duration+args.startup_timeout: raise TimeoutError('inference timeout')
                for name in names:
                    info=dict(infos[name]); info['runs']=[records[name]]
                    write_json(repdir/(name+'.json'),info)
                start=max(r['start_unix_s'] for r in records.values())
                end=min(r['end_unix_s'] for r in records.values())
                union_start=min(r['start_unix_s'] for r in records.values())
                union_end=max(r['end_unix_s'] for r in records.values())
                system=dict(schema=1,kind='concurrent_dnn',device='cuda',backend='float',models=names,
                    execution='single process; three persistent worker threads and dedicated CUDA streams',
                    cudnn_mode='off',start_unix_s=start,end_unix_s=end,
                    scope='intersection of three DNN inference loops; sampled module power, not per-model power',
                    cuda_memory_peak_bytes=torch.cuda.max_memory_allocated(),
                    cuda_memory_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    union_start_unix_s=union_start,union_end_unix_s=union_end,
                    aggregate_gops_union=sum(infos[n]['operations']['operations_per_frame']*records[n]['frames'] for n in names)/(union_end-union_start)/1e9,
                    aggregate_convention='all completed frame MAC-derived operations divided by union wall time; not sum of individual kernel rates')
                write_json(repdir/'system.json',system)
                print('[SHARED %d/%d] DONE | aggregate %.3f GOPS'%(repeat,args.repeat,system['aggregate_gops_union']),flush=True)
            for inbox in inboxes.values(): inbox.put(None)
            for t in threads: t.join(timeout=10.)
        if recorder:
            files=attach_recorded_power(folder,folder/'power.csv',recorder.metadata,args.power_interval)
            write_json(folder/'power_results.json',dict(results=files,source=recorder.metadata))
        print('[SHARED] Complete '+str(folder),flush=True)
    except BaseException as exc:
        gate.abort()
        finished.abort()
        for inbox in inboxes.values(): inbox.put(None)
        write_json(folder/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise
