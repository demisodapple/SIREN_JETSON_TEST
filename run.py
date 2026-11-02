"""Single entry point: check / prepare / bench / compress / decode / together / power."""
import argparse
import contextlib
import csv
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import threading
import signal
from power import integrate_power, sensor_candidates, select_sensor, PowerRecorder, attach_recorded_power, is_input_sensor

ROOT = Path(__file__).resolve().parent


class Progress:
    """Low-rate line progress, also visible over SSH and captured in log files.

    A heartbeat can show elapsed time during one long native call. It cannot
    know that call's completion percentage. Disable for final timing comparisons.
    """
    def __init__(self, label, enabled=True, interval=2.):
        if interval<=0: raise ValueError('--progress-interval must be positive')
        self.label,self.enabled,self.interval=label,enabled,interval
        self.stop=threading.Event(); self.thread=None
        self.state=('starting',time.monotonic(),None,None)
        self.completed=0

    def __enter__(self):
        if self.enabled:
            self.thread=threading.Thread(target=self._run,daemon=True)
            self.thread.start()
        return self

    def stage(self, name, target=None, total=None):
        self.completed=0
        self.state=(name,time.monotonic(),target,total)
        if self.enabled: print('[%s] %s ...'%(self.label,name),flush=True)

    def advance(self,count):
        self.completed=count

    def render(self):
        name,start,target,total=self.state
        elapsed=time.monotonic()-start
        if target is not None:
            fraction=min(.99,elapsed/target)
            n=int(fraction*20)
            detail='[%s%s] %2d%% time %.1f/%.1fs | completed inferences %d'%(
                '#'*n,'-'*(20-n),fraction*100,elapsed,target,self.completed)
            if elapsed>=target: detail+=' | waiting for current inference to finish'
        elif total is not None:
            fraction=min(1.,self.completed/max(total,1))
            n=int(fraction*20)
            detail='[%s%s] %d/%d | %.1fs'%('#'*n,'-'*(20-n),self.completed,total,elapsed)
        else:
            detail='elapsed %.1fs; waiting for stage to finish'%elapsed
        return '[%s] %s | %s'%(self.label,name,detail)

    def _run(self):
        while not self.stop.wait(self.interval):
            print(self.render(),flush=True)

    def __exit__(self,*exc):
        self.stop.set()
        if self.thread: self.thread.join(timeout=self.interval+1)


def failure_status(code):
    if code<0:
        try: return 'signal %s (%d)'%(signal.Signals(-code).name,-code)
        except ValueError: return 'signal %d'%(-code)
    return 'exit code %d'%code


def progress_flags(args):
    flags=['--progress-interval',str(getattr(args,'progress_interval',2.))]
    if getattr(args,'no_progress',False): flags.append('--no-progress')
    return flags


def drain_logs(folder,names,offsets):
    """Mirror new worker lines while retaining the original per-model logs."""
    for name in names:
        path=folder/(name+'.log')
        if path.exists():
            with open(str(path),'r',encoding='utf-8',errors='replace') as f:
                f.seek(offsets.get(name,0)); text=f.read(); offsets[name]=f.tell()
            if text: print(text,end='',flush=True)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def command_output(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=15).decode('utf-8', errors='replace').strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)


def check(args):
    info = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'system': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
            'executable': sys.executable, 'disk_free_bytes': shutil.disk_usage(str(ROOT)).free}
    for name, path in [('board','/proc/device-tree/model'),('os','/etc/os-release'),
                       ('jetson_linux','/etc/nv_tegra_release'),('memory','/proc/meminfo')]:
        p = Path(path)
        info[name] = p.read_text().replace('\0','') if p.exists() else 'not available on this host'
    for name, cmd in [('power_mode',['nvpmodel','-q','--verbose']),('clocks',['jetson_clocks','--show'])]:
        info[name] = command_output(cmd) if shutil.which(cmd[0]) else 'not installed'
    try:
        import torch
        info.update(torch=torch.__version__,cuda_version=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            x=torch.ones(4,device='cuda'); info['gpu_check_sum']=float((x+x).sum().cpu())
    except ImportError as exc:
        info['torch'] = str(exc)
    write_json(args.output,info)
    print(json.dumps(info,ensure_ascii=False,indent=2))


def setup_model(args,progress=None):
    if progress: progress.stage('1/6 Import PyTorch and model code')
    import torch
    from models import Model
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    if hasattr(torch.backends,'cuda') and hasattr(torch.backends.cuda,'matmul'):
        torch.backends.cuda.matmul.allow_tf32=False
    if progress: progress.stage('2/6 Load weights and construct model on '+args.device)
    return Model(args.model,args.device,args.backend)


def prepare(args):
    import numpy as np
    import preprocess as pp
    m=setup_model(args)
    if args.demo_calibration and args.kind not in ('bus','pd'):
        raise ValueError('--demo-calibration applies only to --kind bus or pd')
    if args.kind=='smoke':
        # Explicit synthetic label survives into every benchmark result.
        dtype=np.int16 if m.backend=='reference' else np.float32
        x=np.zeros([1]+m.input_shape,dtype)
        pp.save_input(args.output,x,m,[1 if m.name=='gru' else m.input_shape[0]],'synthetic zero smoke input',True)
    elif args.kind=='dummy':
        pp.prepare_dummy(args,m)
    elif args.kind=='counts':
        if m.name not in ('gru','comp'):
            raise ValueError('counts preparation requires gru or comp')
        pp.prepare_counts(args,m)
    elif args.kind=='bus':
        if m.name!='bus': raise ValueError('--model bus required')
        pp.prepare_bus(args,m)
    elif args.kind=='pd':
        if m.name!='pd': raise ValueError('--model pd required')
        pp.prepare_pd(args,m)
    else:
        if not args.input or not args.valid_rows:
            raise ValueError('tensor import needs --input and --valid-rows')
        p=Path(args.input)
        if p.suffix=='.npy':
            x=np.load(str(p),allow_pickle=False)
        elif p.suffix=='.npz':
            with np.load(str(p),allow_pickle=False) as d: x=d['x']
        else:
            if not args.dtype: raise ValueError('binary import needs explicit --dtype, e.g. int16')
            x=np.fromfile(str(p),dtype=np.dtype(args.dtype)).reshape([-1]+m.input_shape)
        extra={}
        if args.expected:
            extra['expected']=np.load(args.expected,allow_pickle=False)
            if len(extra['expected'])!=len(x): raise ValueError('expected output frame count mismatch')
        pp.save_input(args.output,x,m,np.full(len(x),args.valid_rows),'external prepared tensor:'+str(p),**extra)
    print('Saved '+args.output)


def operations_command(args):
    m=setup_model(args)
    result={'model':m.name,'backend':m.backend,'implementation':m.description,
            'input_shape':[1]+m.input_shape,'weight_sha256':m.sha,'operations':m.operations()}
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


@contextlib.contextmanager
def telemetry(path):
    process=None
    handle=None
    try:
        if path:
            if not shutil.which('tegrastats'): raise ValueError('tegrastats not available; omit --tegrastats on PC')
            handle=open(path,'x',encoding='utf-8')
            process=subprocess.Popen(['tegrastats','--interval','1000'],stdout=handle,stderr=subprocess.STDOUT)
        yield
    finally:
        if process:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        if handle: handle.close()


def gate(args):
    """Child ready barrier: every model loaded and warmed before concurrent timing."""
    if args.gate:
        write_json(args.ready,{'ready':True,'pid':os.getpid()})
        until=time.monotonic()+args.startup_timeout
        while not Path(args.gate).exists():
            if time.monotonic()>until: raise TimeoutError('concurrent start barrier timed out')
            time.sleep(.05)


def benchmark(args):
    with Progress(args.model.upper(),not getattr(args,'no_progress',False),
                  getattr(args,'progress_interval',2.)) as progress:
        _benchmark(args,progress)


def _benchmark(args,progress):
    progress.stage('Import numerical libraries')
    import numpy as np
    import torch
    from preprocess import load_input
    from models import file_sha
    if args.duration<=0 or args.repeat<1 or args.warmup<0:
        raise ValueError('duration/repeat must be positive, warmup nonnegative')
    m=setup_model(args,progress)
    progress.stage('3/6 Load input and transfer to '+args.device)
    x,meta,expected=load_input(args.input,m)
    if meta['calibration_mode']=='arbitrary_demo':
        print('DEMO CALIBRATION: DNN load/power experiment only; detection accuracy is unvalidated.',flush=True)
    if meta['synthetic']:
        print('SYNTHETIC INPUT: DNN load/OPS experiment; task accuracy and representative mission power unvalidated.',flush=True)
    progress.stage('4/6 Count operations from model structure (no GPU forward)')
    ops=m.operations()
    # Validate supplied reference outside measurement, preserving sequential GRU states.
    validation={'expected_provided':expected is not None,'passed':None}
    if expected is not None:
        progress.stage('Validate supplied reference outputs',total=len(x))
        if len(expected)!=len(x): raise ValueError('expected output length mismatch')
        m.reset()
        for i in range(len(x)):
            got=m.forward(x[i:i+1]).cpu().numpy()
            target=expected[i:i+1]
            if m.backend=='reference': np.testing.assert_array_equal(got,target)
            else: np.testing.assert_allclose(got,target,rtol=1e-5,atol=1e-6)
            progress.advance(i+1)
        validation['passed']=True
    progress.stage('5/6 Warmup (first inference may take longer)',total=args.warmup)
    for index in range(args.warmup):
        m.reset(); m.forward(x[:1]); m.synchronize()
        progress.advance(index+1)
    if args.gate: progress.stage('Ready; waiting for other models')
    gate(args)
    result={'schema':1,'kind':'dnn','model':m.name,'backend':m.backend,'implementation':m.description,
            'device':str(m.device),'threads':args.threads,'torch':torch.__version__,'python':sys.version,
            'machine':platform.machine(),'weight_sha256':m.sha,'input_sha256':file_sha(args.input),
            'input_shape':list(x.shape),'input':meta,'validation':validation,'operations':ops,
            'scope':'preloaded-device tensor to DNN output; model load/preprocessing/transfers excluded',
            'power_scope':'board input during inference loop; includes synchronization and loop bookkeeping',
            'cuda_memory_peak_bytes':None,'runs':[],
            'progress':{'enabled':not getattr(args,'no_progress',False),
                        'interval_seconds':getattr(args,'progress_interval',2.),
                        'note':'low-rate heartbeat adds host/output activity; use --no-progress for final comparisons'}}
    if m.device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    with telemetry(args.tegrastats):
        for repeat in range(args.repeat):
            m.reset(); m.synchronize()
            progress.stage('6/6 Measure repeat %d/%d'%(repeat+1,args.repeat),target=args.duration)
            wall_start=time.time(); begin=time.perf_counter()
            measured=0.; frames=0; rows=0
            # Welford accumulators avoid unbounded latency arrays for long runs.
            mean=0.; m2=0.; lo=None; hi=0.
            while frames==0 or time.perf_counter()-begin<args.duration:
                i=frames%len(x)
                if i==0: m.reset()
                m.synchronize(); a=time.perf_counter()
                out=m.forward(x[i:i+1]); m.synchronize()
                elapsed=time.perf_counter()-a
                measured+=elapsed; frames+=1; rows+=meta['valid_rows'][i]
                progress.advance(frames)
                delta=elapsed-mean; mean+=delta/frames; m2+=delta*(elapsed-mean)
                lo=elapsed if lo is None else min(lo,elapsed); hi=max(hi,elapsed)
            end=time.perf_counter(); wall_end=time.time()
            if not bool(torch.isfinite(out).all()): raise ValueError('nonfinite output')
            record={'repeat':repeat+1,'start_unix_s':wall_start,'end_unix_s':wall_end,
                    'wall_seconds':end-begin,'inference_seconds':measured,'frames':frames,'valid_rows':rows,
                    'mean_latency_ms':mean*1000,'min_latency_ms':lo*1000,'max_latency_ms':hi*1000,
                    'latency_std_ms':(m2/frames)**.5*1000,'frames_per_second':frames/measured,
                    'valid_rows_per_second':rows/measured,
                    'operations_per_second':ops['operations_per_frame']*frames/measured,
                    'gops':ops['operations_per_frame']*frames/measured/1e9}
            result['runs'].append(record)
            if progress.enabled:
                print('[%s] [####################] 100%% | repeat %d/%d complete | %d inferences'%(
                    m.name.upper(),repeat+1,args.repeat,frames),flush=True)
            print(json.dumps(record),flush=True)
    if m.device.type=='cuda': result['cuda_memory_peak_bytes']=torch.cuda.max_memory_allocated()
    progress.stage('Save result '+args.output)
    write_json(args.output,result)
    if progress.enabled: print('[%s] DONE'%m.name.upper(),flush=True)


def compress(args):
    import numpy as np
    import torch
    from models import Model, Predictor, BaselinePredictor, load_bundle, file_sha
    from preprocess import read_counts
    from compression import encode,decode
    if args.mode=='neural' and args.model not in ('gru','comp'):
        raise ValueError('neural compression requires --model gru or comp')
    if args.block<1 or not 1<=args.level<=9: raise ValueError('invalid block size/BZ2 level')
    output=Path(args.output)
    report=output.with_suffix(output.suffix+'.json')
    if output.exists() or report.exists(): raise ValueError('output or result already exists')
    bundle,_=load_bundle('gru')
    channels=list(bundle['checkpoint']['channels'])
    scale=int(bundle['checkpoint']['config']['quantization_scale'])
    # Always compress the SAME 28 output channels, including ModernTCN's unsupported channels.
    q,aux=read_counts(args.input,channels,scale)
    if args.limit:
        if args.limit<2: raise ValueError('--limit must be >=2')
        q,aux=q[:args.limit],aux[:args.limit]
    model=setup_model(args) if args.mode=='neural' else None
    predictor=Predictor(model,channels,scale) if model else BaselinePredictor(channels,scale)
    if model:
        predictor.step(q[0],aux[0]); model.synchronize(); predictor.reset()
    output.parent.mkdir(parents=True,exist_ok=True)
    with telemetry(args.tegrastats):
        if model: model.synchronize()
        start=time.time(); tic=time.perf_counter()
        packet,metrics=encode(q,aux,predictor,args.block,args.level,args.mode)
        if model: model.synchronize()
        elapsed=time.perf_counter()-tic; end=time.time()
    # Decoder is independent of original q; receives packet and predictor only.
    print('Encoding complete; checking exact reconstruction outside measurement.',flush=True)
    tic=time.perf_counter(); restored,header=decode(packet,predictor)
    np.testing.assert_array_equal(restored,q)
    decode_seconds=time.perf_counter()-tic
    with open(str(output),'xb') as f: f.write(packet)
    metrics.update(schema=1,kind='compression',model=args.model if model else 'none',mode=args.mode,
                   backend=model.backend if model else 'none',implementation=model.description if model else 'BZ2 baseline',
                   device=str(model.device) if model else 'cpu',threads=args.threads,torch=torch.__version__,
                   input_sha256=file_sha(args.input),packet_sha256=file_sha(output),limit=args.limit,
                   start_unix_s=start,end_unix_s=end,wall_seconds=elapsed,decode_seconds=decode_seconds,
                   exact_reconstruction=True,original_mb_per_second=q.nbytes/1e6/elapsed,
                   output_channels=channels,predicted_channels=len(predictor.predicted_indices),
                   scope='RAM counts+aux to complete packet; all candidate costs included; decode and disk I/O excluded',
                   comparison='same 28 output channels; GRU predicts 28, ModernTCN predicts 13; system comparison')
    write_json(report,metrics)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


def decode_command(args):
    import numpy as np
    from compression import decode,MAGIC
    from models import Predictor,BaselinePredictor
    import struct
    packet=Path(args.input).read_bytes()
    if not packet.startswith(MAGIC): raise ValueError('invalid packet magic')
    at=len(MAGIC)+4
    size=struct.unpack('<I',packet[len(MAGIC):at])[0]
    h=json.loads(packet[at:at+size])
    model=setup_model(args) if h['mode']=='neural' else None
    p=Predictor(model,h['channels'],h['scale']) if model else BaselinePredictor(h['channels'],h['scale'])
    q,_=decode(packet,p)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,'xb') as f:
        np.savez(f,q=q,counts=(q.astype(np.float64)/h['scale']).astype(np.float32),channels=np.asarray(h['channels']))
    print('Exact reconstruction verified by stored SHA-256.')


def power(args):
    data=json.loads(Path(args.result).read_text(encoding='utf-8'))
    runs=data.get('runs',[data])
    for r in runs:
        r['power']=integrate_power(args.input,r['start_unix_s'],r['end_unix_s'],args.max_gap)
        if r.get('frames'): r['power']['joules_per_frame']=r['power']['energy_j']/r['frames']
        if r.get('count_raw_bytes'): r['power']['joules_per_original_mb']=r['power']['energy_j']/(r['count_raw_bytes']/1e6)
    data['power_csv']=str(Path(args.input).resolve())
    write_json(args.output,data)
    print('Saved '+args.output)


def together(args):
    folder=Path(args.output).resolve()
    folder.mkdir(parents=True,exist_ok=False)
    go=folder/'START'
    children=[]; logs=[]; offsets={}
    assignments=[('pd',args.pd_input),('bus',args.bus_input),(args.comp_model,args.comp_input)]
    try:
        for name,inp in assignments:
            log=open(str(folder/(name+'.log')),'x',encoding='utf-8'); logs.append(log)
            cmd=[sys.executable,'-u','-X','faulthandler',str(ROOT/'run.py'),'bench','--model',name,'--backend',args.backend,
                 '--device',args.device,'--threads',str(args.threads),'--input',str(Path(inp).resolve()),
                 '--output',str(folder/(name+'.json')),'--duration',str(args.duration),'--repeat','1',
                 '--warmup',str(args.warmup),'--gate',str(go),'--ready',str(folder/(name+'.ready.json')),
                 '--startup-timeout',str(args.startup_timeout)]+progress_flags(args)
            children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+args.startup_timeout
        last_display=0.
        while not all((folder/(name+'.ready.json')).exists() for name,_ in assignments):
            if not getattr(args,'no_progress',False) and time.monotonic()-last_display>=args.progress_interval:
                drain_logs(folder,[n for n,_ in assignments],offsets); last_display=time.monotonic()
            for (name,_),worker in zip(assignments,children):
                if worker.poll() is not None:
                    drain_logs(folder,[n for n,_ in assignments],offsets)
                    raise RuntimeError('%s worker failed during startup: %s; see %s'%(name,failure_status(worker.returncode),folder/(name+'.log')))
            if time.monotonic()>deadline: raise TimeoutError('model loading/warmup timed out')
            time.sleep(.1)
        with telemetry(str(folder/'tegrastats.txt') if args.tegrastats else None):
            go.write_text('start',encoding='ascii')
            while any(p.poll() is None for p in children):
                if not getattr(args,'no_progress',False) and time.monotonic()-last_display>=args.progress_interval:
                    drain_logs(folder,[n for n,_ in assignments],offsets); last_display=time.monotonic()
                for (name,_),worker in zip(assignments,children):
                    if worker.poll() not in (None,0):
                        drain_logs(folder,[n for n,_ in assignments],offsets)
                        raise RuntimeError('%s worker failed: %s; see %s'%(name,failure_status(worker.returncode),folder/(name+'.log')))
                time.sleep(.2)
        if not getattr(args,'no_progress',False): drain_logs(folder,[n for n,_ in assignments],offsets)
        for (name,_),worker in zip(assignments,children):
            if worker.returncode!=0:
                raise RuntimeError('%s worker failed: %s; see model log'%(name,failure_status(worker.returncode)))
        results=[json.loads((folder/(name+'.json')).read_text(encoding='utf-8')) for name,_ in assignments]
        starts=[r['runs'][0]['start_unix_s'] for r in results]
        ends=[r['runs'][0]['end_unix_s'] for r in results]
        write_json(folder/'system.json',{'kind':'concurrent_dnn','start_unix_s':max(starts),'end_unix_s':min(ends),
                   'union_start_unix_s':min(starts),'union_end_unix_s':max(ends),'models':[n for n,_ in assignments],
                   'scope':'intersection of three DNN workloads; no codec, no mission scheduling',
                   'device':args.device,'threads_per_worker':args.threads,
                   'calibration_modes':{r['model']:r['input']['calibration_mode'] for r in results},
                   'synthetic_inputs':{r['model']:r['input']['synthetic'] for r in results},
                   'results':[n+'.json' for n,_ in assignments]})
        print('Completed '+str(folder))
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
        for log in logs: log.close()


def sensors_command(args):
    found=sensor_candidates()
    result={'sensors':found,'module_input_available':sum(bool(is_input_sensor(s)) for s in found)==1,
            'note':'read-only INA3221 probe; VDD_IN/POM_5V_IN is module input, not full board power'}
    print(json.dumps(result,indent=2))
    if args.output: write_json(args.output,result)


def record_power_command(args):
    if args.duration<=0: raise ValueError('duration must be positive')
    with PowerRecorder(args.output,args.interval,args.sensor) as recorder:
        print('Recording module %s: %s'%(recorder.sensor['rail'],recorder.sensor['path']),flush=True)
        deadline=time.monotonic()+args.duration
        while time.monotonic()<deadline:
            time.sleep(min(.1,max(0,deadline-time.monotonic())))
            if recorder.error: raise RuntimeError(str(recorder.error))
    report=integrate_power(args.output,recorder.first,recorder.last,max(.05,args.interval*5))
    print(json.dumps({'scope':'module main input','result':report,'metadata':recorder.metadata},indent=2))


def trial(args):
    if args.comp_model=='gru' and args.backend!='float':
        raise ValueError('GRU trial comparison currently requires --backend float')
    # Probe before creating result folder or loading models. Missing sensors never
    # silently produce a timing-only "power measurement".
    if args.record_power: select_sensor(args.power_sensor)
    if not args.record_power:
        return _trial(args)
    folder=Path(args.output).resolve()
    if folder.exists(): raise ValueError('output folder exists; choose a new name')
    with PowerRecorder(folder/'power.csv',args.power_interval,args.power_sensor) as recorder:
        _trial(args,folder_created=True)
    files=attach_recorded_power(folder,folder/'power.csv',recorder.metadata,args.power_interval)
    write_json(folder/'power_results.json',{'files':files,'sensor':recorder.metadata,
               'scope':'main module input, not total carrier-board power'})
    print('Module power analysis complete; see '+str(folder/'power_results.json'),flush=True)


def _trial(args,folder_created=False):
    """Bundled dummy DNNs: separate processes, then three concurrent workloads."""
    if args.duration<=0 or args.repeat<1 or args.warmup<0 or args.threads<1:
        raise ValueError('duration/repeat/threads must be positive; warmup nonnegative')
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; check NVIDIA PyTorch or use --device cpu')
    folder=Path(args.output).resolve()
    if not folder_created: folder.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'dummy_data'/'manifest.json').read_text(encoding='utf-8'))
    model_names=('pd','bus',args.comp_model)
    inputs={n:ROOT/manifest['inputs'][n+'_'+args.backend]['file'] for n in model_names}
    from models import file_sha
    for name,path in inputs.items():
        if file_sha(path)!=manifest['inputs'][name+'_'+args.backend]['sha256']:
            raise ValueError('bundled dummy input hash mismatch: '+str(path))
    common=['--backend',args.backend,'--device',args.device,'--threads',str(args.threads),
            '--duration',str(args.duration),'--warmup',str(args.warmup)]+progress_flags(args)
    outputs=[]
    write_json(folder/'trial_config.json',{'models':list(model_names),'synthetic':True,'backend':args.backend,
        'device':args.device,'phase':args.phase,'repeat':args.repeat,'duration_seconds':args.duration,
        'scope':'DNN only; preprocessing, compression and model loading excluded',
        'power_note':'module INA3221 log enabled' if args.record_power else 'no power capture; supply external CSV or --record-power'})
    def child(extra,logname):
        with open(str(folder/logname),'x',encoding='utf-8') as log:
            print('Running: '+' '.join(extra),flush=True)
            p=subprocess.Popen([sys.executable,'-u','-X','faulthandler',str(ROOT/'run.py')]+extra,
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,bufsize=1)
            try:
                for line in p.stdout:
                    log.write(line); log.flush()
                    print(line,end='',flush=True)
                status=p.wait()
            finally:
                if p.poll() is None:
                    p.terminate()
                    try: p.wait(timeout=5)
                    except subprocess.TimeoutExpired: p.kill(); p.wait()
                if p.stdout: p.stdout.close()
            if status!=0:
                raise RuntimeError('trial worker failed: %s; see %s'%(failure_status(status),folder/logname))
    if args.phase in ('single','all'):
        for name in model_names:
            output=folder/(name+'.json')
            extra=['bench','--model',name,'--input',str(inputs[name]),'--output',str(output),
                   '--repeat',str(args.repeat)]+common
            if args.tegrastats: extra+=['--tegrastats',str(folder/(name+'_health.txt'))]
            child(extra,name+'.log'); outputs.append(output)
    if args.phase in ('together','all'):
        for repeat in range(args.repeat):
            output=folder/('together_%02d'%(repeat+1))
            extra=['together','--pd-input',str(inputs['pd']),'--bus-input',str(inputs['bus']),
                   '--comp-input',str(inputs[args.comp_model]),'--comp-model',args.comp_model,'--output',str(output),
                   '--startup-timeout',str(args.startup_timeout)]+common
            if args.tegrastats: extra+=['--tegrastats']
            child(extra,'together_%02d.log'%(repeat+1)); outputs.append(output/'system.json')
    summary=[]
    for path in outputs:
        data=json.loads(path.read_text(encoding='utf-8'))
        summary.append({'file':str(path.relative_to(folder)),'kind':data['kind'],
                        'model':data.get('model'),'backend':args.backend,
                        'runs':data.get('runs'),'start_unix_s':data.get('start_unix_s'),
                        'end_unix_s':data.get('end_unix_s')})
    write_json(folder/'summary.json',{'synthetic':True,'measurements':summary,
        'note':'attach power to each model JSON or together_NN/system.json; never sum individual power peaks'})
    print('Completed dummy DNN trial: '+str(folder),flush=True)


def parser():
    p=argparse.ArgumentParser(description='SIREN Jetson: one CLI for preparation, models, compression and measurements')
    sub=p.add_subparsers(dest='command'); sub.required=True
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--model',choices=['pd','bus','comp','gru'],required=True)
    common.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    common.add_argument('--backend',choices=['reference','float'],default='reference')
    common.add_argument('--threads',type=int,default=1,help='CPU threads per process, not CPU power-off')
    progress_options=argparse.ArgumentParser(add_help=False)
    progress_options.add_argument('--no-progress',action='store_true',help='disable heartbeat/progress output for final measurements')
    progress_options.add_argument('--progress-interval',type=float,default=2.,help='progress heartbeat interval in seconds')
    c=sub.add_parser('check'); c.add_argument('--output',default='results/environment.json'); c.set_defaults(func=check)
    c=sub.add_parser('ops',parents=[common]); c.set_defaults(func=operations_command)
    c.add_argument('--output',required=True)
    c=sub.add_parser('prepare',parents=[common]); c.set_defaults(func=prepare)
    c.add_argument('--kind',choices=['smoke','dummy','tensor','counts','bus','pd'],required=True)
    c.add_argument('--input'); c.add_argument('--output',required=True)
    c.add_argument('--dtype'); c.add_argument('--valid-rows',type=int); c.add_argument('--expected')
    c.add_argument('--frames',type=int,default=4,help='synthetic frames for --kind dummy')
    c.add_argument('--seed',type=int,default=20261102,help='deterministic dummy seed')
    c.add_argument('--stats'); c.add_argument('--time-column',default='time_s')
    c.add_argument('--demo-calibration',help='explicit arbitrary demo JSON for PD/BUS load tests')
    c.add_argument('--bus-channel',help='select EIRSAT Channel; demo defaults to 15 when column exists')
    for flag in ('tables','state','date','context','records','geo','sun'): c.add_argument('--'+flag)
    c=sub.add_parser('bench',parents=[common,progress_options]); c.set_defaults(func=benchmark)
    c.add_argument('--input',required=True); c.add_argument('--output',required=True)
    c.add_argument('--duration',type=float,default=60); c.add_argument('--repeat',type=int,default=3)
    c.add_argument('--warmup',type=int,default=3); c.add_argument('--tegrastats')
    c.add_argument('--gate',help=argparse.SUPPRESS); c.add_argument('--ready',help=argparse.SUPPRESS)
    c.add_argument('--startup-timeout',type=float,default=600,help=argparse.SUPPRESS)
    c=sub.add_parser('compress',parents=[common]); c.set_defaults(func=compress)
    c.add_argument('--input',required=True); c.add_argument('--output',required=True)
    c.add_argument('--mode',choices=['raw','delta','neural'],default='neural')
    c.add_argument('--block',type=int,default=1800); c.add_argument('--level',type=int,default=5)
    c.add_argument('--limit',type=int); c.add_argument('--tegrastats')
    c=sub.add_parser('decode',parents=[common]); c.set_defaults(func=decode_command)
    c.add_argument('--input',required=True); c.add_argument('--output',required=True)
    c=sub.add_parser('together',parents=[progress_options]); c.set_defaults(func=together)
    for name in ('pd','bus','comp'): c.add_argument('--'+name+'-input',required=True)
    c.add_argument('--comp-model',choices=['comp','gru'],default='comp')
    c.add_argument('--backend',choices=['reference','float'],default='reference')
    c.add_argument('--device',choices=['cpu','cuda'],default='cpu'); c.add_argument('--threads',type=int,default=1)
    c.add_argument('--duration',type=float,default=60); c.add_argument('--warmup',type=int,default=3)
    c.add_argument('--startup-timeout',type=float,default=600); c.add_argument('--tegrastats',action='store_true')
    c.add_argument('--output',required=True)
    c=sub.add_parser('power'); c.set_defaults(func=power)
    c.add_argument('--input',required=True,help='external power CSV, Unix epoch time_s and power_w')
    c.add_argument('--result',required=True); c.add_argument('--output',required=True)
    c.add_argument('--max-gap',type=float,default=.05)
    c=sub.add_parser('sensors'); c.set_defaults(func=sensors_command)
    c.add_argument('--output',help='optional save path for discovered sensors')
    c=sub.add_parser('record-power'); c.set_defaults(func=record_power_command)
    c.add_argument('--output',required=True); c.add_argument('--duration',type=float,default=30.)
    c.add_argument('--interval',type=float,default=.1); c.add_argument('--sensor')
    c=sub.add_parser('trial',parents=[progress_options]); c.set_defaults(func=trial)
    c.add_argument('--comp-model',choices=['comp','gru'],default='comp',help='compression predictor; comp=ModernTCN')
    c.add_argument('--record-power',action='store_true',help='record main-module VDD_IN from read-only INA3221 sysfs')
    c.add_argument('--power-interval',type=float,default=.1,help='sensor polling interval, not conversion bandwidth')
    c.add_argument('--power-sensor',help='explicit VDD_IN path shown by sensors command')
    c.add_argument('--backend',choices=['reference','float'],default='float')
    c.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    c.add_argument('--phase',choices=['single','together','all'],default='all')
    c.add_argument('--duration',type=float,default=60); c.add_argument('--repeat',type=int,default=3)
    c.add_argument('--warmup',type=int,default=3); c.add_argument('--threads',type=int,default=1)
    c.add_argument('--startup-timeout',type=float,default=600)
    c.add_argument('--tegrastats',action='store_true'); c.add_argument('--output',required=True)
    return p


if __name__=='__main__':
    import faulthandler
    faulthandler.enable()
    args=parser().parse_args()
    try:
        if getattr(args,'output',None) and args.command not in ('together','trial') and Path(args.output).exists():
            raise ValueError('output exists; choose a new name: '+args.output)
        args.func(args)
    except (ValueError,RuntimeError,OSError,ImportError,KeyError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        sys.exit(1)
