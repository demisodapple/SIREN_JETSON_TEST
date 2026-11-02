"""Read-only Nano INA3221 acquisition and timestamped power-log analysis.

VDD_IN is main MODULE input, not calibrated carrier-board/DC-jack total power.
Never write sensor configuration, limits or update rates through sysfs.
"""
import csv
import json
import math
from pathlib import Path
import threading
import time


def sensor_candidates(root='/sys'):
    root=Path(root)
    matches=list(root.glob('bus/i2c/drivers/ina3221x/*/iio*device*/rail_name_*'))
    matches+=list(root.glob('bus/iio/devices/iio*device*/rail_name_*'))
    seen=set(); result=[]
    for label_path in sorted(matches):
        number=label_path.name.rsplit('_',1)[-1]
        path=label_path.parent/('in_power%s_input'%number)
        if not path.is_file() or str(path.resolve()) in seen: continue
        seen.add(str(path.resolve()))
        entry={'rail':None,'path':str(path),'source_units':'mW','scale_to_w':.001,
               'readable':False,'power_w':None,'error':None}
        try:
            entry['rail']=label_path.read_text().strip()
            raw=float(path.read_text().strip())
            if not math.isfinite(raw) or raw<0: raise ValueError('invalid power value')
            entry.update(readable=True,power_w=raw*.001)
        except (OSError,ValueError) as exc:
            entry['error']=str(exc)
        result.append(entry)
    return result


def select_sensor(path=None,root='/sys'):
    sensors=sensor_candidates(root)
    candidates=[s for s in sensors if s['readable'] and s['rail']=='VDD_IN']
    if path:
        resolved=str(Path(path).resolve())
        candidates=[s for s in candidates if str(Path(s['path']).resolve())==resolved]
    if len(candidates)!=1:
        raise ValueError('need one readable VDD_IN sensor; run "python3 run.py sensors". '
                         'No sensor power is fabricated. Candidates: '+json.dumps(sensors))
    return candidates[0]


class PowerRecorder:
    def __init__(self,output,interval=.1,sensor_path=None,root='/sys'):
        if not math.isfinite(interval) or interval<.01:
            raise ValueError('power interval must be finite and >=0.01 seconds')
        self.sensor=select_sensor(sensor_path,root)
        self.output=Path(output); self.interval=interval
        if self.output.exists() or Path(str(self.output)+'.json').exists():
            raise ValueError('power CSV/metadata exists; choose a new output path')
        self.stop=threading.Event(); self.thread=None; self.error=None
        self.samples=0; self.first=None; self.last=None; self.previous_power=None
        self.changed=0; self.max_gap=0.; self.handle=None

    def _sample(self):
        a=time.time()
        raw=float(Path(self.sensor['path']).read_text().strip())
        b=time.time(); stamp=(a+b)*.5
        if not math.isfinite(raw) or raw<0: raise ValueError('invalid sensor power')
        watts=raw*self.sensor['scale_to_w']
        if self.last is not None:
            if stamp<=self.last: raise ValueError('wall clock changed backwards; discard measurement')
            self.max_gap=max(self.max_gap,stamp-self.last)
        if self.previous_power is not None and watts!=self.previous_power: self.changed+=1
        if self.first is None: self.first=stamp
        self.last=stamp; self.previous_power=watts; self.samples+=1
        self.writer.writerow([stamp,watts])

    def _run(self):
        deadline=time.monotonic()+self.interval
        try:
            while not self.stop.wait(max(0,deadline-time.monotonic())):
                self._sample()
                if self.samples%max(1,int(1/self.interval))==0: self.handle.flush()
                deadline+=self.interval
                if deadline<time.monotonic(): deadline=time.monotonic()+self.interval
        except Exception as exc:
            self.error=exc

    def __enter__(self):
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.output.open('x',newline='',encoding='utf-8')
        self.writer=csv.writer(self.handle); self.writer.writerow(['time_s','power_w'])
        try:
            self._sample(); self.handle.flush()
        except Exception:
            self.handle.close(); raise
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
        return self

    def __exit__(self,exc_type,*unused):
        self.stop.set()
        self.thread.join()
        try:
            if self.error is None: self._sample()  # bracket the final DNN interval
        except Exception as exc:
            self.error=exc
        finally:
            self.handle.flush(); self.handle.close()
        elapsed=(self.last-self.first) if self.samples>1 else 0
        self.metadata={'sensor':self.sensor,'scope':'main module VDD_IN; not full carrier-board/DC-jack total',
            'requested_poll_interval_s':self.interval,'samples':self.samples,
            'mean_observed_poll_rate_hz':(self.samples-1)/elapsed if elapsed else None,
            'max_observed_gap_s':self.max_gap,'observed_value_changes':self.changed,
            'start_unix_s':self.first,'end_unix_s':self.last,'status':'failed' if self.error else 'complete',
            'error':str(self.error) if self.error else None,
            'note':'read/poll rate is NOT proven sensor conversion bandwidth; peak is sampled maximum; logger adds host load'}
        with open(str(self.output)+'.json','x',encoding='utf-8') as f:
            json.dump(self.metadata,f,indent=2); f.write('\n')
        if self.error and exc_type is None: raise RuntimeError('power acquisition failed: '+str(self.error))


def integrate_power(csv_path, start, end, max_gap=.05):
    """CSV time_s = Unix epoch seconds, power_w = watts; no clock guessing."""
    import numpy as np
    with open(csv_path,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'time_s','power_w'}<=set(reader.fieldnames or []):
            raise ValueError('power CSV requires time_s (Unix epoch seconds), power_w (watts)')
        data=np.asarray([(float(r['time_s']),float(r['power_w'])) for r in reader],np.float64)
    if (len(data)<2 or not np.all(np.isfinite(data)) or np.any(np.diff(data[:,0])<=0)
            or np.any(data[:,1]<0) or end<=start or max_gap<=0): raise ValueError('invalid power samples or interval')
    t,p=data[:,0],data[:,1]
    if start<t[0] or end>t[-1]: raise ValueError('power log does not cover complete measured interval; no extrapolation')
    left=max(0,int(np.searchsorted(t,start,side='right'))-1)
    right=min(len(t)-1,int(np.searchsorted(t,end)))
    gaps=np.diff(t[left:right+1])
    if len(gaps)==0 or np.max(gaps)>max_gap: raise ValueError('power log gap exceeds --max-gap; check sampling and clocks')
    inside=(t>start)&(t<end)
    tt=np.concatenate(([start],t[inside],[end])); pp=np.interp(tt,t,p)
    energy=float(np.sum((pp[:-1]+pp[1:])*.5*np.diff(tt)))
    observed=p[(t>=start)&(t<=end)]
    if len(observed)<1: raise ValueError('no measured sample inside interval')
    return {'energy_j':energy,'mean_power_w':energy/(end-start),'observed_max_power_w':float(observed.max()),
            'observed_samples':int(len(observed)),'median_sample_rate_hz':1/float(np.median(gaps)),
            'max_gap_s':float(np.max(gaps)),'method':'trapezoid with interpolated boundaries; sampled maximum, not transient peak'}


def attach_recorded_power(folder,csv_path,metadata,interval):
    """Attach only valid DNN/concurrent windows after recording has fully stopped."""
    files=[]
    for path in sorted(Path(folder).rglob('*.json')):
        if path.name.endswith('_power.json'): continue
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('kind') not in ('dnn','concurrent_dnn'): continue
        for r in data.get('runs',[data]):
            r['power']=integrate_power(csv_path,r['start_unix_s'],r['end_unix_s'],max(.05,interval*5))
            if r.get('frames'): r['power']['joules_per_frame']=r['power']['energy_j']/r['frames']
        data['power_csv']=str(csv_path); data['power_source']=metadata
        data['power_scope']='main module VDD_IN during measured intervals; includes sensor logger overhead; excludes carrier-board total'
        out=path.with_name(path.stem+'_power.json')
        with out.open('x',encoding='utf-8') as f:
            json.dump(data,f,indent=2,allow_nan=False); f.write('\n')
        files.append(str(out))
    return files
