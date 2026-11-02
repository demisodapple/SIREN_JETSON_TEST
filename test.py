"""CPU verification; synthetic inputs are tests, never mission measurements."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import contextlib
import io
import signal
import time

import numpy as np
import torch

import compression as codec
import models
import preprocess as prep
import reference
import run

torch.set_num_threads(1)
TEST_TMP=Path(__file__).resolve().parent/'results'
TEST_TMP.mkdir(exist_ok=True)


class TestModels(unittest.TestCase):
    def test_float_operations_never_run_inference(self):
        expected={'pd':7537850880,'comp':4613276160}
        for name in ('pd','comp'):
            m=models.Model(name,backend='float')
            with mock.patch.object(m,'forward',side_effect=AssertionError('OPS must not run inference')):
                self.assertEqual(m.operations()['operations_per_frame'],expected[name])
    def test_hash_and_metadata(self):
        for name in models.FILES:
            bundle,digest=models.load_bundle(name)
            self.assertEqual(bundle['model'],name)
            self.assertEqual(len(digest),64)

    def test_real_integer_weights_against_numpy_oracle(self):
        rng=np.random.RandomState(37)
        for name in ('bus','pd','comp'):
            with self.subTest(model=name):
                bundle,_=models.load_bundle(name)
                doc=bundle['contract']
                # BUS temporal pooling requires 64; PD/Comp shorten time only, keep ALL real layers/weights.
                if name!='bus': doc['input_shape'][0]=3
                c=reference.load(doc)
                x=rng.randint(-25,26,size=[1]+c['input_shape']).astype(np.int64)
                expected,_=reference.forward(c,x)
                actual,_=models.ref_forward(c,x,prepared=models.ref_prepare(c))
                np.testing.assert_array_equal(actual.numpy(),expected)

    def test_float_parameters_strict_load_and_causality(self):
        for name in ('pd','comp'):
            m=models.Model(name,backend='float')
            channels=m.input_shape[1]
            x=torch.randn(1,6,channels)
            y=m.network(x)
            z=x.clone(); z[:,4:]+=3
            yy=m.network(z)
            torch.testing.assert_close(y[:,:4],yy[:,:4],rtol=1e-5,atol=1e-6)
            self.assertTrue(bool(torch.isfinite(y).all()))
        m=models.Model('bus',backend='float')
        self.assertEqual(tuple(m.forward(m.tensor(np.zeros((1,64,12),np.float32))).shape),(1,1,1))

    def test_gru_original_checkpoint_step_math(self):
        m=models.Model('gru')
        ck=m.checkpoint
        p=models.Predictor(m,m.channels,m.scale)
        q=np.arange(28,dtype=np.int32)
        aux=np.linspace(-1,1,6).astype(np.float32)
        normalized=(np.log1p((q.astype(np.float64)/m.scale).astype(np.float32))-np.asarray(ck['mean']))/np.asarray(ck['std'])
        x=torch.from_numpy(np.concatenate((normalized,aux)).astype(np.float32)[None,None])
        with torch.no_grad(): f,_=m.network(x)
        expected=np.asarray([b[int(np.clip(np.rint(float(f[0,0,i])*(len(b)-1)),0,len(b)-1))]
                             for i,b in enumerate(ck['codebooks'])],np.int32)
        np.testing.assert_array_equal(p.step(q,aux),expected)
        self.assertEqual(m.operations()['macs_per_frame'],3*64*34+3*64*64+64*28)


class FakePredictor:
    channels=['noise','ramp','forecast']
    scale=8
    model_id='test-only'
    description='synthetic-test'
    predicted_indices=[0,1,2]
    def reset(self): self.steps=0
    def step(self,q,aux):
        self.steps+=1
        return np.asarray([0,0,int(aux[0])],np.int32)


class TestCompression(unittest.TestCase):
    def setUp(self):
        rng=np.random.RandomState(5)
        self.q=np.column_stack((rng.randint(0,256,300),100000+np.arange(300),rng.randint(0,1000000,300))).astype(np.int32)
        self.aux=np.zeros((300,6),np.float32); self.aux[:-1,0]=self.q[1:,2]

    def test_all_candidates_roundtrip_and_packet_accounting(self):
        p=FakePredictor()
        packet,metrics=codec.encode(self.q,self.aux,p,63,5,'neural')
        restored,h=codec.decode(packet,p)
        np.testing.assert_array_equal(restored,self.q)
        self.assertEqual(p.steps,299)
        self.assertEqual(set(metrics['mode_blocks']),{'raw','delta','neural'})
        self.assertEqual(metrics['packet_bytes'],len(packet))
        self.assertEqual(metrics['header_bytes']+metrics['selected_count_payload_bytes']+metrics['auxiliary_payload_bytes'],len(packet))
        self.assertEqual(h['records'][-1]['count'],47)

    def test_baselines_do_not_execute_neural_or_store_aux(self):
        p=models.BaselinePredictor(FakePredictor.channels,8)
        for mode in ('raw','delta'):
            packet,metrics=codec.encode(self.q,self.aux,p,63,5,mode)
            np.testing.assert_array_equal(codec.decode(packet,p)[0],self.q)
            self.assertEqual(metrics['auxiliary_payload_bytes'],0)
            self.assertNotIn('neural',metrics['mode_blocks'])

    def test_only_supported_channels_use_neural_candidate(self):
        p=FakePredictor(); p.predicted_indices=[2]
        packet,_=codec.encode(self.q,self.aux,p,63,5)
        restored,h=codec.decode(packet,p)
        np.testing.assert_array_equal(restored,self.q)
        self.assertTrue(all(r['channel']==2 for r in h['records'] if r['mode']=='neural'))

    def test_corruption_and_wrong_predictor_rejected(self):
        p=FakePredictor(); packet,_=codec.encode(self.q,self.aux,p)
        for corrupt in (packet[:-1],packet+b'x',packet[:-1]+bytes([packet[-1]^1]),b'bad'):
            with self.assertRaises(ValueError): codec.decode(corrupt,p)
        p.description='different-backend'
        with self.assertRaises(ValueError): codec.decode(packet,p)

    def test_real_gru_small_roundtrip(self):
        m=models.Model('gru'); p=models.Predictor(m,m.channels,8)
        q=np.tile(np.arange(28,dtype=np.int32),(12,1))
        aux=np.zeros((len(q),6),np.float32)
        packet,_=codec.encode(q,aux,p,5,5)
        np.testing.assert_array_equal(codec.decode(packet,p)[0],q)

    def test_real_comp_step_roundtrip_small_frame(self):
        m=models.Model('comp')
        # Exact history length is 46; shortened frame preserves causal output at row 46.
        m.contract['input_shape'][0]=47; m.input_shape=[47,24]
        b,_=models.load_bundle('gru'); channels=b['checkpoint']['channels']
        p=models.Predictor(m,channels,8)
        q=np.tile(np.arange(28,dtype=np.int32),(3,1))
        aux=np.zeros((len(q),6),np.float32)
        packet,_=codec.encode(q,aux,p,2,5)
        np.testing.assert_array_equal(codec.decode(packet,p)[0],q)


class TestProgress(unittest.TestCase):
    def test_elapsed_bar_does_not_claim_inference_finished(self):
        p=run.Progress('PD',enabled=False)
        p.stage('measure',target=.01); p.state=('measure',time.monotonic()-1,.01,None)
        p.advance(0)
        text=p.render()
        self.assertIn('99%',text)
        self.assertIn('completed inferences 0',text)
        self.assertIn('waiting for current inference',text)

    def test_unknown_stage_has_heartbeat_and_quiet_mode_is_silent(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            with run.Progress('PD',interval=.01) as p:
                p.stage('weights')
                time.sleep(.04)
        self.assertIn('waiting for stage to finish',out.getvalue())
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            with run.Progress('PD',enabled=False) as p: p.stage('weights')
        self.assertEqual(out.getvalue(),'')
        self.assertIn('SIGILL',run.failure_status(-signal.SIGILL))
        self.assertIn('exit code 2',run.failure_status(2))


class TestPreparationAndPower(unittest.TestCase):
    def test_dummy_bounds_padding_reproducibility_and_identity(self):
        for name in ('pd','bus','comp','gru'):
            with self.subTest(model=name):
                m=models.Model(name,backend='float')
                x,valid,details=prep.dummy_input(m)
                again,_,_=prep.dummy_input(m)
                np.testing.assert_array_equal(x,again)
                self.assertEqual(x.shape[0],4)
                self.assertGreater(np.count_nonzero(x),0)
                self.assertLessEqual(float(np.abs(x).max()),.9)
                different,_,_=prep.dummy_input(m,seed=1)
                self.assertFalse(np.array_equal(x,different))
                if name=='comp': self.assertTrue(np.all(x[:,:,19:]==0))
                if name!='gru':
                    r=models.Model(name,backend='reference')
                    codes,rv,rd=prep.dummy_input(r)
                    np.testing.assert_array_equal(valid,rv)
                    np.testing.assert_array_equal(x,codes[:,:,:x.shape[-1]]*2.0**(-rd['grid_qpos']))
                    if name=='pd': self.assertTrue(np.all(codes[:,:,28:]==0))
                with self.assertRaises(ValueError): prep.dummy_input(m,frames=0)
                with self.assertRaises(ValueError): prep.dummy_input(m,seed=-1)

    def test_bundled_dummy_inputs_hashes_and_metadata(self):
        root=Path(__file__).parent
        manifest=json.loads((root/'dummy_data/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['inputs']),7)
        for key,entry in manifest['inputs'].items():
            path=root/entry['file']; self.assertEqual(models.file_sha(path),entry['sha256'])
            name,backend=key.split('_')
            m=models.Model(name,backend=backend)
            tensor,meta,expected=prep.load_input(path,m)
            self.assertEqual(list(tensor.shape),entry['shape'])
            self.assertTrue(meta['synthetic'])
            self.assertEqual(meta['calibration_mode'],'synthetic_dummy')
            self.assertIsNone(expected)
            self.assertEqual(entry['operations'],m.operations())

    def test_explicit_demo_calibration_and_result_metadata(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_TMP)) as temp:
            root=Path(temp); profile=Path(__file__).parent/'demo_calibration.json'
            p=models.Model('pd'); day='20220101'
            summary=np.ones((288,36),np.float32)
            summary[:,prep._C['n_valid']]=150; summary[:,prep._C['ifc']]=0
            np.save(str(root/'summary.npy'),summary)
            args=SimpleNamespace(input=str(root/'summary.npy'),output=str(root/'pd.npz'),date=day,
                    demo_calibration=str(profile),tables=None,state=None,context=None,records=None,geo=None,sun=None)
            prep.prepare_pd(args,p)
            x,meta,expected=prep.load_input(args.output,p)
            self.assertEqual(tuple(x.shape),(2,310,32))
            self.assertEqual(meta['calibration_mode'],'arbitrary_demo')
            self.assertFalse(meta['preparation_details']['accuracy_validated'])
            self.assertFalse(meta['synthetic'])  # raw/supplied observations, arbitrary calibration
            self.assertIsNone(expected)
            self.assertEqual(len(meta['calibration_sha256']),64)
            args.tables='real.bin'
            with self.assertRaises(ValueError): prep.prepare_pd(args,p)

    def test_demo_bus_channel_filter_and_iso_time(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_TMP)) as temp:
            root=Path(temp); profile=Path(__file__).parent/'demo_calibration.json'
            bus_profile=prep.read_demo_calibration(profile)['bus']
            names=bus_profile['channels']
            path=root/'raw.csv'
            with path.open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['OBT','Channel']+names)
                for i in range(80):
                    w.writerow([i*50,15]+bus_profile['std'])
                    w.writerow([i*50,16]+[999]*12)
            args=SimpleNamespace(input=str(path),output=str(root/'bus.npz'),stats=None,
                                 time_column='auto',demo_calibration=str(profile),bus_channel=None)
            m=models.Model('bus'); prep.prepare_bus(args,m)
            x,meta,_=prep.load_input(args.output,m)
            self.assertTrue(bool((x==16).all()))
            self.assertEqual(meta['preparation_details']['bus_channel'],'15')
            self.assertEqual(meta['calibration_mode'],'arbitrary_demo')
            self.assertEqual(prep.time_seconds('2022-01-01T00:00:00Z'),1640995200.)

    def test_demo_netcdf_noaa_time_metadata_and_missing_geometry(self):
        try:
            from netCDF4 import Dataset
        except ImportError:
            self.skipTest('install requirements.txt to test netCDF adapter')
        with tempfile.TemporaryDirectory(dir=str(TEST_TMP)) as temp:
            path=Path(temp)/'l1b.nc'
            with Dataset(str(path),'w') as ds:
                ds.createDimension('time',150)
                t=ds.createVariable('time','u8',('time',))
                t.units='time'; t.long_name='milliseconds since 1970'; t.valid_range=np.asarray([0,0],np.uint64)
                t[:]=1640995200000+np.arange(150)*2000
                for name in (['mep_pro_tel0_flux_p%d'%i for i in range(1,7)]+
                             ['mep_ele_tel0_flux_e%d'%i for i in range(1,4)]):
                    ds.createVariable(name,'f4',('time',))[:]=3
            profile=prep.read_demo_calibration(Path(__file__).parent/'demo_calibration.json')['pd']
            summary,date,details=prep.l1b_demo_summary(path,profile)
            self.assertEqual(date,'20220101')
            self.assertEqual(summary[0,0],3)
            self.assertEqual(summary[0,prep._C['n_valid']],150)
            self.assertIn('abs_mlat',details['placeholder_geometry'])

    def test_counts_preparation_and_bus_statistics(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_TMP)) as temp:
            root=Path(temp)
            g=models.Model('gru')
            counts=np.tile(np.arange(28,dtype=np.float32)/8,(8,1))
            source=root/'counts.npz'
            np.savez(str(source),counts=counts,auxiliary=np.zeros((8,6),np.float32),channels=np.asarray(g.channels))
            for name in ('gru','comp'):
                m=g if name=='gru' else models.Model(name)
                path=root/(name+'.npz')
                prep.prepare_counts(SimpleNamespace(input=str(source),output=str(path)),m)
                _,info,_=prep.load_input(path,m)
                self.assertEqual(sum(info['valid_rows']),7)
            m=models.Model('bus')
            columns=['channel%d'%i for i in range(12)]
            stats=root/'stats.npz'
            np.savez(str(stats),channels=columns,mean=np.ones(12),std=np.ones(12)*2)
            path=root/'bus.csv'
            with path.open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['time_s']+columns)
                for i in range(80): w.writerow([i*50]+[3]*12)
            args=SimpleNamespace(input=str(path),output=str(root/'bus.npz'),stats=str(stats),time_column='time_s')
            prep.prepare_bus(args,m)
            x,info,_=prep.load_input(args.output,m)
            self.assertEqual(tuple(x.shape),(2,64,12))
            self.assertTrue(bool((x==16).all()))
            args.stats=None
            with self.assertRaises(ValueError): prep.prepare_bus(args,m)

    def test_pd_day_pipeline_state_and_context(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_TMP)) as temp:
            root=Path(temp); m=models.Model('pd')
            date='20220101'; day=prep._day_number(date)
            cols=[0,1,2,3,4,5,12,13,14]+list(range(23,31))+[54,55]+[56,57,58,59,60,61,68,69,70]+[prep.PAD_COL]*4
            tables=prep.tables_bytes(np.ones(23),{2021:np.ones((18,23))},cols,np.zeros(32),np.ones(32),9,16)
            (root/'tables.bin').write_bytes(tables)
            np.savez(str(root/'state.npz'),days=np.arange(day-5,day),medians=np.ones((5,18,23),np.float32))
            summary=np.ones((288,36),np.float32)
            summary[:,prep._C['n_valid']]=150; summary[:,prep._C['ifc']]=0
            summary[:,prep._C['abs_mlat']]=65
            np.save(str(root/'summary.npy'),summary)
            np.save(str(root/'context.npy'),np.zeros((288,32),np.int16))
            args=SimpleNamespace(input=str(root/'summary.npy'),output=str(root/'pd.npz'),date=date,
                                 tables=str(root/'tables.bin'),state=str(root/'state.npz'),context=str(root/'context.npy'),
                                 records=None,geo=None,sun=None)
            prep.prepare_pd(args,m)
            x,info,_=prep.load_input(args.output,m)
            self.assertEqual(tuple(x.shape),(2,310,32))
            self.assertEqual(info['valid_rows'],[264,24])
            with np.load(str(root/'pd.state.npz')) as s: self.assertIn(day,s['days'])

    def test_netcdf_raw_counts(self):
        try:
            from netCDF4 import Dataset
        except ImportError:
            self.skipTest('install requirements.txt to test netCDF adapter')
        with tempfile.TemporaryDirectory(dir=str(TEST_TMP)) as temp:
            path=Path(temp)/'day.nc'
            with Dataset(str(path),'w') as ds:
                ds.createDimension('time',4)
                for name,values in [('count',[0,.125,.25,1]),('lat',[-20,-10,0,10]),
                                    ('lon',[100,101,102,103]),('sat_direction',[0,0,0,0])]:
                    ds.createVariable(name,'f4',('time',))[:]=values
            q,aux=prep.read_counts(path,['count'],8)
            np.testing.assert_array_equal(q[:,0],[0,1,2,8])
            np.testing.assert_array_equal(aux[:,4],0)
            np.testing.assert_allclose(aux[:,0],np.sin(np.deg2rad([-20,-10,0,10])),atol=1e-7)

    def test_grid_and_frames(self):
        np.testing.assert_array_equal(prep.exact_q(np.asarray([0,.125,10],np.float32),8),[0,1,80])
        for bad in ([.1],[np.nan],[-1],[1e15]):
            with self.assertRaises(ValueError): prep.exact_q(np.asarray(bad,np.float32),8)
        rows=np.arange(10,dtype=np.int16)[:,None]
        x,valid=prep.framed(rows,6,2)
        np.testing.assert_array_equal(valid,[4,4,2])
        np.testing.assert_array_equal(x[0,:,0],[0,0,0,1,2,3])
        np.testing.assert_array_equal(x[1,:,0],[2,3,4,5,6,7])

    def test_input_identity_and_dtype(self):
        m=models.Model('bus')
        with tempfile.TemporaryDirectory(dir=str(TEST_TMP)) as temp:
            path=Path(temp)/'in.npz'
            prep.save_input(path,np.zeros((1,64,12),np.int16),m,[16],'test',True)
            x,info,_=prep.load_input(path,m)
            self.assertTrue(info['synthetic']); self.assertEqual(tuple(x.shape),(1,64,12))
            with self.assertRaises(ValueError): m.tensor(np.zeros((1,64,12),np.float32))
            m.sha='wrong'
            with self.assertRaises(ValueError): prep.load_input(path,m)

    def test_pd_history_rejects_future_and_uses_previous_days(self):
        state={d:np.ones((18,23),np.float32)*d for d in range(10)}
        bg=prep.background(state,9,history=5,min_periods=5)
        np.testing.assert_array_equal(bg,np.full((18,23),6.))

    def test_power_linear_integral_and_clock_boundaries(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_TMP)) as temp:
            path=Path(temp)/'power.csv'
            with path.open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['time_s','power_w'])
                for t in np.linspace(100,102,201): w.writerow([t,2+(t-100)])
            result=run.integrate_power(path,100.2,101.8)
            self.assertAlmostEqual(result['energy_j'],4.8)
            self.assertAlmostEqual(result['mean_power_w'],3)
            with self.assertRaises(ValueError): run.integrate_power(path,99,101)
            with self.assertRaises(ValueError): run.integrate_power(path,100,101,.001)


if __name__=='__main__': unittest.main(verbosity=2)
