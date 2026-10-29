"""All model loading and CPU/CUDA inference; reference != native INT8 acceleration."""
import functools
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import reference as ic
from reference import need

"""Bit-exact PyTorch port of the mixed integer contract forward (``integer_contract.forward_mixed``).

Covers ``pointwise``, ``depthwise`` (SAME, VALID or CAUSAL), ``residual``, ``layer_norm`` and ``gelu_pwl``
layers of a contract loaded by ``integer_contract.load``, on CPU or CUDA. Every value is int64 with the
oracle's wrap-around, rounding and saturation; matrix products go through float64 only in limbs whose
partial sums stay below 2^53, where any summation order is exact.
"""
KINDS = {'pointwise', 'depthwise', 'residual', 'layer_norm', 'gelu_pwl', 'stride_mean'}
EXACT = (1 << 53) - 1
SQRT_LIMIT = 3 << 61
LN_GROUP = 8
I64 = torch.int64

def ref__int64(values, device=None):
    if torch.is_tensor(values):
        return values.to(device=device, dtype=I64)
    return torch.as_tensor(np.array(values, dtype=np.int64), device=device)

def ref__sign(values):
    return 1 - 2 * (values < 0).to(I64)

def ref_rne_shift(values, shift):
    """Right shift with ties to even (int64; shift 0..62)."""
    if shift == 0:
        return values.clone()
    base = values >> shift
    rest = values - (base << shift)
    half = 1 << shift - 1
    return base + ((rest > half) | (rest == half) & (base & 1 != 0)).to(I64)

def ref_rne_divide(numerator, denominator):
    """Floor-division based rounding to nearest, ties to even, over a positive denominator (as the oracle)."""
    q = torch.div(numerator, denominator, rounding_mode='floor')
    r = numerator - q * denominator
    return q + ((2 * r > denominator) | (2 * r == denominator) & (q & 1 != 0)).to(I64)

def ref_integer_sqrt(values):
    """Floor square root of int64 values in [0, 3 * 2^61): float64 estimate, exact integer correction."""
    if values.numel():
        need(int(values.min()) >= 0 and int(values.max()) < SQRT_LIMIT, 'integer_sqrt operand out of range')
    root = torch.floor(torch.sqrt(values.to(torch.float64))).to(I64)
    while True:
        high = root * root > values
        low = (root + 1) * (root + 1) <= values
        if not bool((high | low).any()):
            return root
        root = root - high.to(I64) + low.to(I64)

def ref__matmul(x, weights, peak):
    """``x @ w`` for int64 ``x`` and float64 integer weights of peak magnitude ``peak``, in exact limbs."""
    out = torch.zeros((x.shape[0], weights.shape[1]), dtype=I64, device=x.device)
    if x.numel() == 0 or weights.numel() == 0 or peak == 0:
        return out
    limit = EXACT // (peak * x.shape[1])
    need(limit >= 2, 'matrix too wide or weights too large for exact float64 limbs')
    bits = limit.bit_length() - 1
    shift, rest = (0, x)
    while max(-int(rest.min()), int(rest.max())) > limit:
        low = rest & (1 << bits) - 1
        out = out + ((low.to(torch.float64) @ weights).to(I64) << shift)
        rest, shift = (rest >> bits, shift + bits)
    return out + ((rest.to(torch.float64) @ weights).to(I64) << shift)

def ref_exact_matmul(x, w):
    """``x @ w`` of int64 matrices, exact (``integer_contract.exact_matmul``)."""
    return ref__matmul(x, w.to(torch.float64), int(w.abs().max()) if w.numel() else 0)

@functools.lru_cache(maxsize=None)
def ref__gelu_arrays():
    return ic.gelu_table()

def ref_gelu_table(device=None):
    """The contract's PWL GELU bases and deltas as int64 tensors."""
    bases, deltas = ref__gelu_arrays()
    return (ref__int64(bases, device), ref__int64(deltas, device))

def ref__shift_entry(shift, device):
    shift = np.asarray(shift, np.int64)
    rounding = np.where(shift > 0, np.left_shift(np.int64(1), np.maximum(shift - 1, 0)), 0)
    return dict(shift=ref__int64(shift, device), rounding=ref__int64(rounding, device))

def ref__prepare_layer(layer, device):
    kind = layer['kind']
    need(kind in KINDS, '%s: layer kind %s is not supported by the torch forward' % (layer['id'], kind))
    if kind == 'layer_norm':
        return dict(gamma=ref__int64(layer['gamma'], device), beta=ref__int64(layer['beta'], device))
    if kind == 'gelu_pwl':
        return {}
    entry = ref__shift_entry(layer['shift'], device)
    if kind == 'residual':
        return entry
    w = layer['weights'][0]
    entry['bias'] = ref__int64(layer['bias'], device)
    if kind == 'depthwise':
        entry['taps'] = ref__int64(w[:, :, 0], device)
    else:
        entry['taps'] = [(torch.as_tensor(w[k].astype(np.float64), device=device), int(np.abs(w[k]).max())) for k in range(w.shape[0])]
    return entry

def ref_prepare(contract, device=None):
    """Device tensors of every layer of a loaded mixed contract, reusable across ``forward`` calls."""
    need(contract['schema'] in ic.MIXED_SCHEMAS, 'the torch forward needs a mixed contract')
    device = torch.device(device if device is not None else 'cpu')
    return dict(device=device, table=ref_gelu_table(device), layers={layer['id']: ref__prepare_layer(layer, device) for layer in contract['layers']})

def ref_accumulate(layer, x, entry=None):
    """Exact MAC accumulator of a pointwise or depthwise layer for (N, T, C) codes, tap by tap."""
    entry = ref__prepare_layer(layer, x.device) if entry is None else entry
    _, kernel, _, _ = layer['weights'].shape
    step = layer['stride'][2]
    left = right = 0
    if layer['padding'] == 'SAME' and kernel > 1:
        left = right = kernel // 2
    elif layer['padding'] == 'CAUSAL':
        left = kernel - 1
    if left or right:
        n, _, c = x.shape
        x = torch.cat([x.new_zeros((n, left, c)), x, x.new_zeros((n, right, c))], dim=1)
    length = (x.shape[1] - kernel) // step + 1
    acc = None
    for k in range(kernel):
        window = x[:, k:k + step * (length - 1) + 1:step, :]
        if layer['kind'] == 'depthwise':
            term = window * entry['taps'][k]
        else:
            weights, peak = entry['taps'][k]
            term = ref__matmul(window.reshape(-1, window.shape[2]), weights, peak).reshape(window.shape[0], length, -1)
        acc = term if acc is None else acc + term
    return acc

def ref_requant(layer, values, entry=None):
    """``integer_contract._requant_mixed``: ReLU if set, per-channel right shift with ties up, saturation."""
    entry = ref__shift_entry(layer['shift'], values.device) if entry is None else entry
    if layer['relu']:
        values = torch.clamp_min(values, 0)
    values = values + entry['rounding'] >> entry['shift']
    low, high = ic.signed_range(layer['output_bits'])
    return torch.clamp(values, low, high)

def ref_layer_norm(layer, x, entry=None):
    """``integer_contract.layer_norm`` (same rounding, grouping of the squares and saturation)."""
    if entry is None:
        entry = dict(gamma=ref__int64(layer['gamma'], x.device), beta=ref__int64(layer['beta'], x.device))
    channels = x.shape[-1]
    total = x.sum(dim=-1, keepdim=True)
    mean = ref_rne_divide(total.abs(), channels) * ref__sign(total)
    centered = x - mean
    padded = torch.cat([centered, centered.new_zeros(x.shape[:-1] + (-channels % LN_GROUP,))], dim=-1)
    partial = (padded * padded).reshape(x.shape[:-1] + (-1, LN_GROUP)).sum(dim=-1)
    quotient = torch.div(partial, channels, rounding_mode='floor')
    remainder = (partial - quotient * channels).sum(dim=-1)
    carry = torch.div(remainder, channels, rounding_mode='floor')
    rest = remainder - carry * channels
    base = quotient.sum(dim=-1) + carry
    variance = base + ((2 * rest > channels) | (2 * rest == channels) & (base & 1 != 0)).to(I64)
    root = ref_integer_sqrt(variance + layer['epsilon_code']).unsqueeze(-1)
    normalized = ref_rne_divide(centered.abs() << ic.LN_NORMAL_QPOS, root) * ref__sign(centered)
    normalized = torch.clamp(normalized, -(1 << 31), (1 << 31) - 1)
    accumulator = normalized * entry['gamma'] + (entry['beta'] << ic.LN_NORMAL_QPOS)
    out = ref_rne_shift(accumulator, ic.LN_NORMAL_QPOS + ic.LN_AFFINE_QPOS - layer['output_qpos'])
    low, high = ic.signed_range(layer['output_bits'])
    return torch.clamp(out, low, high)

def ref_gelu_pwl(layer, x, table=None):
    """``integer_contract.gelu_pwl`` (PWL table, RNE interpolation and output shift, saturation)."""
    bases, deltas = ref_gelu_table(x.device) if table is None else table
    iq, oq = (layer['input_qpos'], layer['output_qpos'])
    magnitude = x.abs()
    fraction_bits = iq - ic.GELU_GRID_BITS
    address = torch.clamp(magnitude >> fraction_bits, 0, ic.GELU_ENTRIES - 1)
    fraction = magnitude & (1 << fraction_bits) - 1
    correction = bases[address] + ref_rne_shift(deltas[address] * fraction, fraction_bits)
    correction = torch.where(magnitude >= 8 << iq, 0, correction)
    y = correction + torch.where(x < 0, 0, magnitude << ic.GELU_TABLE_QPOS - iq)
    low, high = ic.signed_range(layer['output_bits'])
    return torch.clamp(ref_rne_shift(y, ic.GELU_TABLE_QPOS - oq), low, high)

@torch.no_grad()
def ref_forward(contract, codes, device=None, prepared=None):
    """Run a loaded mixed contract on (N, T, C) input codes; return the final codes and every layer's codes
    as int64 tensors, equal to ``integer_contract.forward_mixed`` bit for bit. ``device`` defaults to the
    codes' device (CPU for arrays); a ``prepare`` result fixes the device instead."""
    if prepared is None:
        if device is None:
            device = codes.device if torch.is_tensor(codes) else 'cpu'
        prepared = ref_prepare(contract, device)
    x = ref__int64(codes, prepared['device'])
    low, high = ic.signed_range(contract['input_bits'])
    need(x.ndim == 3 and list(x.shape[1:]) == contract['input_shape'], 'inputs must be (N, time, channels)')
    need(int(x.min()) >= low and int(x.max()) <= high, 'input codes outside the input width')
    values = {'input': x}
    last_use = {}
    for i, layer in enumerate(contract['layers']):
        for key in (layer['input_id'], layer.get('branch_id')):
            if key is not None:
                last_use[key] = i
    for i, layer in enumerate(contract['layers']):
        for key in list(values):
            if last_use.get(key, i) < i:
                del values[key]
        lid, entry = (layer['id'], prepared['layers'][layer['id']])
        source = values[layer['input_id']]
        if layer['kind'] == 'layer_norm':
            values[lid] = ref_layer_norm(layer, source, entry)
            continue
        if layer['kind'] == 'gelu_pwl':
            values[lid] = ref_gelu_pwl(layer, source, prepared['table'])
            continue
        acc_low, acc_high = ic.signed_range(layer['accumulator_bits'])
        if layer['kind'] == 'residual':
            q = max(layer['input_qpos'], layer['branch_qpos'])
            acc = (source << q - layer['input_qpos']) + (values[layer['branch_id']] << q - layer['branch_qpos'])
        else:
            acc = ref_accumulate(layer, source, entry)
        need(int(acc.min()) >= acc_low and int(acc.max()) <= acc_high, lid + ': accumulator overflow')
        values[lid] = ref_requant(layer, acc if layer['kind'] == 'residual' else acc + entry['bias'], entry)
    return (values[contract['layers'][-1]['id']], values)


ROOT = Path(__file__).resolve().parent
FILES = {'pd': 'model1_PD_weight.pth', 'bus': 'model2_BUS_weight.pth',
         'comp': 'model3_COMP_ModernTCN_weight.pth', 'gru': 'model3_COMP_GRU_weight.pth'}


def file_sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_bundle(name):
    path = ROOT / 'weights' / FILES[name]
    manifest = json.loads((ROOT / 'SOURCES.json').read_text(encoding='utf-8'))
    need(file_sha(path) == manifest['models'][name]['sha256'], 'weight SHA-256 mismatch: ' + str(path))
    # Only load repository-owned, hash-checked bundles. Old Jetson torch has no weights_only parameter.
    import inspect
    kw = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kw['weights_only'] = False
    d = torch.load(str(path), **kw)
    need(d['schema'] == 'siren-jetson-model-v1' and d['model'] == name, 'wrong model bundle')
    return d, manifest['models'][name]['sha256']


class CausalBlock(nn.Module):
    def __init__(self, width, kernel, small, expansion):
        super().__init__()
        self.token_norm = nn.LayerNorm(width, eps=1e-5)
        self.large_depthwise = nn.Conv1d(width, width, kernel, groups=width)
        self.small_depthwise = nn.Conv1d(width, width, small, groups=width)
        self.large_left_padding, self.small_left_padding = kernel - 1, small - 1
        self.token_scale = nn.Parameter(torch.ones(width))
        self.ffn_norm = nn.LayerNorm(width, eps=1e-5)
        self.ffn = nn.Sequential(nn.Conv1d(width, expansion, 1), nn.GELU(),
                                 nn.Identity(), nn.Conv1d(expansion, width, 1))
        self.ffn_scale = nn.Parameter(torch.ones(width))

    def forward(self, x):
        y = self.token_norm(x.transpose(1, 2)).transpose(1, 2)
        y = (self.large_depthwise(F.pad(y, (self.large_left_padding, 0))) +
             self.small_depthwise(F.pad(y, (self.small_left_padding, 0))))
        x = x + y * self.token_scale[None, :, None]
        y = self.ffn(self.ffn_norm(x.transpose(1, 2)).transpose(1, 2))
        return x + y * self.ffn_scale[None, :, None]


class ModernTCN(nn.Module):
    """FP32 checkpoint-parameter port of pd_contract.fold topology; no QAT emulation.

    This explicit alternative needs real reference-output validation before scientific use.
    The default contract backend below preserves the published integer computations.
    """
    def __init__(self, state):
        super().__init__()
        width, cin, _ = state['input_projection.weight'].shape
        self.input_projection = nn.Conv1d(cin, width, 1)
        blocks = []
        i = 0
        while 'blocks.%d.token_scale' % i in state:
            prefix = 'blocks.%d.' % i
            blocks.append(CausalBlock(width, state[prefix+'large_depthwise.weight'].shape[-1],
                          state[prefix+'small_depthwise.weight'].shape[-1], state[prefix+'ffn.0.weight'].shape[0]))
            i += 1
        self.blocks = nn.ModuleList(blocks)
        self.output_norm = nn.LayerNorm(width, eps=1e-5)
        mid = state['risk_head.0.weight'].shape[0]
        cout = state['risk_head.3.weight'].shape[0]
        self.risk_head = nn.Sequential(nn.Conv1d(width, mid, 1), nn.GELU(), nn.Identity(), nn.Conv1d(mid, cout, 1))
        self.load_state_dict(state, strict=True)

    def forward(self, x):
        x = self.input_projection(x.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        return self.risk_head(self.output_norm(x.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)


class BUSFloat(nn.Module):
    """Unquantized checkpoint parameters: diagnostic FP32 baseline, not W8A8 inference."""
    def __init__(self, checkpoint, contract):
        super().__init__()
        self.graph = contract['layers']
        self.weight, self.bias = nn.ParameterDict(), nn.ParameterDict()
        for key, value in checkpoint['state'].items():
            group, name = key.split('.', 1)
            getattr(self, group)[name] = nn.Parameter(value.float(), requires_grad=False)

    def forward(self, x):
        values = {'input': x}
        for layer in self.graph:
            name, kind = layer['id'], layer['kind']
            v = values[layer['input_id']]
            if kind == 'residual':
                out = v + values[layer['branch_id']]
            else:
                w = self.weight[name][0]
                if kind == 'depthwise':
                    out = F.conv1d(v.transpose(1, 2), w[:, :, 0].T[:, None], self.bias[name],
                                   padding=w.shape[0]//2, groups=v.shape[-1]).transpose(1, 2)
                elif kind == 'stride_mean':
                    out = F.conv1d(v.transpose(1, 2), w.permute(2, 1, 0), self.bias[name],
                                   stride=layer['stride'][2]).transpose(1, 2)
                else:
                    out = v @ w[0] + self.bias[name]
            values[name] = torch.relu(out) if layer['relu'] else out
        return values[self.graph[-1]['id']]


class StageGRU(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.gru = nn.GRU(channels+6, hidden, batch_first=True)
        self.output = nn.Linear(hidden, channels)

    def forward(self, x, hidden=None):
        seq, hidden = self.gru(x, hidden)
        return torch.sigmoid(self.output(seq)), hidden


class Model:
    def __init__(self, name, device='cpu', backend='reference'):
        need(name in FILES, 'unknown model')
        need(backend in ('reference', 'float'), 'backend must be reference or float')
        need(device in ('cpu', 'cuda'), 'device must be cpu or cuda')
        need(device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable in this Python environment')
        bundle, self.sha = load_bundle(name)
        self.name, self.device, self.backend = name, torch.device(device), backend
        self.checkpoint = bundle['checkpoint']
        self.contract = ic.load(bundle['contract']) if bundle['contract'] else None
        self.hidden = None
        if name == 'gru':
            ck = self.checkpoint
            self.channels = list(ck['channels'])
            self.scale = int(ck['config']['quantization_scale'])
            self.network = StageGRU(len(self.channels), int(ck['config']['hidden_dim']))
            self.network.load_state_dict(ck['state_dict'], strict=True)
            self.backend = 'float'
            self.input_shape = [1, len(self.channels)+6]
            self.description = 'GRU checkpoint FP32, sequential state'
        else:
            self.input_shape = self.contract['input_shape']
            self.channels = self.contract.get('source', {}).get('channels', [])
            self.scale = 8  # count grid for the supplied SEM2 GRU/Comp comparison
            if backend == 'reference':
                self.prepared = ref_prepare(self.contract, self.device)
                self.network = None
                self.description = 'integer-contract reference; INT64 and exact FP64 limbs, NOT native INT8'
            else:
                if name == 'bus':
                    self.network = BUSFloat(self.checkpoint, self.contract)
                else:
                    state = self.checkpoint['model'] if name == 'pd' else {
                        k[2:]: v for k, v in self.checkpoint['state'].items()}
                    self.network = ModernTCN(state)
                    self.input_shape = [310, state['input_projection.weight'].shape[1]]
                self.description = 'FP32 checkpoint-parameter port; QAT disabled; task accuracy not yet validated'
        if self.network is not None:
            self.network = self.network.float().to(self.device).eval()

    def reset(self):
        self.hidden = None

    def tensor(self, x):
        arr = np.asarray(x)
        need(arr.ndim == 3 and list(arr.shape[1:]) == self.input_shape, 'input must be [N,%s]' % self.input_shape)
        need(len(arr) > 0 and np.all(np.isfinite(arr)), 'empty/nonfinite input')
        if self.backend == 'reference':
            need(arr.dtype.kind in 'iu', 'reference backend requires integer codes, not normalized floats')
            low, high = ic.signed_range(self.contract['input_bits'])
            need(arr.min() >= low and arr.max() <= high, 'input codes outside width')
        return torch.as_tensor(arr, dtype=torch.int64 if self.backend == 'reference' else torch.float32, device=self.device)

    @torch.no_grad()
    def forward(self, x):
        if self.name == 'gru':
            out, self.hidden = self.network(x, self.hidden)
            return out
        if self.backend == 'reference':
            return ref_forward(self.contract, x, prepared=self.prepared)[0]
        return self.network(x)

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def operations(self):
        """Per batch=1 frame; matrix/conv MACs only, not a count of all instructions."""
        if self.name == 'gru':
            c = len(self.channels)
            h = int(self.checkpoint['config']['hidden_dim'])
            macs = 3*h*(c+6) + 3*h*h + h*c
            excluded = 'bias, sigmoid/tanh, gate elementwise products, preprocessing'
        elif self.backend == 'reference':
            macs = 0
            for layer in self.contract['layers']:
                if layer['kind'] in ic.MAC_KINDS:
                    _, k, cin, cout = layer['weights'].shape
                    macs += layer['output_shape'][0] * k * cin * cout
            excluded = 'bias, residual add, LayerNorm, GELU, requantization, emulation overhead'
        else:
            total = [0]
            handles = []
            def hook(module, args, output):
                total[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
            if self.name == 'bus':
                macs = sum(l['output_shape'][0] * int(np.prod(l['weights'].shape))
                           for l in self.contract['layers'] if l['kind'] in ic.MAC_KINDS)
            else:
                for module in self.network.modules():
                    if isinstance(module, nn.Conv1d):
                        handles.append(module.register_forward_hook(hook))
                try:
                    self.forward(self.tensor(np.zeros([1]+self.input_shape, np.float32)))
                    macs = total[0]
                finally:
                    for handle in handles:
                        handle.remove()
            excluded = 'bias, residual add, LayerNorm, GELU, preprocessing'
        return {'macs_per_frame': int(macs), 'operations_per_frame': int(2*macs),
                'convention': '1 MAC = 2 operations; matrix/conv only', 'excluded': excluded}


class Predictor:
    """Same selected output file for GRU and Comp; unsupported channels use raw/delta.

    Sequential, same-shape inference is intentional for reversible float predictions.
    Compression timing includes these calls and input transformations.
    """
    def __init__(self, model, channels, scale):
        need(model.name in ('gru', 'comp'), 'compression needs gru or comp')
        self.m, self.channels, self.scale = model, list(channels), int(scale)
        need(len(set(channels)) == len(channels), 'duplicate output channels')
        need(self.scale == model.scale, 'count scale mismatch')
        need(all(c in channels for c in model.channels), 'input must include every trained model channel')
        self.predicted_indices = [self.channels.index(c) for c in model.channels]
        self.model_id = model.sha
        self.description = model.name + ':' + model.backend + ':' + str(model.device) + ':step-v1'
        if model.name == 'gru':
            ck = model.checkpoint
            self.mean, self.std = np.asarray(ck['mean'], np.float32), np.asarray(ck['std'], np.float32)
            self.books = [np.asarray(b, np.int32) for b in ck['codebooks']]
        else:
            src = model.contract['source']
            self.mean, self.std = np.asarray(src['mean']), np.asarray(src['std'])
            self.rf = sum(l['weight_shape'][1]-1 for l in model.contract['layers'] if l['kind']=='depthwise')
        self.reset()

    def reset(self):
        self.m.reset()
        if self.m.name == 'comp':
            self.history = np.zeros((self.rf+1, self.m.input_shape[1]),
                                    np.int16 if self.m.backend=='reference' else np.float32)

    def step(self, previous_q, auxiliary):
        from preprocess import comp_rows, comp_output_counts
        q = np.asarray(previous_q)[self.predicted_indices]
        if self.m.name == 'gru':
            counts = (q.astype(np.float64)/self.scale).astype(np.float32)
            x = np.concatenate(((np.log1p(counts)-self.mean)/self.std, auxiliary)).astype(np.float32)
            frac = self.m.forward(self.m.tensor(x[None, None]))[0, 0].cpu().numpy()
            need(np.all(np.isfinite(frac)), 'nonfinite GRU prediction')
            pred = [b[int(np.clip(np.rint(frac[i]*(len(b)-1)), 0, len(b)-1))] for i,b in enumerate(self.books)]
        else:
            row = comp_rows(q[None], np.asarray(auxiliary)[None], self.m)[0]
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = row
            frame = np.zeros([1]+self.m.input_shape, self.history.dtype)
            frame[0, :len(self.history)] = self.history
            out = self.m.forward(self.m.tensor(frame))[0, self.rf].cpu().numpy()
            pred = comp_output_counts(out, self.m)
        result = np.zeros(len(self.channels), np.int32)
        result[self.predicted_indices] = pred
        return result


class BaselinePredictor:
    def __init__(self, channels, scale):
        self.channels, self.scale = list(channels), int(scale)
        self.model_id, self.description = 'none', 'no-neural-network'
        self.predicted_indices = []

    def reset(self):
        pass

    def step(self, previous_q, auxiliary):
        raise RuntimeError('baseline must never run a neural predictor')
