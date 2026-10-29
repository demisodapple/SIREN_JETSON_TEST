# Adapted from f_astra/integer_contract.py; see SOURCES.json.
"""Power-of-two integer network contracts and their exact host reference.

A contract fixes every layer's integer weight and bias codes, binary-point
positions (``qpos``), rounding (nearest, ties up) and saturation. ``forward`` is
the host integer oracle for that contract. ``thresholds`` restates each
requantization as the per-channel threshold table that a FINN ``MultiThreshold``
evaluates, so the FINN graph built from the contract computes the same codes.
NumPy only: this module runs in the repository and the FINN environments.
"""

import numpy as np

UNIFORM = "f-astra/finn-modela-int8/v1"
MIXED = "f-astra/integer-network/v2"
MIXED3 = "f-astra/integer-network/v3"
SCHEMAS = {UNIFORM, MIXED, MIXED3}
MIXED_SCHEMAS = {MIXED, MIXED3}
MAC_KINDS = {"pointwise", "depthwise", "stride_mean"}
KINDS = MAC_KINDS | {"residual"}
# v3 adds two non-MAC kinds that FINN lowers to F-ASTRA custom operators (docs/pd-finn-custom-ops.md).
NORM_KINDS = {"layer_norm", "gelu_pwl"}
KINDS3 = KINDS | NORM_KINDS
LN_NORMAL_QPOS = 26          # normalized value (x - mean) / root
LN_AFFINE_QPOS = 30          # gamma and beta codes
GELU_GRID_BITS = 11          # table step 2^-11 over [0, 8)
GELU_TABLE_QPOS = 28
GELU_ENTRIES = 8 << GELU_GRID_BITS
MAX_SHIFT = 30
FLOAT32_EXACT = 1 << 24


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _int(value, low, high, message):
    need(type(value) is int and low <= value <= high, message)
    return value


def signed_range(bits):
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def load(document):
    """Validate a contract document; return it with weights and biases as int64 arrays."""
    need(isinstance(document, dict) and document.get("schema") in SCHEMAS, "unsupported integer contract schema")
    need(document.get("rounding") == "nearest_half_up", "nearest-half-up rounding contract required")
    if document["schema"] in MIXED_SCHEMAS:
        return load_mixed(document)
    need(document.get("mixed_precision") is False, "uniform-width contract required by this schema")
    widths = {name: _int(document.get(name), 2, 64, "invalid " + name)
              for name in ("weight_bits", "activation_bits", "accumulator_bits")}
    shape = document.get("input_shape")
    need(isinstance(shape, list) and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape),
         "input_shape must be [time, channels]")
    _int(document.get("input_qpos"), -MAX_SHIFT, MAX_SHIFT, "invalid input_qpos")
    layers = document.get("layers")
    need(isinstance(layers, list) and layers, "contract layers required")
    wmin, wmax = signed_range(widths["weight_bits"])
    seen = {"input": (shape[0], shape[1])}
    result = []
    for raw in layers:
        need(isinstance(raw, dict) and raw.get("kind") in KINDS, "unsupported contract layer kind")
        layer = dict(raw)
        lid = layer.get("id")
        need(isinstance(lid, str) and lid not in seen, "unique layer id required")
        need(layer.get("input_id") in seen, lid + ": unknown input")
        for field in ("input_qpos", "output_qpos"):
            _int(layer.get(field), -MAX_SHIFT, MAX_SHIFT, lid + ": invalid " + field)
        length, channels = seen[layer["input_id"]]
        if layer["kind"] == "residual":
            need(layer.get("branch_id") in seen and seen[layer["branch_id"]] == (length, channels),
                 lid + ": residual branch shape differs")
            _int(layer.get("branch_qpos"), -MAX_SHIFT, MAX_SHIFT, lid + ": invalid branch_qpos")
            # Schema f-astra/finn-modela-int8/v1 always applies ReLU after the join.
            layer["relu"] = True
            seen[lid] = (length, channels)
            layer["output_shape"] = [length, channels]
        else:
            for field in ("weight_qpos", "bias_qpos"):
                _int(layer.get(field), -MAX_SHIFT, MAX_SHIFT, lid + ": invalid " + field)
            need(type(layer.get("relu")) is bool, lid + ": explicit relu flag required")
            wshape = layer.get("weight_shape")
            need(isinstance(wshape, list) and len(wshape) == 4 and wshape[0] == 1
                 and all(type(v) is int and v > 0 for v in wshape), lid + ": weight_shape [1,k,cin,cout] required")
            _, kernel, cin, cout = wshape
            need(cin == channels, lid + ": input channels differ")
            stride = layer.get("stride")
            need(isinstance(stride, list) and len(stride) == 4 and stride[0] == stride[1] == stride[3] == 1
                 and type(stride[2]) is int and stride[2] > 0, lid + ": temporal stride [1,1,s,1] required")
            step = stride[2]
            need(layer.get("padding") in {"SAME", "VALID"}, lid + ": SAME or VALID padding required")
            if layer["kind"] == "depthwise":
                need(cout == 1, lid + ": depthwise multiplier must be one")
                cout = cin
            if layer["kind"] == "pointwise":
                need(kernel == 1 and step == 1, lid + ": pointwise needs kernel and stride one")
            if layer["kind"] == "stride_mean":
                need(cin == cout == 1 and layer["padding"] == "VALID" and kernel == step and length % step == 0,
                     lid + ": stride mean needs one channel, VALID, kernel == stride")
            if layer["padding"] == "SAME" and kernel > 1:
                need(kernel % 2 == 1 and step == 1, lid + ": SAME needs odd kernel and stride one")
                out_length = length
            else:
                need(length >= kernel and (length - kernel) % step == 0, lid + ": VALID geometry is not exact")
                out_length = (length - kernel) // step + 1
            weights = np.asarray(layer.get("weight_codes"), dtype=np.int64)
            need(weights.size == kernel * cin * wshape[3] and weights.min() >= wmin and weights.max() <= wmax,
                 lid + ": weight codes outside the declared width")
            layer["weights"] = weights.reshape(wshape)
            bias = np.asarray(layer.get("bias_codes"), dtype=np.int64)
            need(bias.shape == (cout,), lid + ": one bias code per output channel required")
            layer["bias"] = bias
            for shift in (_product_qpos(layer) - layer["bias_qpos"],):
                need(abs(shift) <= MAX_SHIFT, lid + ": bias alignment shift too large")
            seen[lid] = (out_length, cout)
            layer["output_shape"] = [out_length, cout]
        result.append(layer)
    out = dict(document)
    out["layers"] = result
    out["output_shape"] = list(seen[result[-1]["id"]])
    return out


def _product_qpos(layer):
    return layer["input_qpos"] + layer["weight_qpos"]


def requant(values, source_q, target_q, relu, bits):
    """Shift from source to target binary point, ties rounded up, then saturate."""
    values = np.asarray(values, dtype=np.int64)
    if relu:
        values = np.maximum(values, 0)
    shift = source_q - target_q
    need(abs(shift) <= MAX_SHIFT, "requantization shift too large")
    if shift > 0:
        values = (values + (1 << (shift - 1))) >> shift
    elif shift < 0:
        values = values << (-shift)
    low, high = signed_range(bits)
    return np.clip(values, low, high)


def affine(layer, acc, bits, channel=None):
    """Bias-aligned requantization of a MAC accumulator (one channel or all)."""
    product, bias_q = _product_qpos(layer), layer["bias_qpos"]
    aligned = max(product, bias_q)
    bias = layer["bias"] if channel is None else layer["bias"][channel]
    values = (np.asarray(acc, np.int64) << (aligned - product)) + (np.asarray(bias, np.int64) << (aligned - bias_q))
    return requant(values, aligned, layer["output_qpos"], layer["relu"], bits)


def residual_sum(layer, x, y):
    q = max(layer["input_qpos"], layer["branch_qpos"])
    return (np.asarray(x, np.int64) << (q - layer["input_qpos"])) + (np.asarray(y, np.int64) << (q - layer["branch_qpos"])), q


def exact_matmul(x, w):
    """``x @ w`` of int64 matrices, exact, through float64 BLAS.

    ``x`` is split into limbs (low limbs unsigned, the top limb signed) small enough that every partial
    sum of ``limb * w`` stays below 2^53, where float64 is exact whatever the summation order. The
    limb products are recombined in int64; the result is exact whenever the true product fits int64.
    """
    x, w = np.asarray(x, np.int64), np.asarray(w, np.int64)
    out = np.zeros((x.shape[0], w.shape[1]), np.int64)
    if x.size == 0 or w.size == 0 or not w.any():
        return out
    limit = ((1 << 53) - 1) // (int(np.abs(w).max()) * x.shape[1])
    need(limit >= 2, "matrix too wide or weights too large for exact float64 limbs")
    bits = limit.bit_length() - 1
    weights = w.astype(np.float64)
    shift, rest = 0, x
    while max(-int(rest.min()), int(rest.max())) > limit:
        low = rest & ((np.int64(1) << bits) - 1)
        out += (low.astype(np.float64) @ weights).astype(np.int64) << shift
        rest, shift = rest >> bits, shift + bits
    out += (rest.astype(np.float64) @ weights).astype(np.int64) << shift
    return out


def accumulate(layer, x):
    """Exact MAC accumulator for (N, T, C) integer codes, tap by tap."""
    _, kernel, _, _ = layer["weights"].shape
    step = layer["stride"][2]
    x = np.asarray(x, np.int64)
    if layer["padding"] == "SAME" and kernel > 1:
        x = np.pad(x, ((0, 0), (kernel // 2, kernel // 2), (0, 0)))
    elif layer["padding"] == "CAUSAL":
        x = np.pad(x, ((0, 0), (kernel - 1, 0), (0, 0)))
    length = (x.shape[1] - kernel) // step + 1
    w = layer["weights"][0]  # K, Cin, Cout (or K, C, 1 for depthwise)
    acc = None
    for k in range(kernel):
        window = x[:, k:k + step * (length - 1) + 1:step, :]
        if layer["kind"] == "depthwise":
            term = window * w[k, :, 0]
        else:
            term = exact_matmul(window.reshape(-1, window.shape[2]), w[k]).reshape(window.shape[0], length, -1)
        acc = term if acc is None else acc + term
    return acc


def _geometry(layer, length, channels, causal=False):
    """Validate a MAC layer's kernel, stride and padding; return (output length, output channels).

    ``causal`` (schema v3) also admits CAUSAL padding: K - 1 zeros before the first step, none after,
    stride one, so output t depends on inputs t - K + 1 .. t only."""
    lid = layer["id"]
    wshape = layer.get("weight_shape")
    need(isinstance(wshape, list) and len(wshape) == 4 and wshape[0] == 1
         and all(type(v) is int and v > 0 for v in wshape), lid + ": weight_shape [1,k,cin,cout] required")
    _, kernel, cin, cout = wshape
    need(cin == channels, lid + ": input channels differ")
    stride = layer.get("stride")
    need(isinstance(stride, list) and len(stride) == 4 and stride[0] == stride[1] == stride[3] == 1
         and type(stride[2]) is int and stride[2] > 0, lid + ": temporal stride [1,1,s,1] required")
    step = stride[2]
    paddings = {"SAME", "VALID", "CAUSAL"} if causal else {"SAME", "VALID"}
    need(layer.get("padding") in paddings, lid + ": padding must be one of " + ", ".join(sorted(paddings)))
    if layer["kind"] == "depthwise":
        need(cout == 1, lid + ": depthwise multiplier must be one")
        cout = cin
    if layer["kind"] == "pointwise":
        need(kernel == 1 and step == 1, lid + ": pointwise needs kernel and stride one")
    if layer["kind"] == "stride_mean":
        need(cin == cout == 1 and layer["padding"] == "VALID" and kernel == step and length % step == 0,
             lid + ": stride mean needs one channel, VALID, kernel == stride")
    if layer["padding"] == "SAME" and kernel > 1:
        need(kernel % 2 == 1 and step == 1, lid + ": SAME needs odd kernel and stride one")
        return length, cout
    if layer["padding"] == "CAUSAL":
        need(step == 1, lid + ": CAUSAL needs stride one")
        return length, cout
    need(length >= kernel and (length - kernel) % step == 0, lid + ": VALID geometry is not exact")
    return (length - kernel) // step + 1, cout


def load_mixed(document):
    """Schema v2: per-layer widths, per-channel weight binary points, bias at the accumulator point.

    A MAC layer computes ``acc = sum(x*w)`` (checked against ``accumulator_bits``), adds the
    bias of each output channel at that channel's accumulator point ``input_qpos + weight_qpos``,
    shifts right to ``output_qpos`` with ties rounded up, applies ReLU if set and saturates to
    ``output_bits``. A residual aligns both inputs to the finer point, adds them and requantizes
    the same way. Right shifts only: every output point is at or below its source point.
    """
    need(document.get("saturation") is True, "saturating contract required")
    shape = document.get("input_shape")
    need(isinstance(shape, list) and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape),
         "input_shape must be [time, channels]")
    _int(document.get("input_qpos"), -MAX_SHIFT, MAX_SHIFT, "invalid input_qpos")
    _int(document.get("input_bits"), 2, 32, "invalid input_bits")
    layers = document.get("layers")
    need(isinstance(layers, list) and layers, "contract layers required")
    seen = {"input": (shape[0], shape[1], document["input_qpos"], document["input_bits"])}
    kinds = KINDS3 if document["schema"] == MIXED3 else KINDS
    result = []
    for raw in layers:
        need(isinstance(raw, dict) and raw.get("kind") in kinds, "unsupported contract layer kind")
        layer = dict(raw)
        lid = layer.get("id")
        need(isinstance(lid, str) and lid not in seen, "unique layer id required")
        need(layer.get("input_id") in seen, lid + ": unknown input")
        length, channels, q_in, bits_in = seen[layer["input_id"]]
        need(layer.get("input_qpos") == q_in, lid + ": input_qpos differs from its producer")
        _int(layer.get("output_qpos"), -MAX_SHIFT, MAX_SHIFT, lid + ": invalid output_qpos")
        _int(layer.get("output_bits"), 2, 32, lid + ": invalid output_bits")
        layer["input_bits"] = bits_in
        if layer["kind"] in NORM_KINDS:
            _load_norm(layer, lid, channels, q_in, bits_in)
            layer["output_shape"] = [length, channels]
            seen[lid] = (length, channels, layer["output_qpos"], layer["output_bits"])
            result.append(layer)
            continue
        _int(layer.get("accumulator_bits"), 2, 64, lid + ": invalid accumulator_bits")
        need(type(layer.get("relu")) is bool, lid + ": explicit relu flag required")
        if layer["kind"] == "residual":
            need(layer.get("branch_id") in seen, lid + ": unknown branch")
            b_length, b_channels, q_branch, _ = seen[layer["branch_id"]]
            need((b_length, b_channels) == (length, channels), lid + ": residual branch shape differs")
            need(layer.get("branch_qpos") == q_branch, lid + ": branch_qpos differs from its producer")
            q = max(q_in, q_branch)
            layer["shift"] = np.full(channels, q - layer["output_qpos"], np.int64)
            out_length, cout = length, channels
        else:
            out_length, cout = _geometry(layer, length, channels, causal=document["schema"] == MIXED3)
            _int(layer.get("weight_bits"), 2, 32, lid + ": invalid weight_bits")
            wq = layer.get("weight_qpos")
            wq = np.full(cout, wq, np.int64) if type(wq) is int else np.asarray(wq, np.int64)
            need(wq.shape == (cout,) and np.all(np.abs(wq) <= MAX_SHIFT), lid + ": one weight_qpos per output channel")
            layer["weight_qpos_array"] = wq
            wmin, wmax = signed_range(layer["weight_bits"])
            weights = np.asarray(layer.get("weight_codes"), dtype=np.int64)
            need(weights.size == int(np.prod(layer["weight_shape"])) and weights.min() >= wmin and weights.max() <= wmax,
                 lid + ": weight codes outside the declared width")
            layer["weights"] = weights.reshape(layer["weight_shape"])
            bias = np.asarray(layer.get("bias_codes"), dtype=np.int64)
            need(bias.shape == (cout,), lid + ": one bias code per output channel required")
            layer["bias"] = bias
            layer["shift"] = q_in + wq - layer["output_qpos"]
        need(np.all(layer["shift"] >= 0) and np.all(layer["shift"] <= MAX_SHIFT), lid + ": requantization must shift right")
        layer["output_shape"] = [out_length, cout]
        seen[lid] = (out_length, cout, layer["output_qpos"], layer["output_bits"])
        result.append(layer)
    out = dict(document)
    out["layers"] = result
    out["output_shape"] = list(seen[result[-1]["id"]][:2])
    out["output_qpos"] = result[-1]["output_qpos"]
    return out


def _load_norm(layer, lid, channels, q_in, bits_in):
    """Validate a v3 ``layer_norm`` or ``gelu_pwl`` layer (no weights, no accumulator, no ReLU)."""
    need(layer.get("relu", False) is False, lid + ": no ReLU on a normalization layer")
    layer["relu"] = False
    if layer["kind"] == "layer_norm":
        need(bits_in <= 30 and channels >= 2, lid + ": LayerNorm input up to 30 bits over at least two channels")
        need(0 <= q_in <= 29, lid + ": LayerNorm input_qpos 0..29 (the variance point is twice it)")
        eps = layer.get("epsilon_code")
        _int(eps, 1, (1 << 62) - 1, lid + ": epsilon_code (epsilon at twice the input point) required")
        for name in ("gamma_codes", "beta_codes"):
            codes = np.asarray(layer.get(name), dtype=np.int64)
            need(codes.shape == (channels,) and codes.min() >= -(1 << 31) and codes.max() <= (1 << 31) - 1,
                 lid + ": one signed32 Q30 %s per channel" % name)
            layer[name.replace("_codes", "")] = codes
        need(0 <= LN_NORMAL_QPOS + LN_AFFINE_QPOS - layer["output_qpos"] <= 62, lid + ": output point out of range")
    else:
        need(GELU_GRID_BITS <= q_in <= GELU_TABLE_QPOS, lid + ": GELU input_qpos between the grid and table points")
        need(layer["output_qpos"] <= GELU_TABLE_QPOS, lid + ": GELU output_qpos at or below the table point")
        need(bits_in <= 32, lid + ": GELU input up to 32 bits")
        sha = layer.get("table_sha256")
        need(sha is None or sha == gelu_table_sha256(), lid + ": GELU table differs from the contract's")


def rne_shift(values, shift):
    """Right shift with ties to even (int64; shift 0..62)."""
    values = np.asarray(values, np.int64)
    if shift == 0:
        return values.copy()
    base = values >> shift
    rest = values - (base << shift)
    half = np.int64(1) << (shift - 1)
    return base + ((rest > half) | ((rest == half) & ((base & 1) != 0))).astype(np.int64)


def rne_divide(numerator, denominator):
    """Non-negative numerator over a positive denominator, rounded to nearest with ties to even."""
    q = numerator // denominator
    r = numerator - q * denominator
    return q + ((2 * r > denominator) | ((2 * r == denominator) & ((q & 1) != 0))).astype(np.int64)


def integer_sqrt(values):
    """Floor square root of non-negative int64 values (float estimate, exact integer correction)."""
    values = np.asarray(values, np.int64)
    root = np.floor(np.sqrt(values.astype(np.float64))).astype(np.int64)
    for _ in range(2):
        root = np.where(root * root > values, root - 1, root)
        root = np.where((root + 1) * (root + 1) <= values, root + 1, root)
    return root


def layer_norm(layer, x, moments=False):
    """Integer LayerNorm over the last axis: RNE mean, exact centered second moment (RNE), floor sqrt of
    variance + epsilon, RNE normalization to Q26, Q30 affine, RNE to the output point, saturation.
    ``moments`` also returns (mean, variance, root) per row."""
    x = np.asarray(x, np.int64)
    channels = x.shape[-1]
    total = x.sum(axis=-1, keepdims=True)
    mean = rne_divide(np.abs(total), channels) * np.where(total < 0, -1, 1)
    centered = x - mean
    # |centered| < 2^30, squares < 2^60: groups of eight stay below 2^63. variance = RNE(sum / C) exactly: the
    # remainders' carry joins the quotient before a tie goes to even (the tie parity is that of the whole quotient).
    group = 8
    padded = np.concatenate([centered, np.zeros(x.shape[:-1] + (-channels % group,), np.int64)], axis=-1)
    partial = np.square(padded).reshape(x.shape[:-1] + (-1, group)).sum(axis=-1)
    quotient, remainder = partial // channels, partial % channels
    carry, rest = np.divmod(remainder.sum(axis=-1), channels)
    base = quotient.sum(axis=-1) + carry
    variance = base + ((2 * rest > channels) | ((2 * rest == channels) & ((base & 1) != 0))).astype(np.int64)
    root = integer_sqrt(variance + layer["epsilon_code"])[..., None]
    normalized = rne_divide(np.abs(centered) << LN_NORMAL_QPOS, root) * np.where(centered < 0, -1, 1)
    normalized = np.clip(normalized, -(1 << 31), (1 << 31) - 1)
    accumulator = normalized * layer["gamma"] + (layer["beta"] << LN_NORMAL_QPOS)
    out = rne_shift(accumulator, LN_NORMAL_QPOS + LN_AFFINE_QPOS - layer["output_qpos"])
    low, high = signed_range(layer["output_bits"])
    if moments:
        return np.clip(out, low, high), mean[..., 0], variance, root[..., 0]
    return np.clip(out, low, high)


def gelu_table():
    """PWL GELU correction c(a) = -a * Phi(-a) on [0, 8): signed32 bases at Q28 and signed18 deltas."""
    import math
    values = np.asarray([round(-.5 * (i / (1 << GELU_GRID_BITS)) * math.erfc((i / (1 << GELU_GRID_BITS)) / math.sqrt(2))
                               * (1 << GELU_TABLE_QPOS)) for i in range(GELU_ENTRIES + 1)], dtype=np.int64)
    delta = np.diff(values)
    need(values.min() >= -(1 << 31) and values.max() < (1 << 31) and delta.min() >= -(1 << 17) and delta.max() < (1 << 17),
         "GELU table exceeds signed32 bases or signed18 deltas")
    return values[:-1], delta


def gelu_table_sha256():
    import hashlib
    bases, deltas = gelu_table()
    return hashlib.sha256(bases.astype("<i8").tobytes() + deltas.astype("<i8").tobytes()).hexdigest()


def gelu_pwl(layer, x, table=None):
    """Integer GELU: GELU(+a) = a + c(a), GELU(-a) = c(a), c from the PWL table with RNE interpolation;
    c = 0 at and beyond 8; RNE to the output point and saturation."""
    x = np.asarray(x, np.int64)
    bases, deltas = table if table is not None else gelu_table()
    iq, oq = layer["input_qpos"], layer["output_qpos"]
    magnitude = np.abs(x)
    fraction_bits = iq - GELU_GRID_BITS
    address = np.clip(magnitude >> fraction_bits, 0, GELU_ENTRIES - 1)
    fraction = magnitude & ((np.int64(1) << fraction_bits) - 1)
    correction = bases[address] + rne_shift(deltas[address] * fraction, fraction_bits)
    correction = np.where(magnitude >= (np.int64(8) << iq), 0, correction)
    y = correction + np.where(x < 0, 0, magnitude << (GELU_TABLE_QPOS - iq))
    low, high = signed_range(layer["output_bits"])
    return np.clip(rne_shift(y, GELU_TABLE_QPOS - oq), low, high)


def requant_steps(layer):
    """The FINN restatement of one requantization: clip in the accumulator domain, then shift.

    ``out = min(max(v + B, L), T) >> s`` per channel equals ties-up rounding, ReLU and
    saturation of ``v`` (accumulator plus bias, or the aligned residual sum).
    """
    s = layer["shift"]
    low, high = signed_range(layer["output_bits"])
    rounding = np.where(s > 0, np.left_shift(np.int64(1), np.maximum(s - 1, 0)), 0)
    offset = rounding + (layer["bias"] if layer["kind"] != "residual" else 0)
    floor = (0 if layer["relu"] else low) * (np.int64(1) << s)
    ceiling = (np.int64(high + 1) << s) - 1
    return dict(offset=offset.astype(np.int64), floor=np.asarray(floor, np.int64), ceiling=ceiling.astype(np.int64),
                shift=s.astype(np.int64))


def reachable_ranges(contract):
    """Sound (low, high) output codes of every layer of a loaded mixed contract.

    Interval arithmetic over the whole input range with the requantization steps, as the FINN lowering
    uses to decide which clips can act. Any output outside these bounds cannot come from the contract.
    """
    need(contract["schema"] in MIXED_SCHEMAS, "reachable ranges need a mixed contract")
    ranges = {"input": signed_range(contract["input_bits"])}
    for layer in contract["layers"]:
        if layer["kind"] in NORM_KINDS:
            ranges[layer["id"]] = signed_range(layer["output_bits"])
            continue
        steps = requant_steps(layer)
        if layer["kind"] == "residual":
            q = max(layer["input_qpos"], layer["branch_qpos"])
            (alo, ahi), (blo, bhi) = ranges[layer["input_id"]], ranges[layer["branch_id"]]
            sa, sb = 1 << (q - layer["input_qpos"]), 1 << (q - layer["branch_qpos"])
            cout = layer["output_shape"][1]
            acc_lo, acc_hi = np.full(cout, alo * sa + blo * sb), np.full(cout, ahi * sa + bhi * sb)
        else:
            lo_in, hi_in = ranges[layer["input_id"]]
            taps = layer["weights"][0].reshape(-1, layer["weights"].shape[3]).astype(np.int64)
            if layer["kind"] == "depthwise":
                taps = layer["weights"][0, :, :, 0].astype(np.int64)
            acc_hi = np.sum(np.maximum(taps * lo_in, taps * hi_in), axis=0)
            acc_lo = np.sum(np.minimum(taps * lo_in, taps * hi_in), axis=0)
        reach_lo = (acc_lo + steps["offset"]) >> steps["shift"]
        reach_hi = (acc_hi + steps["offset"]) >> steps["shift"]
        low, high = signed_range(layer["output_bits"])
        floor = 0 if layer["relu"] else low
        ranges[layer["id"]] = (max(int(np.min(reach_lo)), floor), min(int(np.max(reach_hi)), high))
    return ranges


def _requant_mixed(layer, values):
    s = layer["shift"]
    values = np.asarray(values, np.int64)
    if layer["relu"]:
        values = np.maximum(values, 0)
    rounding = np.where(s > 0, np.left_shift(np.int64(1), np.maximum(s - 1, 0)), 0)
    values = (values + rounding) >> s
    low, high = signed_range(layer["output_bits"])
    return np.clip(values, low, high)


def forward_mixed(contract, inputs):
    x = np.asarray(inputs, np.int64)
    low, high = signed_range(contract["input_bits"])
    need(x.ndim == 3 and list(x.shape[1:]) == contract["input_shape"], "inputs must be (N, time, channels)")
    need(x.min() >= low and x.max() <= high, "input codes outside the input width")
    values, points = {"input": x}, {"input": contract["input_qpos"]}
    for layer in contract["layers"]:
        source = values[layer["input_id"]]
        if layer["kind"] == "layer_norm":
            values[layer["id"]] = layer_norm(layer, source)
            continue
        if layer["kind"] == "gelu_pwl":
            values[layer["id"]] = gelu_pwl(layer, source)
            continue
        acc_low, acc_high = signed_range(layer["accumulator_bits"])
        if layer["kind"] == "residual":
            q = max(layer["input_qpos"], layer["branch_qpos"])
            acc = (source << (q - layer["input_qpos"])) + (values[layer["branch_id"]] << (q - layer["branch_qpos"]))
            need(acc.min() >= acc_low and acc.max() <= acc_high, layer["id"] + ": accumulator overflow")
            values[layer["id"]] = _requant_mixed(layer, acc)
        else:
            acc = accumulate(layer, source)
            need(acc.min() >= acc_low and acc.max() <= acc_high, layer["id"] + ": accumulator overflow")
            values[layer["id"]] = _requant_mixed(layer, acc + layer["bias"])
    return values[contract["layers"][-1]["id"]], values


def forward_stream(contract, inputs, chunk=8):
    """``forward_mixed`` in chunks of samples, keeping only live tensors; returns the final codes and,
    per layer, the output code range and how many codes sit on a saturation bound (zero proves no clip)."""
    need(contract["schema"] in MIXED_SCHEMAS, "streaming needs a mixed contract")
    layers = contract["layers"]
    last_use = {}
    for index, layer in enumerate(layers):
        for source in (layer["input_id"], layer.get("branch_id")):
            if source is not None:
                last_use[source] = index
    stats = {layer["id"]: dict(min=None, max=None, at_bounds=0) for layer in layers}
    outputs = []
    for start in range(0, len(inputs), chunk):
        outputs.append(_stream_chunk(contract, layers, last_use, inputs[start:start + chunk], stats))
    return np.concatenate(outputs), stats


def _stream_chunk(contract, layers, last_use, x, stats):
    x = np.asarray(x, np.int64)
    low, high = signed_range(contract["input_bits"])
    need(x.ndim == 3 and list(x.shape[1:]) == contract["input_shape"], "inputs must be (N, time, channels)")
    need(x.min() >= low and x.max() <= high, "input codes outside the input width")
    values = {"input": x}
    table = gelu_table()
    for index, layer in enumerate(layers):
        source = values[layer["input_id"]]
        if layer["kind"] == "layer_norm":
            out = layer_norm(layer, source)
        elif layer["kind"] == "gelu_pwl":
            out = gelu_pwl(layer, source, table)
        else:
            acc_low, acc_high = signed_range(layer["accumulator_bits"])
            if layer["kind"] == "residual":
                q = max(layer["input_qpos"], layer["branch_qpos"])
                acc = (source << (q - layer["input_qpos"])) + (values[layer["branch_id"]] << (q - layer["branch_qpos"]))
            else:
                acc = accumulate(layer, source)
            need(acc.min() >= acc_low and acc.max() <= acc_high, layer["id"] + ": accumulator overflow")
            out = _requant_mixed(layer, acc if layer["kind"] == "residual" else acc + layer["bias"])
        o_low, o_high = signed_range(layer["output_bits"])
        entry = stats[layer["id"]]
        entry["min"] = int(out.min()) if entry["min"] is None else min(entry["min"], int(out.min()))
        entry["max"] = int(out.max()) if entry["max"] is None else max(entry["max"], int(out.max()))
        entry["at_bounds"] += int(np.count_nonzero(out == o_high)) + (0 if layer["relu"] else int(np.count_nonzero(out == o_low)))
        values[layer["id"]] = out
        for name in [n for n, last in last_use.items() if last == index]:
            values.pop(name, None)
    return values[layers[-1]["id"]]


def forward(contract, inputs):
    """Run the contract on (N, T, C) input codes; return the final codes and every layer's codes."""
    if contract["schema"] in MIXED_SCHEMAS:
        return forward_mixed(contract, inputs)
    bits = contract["activation_bits"]
    acc_low, acc_high = signed_range(contract["accumulator_bits"])
    low, high = signed_range(bits)
    x = np.asarray(inputs, np.int64)
    need(x.ndim == 3 and list(x.shape[1:]) == contract["input_shape"], "inputs must be (N, time, channels)")
    need(x.min() >= low and x.max() <= high, "input codes outside the activation width")
    values = {"input": x}
    for layer in contract["layers"]:
        source = values[layer["input_id"]]
        if layer["kind"] == "residual":
            acc, q = residual_sum(layer, source, values[layer["branch_id"]])
            need(acc.min() >= acc_low and acc.max() <= acc_high, layer["id"] + ": accumulator overflow")
            values[layer["id"]] = requant(acc, q, layer["output_qpos"], True, bits)
        else:
            acc = accumulate(layer, source)
            need(acc.min() >= acc_low and acc.max() <= acc_high, layer["id"] + ": accumulator overflow")
            values[layer["id"]] = affine(layer, acc, bits)
    return values[contract["layers"][-1]["id"]], values
