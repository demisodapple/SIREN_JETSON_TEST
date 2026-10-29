"""Input preparation for all three missions and compression comparison."""
import csv
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from reference import need

SUMMARY_COLUMNS = (["f_mep_pro_tel%d_flux_p%d" % (t, k) for t in (0, 90) for k in range(1, 7)]
                   + ["f_mep_ele_tel%d_flux_e%d" % (t, k) for t in (0, 90) for k in range(1, 5)]
                   + ["f_mep_omni_flux_p%d" % k for k in (1, 2, 3)]
                   + ["pc_mep_pro_tel0_flux_p6", "pc_mep_pro_tel90_flux_p6", "pc_mep_omni_flux_p1", "pc_mep_omni_flux_p2",
                      "pc_mep_omni_flux_p3"]
                   + ["n_valid", "ifc", "polar", "abs_mlat", "L", "mlt_sin", "mlt_cos", "saa"])
STEPS = 288
NFLUX = 23
NBINS = 18
FEATURES = 79                 # raw 0-22, geo 23-30, orb 31-53 (not on the board), qual 54-55, zt 56-78
ZT = 56
PAD_COL = 0xFFFF                  # a padding input channel (zero weights in the contract): code 0
HISTORY = 27
MIN_PERIODS = 5
PROGRAM = "f-astra-pd-zt"
SOURCE = Path(__file__).with_name("f_astra_pd_zt.c")
PREP_FILE = "pd-zt-prep.txt"
HOST_REFERENCE = "pd-zt-inputs.sha256"
RECORD = "f-astra/directrtl-pd-zt-bundle/v1"
_C = {n: i for i, n in enumerate(SUMMARY_COLUMNS)}


def _day_number(ymd):
    import datetime
    return (datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])) - datetime.date(1970, 1, 1)).days


def tables_bytes(eps, scales, cols, mu, sd, qpos, bits, history=HISTORY, min_periods=MIN_PERIODS):
    """The board's table file (``ASTRAZT1``): eps (23 float64), the model's feature columns (indices into the 79
    features; orb columns are not on the board), normalization (float32), and per year the 18 x 23 scale table
    (float64). A column PAD_COL is a padding input channel of the contract (zero weights; its code is 0)."""
    import numpy as np
    eps = np.asarray(eps, np.float64)
    cols = [int(c) for c in cols]
    need(eps.shape == (NFLUX,), "eps: 23 channels")
    need(all(c == PAD_COL or (0 <= c < FEATURES and not 31 <= c < 54) for c in cols), "columns: raw, geo, qual and zt only")
    mu, sd = np.asarray(mu, np.float32), np.asarray(sd, np.float32)
    need(mu.shape == sd.shape == (len(cols),), "normalization per column")
    need(1 <= min_periods <= history and 2 <= bits <= 16 and scales, "history, bits and scale tables")
    out = [b"ASTRAZT1", struct.pack("<10I", 1, NFLUX, NBINS, history, min_periods, qpos & 0xFFFFFFFF, bits, len(cols),
                                    len(scales), 0),
           eps.astype("<f8").tobytes(), np.asarray(cols, "<u2").tobytes(), mu.astype("<f4").tobytes(), sd.astype("<f4").tobytes()]
    for year in sorted(scales):
        table = np.asarray(scales[year], np.float64)
        need(table.shape == (NBINS, NFLUX), "scale table: 18 bins x 23 channels")
        out += [struct.pack("<i", int(year)), table.astype("<f8").tobytes()]
    return b"".join(out)


def read_tables(data):
    import numpy as np
    need(data[:8] == b"ASTRAZT1", "ASTRAZT1 tables required")
    version, nflux, nbins, history, min_periods, qpos, bits, ncols, nyears = struct.unpack("<9I", data[8:44])
    need(version == 1 and nflux == NFLUX and nbins == NBINS, "tables version 1, 23 channels, 18 bins")
    p = 48
    eps = np.frombuffer(data, "<f8", NFLUX, p); p += 8 * NFLUX
    cols = np.frombuffer(data, "<u2", ncols, p).astype(int); p += 2 * ncols
    mu = np.frombuffer(data, "<f4", ncols, p); p += 4 * ncols
    sd = np.frombuffer(data, "<f4", ncols, p); p += 4 * ncols
    scales = {}
    for _ in range(nyears):
        year = struct.unpack("<i", data[p:p + 4])[0]
        scales[year] = np.frombuffer(data, "<f8", NBINS * NFLUX, p + 4).reshape(NBINS, NFLUX); p += 4 + 8 * NBINS * NFLUX
    need(p == len(data), "tables size")
    qpos = qpos - (1 << 32) if qpos >= 1 << 31 else qpos
    return dict(eps=eps, cols=cols, mu=mu, sd=sd, scales=scales, history=history, min_periods=min_periods, qpos=qpos,
                bits=bits)


def state_bytes(state, history=HISTORY):
    """The board's state file (``ASTRAZS1``): per ring slot (day number mod history) the day number (days since
    1970-01-01, INT32_MIN when empty) and the day's 18 x 23 bin medians (float32)."""
    import numpy as np
    out = [b"ASTRAZS1", struct.pack("<4I", 1, history, NBINS, NFLUX)]
    slots = {d % history: d for d in state}
    need(len(slots) == len(state), "state: one day per slot")
    for k in range(history):
        if k in slots:
            out += [struct.pack("<i", slots[k]), np.asarray(state[slots[k]], np.float32).astype("<f4").tobytes()]
        else:
            out += [struct.pack("<i", -2 ** 31), np.full(NBINS * NFLUX, np.nan, "<f4").tobytes()]
    return b"".join(out)


ROW_BYTES = 281                   # POES l1a row (directrtl_orbit.orbit_object)
GLAT, GLON = 181, 361             # 1-degree geographic grid, latitude -90..90, longitude 0..360


def geo_bytes(src, factor, grids):
    """The board's geomagnetic table (``ASTRAGT1``): per flux channel its l1a count-rate index (-1: the board cannot
    make it) and count-rate-to-flux factor (float64), then four float32 181 x 361 grids at the satellite's altitude:
    |mlat|, min(L, 20), cosine and sine of the magnetic longitude."""
    import numpy as np
    src, factor = np.asarray(src, "<i4"), np.asarray(factor, np.float64)
    need(src.shape == factor.shape == (NFLUX,) and src.min() >= -1 and src.max() < 46, "per channel source and factor")
    need(len(grids) == 4 and all(np.shape(g) == (GLAT, GLON) for g in grids), "four 181 x 361 grids")
    return b"".join([b"ASTRAGT1", struct.pack("<6I", 1, GLAT, GLON, NFLUX, 0, 0), src.tobytes(), factor.astype("<f8").tobytes()]
                    + [np.asarray(g, "<f4").tobytes() for g in grids])


def read_geo(data):
    import numpy as np
    need(data[:8] == b"ASTRAGT1" and struct.unpack("<4I", data[8:24]) == (1, GLAT, GLON, NFLUX), "ASTRAGT1 table required")
    cells = GLAT * GLON
    need(len(data) == 32 + 12 * NFLUX + 16 * cells, "geo table size")
    src = np.frombuffer(data, "<i4", NFLUX, 32)
    factor = np.frombuffer(data, "<f8", NFLUX, 32 + 4 * NFLUX)
    grids = [np.frombuffer(data, "<f4", cells, 32 + 12 * NFLUX + 4 * cells * k).reshape(GLAT, GLON) for k in range(4)]
    return dict(src=src, factor=factor, grids=grids)


def _grid_at(q, lat, lon):
    """Bilinear interpolation on the grid, operation for operation as the board's grid_at (NaN for a missing
    position)."""
    import numpy as np
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = np.where(ok, lat, 0.0), np.where(ok, lon, 0.0)
    y = lat + 90.0
    x = lon - 360.0 * np.floor(lon / 360.0)
    i0 = np.clip(np.floor(y), 0, GLAT - 2).astype(np.int64)
    j0 = np.clip(np.floor(x), 0, GLON - 2).astype(np.int64)
    fy, fx = y - i0, x - j0
    q = np.asarray(q, np.float64)
    v = q[i0, j0] * (1.0 - fy) * (1.0 - fx) + q[i0 + 1, j0] * fy * (1.0 - fx) + q[i0, j0 + 1] * (1.0 - fy) * fx \
        + q[i0 + 1, j0 + 1] * fy * fx
    return np.where(ok, v, np.nan)


def day_summary(rows, geo, sun, ymd):
    """The day's five-minute summary (float32 [288, 36], ``SUMMARY_COLUMNS``) from its raw l1a rows: the board's
    ``sum`` job. ``geo`` as ``read_geo``, ``sun`` float32 [1440, 2] (cosine, sine of the sun's magnetic longitude per UT
    minute). Sums run in float64 in row order (as the C loop)."""
    import numpy as np
    n = len(rows) // ROW_BYTES
    need(len(rows) == n * ROW_BYTES, "whole 281-byte records required")
    r = np.frombuffer(rows, np.uint8).reshape(n, ROW_BYTES)
    t = r[:, 0:8].copy().view("<u8")[:, 0].astype(np.int64)
    f = r[:, 22:22 + 252].copy().view("<f4")
    ifc_on = r[:, 22 + 252 + 4].view(np.int8)
    day0 = _day_number(ymd) * 86400000
    step = np.where(t >= day0, (t - day0) // 300000, -1)
    valid = (step >= 0) & (step < STEPS)
    k = np.where(valid, step, 0)
    lat, lon = f[:, 1].astype(np.float64), f[:, 2].astype(np.float64)
    am = _grid_at(geo["grids"][0], lat, lon)
    L = _grid_at(geo["grids"][1], lat, lon)
    is_polar = am >= 60.0
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    saa = (lat > -50.0) & (lat < 0.0) & (lon180 > -90.0) & (lon180 < 40.0) & (L < 2.5)
    count = lambda m: np.bincount(k[m & valid], minlength=STEPS)
    cnt = count(np.ones(n, bool))
    out = np.full((STEPS, len(SUMMARY_COLUMNS)), np.nan, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in range(NFLUX):
            if geo["src"][c] < 0:
                continue
            v = f[:, 3 + geo["src"][c]]
            ok = valid & np.isfinite(v) & (v >= 0)
            fl = v.astype(np.float64) * geo["factor"][c]
            s, m = np.bincount(k[ok], weights=fl[ok], minlength=STEPS), np.bincount(k[ok], minlength=STEPS)
            out[:, c] = np.where(m > 0, s / np.maximum(m, 1), np.nan)
            if c == 5:
                okp = ok & is_polar
                s, m = np.bincount(k[okp], weights=fl[okp], minlength=STEPS), np.bincount(k[okp], minlength=STEPS)
                out[:, 23] = np.where(m > 0, s / np.maximum(m, 1), np.nan)
        out[:, _C["n_valid"]] = cnt
        out[:, _C["ifc"]] = count(ifc_on > 0) / np.maximum(cnt, 1)
        out[:, _C["polar"]] = count(is_polar) / np.maximum(cnt, 1)
        out[:, _C["saa"]] = count(saa) / np.maximum(cnt, 1)
        ca, sa = _grid_at(geo["grids"][2], lat, lon), _grid_at(geo["grids"][3], lat, lon)
        norm = np.sqrt(ca * ca + sa * sa)
        minute = np.minimum(np.where(valid, (t - day0) // 60000, 0), 1439)
        sun = np.asarray(sun, np.float32).astype(np.float64)
        cb, sb = sun[minute, 0], sun[minute, 1]
        ca, sa = ca / norm, sa / norm
        s_ = -(sa * cb - ca * sb)
        c_ = -(ca * cb + sa * sb)
        ok = valid & (norm > 0) & np.isfinite(s_) & np.isfinite(c_)
        m = np.bincount(k[ok], minlength=STEPS)
        out[:, _C["mlt_sin"]] = np.where(m > 0, np.bincount(k[ok], weights=s_[ok], minlength=STEPS) / np.maximum(m, 1), np.nan)
        out[:, _C["mlt_cos"]] = np.where(m > 0, np.bincount(k[ok], weights=c_[ok], minlength=STEPS) / np.maximum(m, 1), np.nan)
    Lc = np.where(np.isfinite(L) & (L < 20.0), L, 20.0)
    for j in range(STEPS):
        sel = valid & (k == j)
        a = am[sel]; a = a[np.isfinite(a)]
        out[j, _C["abs_mlat"]] = np.median(a) if len(a) else np.nan
        b = Lc[sel]
        out[j, _C["L"]] = np.median(b) if len(b) else np.nan
    out[cnt == 0] = np.nan
    out[cnt == 0, _C["n_valid"]] = 0
    return out.astype(np.float32)


_INV = [1.0 / k for k in (21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1)]


def det_ln(v):
    """Natural logarithm from IEEE +, -, *, / only, operation for operation as the board program (``det_ln``): libm
    and numpy logarithms differ in the last bit between libraries, these do not. Within a few units in the last place."""
    import numpy as np
    v = np.asarray(v, np.float64)
    with np.errstate(all="ignore"):
        m, e = np.frexp(v)
        low = m < 0.7071067811865476
        m = np.where(low, m * 2.0, m)
        e = np.where(low, e - 1, e).astype(np.float64)
        s = (m - 1.0) / (m + 1.0)
        s2 = s * s
        p = np.full_like(s, 1.0 / 23.0)
        for c in _INV:
            p = p * s2 + c
        out = e * 0.6931471805599453 + 2.0 * s * p
    out = np.where(v == 0, -np.inf, out)
    out = np.where(np.isinf(v) & (v > 0), np.inf, out)
    return np.where(np.isnan(v) | (v < 0), np.nan, out)


def det_log10(v):
    """log10 as the board program (``det_log10``): ``det_ln`` times log10(e)."""
    return det_ln(v) * 0.4342944819032518


def _bins(am):
    import numpy as np
    return np.clip(np.digitize(np.nan_to_num(am, nan=0.0), np.arange(0, 95, 5)) - 1, 0, NBINS - 1)


def day_features(summary, eps):
    """The day's 79 features as float16 (orb columns zero) and the residual's log flux (bad steps missing), as the
    study's dataset forms them (build_dataset.py, zt_build.py)."""
    import numpy as np
    x = np.asarray(summary, np.float32)
    need(x.shape == (STEPS, len(SUMMARY_COLUMNS)), "summary: 288 x 36 float32")
    lf = det_log10(x[:, :NFLUX] + np.asarray(eps, np.float64))
    am = x[:, _C["abs_mlat"]]
    geo = np.stack([am / 90.0, np.minimum(x[:, _C["L"]], 20) / 20.0, x[:, _C["mlt_sin"]], x[:, _C["mlt_cos"]],
                    x[:, _C["polar"]], x[:, _C["saa"]], x[:, _C["ifc"]], x[:, _C["n_valid"]] / 150.0], -1)
    bad = (x[:, _C["ifc"]] > 0) | ~(x[:, _C["n_valid"]] >= 75)
    since = np.zeros(STEPS, np.float32)
    c = STEPS
    for k in range(STEPS):
        c = 0 if am[k] >= 60 else min(c + 1, STEPS)
        since[k] = c / 288.0
    feats = np.concatenate([lf, geo, np.zeros((STEPS, 23)), bad[:, None].astype(np.float32), since[:, None],
                            np.zeros((STEPS, NFLUX))], -1)
    feats = np.nan_to_num(feats.astype(np.float32), nan=0.0)
    lfm = det_log10(np.where(np.isfinite(x[:, :NFLUX]), x[:, :NFLUX], np.nan) + np.asarray(eps, np.float64))
    lfm[bad | ~np.isfinite(am)] = np.nan
    return feats.astype(np.float16), lfm, _bins(am)


def background(state, day, history=HISTORY, min_periods=MIN_PERIODS):
    """Per bin and channel the median of the stored daily medians of days day-history .. day-1 (pandas rolling median,
    closed left, min_periods non-missing)."""
    import numpy as np
    days = [d for d in state if day - history <= d < day]
    if not days:
        return np.full((NBINS, NFLUX), np.nan)
    v = np.stack([np.asarray(state[d], np.float64) for d in days])
    n = np.isfinite(v).sum(0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(v, axis=0)
    return np.where(n >= min_periods, med, np.nan)


def daily_medians(lfm, bins):
    import numpy as np
    import warnings
    out = np.full((NBINS, NFLUX), np.nan, np.float32)
    for j in range(NBINS):
        v = np.where((bins == j)[:, None], lfm, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[j] = np.nanmedian(v, axis=0)
    return out


def day_codes(summary, state, tables, ymd):
    """The day's input codes (int16 [288, C]) and the state with the day's medians: the board's ``day`` job."""
    import numpy as np
    day = _day_number(ymd)
    year = int(ymd[:4])
    scl = tables["scales"].get(year - 1)
    scl = tables["scales"][year] if scl is None else scl
    h, lfm, bins = day_features(summary, tables["eps"])
    bg = background(state, day, tables["history"], tables["min_periods"])
    with np.errstate(all="ignore"):
        z = (lfm - bg[bins]) / scl[bins]
    h = h.copy()
    h[:, ZT:] = np.nan_to_num(z, nan=0.0).clip(-20, 50).astype(np.float16)
    cols = np.asarray(tables["cols"])
    pad = cols == PAD_COL
    f = (np.asarray(h[:, np.where(pad, 0, cols)], np.float64) - tables["mu"]) / tables["sd"]
    top = (1 << (tables["bits"] - 1)) - 1
    codes = np.clip(np.floor(f * 2.0 ** tables["qpos"] + 0.5), -top - 1, top).astype(np.int16)
    codes[:, pad] = 0
    new = {d: m for d, m in state.items() if d % tables["history"] != day % tables["history"]}
    new[day] = daily_medians(lfm, bins)
    return codes, new


# IGRF-13 centered-dipole terms at 2020.0 (nT) and their secular variation (nT / year), valid to 2025.0; another epoch
# needs its terms (``sun_table`` dipole)
IGRF13_DIPOLE = (-29404.8, -1450.9, 4652.5, 5.7, 7.4, -25.9)
# the NOAA SEM-2 L1b MLT convention against the IGRF dipole: the median of 12 days of 2021 (MetOp-01, |mlat| < 60 deg)
SUN_OFFSET_DEG = -3.658


def _sun_geo(year, doy, minutes):
    """Unit vectors to the sun (geographic Cartesian: x to longitude 0, z north) at UT minutes of a day: the NOAA
    low-precision solar position (declination and equation of time, about 0.05 degrees)."""
    import numpy as np
    ndays = 366 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 365
    hours = np.asarray(minutes, np.float64) / 60.0
    gamma = 2 * np.pi / ndays * (doy - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma)
                       - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    lon = np.radians(-15.0 * (hours - 12.0) - eqtime / 4.0)
    return np.stack([np.cos(decl) * np.cos(lon), np.cos(decl) * np.sin(lon), np.sin(decl)], -1)


def sun_table(ymd, offset_deg=SUN_OFFSET_DEG, dipole=None):
    """The ``sum`` job's SUN table of a day (YYYYMMDD) from a ground model, without the day's data: cosine and sine of
    the sun's magnetic longitude per UT minute (float32 [1440, 2], minute centres): the subsolar direction in the
    centered dipole of the date (IGRF-13 by default, 2020.0-2025.0; ``dipole`` = g10, g11, h11 in nT for another
    epoch), turned by ``offset_deg`` (the L1b MLT convention, ``SUN_OFFSET_DEG``). On MetOp-01, January 2022, the 0-degree
    student's alarms over 25 days are those with the table fitted to the day's L1b MLT, step for step."""
    import datetime
    import numpy as np
    day = datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
    doy = day.timetuple().tm_yday
    if dipole is None:
        year = day.year + (doy - 0.5) / 365.25
        need(2020.0 <= year < 2025.0, "IGRF-13 dipole: 2020-2024; give the epoch's dipole terms")
        g10, g11, h11, d10, d11, dh11 = IGRF13_DIPOLE
        dt = year - 2020.0
        dipole = (g10 + d10 * dt, g11 + d11 * dt, h11 + dh11 * dt)
    g10, g11, h11 = dipole
    z = -np.array([g11, h11, g10], np.float64)
    z /= np.linalg.norm(z)
    y = np.cross([0.0, 0.0, 1.0], z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    s = _sun_geo(day.year, doy, np.arange(1440) + 0.5)
    lam = np.arctan2(s @ y, s @ x) + np.radians(offset_deg)
    return np.stack([np.cos(lam), np.sin(lam)], -1).astype(np.float32)


def windows(day_code_list, window, stride, first, frames=None):
    """The board's ``win`` job, or with ``frames`` its ``winpad`` job (that many windows, rows past the stream's end
    as code 0)."""
    import numpy as np
    stream = np.concatenate(day_code_list)
    if frames is None:
        starts = list(range(first, len(stream) - window + 1, stride))
        need(starts, "no whole window")
        return np.stack([stream[s:s + window] for s in starts])
    need(frames >= 1 and first < len(stream), "no window")
    out = np.zeros((frames, window) + stream.shape[1:], stream.dtype)
    for f in range(frames):
        part = stream[first + f * stride:first + f * stride + window]
        out[f, :len(part)] = part
    return out


def receptive_field(contract):
    """Rows of history an output needs: the sum of (kernel - 1) over the contract's depthwise layers."""
    return sum(layer["weight_shape"][1] - 1 for layer in contract["layers"] if layer["kind"] == "depthwise")


def exact_q(counts, scale):
    counts = np.asarray(counts)
    need(counts.dtype == np.float32, 'counts must be float32')
    wide = np.rint(counts.astype(np.float64)*scale)
    need(np.all(np.isfinite(counts)) and np.all(counts >= 0) and np.all(wide <= 2147483647),
         'counts must be finite, nonnegative, int32 representable')
    q = wide.astype(np.int32)
    need(np.array_equal((q.astype(np.float64)/scale).astype(np.float32), counts),
         'lossy count conversion rejected: not exactly on 1/scale grid')
    return q


def read_counts(path, channels, scale=8):
    """Read L1a netCDF or NPZ; never infer count-channel order from folder names."""
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(str(path), allow_pickle=False) as d:
            names = d['channels'].tolist()
            need(len(names)==len(set(names)) and all(c in names for c in channels), 'missing/duplicate count channels')
            counts = d['counts'][:, [names.index(c) for c in channels]]
            aux = d['auxiliary']
    else:
        from netCDF4 import Dataset
        with Dataset(str(path), 'r') as ds:
            def read(name):
                a = ds[name][:]
                need(not np.ma.isMaskedArray(a) or not np.any(np.ma.getmaskarray(a)), 'masked values: '+name)
                a = np.asarray(a, np.float32).reshape(-1)
                need(np.all(np.isfinite(a)), 'nonfinite values: '+name)
                return a
            counts = np.column_stack([read(c) for c in channels])
            lat, lon = read('lat'), read('lon')
            need(len(lat)>=2 and len(lat)==len(lon)==len(counts), 'coordinate length mismatch')
            need(np.all(np.abs(lat)<=90) and np.all(np.abs(lon)<=360), 'invalid coordinates')
            derived = (np.gradient(lat.astype(np.float64)) < 0).astype(np.float32)
            direction = read('sat_direction') if 'sat_direction' in ds.variables else derived
            need(len(direction)==len(counts), 'direction length mismatch')
            if len(np.unique(direction)) <= 1:
                direction = derived
        need(np.all((direction==0)|(direction==1)), 'sat_direction must use 0/1 convention')
        phase = np.mod(np.where(direction < .5, (lat.astype(np.float64)+90)/360,
                      .5+(90-lat.astype(np.float64))/360), 1).astype(np.float32)
        la, lo = np.deg2rad(lat.astype(np.float64)), np.deg2rad(lon.astype(np.float64))
        aux = np.column_stack((np.sin(la),np.cos(la),np.sin(lo),np.cos(lo),direction,phase)).astype(np.float32)
    need(counts.ndim==2 and len(counts)>=2 and aux.shape==(len(counts),6) and np.all(np.isfinite(aux)),
         'counts [T,C], auxiliary [T,6], T>=2 required')
    return exact_q(counts, scale), np.asarray(aux, np.float32)


def save_counts(path, q, aux, channels, scale=8):
    need(not Path(path).exists(), 'output already exists: '+str(path))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'wb') as f:
        np.savez(f, counts=(q.astype(np.float64)/scale).astype(np.float32), auxiliary=aux,
                 channels=np.asarray(channels), scale=np.asarray(scale))


def comp_rows(q, aux, model):
    """Match directrtl_comp_mtcn.row_codes for reference; continuous FP32 for float port."""
    src = model.contract['source']
    c = len(src['channels'])
    need(q.ndim==2 and q.shape[1]==c and aux.shape==(len(q),6), 'Comp input shape mismatch')
    out = np.zeros((len(q), model.input_shape[1]), np.float64)
    out[:, :c] = (det_ln((q.astype(np.float64)+model.scale)/model.scale)-np.asarray(src['mean']))/np.asarray(src['std'])
    out[:, c:c+6] = aux
    if model.backend == 'reference':
        out = np.clip(np.floor(out*2.0**model.contract['input_qpos']+.5), -32768, 32767).astype(np.int16)
    else:
        out = out.astype(np.float32)
    return out


def comp_output_counts(output, model):
    # Deterministic expm1 polynomial from directrtl_comp_mtcn.
    import math
    src = model.contract['source']
    normalized = np.asarray(output, np.float64)
    if model.backend == 'reference':
        normalized *= 2.0**(-model.contract['output_qpos'])
    y = normalized*np.asarray(src['std'])+np.asarray(src['mean'])
    need(np.all(np.isfinite(y)), 'nonfinite Comp prediction')
    with np.errstate(all='ignore'):
        k = np.floor(y*1.4426950408889634+.5)
        r = (y-k*6.93147180369123816490e-01)-k*1.90821492927058770002e-10
        v = np.full_like(r, 1.0/math.factorial(13))
        for n in range(12, 0, -1):
            v = v*r+1.0/math.factorial(n)
        v = np.ldexp(v*r+1.0, np.clip(k,-1100,1100).astype(np.int32))-1.0
    v = np.where(y>709, np.inf, np.where(y < -708, -1.0, v))
    return np.rint(np.minimum(np.maximum(v,0)*model.scale,2147483647)).astype(np.int32)


def framed(rows, length, history, context=None):
    """No future history; pad final frame and report only real output rows."""
    stride = length-history
    need(len(rows)>0 and stride>0, 'invalid frame geometry')
    if context is None:
        context = np.zeros((history,rows.shape[1]), rows.dtype)
    need(context.shape==(history,rows.shape[1]), 'context must contain exactly history rows')
    stream = np.concatenate((context, rows))
    frames, valid = [], []
    for start in range(0,len(rows),stride):
        frame = np.zeros((length,rows.shape[1]),rows.dtype)
        part = stream[start:start+length]
        frame[:len(part)] = part
        frames.append(frame)
        valid.append(min(stride,len(rows)-start))
    return np.stack(frames), np.asarray(valid,np.int64)


def save_input(path, x, model, valid_rows, source, synthetic=False, **extra):
    model.tensor(x[:1])  # validate dtype/range/shape before writing
    need(np.all(np.isfinite(x)), 'nonfinite input')
    valid = np.asarray(valid_rows,np.int64)
    need(valid.shape==(len(x),) and np.all(valid>0) and np.all(valid<=x.shape[1]), 'invalid valid_rows')
    path = Path(path)
    need(not path.exists(), 'output already exists: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(str(path),'wb') as f:
        np.savez(f,x=x,model=np.asarray(model.name),backend=np.asarray(model.backend),
                 weight_sha256=np.asarray(model.sha),valid_rows=valid,synthetic=np.asarray(synthetic),
                 source=np.asarray(source),**extra)


def load_input(path, model):
    with np.load(str(path),allow_pickle=False) as d:
        need(str(d['model'])==model.name and str(d['backend'])==model.backend, 'input model/backend mismatch; prepare again')
        need(str(d['weight_sha256'])==model.sha, 'input prepared with different weights')
        x, valid = d['x'], d['valid_rows']
        synthetic = bool(d['synthetic'])
        source = str(d['source'])
        expected = d['expected'] if 'expected' in d else None
    need(len(x)>0 and valid.shape==(len(x),) and valid.dtype.kind in 'iu' and np.all(valid>0)
         and np.all(valid<=x.shape[1]), 'invalid input metadata')
    return model.tensor(x), {'synthetic':synthetic,'source':source,'valid_rows':valid.tolist()}, expected


def prepare_counts(args, model):
    q, aux = read_counts(args.input,model.channels,model.scale)
    if model.name=='gru':
        ck=model.checkpoint
        counts=(q.astype(np.float64)/model.scale).astype(np.float32)
        x=np.concatenate(((np.log1p(counts)-np.asarray(ck['mean'],np.float32))/np.asarray(ck['std'],np.float32),aux),axis=1)
        x=x[:-1,None,:].astype(np.float32)
        valid=np.ones(len(x),np.int64)
    else:
        rows=comp_rows(q[:-1],aux[:-1],model)
        x,valid=framed(rows,model.input_shape[0],receptive_field(model.contract))
    save_input(args.output,x,model,valid,'counts:'+str(args.input),cold_start=np.asarray(True))


def prepare_bus(args, model):
    need(args.stats is not None, 'BUS requires training statistics NPZ: channels, mean, std. Not bundled upstream.')
    with np.load(args.stats,allow_pickle=False) as s:
        names=s['channels'].tolist(); mean=s['mean']; std=s['std']
    need(len(names)==12 and len(set(names))==12 and mean.shape==std.shape==(12,)
         and np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std>0), 'invalid BUS statistics')
    with open(args.input,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        need(args.time_column in reader.fieldnames and all(n in reader.fieldnames for n in names), 'CSV columns missing')
        records=list(reader)
    times=np.asarray([float(r[args.time_column]) for r in records],np.float64)
    need(len(times)>=2 and np.all(np.isfinite(times)) and np.all(np.diff(times)>0), 'time column must be increasing seconds')
    grid=np.arange(times[0],times[-1]+1e-8,50.0)
    columns=[]
    for name in names:
        vals=np.asarray([float(r[name]) if r[name].strip() else np.nan for r in records])
        ok=np.isfinite(vals)
        need(np.count_nonzero(ok)>=2 and np.mean(~ok)<.25, 'insufficient valid BUS column: '+name)
        columns.append(np.interp(grid,times[ok],vals[ok]))
    rows=(np.stack(columns,axis=1)-mean)/std
    need(len(rows)>=64, 'BUS requires at least 64 resampled rows')
    x=np.stack([rows[i:i+64] for i in range(0,len(rows)-63,16)])
    if model.backend=='reference':
        low,high=ic_range(model.contract['input_bits'])
        x=np.clip(np.floor(x*2.0**model.contract['input_qpos']+.5),low,high).astype(np.int16)
    else:
        x=x.astype(np.float32)
    save_input(args.output,x,model,np.full(len(x),16),str(args.input),
               preprocessing=np.asarray('50s linear interpolation; origin first timestamp; requires reference-output validation'),
               stats_sha256=np.asarray(hashlib.sha256(Path(args.stats).read_bytes()).hexdigest()))


def ic_range(bits):
    return -(1<<(bits-1)),(1<<(bits-1))-1


def prepare_pd(args, model):
    need(args.tables is not None, 'PD requires mentor ASTRAZT1 tables (eps, cols, mu, sd, yearly scales)')
    need(args.state is not None, 'PD requires previous-day history NPZ; do not silently replace by an empty state')
    need(args.date is not None, 'PD requires --date YYYYMMDD')
    state_path=Path(args.output).with_suffix('.state.npz')
    need(not state_path.exists(), 'state output already exists')
    tables=read_tables(Path(args.tables).read_bytes())
    need(np.all(np.isfinite(tables['mu'])) and np.all(np.isfinite(tables['sd'])) and np.all(tables['sd']>0), 'invalid PD normalization')
    need(tables['qpos']==model.contract['input_qpos'] and tables['bits']==model.contract['input_bits'], 'PD table/contract precision mismatch')
    with np.load(args.state,allow_pickle=False) as s:
        days, medians=s['days'],s['medians']
        need(medians.shape==(len(days),NBINS,NFLUX), 'history shape must be [days,18,23]')
        state={int(d):m for d,m in zip(days,medians)}
    day=_day_number(args.date)
    need(all(day-tables['history']<=d<day for d in state) and len(state)>=tables['min_periods'], 'PD history must precede target day')
    if args.records:
        need(args.geo is not None and args.sun is not None, 'raw PD records need geo and sun tables')
        summary=day_summary(Path(args.records).read_bytes(),read_geo(Path(args.geo).read_bytes()),np.load(args.sun),args.date)
    else:
        need(args.input is not None, 'give --input summary.npy or --records binary')
        summary=np.load(args.input,allow_pickle=False)
    rows,new_state=day_codes(summary,state,tables,args.date)
    need(rows.shape[1]==model.contract['input_shape'][1], 'PD table columns differ from contract')
    need(args.context is not None, 'PD requires previous-day input codes .npy for causal context')
    if Path(args.context).suffix=='.npz':
        with np.load(args.context,allow_pickle=False) as ctx:
            previous=ctx['day_codes']
    else:
        previous=np.load(args.context,allow_pickle=False)
    need(previous.ndim==2 and previous.shape[1]==rows.shape[1], 'PD context shape mismatch')
    rf=receptive_field(model.contract)
    # Checkpoints use 28 inputs; integer contract pads four zero-weight inputs.
    current_codes=rows.copy()
    if model.backend=='float':
        need(np.all(tables['cols'][28:]==PAD_COL), 'only trailing padded inputs can be removed')
        rows=(rows[:,:28]*2.0**(-tables['qpos'])).astype(np.float32)
        previous=(previous[:,:28]*2.0**(-tables['qpos'])).astype(np.float32)
    x,valid=framed(rows,model.input_shape[0],rf,previous[-rf:])
    save_input(args.output,x,model,valid,'PD:'+args.date,
               tables_sha256=np.asarray(hashlib.sha256(Path(args.tables).read_bytes()).hexdigest()),
               day_codes=current_codes)
    np.savez(str(state_path),days=np.asarray(sorted(new_state)),medians=np.stack([new_state[d] for d in sorted(new_state)]))
