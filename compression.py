# Adapted common BZ2 codec; neural prediction cost remains inside encode.
"""BZ2 hybrid 파일 프로토콜 v1. 채널별 압축, 모든 채널을 시점별로 동기 복원."""
import bz2
import hashlib
import json
import struct
import time
import zlib
from collections import Counter

import numpy as np

MAGIC = b"SJTC2\x01\r\n"
I32 = np.iinfo(np.int32)


def checked_i32(values):
    values = np.asarray(values)
    # 공개 encode API에 실수/문자열이 들어오면 int64 변환에서 조용히 잘릴 수 있다.
    # 원본 정수의 무손실 계약을 지키기 위해 정수 dtype만 허용한다.
    if values.dtype.kind not in "iu":
        raise ValueError("count/residual values must be integers")
    if np.any(values < I32.min) or np.any(values > I32.max):
        raise ValueError("integer difference/reconstruction exceeds int32")
    return values.astype("<i4")


def zigzag(values):
    # 부호를 최하위 비트로 옮겨 작은 음수도 작은 양수처럼 표현한다.
    r = checked_i32(values).astype(np.int64)
    return ((r << 1) ^ (r >> 63)).astype("<u4")


def unzigzag(values):
    u = values.astype(np.int64)
    return checked_i32((u >> 1) ^ -(u & 1))


def encode(q, auxiliary, predictor, block_samples=1800, level=5, mode="neural"):
    if mode not in ("raw", "delta", "neural"):
        raise ValueError("mode must be raw, delta or neural")
    """완전한 파일과 실제 파일 전체 크기 기준의 메트릭을 반환한다."""
    started = time.perf_counter()
    q = checked_i32(q)
    c = len(predictor.channels)
    auxiliary = np.asarray(auxiliary, dtype="<f4")
    if (q.ndim != 2 or q.shape[1] != c or len(q) < 2 or np.any(q < 0)
            or auxiliary.shape != (len(q), 6) or not np.all(np.isfinite(auxiliary))
            or block_samples <= 0 or not 1 <= level <= 9):
        raise ValueError("invalid input shapes, values or codec settings")
    predictor.reset()
    n = len(q)-1
    predictions = np.empty((n, c), dtype="<i4")
    if mode == "neural":
        for t in range(n):
            predictions[t] = predictor.step(q[t], auxiliary[t])
    else:
        predictions.fill(0)
    # 빼기는 반드시 int64에서 한다. int32 차이가 넘치면 조용히 잘라내지 않는다.
    delta = checked_i32(q[1:].astype(np.int64)-q[:-1].astype(np.int64))
    residual = checked_i32(q[1:].astype(np.int64)-predictions.astype(np.int64))
    records, payloads = [], []
    modes = Counter()
    for block, start in enumerate(range(0, n, block_samples)):
        end = min(start+block_samples, n)
        for channel in range(c):
            candidates = {"raw": bz2.compress(q[1+start:1+end, channel].astype("<i4").tobytes(), level)}
            if mode in ("delta", "neural"):
                candidates["delta"] = bz2.compress(zigzag(delta[start:end, channel]).tobytes(), level)
            if mode == "neural" and channel in predictor.predicted_indices:
                candidates["neural"] = bz2.compress(zigzag(residual[start:end, channel]).tobytes(), level)
            # 실제 메타데이터 길이까지 포함해 선택한다. 동률은 raw→delta→gru.
            options = []
            for mode, payload in candidates.items():
                record = dict(block=block, start=start, count=end-start, channel=channel,
                              mode=mode, bytes=len(payload), crc32=zlib.crc32(payload))
                options.append((len(payload)+len(json_bytes(record)), record, payload))
            _, record, payload = min(options, key=lambda x: x[0])
            records.append(record)
            payloads.append(payload)
            modes[record["mode"]] += 1
    # 지상 예측을 위해 초기 전체 채널 q와 각 시점의 6개 보조 입력을 함께 보낸다.
    # 보조 입력 비용도 파일 전체 메트릭에 포함하며 숨은 외부 데이터에 의존하지 않는다.
    aux_payload = bz2.compress(auxiliary[:-1].tobytes(), level) if mode == "neural" else b""
    header = dict(version=1, mode=mode, predictor=predictor.description, codec="bz2", level=level, dtype="int32-le", samples=len(q),
                  channels=predictor.channels, scale=predictor.scale, model_sha256=predictor.model_id,
                  block_samples=block_samples, initial_q=q[0].tolist(),
                  aux_bytes=len(aux_payload), aux_crc32=zlib.crc32(aux_payload), records=records,
                  q_sha256=hashlib.sha256(q.astype("<i4").tobytes()).hexdigest())
    h = json_bytes(header)
    packet = MAGIC + struct.pack("<I", len(h)) + h + aux_payload + b"".join(payloads)
    raw_bytes = int(q.nbytes)
    metrics = dict(codec="BZ2 hybrid", count_raw_bytes=raw_bytes, packet_bytes=len(packet),
                   count_to_packet_ratio=raw_bytes/len(packet),
                   auxiliary_payload_bytes=len(aux_payload),
                   selected_count_payload_bytes=sum(map(len, payloads)),
                   header_bytes=len(MAGIC)+4+len(h), mode_blocks=dict(modes),
                   samples=len(q), channels=c, block_samples=block_samples,
                   encode_seconds=time.perf_counter()-started, model_sha256=predictor.model_id)
    return packet, metrics


def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def decompress_exact(payload, expected):
    # 헤더에서 선언한 크기 이상으로 압축 해제하지 않는다. 붙은 스트림/잔여 데이터도 거부.
    decoder = bz2.BZ2Decompressor()
    data = decoder.decompress(payload, max_length=expected+1)
    if len(data) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError("BZ2 payload size/stream mismatch")
    return data


def decode(packet, predictor):
    """초기 q 및 전송 보조값으로 동일 GRU를 실행해 원본 정수 배열을 복원한다."""
    if not packet.startswith(MAGIC) or len(packet) < len(MAGIC)+4:
        raise ValueError("invalid packet magic/version")
    offset = len(MAGIC)+4
    header_size = struct.unpack("<I", packet[len(MAGIC):offset])[0]
    if header_size > len(packet)-offset:
        raise ValueError("truncated header")
    h = json.loads(packet[offset:offset+header_size])
    offset += header_size
    if (h["version"] != 1 or h["codec"] != "bz2" or h["dtype"] != "int32-le"
            or h["model_sha256"] != predictor.model_id or h["channels"] != predictor.channels
            or h["scale"] != predictor.scale or h["predictor"] != predictor.description
            or h["mode"] not in ("raw", "delta", "neural")):
        raise ValueError("packet/model metadata mismatch")
    n, c, block_samples = int(h["samples"])-1, len(h["channels"]), int(h["block_samples"])
    if n < 1 or block_samples < 1:
        raise ValueError("invalid sample/block counts")

    def take(size, crc):
        nonlocal offset
        if not isinstance(size, int) or size < 0 or size > len(packet)-offset:
            raise ValueError("truncated/invalid payload length")
        data = packet[offset:offset+size]
        offset += size
        if zlib.crc32(data) != crc:
            raise ValueError("payload CRC32 mismatch")
        return data

    aux = (np.frombuffer(decompress_exact(take(h["aux_bytes"], h["aux_crc32"]), n*6*4), dtype="<f4").reshape(n, 6)
           if h["mode"] == "neural" else np.zeros((n, 6), np.float32))
    if not np.all(np.isfinite(aux)):
        raise ValueError("nonfinite auxiliary inputs")
    q = np.empty((n+1, c), dtype="<i4")
    initial = checked_i32(h["initial_q"])
    if initial.shape != (c,) or np.any(initial < 0):
        raise ValueError("invalid initial q")
    q[0] = initial
    records = iter(h["records"])
    predictor.reset()
    for block, start in enumerate(range(0, n, block_samples)):
        count = min(block_samples, n-start)
        arrays, modes = [], []
        for channel in range(c):
            r = next(records, None)
            if (r is None or (r["block"], r["start"], r["count"], r["channel"])
                    != (block, start, count, channel) or r["mode"] not in ("raw", "delta", "neural")):
                raise ValueError("invalid block/channel ordering or mode")
            data = decompress_exact(take(r["bytes"], r["crc32"]), count*4)
            values = np.frombuffer(data, dtype="<i4" if r["mode"] == "raw" else "<u4")
            arrays.append(values if r["mode"] == "raw" else unzigzag(values))
            modes.append(r["mode"])
        for i in range(count):
            t = start+i
            # raw/delta 구간에서도 GRU 상태를 진행해야 다음 GRU 구간이 맞는다.
            prediction = predictor.step(q[t], aux[t]) if h["mode"] == "neural" else np.zeros(c, np.int32)
            restored = np.asarray([int(arrays[ch][i]) +
                                   (int(q[t, ch]) if modes[ch] == "delta" else
                                    int(prediction[ch]) if modes[ch] == "neural" else 0)
                                   for ch in range(c)], dtype=np.int64)
            q[t+1] = checked_i32(restored)
    if next(records, None) is not None or offset != len(packet) or np.any(q < 0):
        raise ValueError("extra records/bytes or invalid restored q")
    # 부동소수점 런타임 차이로 예측 정수가 달라져도 성공으로 보고하지 않는다.
    if hashlib.sha256(q.tobytes()).hexdigest() != h["q_sha256"]:
        raise ValueError("exact reconstruction hash mismatch; check model/runtime/preprocessing")
    return q, h
