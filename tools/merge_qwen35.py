#!/usr/bin/env python3
"""Merge Qwen/Qwen3.5-4B's two shards into the single file ComfyUI's text_encoders folder expects.

    python tools/merge_qwen35.py ~/AI/hf_originals/Qwen3.5-4B ~/AI/ComfyUI/models/text_encoders/qwen3.5_4b_bf16.safetensors

The tensors are copied byte for byte (bf16, plus the fp32 tensors Qwen ships), one at a time, with the keys unchanged,
so this needs no torch and almost no memory. The output also loads in the stock Load CLIP node.
"""
import json
import os
import struct
import sys


def read_header(path):
    with open(path, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def main(source, output):
    with open(os.path.join(source, "model.safetensors.index.json")) as fh:
        index = json.load(fh)["weight_map"]
    shards = {}
    entries = {}
    for shard in sorted(set(index.values())):
        path = os.path.join(source, shard)
        header, start = read_header(path)
        shards[shard] = (path, start)
        for key, info in header.items():
            if index.get(key) != shard:
                raise SystemExit(f"{key} is in {shard}, but the index says {index.get(key)}")
            entries[key] = (shard, info)
    if set(entries) != set(index):
        raise SystemExit(f"index and shards disagree on {len(set(entries) ^ set(index))} keys")

    offset = 0
    out_header = {"__metadata__": {"format": "pt", "source": "Qwen/Qwen3.5-4B"}}
    order = sorted(entries)
    for key in order:
        _, info = entries[key]
        begin, end = info["data_offsets"]
        out_header[key] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)

    part = output + ".part"
    handles = {shard: open(path, "rb") for shard, (path, _) in shards.items()}
    try:
        with open(part, "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for key in order:
                shard, info = entries[key]
                begin, end = info["data_offsets"]
                handle = handles[shard]
                handle.seek(shards[shard][1] + begin)
                remaining = end - begin
                while remaining:
                    chunk = handle.read(min(remaining, 1 << 24))
                    if not chunk:
                        raise SystemExit(f"{shard} ended early at {key}")
                    out.write(chunk)
                    remaining -= len(chunk)
    finally:
        for handle in handles.values():
            handle.close()
    expected = 8 + len(blob) + offset
    if os.path.getsize(part) != expected:
        raise SystemExit(f"wrote {os.path.getsize(part)} bytes, expected {expected}")
    os.replace(part, output)
    print(f"{output}: {len(order)} tensors, {expected / 2**30:.2f} GiB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
