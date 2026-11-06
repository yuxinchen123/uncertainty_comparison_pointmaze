"""Rebuild the alternative sprite pack with Montezuma room images whose geometry matches the game.

The alternative pack (archive sha256 7f4ef286..., the only non-ROM pack JAXtari offers) ships each
Montezuma room image twice: a .png dated 2026-05-02 and a .npy dated 2026-10-01. The game and the
renderer read only the .npy files. For five rooms (the start room, the level-1 middle room, the pit
room and the level-2 rooms 0 and 6) the .npy collision map and background are a copy of the corridor
room: solid everywhere below row 47. The player can then reach the lower part of the start room only
by climbing, cannot walk there, is carried by the conveyor to x = 54 and stays there forever, and no
fall is ever long enough to lose a life. The .png images of the same pack have the real room layout
(the corridor .png and .npy collision maps are identical, as are the bonus-room ones).

This script copies the pack and replaces every Montezuma background and collision-map .npy that has a
same-size .png by that .png's pixels (all such .png images are opaque, so RGB is exact). Character,
item, enemy, door, digit and floor sprites keep the pack's .npy art. Standard library only: no
numerical runtime is started, so it may run on a login node.

    python3 build_montezuma_room_geometry_assets.py SOURCE_ROOT DESTINATION_ROOT

SOURCE_ROOT and DESTINATION_ROOT are XDG data roots: each holds jaxtari/sprites/.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import zlib


def sha256(path):
    """Hex digest of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_png(path):
    """Width, height, and rows of RGBA tuples of a non-interlaced 8-bit PNG."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    position, chunks, compressed = 8, {}, b""
    while position < len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        body = data[position + 8:position + 8 + length]
        if kind == b"IDAT":
            compressed += body
        else:
            chunks.setdefault(kind, body)
        position += 12 + length
    width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    if depth != 8 or interlace != 0 or colour not in (2, 6):
        raise ValueError(f"{path}: unsupported PNG (depth {depth}, colour type {colour}, interlace {interlace})")
    channels = 3 if colour == 2 else 4
    raw, stride = zlib.decompress(compressed), width * channels
    rows, previous = [], bytearray(stride)
    for row in range(height):
        start = row * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        for i in range(stride):
            left = line[i - channels] if i >= channels else 0
            up = previous[i]
            upper_left = previous[i - channels] if i >= channels else 0
            if kind == 1:
                line[i] = (line[i] + left) & 255
            elif kind == 2:
                line[i] = (line[i] + up) & 255
            elif kind == 3:
                line[i] = (line[i] + (left + up) // 2) & 255
            elif kind == 4:
                estimate = left + up - upper_left
                a, b, c = abs(estimate - left), abs(estimate - up), abs(estimate - upper_left)
                line[i] = (line[i] + (left if a <= b and a <= c else up if b <= c else upper_left)) & 255
        rows.append([tuple(line[c * channels:(c + 1) * channels]) + ((255,) if channels == 3 else ())
                     for c in range(width)])
        previous = line
    return width, height, rows


def npy_header(path):
    """(shape, dtype description) of a .npy file."""
    data = Path(path).read_bytes()
    major = data[6]
    length = int.from_bytes(data[8:10] if major == 1 else data[8:12], "little")
    offset = 10 if major == 1 else 12
    header = ast.literal_eval(data[offset:offset + length].decode("latin1"))
    return tuple(header["shape"]), header["descr"]


def write_npy(path, height, width, channels, values):
    """Write a C-ordered uint8 [height, width, channels] array in .npy version 1.0 format."""
    header = repr({"descr": "|u1", "fortran_order": False, "shape": (height, width, channels)})
    header = header.encode("latin1")
    padding = 64 - (10 + len(header) + 1) % 64
    header += b" " * (padding % 64) + b"\n"
    Path(path).write_bytes(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + bytes(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    args = parser.parse_args()
    source, destination = args.source_root / "jaxtari", args.destination_root / "jaxtari"
    if destination.exists():
        raise SystemExit(f"{destination} exists; choose a new destination")
    shutil.copytree(source, destination, symlinks=False)
    backgrounds = destination / "sprites" / "montezuma" / "backgrounds"
    replaced, kept = [], []
    for npy in sorted(backgrounds.glob("*.npy")):
        png = npy.with_suffix(".png")
        shape, description = npy_header(npy)
        if not png.exists():
            kept.append({"file": npy.name, "reason": "no png"})
            continue
        width, height, rows = decode_png(png)
        if description != "|u1" or len(shape) != 3 or shape[:2] != (height, width):
            kept.append({"file": npy.name, "reason": f"npy {shape} {description} against png {height}x{width}"})
            continue
        if any(pixel[3] != 255 for row in rows for pixel in row):
            kept.append({"file": npy.name, "reason": "png has transparent pixels"})
            continue
        channels = shape[2]
        values = bytearray()
        for row in rows:
            for pixel in row:
                values.extend(pixel[:channels])
        before = sha256(npy)
        write_npy(npy, height, width, channels, values)
        after = sha256(npy)
        replaced.append({"file": npy.name, "png_sha256": sha256(png), "npy_sha256_before": before,
                         "npy_sha256_after": after, "changed": before != after})
    source_record = json.loads((source / "asset_source.json").read_text())
    record = {
        "base_pack": source_record,
        "base_root": str(source),
        "built_by": str(Path(__file__).resolve()),
        "rule": ("Montezuma backgrounds/*.npy with a same-size opaque .png are replaced by that .png's RGB "
                 "pixels; every other file is copied unchanged."),
        "reason": ("The pack's 2026-10-01 .npy collision maps and backgrounds of the start room, the level-1 "
                   "middle room, the pit room and the level-2 rooms 0 and 6 are a copy of the corridor room "
                   "(solid below row 47), which traps the player on the start room's conveyor and prevents "
                   "every fall death; the pack's 2026-05-02 .png images have the real room layout."),
        "replaced": replaced,
        "kept": kept,
    }
    (destination / "asset_source.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"replaced_and_changed": [r["file"] for r in replaced if r["changed"]],
                      "replaced_identical": [r["file"] for r in replaced if not r["changed"]],
                      "kept": kept}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
