#!/usr/bin/env python3
"""Compare startup-matrix captures against a platform baseline.

Pure-Python PNG decoding (8-bit non-interlaced grayscale/RGB/RGBA), no third
party dependencies. Reports the differing-pixel fraction and channel deltas;
fails on dimension mismatch, blank frames, or differences beyond tolerance.

Usage: compare_captures.py BASELINE.png CANDIDATE.png [--max-diff-fraction F]
       [--max-channel-delta D] [--label NAME]
"""
import argparse
import struct
import sys
import zlib


def decode_png(path):
    data = open(path, 'rb').read()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'{path}: not a PNG file')
    pos = 8
    idat = b''
    width = height = channels = None
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if chunk_type == b'IHDR':
            width, height, depth, color, _, _, interlace = struct.unpack('>IIBBBBB', chunk)
            if depth != 8 or interlace != 0:
                raise ValueError(f'{path}: unsupported PNG format (depth {depth}, interlace {interlace})')
            channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
        elif chunk_type == b'IDAT':
            idat += chunk
        elif chunk_type == b'IEND':
            break
    raw = zlib.decompress(idat)
    stride = width * channels
    out = bytearray(height * stride)
    prev = bytearray(stride)
    p = 0
    for y in range(height):
        filt = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if filt == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 255
        elif filt == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif filt == 3:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif filt == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 255
        elif filt != 0:
            raise ValueError(f'{path}: unknown PNG filter {filt}')
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return width, height, channels, out


def decode_ppm(path):
    data = open(path, 'rb').read()
    if data[:2] != b'P6':
        raise ValueError(f'{path}: not a binary PPM file')
    pos = 2
    tokens = []
    while len(tokens) < 3:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            while pos < len(data) and data[pos:pos + 1] != b'\n':
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(int(data[start:pos]))
    pos += 1  # single whitespace after maxval
    width, height, maxval = tokens
    if maxval != 255:
        raise ValueError(f'{path}: unsupported PPM maxval {maxval}')
    pixels = data[pos:pos + width * height * 3]
    if len(pixels) != width * height * 3:
        raise ValueError(f'{path}: truncated PPM pixel data')
    return width, height, 3, bytearray(pixels)


def decode_image(path):
    with open(path, 'rb') as f:
        magic = f.read(2)
    if magic == b'P6':
        return decode_ppm(path)
    return decode_png(path)


def luminance_stats(pixels, channels):
    # Sampled variance to detect blank frames without a full second pass.
    total = 0
    total_sq = 0
    count = 0
    step = max(channels, len(pixels) // 200000 // channels * channels)
    for i in range(0, len(pixels) - channels + 1, step):
        if channels >= 3:
            lum = (pixels[i] * 299 + pixels[i + 1] * 587 + pixels[i + 2] * 114) // 1000
        else:
            lum = pixels[i]
        total += lum
        total_sq += lum * lum
        count += 1
    mean = total / count
    variance = total_sq / count - mean * mean
    return mean, variance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    parser.add_argument('--max-diff-fraction', type=float, default=0.05)
    parser.add_argument('--max-channel-delta', type=int, default=32)
    parser.add_argument('--max-high-delta-fraction', type=float, default=0.01)
    parser.add_argument('--max-mean-delta', type=float, default=60.0)
    parser.add_argument('--label', default='capture')
    args = parser.parse_args()

    bw, bh, bc, base = decode_image(args.baseline)
    cw, ch, cc, cand = decode_image(args.candidate)
    label = args.label
    if (bw, bh) != (cw, ch):
        print(f'{label}: FAIL dimension mismatch {bw}x{bh} vs {cw}x{ch}')
        return 1
    bmean, bvar = luminance_stats(base, bc)
    cmean, cvar = luminance_stats(cand, cc)
    if bvar < 16.0 or cvar < 16.0:
        print(f'{label}: FAIL blank frame (baseline var {bvar:.1f}, candidate var {cvar:.1f})')
        return 1
    n = min(bc, cc)
    stride_b = bw * bc
    stride_c = cw * cc
    diff_pixels = 0
    over_delta = 0
    max_delta = 0
    sum_delta = 0
    for y in range(bh):
        rb = base[y * stride_b:(y + 1) * stride_b]
        rc = cand[y * stride_c:(y + 1) * stride_c]
        for x in range(bw):
            ob = x * bc
            oc = x * cc
            d = 0
            for k in range(n):
                delta = abs(rb[ob + k] - rc[oc + k])
                if delta > d:
                    d = delta
            if d > 0:
                diff_pixels += 1
                sum_delta += d
                if d > max_delta:
                    max_delta = d
                if d > args.max_channel_delta:
                    over_delta += 1
    total = bw * bh
    fraction = diff_pixels / total
    mean_delta = sum_delta / diff_pixels if diff_pixels else 0.0
    print(f'{label}: diff {diff_pixels}/{total} ({fraction * 100:.3f}%), '
          f'mean delta {mean_delta:.2f}, max delta {max_delta}, '
          f'over-threshold pixels {over_delta}')
    if fraction > args.max_diff_fraction:
        print(f'{label}: FAIL differing fraction {fraction:.4f} > {args.max_diff_fraction}')
        return 1
    if over_delta > total * args.max_high_delta_fraction:
        print(f'{label}: FAIL too many high-delta pixels {over_delta}')
        return 1
    if mean_delta > args.max_mean_delta:
        print(f'{label}: FAIL mean delta {mean_delta:.2f} > {args.max_mean_delta}')
        return 1
    print(f'{label}: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())