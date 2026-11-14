#!/usr/bin/env python3
"""Generate the 32x32 scene node icons used by the Viewer scene graph window.

Requires Pillow. Output: Asset/Textures/Icons/scene_<kind>.png
"""
import math
import os

from PIL import Image, ImageDraw

SIZE = 32
SCALE = 8  # supersampling factor for anti-aliasing
S = SIZE * SCALE
WHITE = (255, 255, 255, 255)


def canvas():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def background(draw, color):
    draw.rounded_rectangle((S * 0.06, S * 0.06, S * 0.94, S * 0.94),
                           radius=S * 0.2, fill=color)


def p(x, y):
    return (x * S, y * S)


def icon_empty():
    img, d = canvas()
    background(d, (120, 130, 145, 255))
    d.ellipse((S * 0.3, S * 0.3, S * 0.7, S * 0.7), outline=WHITE,
              width=int(S * 0.07))
    d.ellipse((S * 0.45, S * 0.45, S * 0.55, S * 0.55), fill=WHITE)
    return img


def icon_geometry():
    img, d = canvas()
    background(d, (45, 120, 210, 255))
    top = [p(0.5, 0.2), p(0.8, 0.35), p(0.5, 0.5), p(0.2, 0.35)]
    left = [p(0.2, 0.35), p(0.5, 0.5), p(0.5, 0.82), p(0.2, 0.67)]
    right = [p(0.8, 0.35), p(0.5, 0.5), p(0.5, 0.82), p(0.8, 0.67)]
    d.polygon(top, fill=(255, 255, 255, 255))
    d.polygon(left, fill=(200, 220, 250, 255))
    d.polygon(right, fill=(150, 185, 235, 255))
    return img


def icon_light():
    img, d = canvas()
    background(d, (235, 170, 20, 255))
    cx, cy = S * 0.5, S * 0.5
    d.ellipse((cx - S * 0.14, cy - S * 0.14, cx + S * 0.14, cy + S * 0.14),
              fill=WHITE)
    for i in range(8):
        a = i * math.pi / 4
        r0, r1 = S * 0.22, S * 0.34
        d.line((cx + r0 * math.cos(a), cy + r0 * math.sin(a),
                cx + r1 * math.cos(a), cy + r1 * math.sin(a)),
               fill=WHITE, width=int(S * 0.05))
    return img


def icon_camera():
    img, d = canvas()
    background(d, (50, 160, 90, 255))
    d.rounded_rectangle((S * 0.16, S * 0.32, S * 0.7, S * 0.7),
                        radius=S * 0.06, fill=WHITE)
    d.polygon([p(0.72, 0.46), p(0.86, 0.36), p(0.86, 0.66), p(0.72, 0.56)],
              fill=WHITE)
    d.ellipse((S * 0.28, S * 0.4, S * 0.5, S * 0.62), fill=(50, 160, 90, 255))
    return img


def icon_bone():
    img, d = canvas()
    background(d, (200, 95, 45, 255))
    a, b = p(0.28, 0.72), p(0.72, 0.28)
    d.line((*a, *b), fill=WHITE, width=int(S * 0.1))
    r = S * 0.075
    # two knobs at each end of the bone
    for (cx, cy), (dx, dy) in ((a, (-1, 1)), (b, (1, -1))):
        for ox, oy in ((dx, 0), (0, dy)):
            kx, ky = cx + ox * r * 0.9, cy + oy * r * 0.9
            d.ellipse((kx - r, ky - r, kx + r, ky + r), fill=WHITE)
    return img


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "Asset", "Textures", "Icons")
    os.makedirs(out, exist_ok=True)
    icons = {
        "empty": icon_empty,
        "geometry": icon_geometry,
        "light": icon_light,
        "camera": icon_camera,
        "bone": icon_bone,
    }
    for name, fn in icons.items():
        img = fn().resize((SIZE, SIZE), Image.LANCZOS)
        img.save(os.path.join(out, f"scene_{name}.png"))


if __name__ == "__main__":
    main()
