from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 1100
BG = (250, 248, 244)
BOX = (255, 255, 255)
OUTLINE = (70, 70, 70)
ACCENT = (34, 85, 130)
ACCENT2 = (168, 89, 50)
TEXT = (20, 20, 20)
SOFT = (235, 240, 246)
SOFT2 = (245, 236, 230)


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(36, bold=True)
FONT_SUB = load_font(24, bold=True)
FONT_BODY = load_font(21, bold=False)
FONT_SMALL = load_font(18, bold=False)


def box(draw, xy, title, lines, fill=BOX, accent=ACCENT):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=OUTLINE, width=3)
    draw.rounded_rectangle((x1, y1, x2, y1 + 44), radius=18, fill=accent, outline=accent)
    draw.text((x1 + 18, y1 + 8), title, font=FONT_SUB, fill=(255, 255, 255))
    y = y1 + 62
    for line in lines:
        draw.text((x1 + 18, y), line, font=FONT_BODY, fill=TEXT)
        y += 30


def arrow(draw, p1, p2, color=ACCENT, width=6):
    x1, y1 = p1
    x2, y2 = p2
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if x1 == x2:
        direction = 1 if y2 > y1 else -1
        tip = (x2, y2)
        left = (x2 - 12, y2 - 20 * direction)
        right = (x2 + 12, y2 - 20 * direction)
    else:
        direction = 1 if x2 > x1 else -1
        tip = (x2, y2)
        left = (x2 - 20 * direction, y2 - 12)
        right = (x2 - 20 * direction, y2 + 12)
    draw.polygon([tip, left, right], fill=color)


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((60, 35), "Current Strength-Control Diffusion Framework (schematic)", font=FONT_TITLE, fill=TEXT)
    draw.text((60, 88), "BetterTSE / TEdit strength line, aligned to current code path on 2026-04-16", font=FONT_SMALL, fill=(80, 80, 80))

    box(
        draw,
        (70, 170, 470, 380),
        "1. Family Sample",
        [
            "source_ts: original series",
            "target_ts: weak / medium / strong target",
            "mask_gt: editable region",
            "instruction_text + strength_label + strength_scalar",
        ],
        fill=SOFT,
    )

    box(
        draw,
        (560, 170, 960, 380),
        "2. Strength Encoder",
        [
            "StrengthProjector",
            "label embedding",
            "scalar MLP",
            "text embedding + mean pooling",
            "fused into one strength vector",
        ],
        fill=SOFT,
    )

    box(
        draw,
        (1050, 150, 1720, 430),
        "3. Diffusion Denoiser Backbone",
        [
            "pretrained synthetic TEdit backbone",
            "base conditioning remains active",
            "strength path adds residual modulation",
            "delta_gamma / delta_beta scaled by strength",
            "recent diagnosis: internal signal exists here",
        ],
        fill=(238, 244, 252),
    )

    box(
        draw,
        (1050, 500, 1720, 760),
        "4. Edited Output",
        [
            "model prediction on edited series",
            "measure actual edit gain in masked region",
            "check whether weak < medium < strong",
            "recent issue: often flat or direction-reversed",
        ],
        fill=(247, 238, 232),
        accent=ACCENT2,
    )

    box(
        draw,
        (70, 500, 960, 920),
        "5. Training Objectives",
        [
            "diffusion realism loss: keep base editing ability",
            "edit-region loss: editable segment should fit target",
            "background loss: non-edit region should stay stable",
            "gain-match loss: actual edit magnitude should match target magnitude",
            "monotonic loss: strong should not edit less than weak",
            "family-relative loss: gaps between weak/medium/strong should stay meaningful",
            "constant-gain penalty: forbid all three strengths collapsing together",
            "numeric-only loss: force scalar control to matter even without text",
            "beta-direction loss: recent repair term for wrong-direction behavior",
        ],
        fill=SOFT2,
        accent=ACCENT2,
    )

    arrow(draw, (470, 275), (560, 275))
    arrow(draw, (960, 275), (1050, 275))
    arrow(draw, (1385, 430), (1385, 500), color=ACCENT2)
    arrow(draw, (1385, 760), (1385, 880), color=ACCENT2)
    arrow(draw, (1385, 880), (960, 880), color=ACCENT2)
    arrow(draw, (960, 880), (960, 710), color=ACCENT2)

    draw.text((1090, 805), "Observed failure mode in Apr 15-16 diagnosis:", font=FONT_SUB, fill=TEXT)
    draw.text((1090, 845), "modulation differences are visible, but final edit strength often becomes", font=FONT_BODY, fill=TEXT)
    draw.text((1090, 875), "flat or reversed. Beta-path sign / output mapping is currently the main suspect.", font=FONT_BODY, fill=TEXT)

    draw.text((70, 975), "Note: this is a reporting schematic, not an exact tensor-shape graph.", font=FONT_SMALL, fill=(90, 90, 90))

    image.save("/root/autodl-tmp/BetterTSE-main/tmp/strength_diffusion_framework_20260416.png")


if __name__ == "__main__":
    main()
