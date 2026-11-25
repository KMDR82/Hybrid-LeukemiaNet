"""Figures 1 and 2 of the revised manuscript as draw.io sources plus matching SVG.

One geometric specification per figure is written twice:
  <name>.drawio  editable in draw.io / diagrams.net (File > Open)
  <name>.svg     the same layout, rendered to PDF/PNG by render_figures.js

  python revision/figures_drawio.py --out revision/results/figures_drawio
"""
import argparse
from pathlib import Path
import re
from xml.sax.saxutils import escape

SUB = re.compile(r"_\{([^}]*)\}")


def sub_html(t):
    return SUB.sub(lambda m: f"<sub>{m.group(1)}</sub>", escape(t))


def sub_svg(t, fs):
    out, pos, shift = [], 0, 0.0
    for m in SUB.finditer(t):
        pre = escape(t[pos:m.start()])
        if pre:
            out.append(f'<tspan dy="{-shift:.1f}">{pre}</tspan>' if shift else pre)
            shift = 0.0
        out.append(f'<tspan dy="{fs * 0.25 - shift:.1f}" font-size="{fs * 0.75:.1f}">{escape(m.group(1))}</tspan>')
        shift = fs * 0.25
        pos = m.end()
    rest = escape(t[pos:])
    out.append(f'<tspan dy="{-shift:.1f}">{rest or " "}</tspan>' if shift else rest)
    return "".join(out)

FONT = "Helvetica"
SVG_FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"
INK = "#262626"

PAL = {  # fill, stroke
    "source": ("#DCE9F6", "#2F6699"),
    "target": ("#FBE8D3", "#B3611B"),
    "test": ("#F7D9D9", "#A82424"),
    "model": ("#E2F0DE", "#3B7D3A"),
    "eval": ("#ECECEC", "#555555"),
    "input": ("#F2F2F2", "#555555"),
    "panel": ("#FAFAFA", "#C4C4C4"),
    "cnn": ("#DCE9F6", "#2F6699"),
    "vit": ("#FBE8D3", "#B3611B"),
    "fuse": ("#E2F0DE", "#3B7D3A"),
    "white": ("#FFFFFF", "#555555"),
}
EDGE = {"solid": (INK, False), "dashed": (INK, True), "red": ("#A82424", False)}


class Fig:
    def __init__(self, w, h, k=1.0):
        self.w, self.h, self.k = w, h, k  # k scales every font size
        self.items = []  # (kind, dict)
        self.boxes = {}

    def panel(self, x, y, w, h, title, pal="panel", dashed=False, fs=13, align="left"):
        fs = round(fs * self.k, 1)
        self.items.append(("panel", dict(x=x, y=y, w=w, h=h, title=title, pal=pal, dashed=dashed, fs=fs, align=align)))

    def box(self, bid, x, y, w, h, lines, pal, fs=12.5, bold_first=True, rounded=True, sw=1.3):
        fs = round(fs * self.k, 1)
        d = dict(id=bid, x=x, y=y, w=w, h=h, lines=lines, pal=pal, fs=fs, bold_first=bold_first,
                 rounded=rounded, sw=sw)
        self.boxes[bid] = d
        self.items.append(("box", d))

    def text(self, x, y, w, h, lines, fs=11.5, italic=False, align="center", color=INK, bold=False):
        fs = round(fs * self.k, 1)
        self.items.append(("text", dict(x=x, y=y, w=w, h=h, lines=lines, fs=fs, italic=italic,
                                        align=align, color=color, bold=bold)))

    def edge(self, src, tgt, pts, kind="solid", label=None, label_pos=None):
        """pts: absolute polyline, first point on src border, last on tgt border."""
        self.items.append(("edge", dict(src=src, tgt=tgt, pts=pts, kind=kind, label=label, label_pos=label_pos)))

    # ------------------------------------------------------------------ draw.io
    def drawio(self, name):
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        n = 0

        def nid():
            nonlocal n
            n += 1
            return f"c{n}"

        def html(lines, bold_first, italic=False, bold=False):
            out = []
            for i, l in enumerate(lines):
                t = sub_html(l)
                if (bold_first and i == 0) or bold:
                    t = f"<b>{t}</b>"
                if italic:
                    t = f"<i>{t}</i>"
                out.append(t)
            return "<br>".join(out)

        for kind, d in self.items:
            if kind == "panel":
                f, s = PAL[d["pal"]]
                st = (f"rounded=1;arcSize=3;absoluteArcSize=0;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};"
                      f"verticalAlign=top;align={d['align']};spacingLeft=10;spacingTop=4;fontFamily={FONT};"
                      f"fontSize={d['fs']};fontStyle=1;fontColor={INK};strokeWidth=1;"
                      + ("dashed=1;dashPattern=4 3;" if d["dashed"] else ""))
                cells.append(f'<mxCell id="{nid()}" value="{escape(escape(d["title"]))}" style="{st}" vertex="1" parent="1">'
                             f'<mxGeometry x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" as="geometry"/></mxCell>')
            elif kind == "box":
                f, s = PAL[d["pal"]]
                st = (f"rounded={1 if d['rounded'] else 0};arcSize=8;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};"
                      f"strokeWidth={d['sw']};fontFamily={FONT};fontSize={d['fs']};fontColor={INK};spacing=4;")
                d["cid"] = nid()
                cells.append(f'<mxCell id="{d["cid"]}" value="{escape(html(d["lines"], d["bold_first"]), {chr(34): "&quot;"})}" '
                             f'style="{st}" vertex="1" parent="1">'
                             f'<mxGeometry x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" as="geometry"/></mxCell>')
            elif kind == "text":
                st = (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align={d['align']};"
                      f"verticalAlign=middle;fontFamily={FONT};fontSize={d['fs']};fontColor={d['color']};spacing=0;")
                cells.append(f'<mxCell id="{nid()}" value="{escape(html(d["lines"], False, d["italic"], d["bold"]), {chr(34): "&quot;"})}" '
                             f'style="{st}" vertex="1" parent="1">'
                             f'<mxGeometry x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" as="geometry"/></mxCell>')
        for kind, d in self.items:
            if kind != "edge":
                continue
            col, dash = EDGE[d["kind"]]
            a, b = self.boxes.get(d["src"]), self.boxes.get(d["tgt"])
            p0, p1 = d["pts"][0], d["pts"][-1]
            st = (f"edgeStyle=none;html=1;rounded=0;endArrow=blockThin;endFill=1;endSize=7;strokeColor={col};"
                  f"strokeWidth={1.6 if d['kind'] == 'red' else 1.3};fontFamily={FONT};fontSize=11;fontColor={INK};"
                  + ("dashed=1;dashPattern=5 4;" if dash else ""))
            if a:
                st += f"exitX={(p0[0] - a['x']) / a['w']:.4f};exitY={(p0[1] - a['y']) / a['h']:.4f};exitDx=0;exitDy=0;"
            if b:
                st += f"entryX={(p1[0] - b['x']) / b['w']:.4f};entryY={(p1[1] - b['y']) / b['h']:.4f};entryDx=0;entryDy=0;"
            wp = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in d["pts"][1:-1])
            src = f' source="{a["cid"]}"' if a else ""
            tgt = f' target="{b["cid"]}"' if b else ""
            geo = (f'<mxGeometry relative="1" as="geometry"><mxPoint x="{p0[0]}" y="{p0[1]}" as="sourcePoint"/>'
                   f'<mxPoint x="{p1[0]}" y="{p1[1]}" as="targetPoint"/>'
                   + (f'<Array as="points">{wp}</Array>' if wp else "") + "</mxGeometry>")
            cells.append(f'<mxCell id="{nid()}" value="" style="{st}" edge="1" parent="1"{src}{tgt}>{geo}</mxCell>')
            if d["label"]:
                lx, ly = d["label_pos"]
                st = (f"text;html=1;align=center;verticalAlign=middle;fontFamily={FONT};fontSize=11;"
                      f"fontColor={INK};labelBackgroundColor=#FFFFFF;fontStyle=2;")
                cells.append(f'<mxCell id="{nid()}" value="{escape(d["label"])}" style="{st}" vertex="1" parent="1">'
                             f'<mxGeometry x="{lx - 45}" y="{ly - 9}" width="90" height="18" as="geometry"/></mxCell>')
        model = (f'<mxGraphModel dx="{self.w}" dy="{self.h}" grid="1" gridSize="5" guides="1" tooltips="1" connect="1" '
                 f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" '
                 f'background="#FFFFFF" math="0" shadow="0"><root>{"".join(cells)}</root></mxGraphModel>')
        return (f'<mxfile host="Electron" type="device"><diagram id="{name}" name="{name}">{model}</diagram></mxfile>\n')

    # ---------------------------------------------------------------------- SVG
    def svg(self):
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
             f'font-family="{SVG_FONT}">',
             '<defs>'] + [
            f'<marker id="ah_{k}" viewBox="0 0 10 10" refX="9.2" refY="5" markerWidth="{7.5 if k != "red" else 6.6}" '
            f'markerHeight="{7.5 if k != "red" else 6.6}" markerUnits="userSpaceOnUse" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for k, (c, _) in EDGE.items()] + [
            '</defs>', f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>']

        def lines_svg(cx, x, w, cy, lines, fs, bold_first=False, italic=False, align="center", color=INK, bold=False):
            lh = fs * 1.22
            y0 = cy - lh * (len(lines) - 1) / 2 + fs * 0.35
            anchor, tx = {"center": ("middle", cx), "left": ("start", x), "right": ("end", x + w)}[align]
            out = []
            for i, l in enumerate(lines):
                wgt = "bold" if (bold_first and i == 0) or bold else "normal"
                sty = "italic" if italic else "normal"
                out.append(f'<text x="{tx:.1f}" y="{y0 + i * lh:.1f}" font-size="{fs}" font-weight="{wgt}" '
                           f'font-style="{sty}" fill="{color}" text-anchor="{anchor}">{sub_svg(l, fs)}</text>')
            return out

        for kind, d in self.items:
            if kind == "panel":
                f, s = PAL[d["pal"]]
                dash = ' stroke-dasharray="4 3"' if d["dashed"] else ""
                o.append(f'<rect x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" rx="6" fill="{f}" '
                         f'stroke="{s}" stroke-width="1"{dash}/>')
                tx = d["x"] + 10 if d["align"] == "left" else d["x"] + d["w"] / 2
                anc = "start" if d["align"] == "left" else "middle"
                o.append(f'<text x="{tx}" y="{d["y"] + 4 + d["fs"]}" font-size="{d["fs"]}" font-weight="bold" '
                         f'fill="{INK}" text-anchor="{anc}">{escape(d["title"])}</text>')
            elif kind == "box":
                f, s = PAL[d["pal"]]
                rx = min(d["w"], d["h"]) * 0.08 if d["rounded"] else 0
                o.append(f'<rect x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" rx="{rx:.1f}" fill="{f}" '
                         f'stroke="{s}" stroke-width="{d["sw"]}"/>')
                o += lines_svg(d["x"] + d["w"] / 2, d["x"], d["w"], d["y"] + d["h"] / 2, d["lines"], d["fs"], d["bold_first"])
            elif kind == "text":
                o += lines_svg(d["x"] + d["w"] / 2, d["x"], d["w"], d["y"] + d["h"] / 2, d["lines"], d["fs"],
                               italic=d["italic"], align=d["align"], color=d["color"], bold=d["bold"])
        for kind, d in self.items:
            if kind != "edge":
                continue
            col, dash = EDGE[d["kind"]]
            pts = " ".join(f"{x},{y}" for x, y in d["pts"])
            da = ' stroke-dasharray="5 4"' if dash else ""
            sw = 1.6 if d["kind"] == "red" else 1.3
            o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}"{da} '
                     f'marker-end="url(#ah_{d["kind"]})"/>')
            if d["label"]:
                lx, ly = d["label_pos"]
                o.append(f'<rect x="{lx - 22}" y="{ly - 9}" width="44" height="17" fill="#FFFFFF"/>')
                o.append(f'<text x="{lx}" y="{ly + 4}" font-size="11" font-style="italic" fill="{INK}" '
                         f'text-anchor="middle">{escape(d["label"])}</text>')
        o.append("</svg>")
        return "\n".join(o)


# ============================================================== Figure 1
def figure_protocol():
    F = Fig(1060, 640, k=1.1)
    # Stage 1 ----------------------------------------------------------------
    F.panel(10, 10, 1040, 175, "Stage 1   Source phase: C-NMC 2019 training phase")
    F.box("S", 30, 48, 220, 92, ["Training-phase data", "73 subjects (47 ALL / 26 HEM)", "10,661 cells"], "source")
    F.box("CV", 300, 48, 340, 92, ["Subject-level 5-fold cross-validation", "StratifiedGroupKFold (by subject, by class)",
                                   "inner validation: 10% of training subjects", "early stopping on validation loss"], "source")
    F.box("FM", 700, 48, 330, 92, ["Final source model", "trained on all 73 subjects",
                                   "for the median early-stopping epoch", "(no fold model selected)"], "model")
    F.edge("S", "CV", [(250, 94), (300, 94)])
    F.edge("CV", "FM", [(640, 94), (700, 94)])
    F.text(300, 148, 340, 26, ["Out-of-fold predictions: CV metrics and secondary threshold (Youden J)"],
           fs=11, italic=True)
    # Stage 2 ----------------------------------------------------------------
    F.panel(10, 200, 1040, 175, "Stage 2   Target phase: C-NMC 2019 preliminary-test phase, five random subject-level splits")
    F.box("T", 30, 245, 220, 92, ["Preliminary-test-phase data", "28 subjects (13 ALL / 15 HEM)", "1,867 cells"], "target")
    F.panel(290, 232, 365, 128, "", pal="white", dashed=True)
    F.text(300, 236, 350, 16, ["Split r = 1, …, 5 (seeds 42–46, stratified by class)"], fs=11, italic=True, align="left")
    F.box("AT", 300, 258, 165, 90, ["Adaptation-train", "4 subjects", "(2 ALL / 2 HEM)"], "target")
    F.box("AV", 480, 258, 165, 90, ["Adaptation-validation", "2 subjects", "(1 ALL / 1 HEM)"], "target")
    F.box("TE", 840, 245, 190, 103, ["Held-out test", "22 subjects", "(10 ALL / 12 HEM)", "never used for any", "selection"],
          "test")
    F.edge("T", None, [(250, 291), (290, 291)])
    # Stage 3 ----------------------------------------------------------------
    F.panel(10, 390, 1040, 210, "Stage 3   Evaluation on the held-out test subjects of each split")
    F.box("EV", 30, 430, 220, 120, ["Evaluation", "22 test subjects per split", "without / with TTA (K = 3)",
                                    "subject-bootstrap 95% CIs", "paired comparisons"], "eval")
    F.box("AD", 300, 430, 345, 62, ["Adapted model", "final source model fine-tuned on adaptation-train,",
                                    "early stopping on adaptation-validation loss"], "model", fs=12)
    F.box("ZS", 300, 512, 345, 50, ["Zero-shot model", "final source model, no update"], "model", fs=12)
    # model flow from the final source model (solid)
    F.edge("FM", "AD", [(752, 140), (752, 461), (645, 461)])
    F.edge(None, "ZS", [(752, 461), (752, 537), (645, 537)])
    # adaptation data (dashed)
    F.edge("AT", "AD", [(382, 348), (382, 430)], kind="dashed")
    F.edge("AV", "AD", [(562, 348), (562, 430)], kind="dashed")
    # models -> evaluation
    F.edge("AD", "EV", [(300, 461), (250, 461)])
    F.edge("ZS", "EV", [(300, 537), (250, 537)])
    # test data (red)
    F.edge("TE", "EV", [(935, 348), (935, 580), (140, 580), (140, 550)], kind="red")
    # legend ----------------------------------------------------------------
    lx, ly = 30, 618
    F.items.append(("edge", dict(src=None, tgt=None, pts=[(lx, ly), (lx + 40, ly)], kind="solid", label=None, label_pos=None)))
    F.text(lx + 48, ly - 9, 160, 18, ["model initialisation / flow"], fs=11, align="left")
    F.items.append(("edge", dict(src=None, tgt=None, pts=[(lx + 230, ly), (lx + 270, ly)], kind="dashed", label=None, label_pos=None)))
    F.text(lx + 278, ly - 9, 260, 18, ["data used for fine-tuning or early stopping"], fs=11, align="left")
    F.items.append(("edge", dict(src=None, tgt=None, pts=[(lx + 545, ly), (lx + 585, ly)], kind="red", label=None, label_pos=None)))
    F.text(lx + 593, ly - 9, 200, 18, ["data used only for final evaluation"], fs=11, align="left")
    F.text(820, ly - 9, 220, 18, ["Same protocol for all three models"], fs=11, italic=True, align="right")
    return F


# ============================================================== Figure 2
def figure_architecture():
    F = Fig(1180, 450, k=1.12)
    F.box("IN", 15, 160, 130, 120, ["Input cell image", "3 × 224 × 224", "RGB, ImageNet", "normalisation"], "input")
    # CNN stream
    F.panel(185, 15, 600, 145, "EfficientNet-B0 stream (ImageNet initialisation, all layers trained)", pal="cnn")
    y, h = 55, 82
    F.box("C1", 200, y, 132, h, ["Stem", "Conv 3×3, stride 2", "32 channels"], "white", fs=11.5)
    F.box("C2", 346, y, 132, h, ["MBConv blocks", "16 blocks, 7 stages", "(depthwise + SE)"], "white", fs=11.5)
    F.box("C3", 492, y, 132, h, ["Head conv", "Conv 1×1", "320 → 1280 channels"], "white", fs=11.5)
    F.box("C4", 638, y, 132, h, ["Global average", "pooling", "7×7×1280 → 1280"], "white", fs=11.5)
    F.box("VC", 825, 50, 105, 92, ["v_{cnn}", "1280-d"], "cnn", fs=13)
    # ViT stream
    F.panel(185, 290, 600, 145, "ViT-B/16 stream (ImageNet initialisation, all layers trained)", pal="vit")
    y = 330
    F.box("V1", 200, y, 132, h, ["Patch embedding", "16×16 patches", "196 tokens × 768"], "white", fs=11.5)
    F.box("V2", 346, y, 132, h, ["+ [CLS] token", "+ position", "embeddings"], "white", fs=11.5)
    F.box("V3", 492, y, 132, h, ["Encoder", "12 Transformer layers", "12 attention heads"], "white", fs=11.5)
    F.box("V4", 638, y, 132, h, ["[CLS] output", "after final", "LayerNorm → 768"], "white", fs=11.5)
    F.box("VV", 825, 325, 105, 92, ["v_{vit}", "768-d"], "vit", fs=13)
    for a, b, yy in (("C1", "C2", 96), ("C2", "C3", 96), ("C3", "C4", 96), ("V1", "V2", 371), ("V2", "V3", 371),
                     ("V3", "V4", 371)):
        x0 = F.boxes[a]["x"] + F.boxes[a]["w"]
        F.edge(a, b, [(x0, yy), (F.boxes[b]["x"], yy)])
    F.edge("IN", "C1", [(80, 160), (80, 96), (200, 96)])
    F.edge("IN", "V1", [(80, 280), (80, 371), (200, 371)])
    F.edge("C4", "VC", [(770, 96), (825, 96)])
    F.edge("V4", "VV", [(770, 371), (825, 371)])
    # fusion + head
    F.box("CAT", 812, 200, 130, 70, ["Concatenation", "F = v_{cnn} ⊕ v_{vit}", "2048-d"], "fuse", fs=11.5)
    F.edge("VC", "CAT", [(877, 142), (877, 200)])
    F.edge("VV", "CAT", [(877, 325), (877, 270)])
    F.panel(960, 15, 205, 420, "Classifier head (MLP)", pal="fuse", align="center")
    F.box("H1", 975, 55, 175, 80, ["FC 2048 → 512", "BatchNorm, ReLU", "Dropout 0.4"], "white", fs=11.5)
    F.box("H2", 975, 175, 175, 70, ["FC 512 → 128", "BatchNorm, ReLU"], "white", fs=11.5)
    F.box("H3", 975, 285, 175, 60, ["FC 128 → 2", "logits z"], "white", fs=11.5)
    F.box("H4", 975, 375, 175, 45, ["Softmax", "p̂ = σ(z)₁ = P(ALL)"], "white", fs=11.5)
    F.edge("CAT", "H1", [(942, 235), (952, 235), (952, 95), (975, 95)])
    F.edge("H1", "H2", [(1062, 135), (1062, 175)])
    F.edge("H2", "H3", [(1062, 245), (1062, 285)])
    F.edge("H3", "H4", [(1062, 345), (1062, 375)])
    F.text(185, 186, 600, 90, ["Single-backbone baselines (trained independently): the same head on v_{cnn} alone",
                               "(1280 → 512 → 128 → 2) or on v_{vit} alone (768 → 512 → 128 → 2).",
                               "Stream masking (secondary analysis): v_{cnn} or v_{vit} of the trained",
                               "dual-stream model set to zero at inference."], fs=11, italic=True, color="#444444")
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="revision/results/figures_drawio")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fig_protocol", figure_protocol), ("fig_architecture", figure_architecture)):
        F = fn()
        (out / f"{name}.drawio").write_text(F.drawio(name), encoding="utf-8")
        (out / f"{name}.svg").write_text(F.svg(), encoding="utf-8")
        print("wrote", out / f"{name}.drawio", out / f"{name}.svg")


if __name__ == "__main__":
    main()
