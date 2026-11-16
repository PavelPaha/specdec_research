"""Square, phone-readable figures of the EAGLE-4 OH500 dataset: HTML (inline SVG) -> PNG 1280x1280 via headless Chrome.

usage: python -I make_figures.py <data_dir: …/data/open-swe-traces/eagle4_oh500> <out_dir> [--chrome PATH]

  1_lengths.png      distribution of trace lengths (L_max) with the 5 strata of 100
  2_composition.png  what the 500 tasks are: repos, languages, task types, task sources
  3_points.png       measurement points: calls / thinking ends / tool calls, context coverage, output composition
Palette, marks and type follow the dataviz reference (light surface; palette validated with validate_palette.js).
"""
import argparse, html, math, os, shutil, subprocess, tempfile
import numpy as np
import pyarrow.parquet as pq

SIZE = 640  # CSS px; screenshots at 2x -> 1280x1280
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FOOT = "Open-SWE-Traces v1.2 · OpenHands · Qwen3.8-27B"

CSS = """
:root{--surface:#fcfcfb;--band:#f3f2ee;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100}
html,body{margin:0;width:640px;height:640px;background:var(--surface);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.card{box-sizing:border-box;width:640px;height:640px;padding:28px 32px 20px;display:flex;flex-direction:column}
h1{font-size:29px;line-height:1.15;font-weight:700;margin:0 0 7px;letter-spacing:-.015em}
.sub{font-size:16px;line-height:1.38;color:var(--ink2);margin:0 0 16px}
.tiles{display:flex;gap:10px;margin:0 0 16px}
.tile{flex:1;border:1px solid var(--ring);border-radius:12px;padding:9px 13px 10px}
.tv{font-size:31px;font-weight:650;line-height:1.1;letter-spacing:-.01em}
.tl{font-size:15px;color:var(--ink2);line-height:1.25;margin-top:3px}
.sec{font-size:17px;font-weight:600;margin:0 0 7px}
.sec span{font-weight:400;color:var(--ink2)}
.legend{display:flex;flex-wrap:wrap;gap:3px 16px;font-size:15px;color:var(--ink2);margin-top:8px}
.legend b{font-weight:600;color:var(--ink)}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.seg{display:flex;gap:2px;height:28px}
.seg>div{border-radius:4px;font-size:15px;font-weight:600;display:flex;align-items:center;padding-left:8px;
box-sizing:border-box;white-space:nowrap;color:#fff}
.block{margin-bottom:14px}
.lrow{display:flex;align-items:center;height:24px}
.ll{width:112px;font-size:16px}
.lt{flex:1;display:flex;align-items:center}
.lb{height:17px;background:var(--s1);border-radius:0 4px 4px 0}
.lv{font-size:15px;color:var(--ink2);margin-left:8px;white-space:nowrap}
.lv b{color:var(--ink);font-weight:600}
svg{display:block;overflow:visible}
svg text{font-family:inherit}
.tick{font-size:14px;fill:var(--muted)}
.ann{font-size:15px;fill:var(--ink2)}
.bl{font-size:16px;font-weight:600;fill:var(--ink2)}
.foot{margin-top:auto;font-size:13px;color:var(--muted)}
"""


def nb(n):
    """Russian-style thousands with a no-break space: 41 753 (the narrow one is invisible at display sizes)."""
    return f"{int(round(n)):,}".replace(",", " ")


def page(title, sub, body):
    return (f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><div class='card'><h1>{title}</h1><p class='sub'>{sub}</p>"
            f"{body}<div class='foot'>{FOOT}</div></div></body></html>")


def tiles(items):
    return "<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='tv'>{v}</div><div class='tl'>{l}</div></div>" for v, l in items) + "</div>"


def bar_path(x0, x1, y0, yb, r=4.0):
    """Column with a 4px rounded data end (top) and a square baseline."""
    h = yb - y0
    if h <= 0.3:
        return ""
    r = min(r, h, (x1 - x0) / 2)
    return (f"M{x0:.2f},{yb:.2f}V{y0 + r:.2f}Q{x0:.2f},{y0:.2f} {x0 + r:.2f},{y0:.2f}H{x1 - r:.2f}"
            f"Q{x1:.2f},{y0:.2f} {x1:.2f},{y0 + r:.2f}V{yb:.2f}Z")


def segbar(parts, width_px=576, min_label_px=48):
    """Tall segmented bar (one whole -> parts). parts: [(name, value, shown_value, css_color, text_color)].
    The share is printed inside a block only when it fits; otherwise it goes beside the name in the legend."""
    tot = sum(p[1] for p in parts)
    blocks, legend = [], []
    for name, v, shown, col, tcol in parts:
        share = v / tot
        fits = share * width_px >= min_label_px
        pct = f"{share * 100:.0f}%"
        blocks.append(f"<div style='flex:{v} 1 0;background:{col};color:{tcol}'>{pct if fits else ''}</div>")
        legend.append(f"<span><i class='sw' style='background:{col}'></i>{name} <b>{shown}</b>"
                      f"{'' if fits else ' · ' + pct}</span>")
    return f"<div class='seg'>{''.join(blocks)}</div><div class='legend'>{''.join(legend)}</div>"


# ---------------------------------------------------------------------------------------------------- figure 1
def fig_lengths(t, c):
    L = t["L_max"]
    E = [32768, 65536, 100000, 131072, 200000, 262144]
    edges = np.arange(E[0], E[-1] + 1, 8192)  # 28 columns of 8K tokens
    h, _ = np.histogram(L, edges)
    W, H = 576, 326
    ml, mr, mt, mb = 30, 12, 50, 28
    pw, ph = W - ml - mr, H - mt - mb
    X = lambda v: ml + (v - E[0]) / (E[-1] - E[0]) * pw
    ytop = int(math.ceil(h.max() / 10) * 10)
    Y = lambda n: mt + ph - n / ytop * ph
    s = [f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"]
    for i in range(5):  # strata as alternating bands, header = how many traces
        x0, x1 = X(E[i]), X(E[i + 1])
        if i % 2 == 0:
            s.append(f"<rect x='{x0:.1f}' y='{mt - 30}' width='{x1 - x0:.1f}' height='{ph + 30}' fill='var(--band)'/>")
        s.append(f"<text class='bl' x='{(x0 + x1) / 2:.1f}' y='{mt - 10}' text-anchor='middle'>100</text>")
    for v in range(0, ytop + 1, 10):
        s.append(f"<line x1='{ml}' x2='{ml + pw}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='var(--grid)' stroke-width='1'/>")
        s.append(f"<text class='tick' x='{ml - 7}' y='{Y(v):.1f}' dy='.35em' text-anchor='end'>{v}</text>")
    for k, n in enumerate(h):
        s.append(f"<path d='{bar_path(X(edges[k]) + 1, X(edges[k + 1]) - 1, Y(n), Y(0))}' fill='var(--s1)'/>")
    s.append(f"<line x1='{ml}' x2='{ml + pw}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='var(--axis)' stroke-width='1'/>")
    for i, e in enumerate(E):
        anchor = "end" if i == len(E) - 1 else "middle"
        s.append(f"<text class='tick' x='{X(e) + (4 if anchor == 'end' else 0):.1f}' y='{mt + ph + 19}' "
                 f"text-anchor='{anchor}'>{e // 1024 if e % 1024 == 0 else e // 1000}K</text>")
    s.append(f"<text class='tick' x='0' y='{mt - 34}'>трейсов</text>")
    s.append("</svg>")
    calls_per = c["calls_per_traj"]
    body = (tiles([(f"{np.median(L) / 1000:.0f}K", "медиана L_max"),
                   (f"{L.max() / 1000:.0f}K", "максимум; окно модели — 262K"),
                   (f"{int(np.median(calls_per))}", f"вызовов на трейс, медиана; до {int(calls_per.max())}")])
            + "<div class='sec'>Трейсы по L_max <span>· столбик = 8K токенов</span></div>" + "".join(s))
    return page("Длинные трейсы: до 258K токенов",
                "L_max — самый длинный вход модели в трейсе. 500 трейсов: по&nbsp;100 в каждом из 5 диапазонов "
                "длины, от 32K до окна модели.", body)


# ---------------------------------------------------------------------------------------------------- figure 2
LANG = {"python": "Python", "ts": "TypeScript", "go": "Go", "js": "JavaScript", "php": "PHP", "java": "Java",
        "rust": "Rust", "c": "C", "cpp": "C++"}
CAT = [("bug-fix", "баг-фикс"), ("feature-request", "новая фича"), ("enhancement", "улучшение"),
       ("other", "другое")]


def fig_composition(t):
    n = len(t["trajectory_id"])
    repos, rc = np.unique(t["repo"], return_counts=True)
    langs, lc = np.unique(t["language"], return_counts=True)
    order = np.argsort(-lc)
    top = [(LANG.get(langs[i], langs[i]), int(lc[i])) for i in order[:5]]
    rest = [(LANG.get(langs[i], langs[i]), int(lc[i])) for i in order[5:]]
    rows = top + ([("другие", sum(v for _, v in rest))] if rest else [])
    vmax = max(v for _, v in rows)
    lr = []
    for name, v in rows:
        extra = f" <span style='color:var(--muted)'>({', '.join(k for k, _ in rest)})</span>" if name == "другие" else ""
        lr.append(f"<div class='lrow'><div class='ll'>{name}</div><div class='lt'><div class='lb' "
                  f"style='width:{v / vmax * 78:.2f}%'></div><span class='lv'><b>{v}</b> · {v / n * 100:.0f}%{extra}"
                  f"</span></div></div>")
    cats = {k: int((t["category"] == k).sum()) for k, _ in CAT}
    src = {s: int((t["source"] == s).sum()) for s in ("scale-swe", "swe-rebench-v2")}
    body = (tiles([(nb(n), "задач; трейс = задача"), (nb(len(repos)), f"репозиториев, ≤{rc.max()} задач на один"),
                   (str(len(langs)), "языков программирования")])
            + "<div class='block'><div class='sec'>Языки</div>" + "".join(lr) + "</div>"
            + "<div class='block'><div class='sec'>Тип задачи</div>"
            + segbar([(ru, cats[k], nb(cats[k]), f"var(--s{i + 1})", "#0b0b0b" if i == 3 else "#fff")
                      for i, (k, ru) in enumerate(CAT)]) + "</div>"
            + "<div class='block'><div class='sec'>Откуда задачи</div>"
            + segbar([("Scale-SWE", src["scale-swe"], nb(src["scale-swe"]), "var(--s1)", "#fff"),
                      ("SWE-rebench-V2", src["swe-rebench-v2"], nb(src["swe-rebench-v2"]), "var(--s2)", "#fff")])
            + "</div>")
    return page("Состав: 500 реальных задач с GitHub",
                "Агент OpenHands на Qwen3.8-27B, все задачи решены (resolved).", body)


# ---------------------------------------------------------------------------------------------------- figure 3
def fig_points(c):
    inp = c["input_len"]
    edges = np.arange(0, 262144 + 1, 16384)  # 16 columns of 16K
    h, _ = np.histogram(inp, edges)
    W, H = 576, 214
    ml, mr, mt, mb = 34, 12, 10, 28
    pw, ph = W - ml - mr, H - mt - mb
    X = lambda v: ml + v / 262144 * pw
    ytop = int(math.ceil(h.max() / 2000) * 2000)
    Y = lambda n: mt + ph - n / ytop * ph
    s = [f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"]
    for v in range(0, ytop + 1, 2000):
        s.append(f"<line x1='{ml}' x2='{ml + pw}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='var(--grid)' stroke-width='1'/>")
        s.append(f"<text class='tick' x='{ml - 7}' y='{Y(v):.1f}' dy='.35em' text-anchor='end'>"
                 f"{'0' if v == 0 else f'{v // 1000}K'}</text>")
    for k, n in enumerate(h):
        cx = (X(edges[k]) + X(edges[k + 1])) / 2
        bw = min(24, X(edges[k + 1]) - X(edges[k]) - 2)
        s.append(f"<path d='{bar_path(cx - bw / 2, cx + bw / 2, Y(n), Y(0))}' fill='var(--s1)'/>")
    s.append(f"<line x1='{ml}' x2='{ml + pw}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='var(--axis)' stroke-width='1'/>")
    for v in (0, 65536, 131072, 196608, 262144):
        anchor = "end" if v == 262144 else ("start" if v == 0 else "middle")
        s.append(f"<text class='tick' x='{X(v):.1f}' y='{mt + ph + 19}' text-anchor='{anchor}'>"
                 f"{f'{v // 1024}K' if v else '0'}</text>")
    # annotation: long-context calls
    n128, n200 = int((inp >= 131072).sum()), int((inp >= 200000).sum())
    i0 = 8
    yb = Y(h[i0:].max()) - 12
    x0, x1 = X(131072) + 3, X(262144) - 3
    s.append(f"<path d='M{x0:.1f},{yb + 6:.1f}V{yb:.1f}H{x1:.1f}V{yb + 6:.1f}' fill='none' stroke='var(--ink2)' "
             f"stroke-width='1'/>")
    s.append(f"<text class='ann' x='{(x0 + x1) / 2:.1f}' y='{yb - 8:.1f}' text-anchor='middle'>"
             f"≥128K: <tspan font-weight='600' fill='var(--ink)'>{nb(n128)}</tspan> · ≥200K: "
             f"<tspan font-weight='600' fill='var(--ink)'>{nb(n200)}</tspan></text>")
    s.append("</svg>")
    out = c["output_tok"]
    body = (tiles([(nb(len(inp)), "стартов вызова — реальных входов модели"),
                   (nb(len(inp)), "концов thinking (он есть в каждом ходе)"),
                   (nb(c["n_tool_calls"]), "tool calls; в " + nb(c["multi_tc_turns"]) + " ходах — параллельные")])
            + "<div class='sec'>Вызовы по длине контекста на входе <span>· столбик = 16K</span></div>" + "".join(s)
            + f"<div class='block' style='margin-top:14px'><div class='sec'>Что генерирует модель "
              f"<span>· {out['total'] / 1e6:.1f}M токенов</span></div>"
            + segbar([(lab, out[key], f"{out[key] / 1e6:.1f}M", f"var(--s{i + 1})", "#fff")
                      for i, (key, lab) in enumerate((("reasoning", "thinking"), ("tool_calls", "tool calls"),
                                                      ("other", "текст и разметка")))]) + "</div>")
    return page("Много точек для старта драфтинга",
                "Каждый вызов модели — реальный вход с точно известным префиксом. Внутри хода можно стартовать "
                "после thinking и после tool call.", body)


def load(data_dir):
    tr = pq.read_table(os.path.join(data_dir, "trajectories.parquet"),
                       columns=["trajectory_id", "repo", "language", "L_max", "src_source", "audit_category"])
    t = {"trajectory_id": np.array(tr.column("trajectory_id").to_pylist()),
         "repo": np.array(tr.column("repo").to_pylist()), "language": np.array(tr.column("language").to_pylist()),
         "L_max": np.array(tr.column("L_max").to_pylist()), "source": np.array(tr.column("src_source").to_pylist()),
         "category": np.array(tr.column("audit_category").to_pylist())}
    ca = pq.read_table(os.path.join(data_dir, "calls.parquet"),
                       columns=["trajectory_id", "input_len", "output_len", "n_reasoning_tokens", "n_tool_calls",
                                "tool_call_starts", "tool_call_ends"]).to_pydict()
    tc_tok = [sum(e - s + 1 for s, e in zip(a, b)) for a, b in zip(ca["tool_call_starts"], ca["tool_call_ends"])]
    out_total, reas, tct = sum(ca["output_len"]), sum(ca["n_reasoning_tokens"]), sum(tc_tok)
    _, per = np.unique(ca["trajectory_id"], return_counts=True)
    c = {"input_len": np.array(ca["input_len"]), "calls_per_traj": per,
         "n_tool_calls": sum(ca["n_tool_calls"]), "multi_tc_turns": sum(1 for x in ca["n_tool_calls"] if x > 1),
         "output_tok": {"total": out_total, "reasoning": reas, "tool_calls": tct, "other": out_total - reas - tct}}
    return t, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--chrome", default=CHROME)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t, c = load(a.data_dir)
    pages = {"1_lengths": fig_lengths(t, c), "2_composition": fig_composition(t), "3_points": fig_points(c)}
    profile = tempfile.mkdtemp(prefix="chrome-profile-")
    try:
        for name, doc in pages.items():
            hp = os.path.abspath(os.path.join(a.out_dir, name + ".html"))
            with open(hp, "w") as f:
                f.write(doc)
            pp = os.path.abspath(os.path.join(a.out_dir, name + ".png"))
            subprocess.run([a.chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                            "--force-device-scale-factor=2", f"--window-size={SIZE},{SIZE}",
                            f"--user-data-dir={profile}", "--virtual-time-budget=2000", f"--screenshot={pp}",
                            "file://" + hp], check=True, capture_output=True, timeout=180)
            print(name, pp)
    finally:
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
