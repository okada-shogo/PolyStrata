#!/usr/bin/env python3
"""Bench reports side by side: `strata-poly --bench-report` files from several PCs in one table per card, with the
numbers that are far from the same card's other reports flagged.

    python tools/bench_reports.py report1.json report2.json ...        a Markdown table per card, and the outliers
    python tools/bench_reports.py --json reports/*.json                 the same as JSON
    python tools/bench_reports.py --check-privacy report.json ...      exit 1 if a report holds a path, an e-mail
                                                                         address, a serial or an address on a network
    python tools/bench_reports.py --check-privacy --private alice,box report.json   ... or one of these words

A number is flagged when it is under OUTLIER (0.75) of the median of the other reports of the same card and model
(at least two others), or for the probes of the same card whatever the model.  The engine writes the reports without
paths, user or host names (engine/src/poly/bench_report.hpp); --check-privacy checks a file before it is posted.
Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

OUTLIER = 0.75

# what must never be in a report that is posted
PRIVACY = [
    ("a path", re.compile(r"(?:/home/|/Users/|/root/|/mnt/|/media/|/tmp/|/var/|/work/|/workspace/|/srv/|/data/|/run/user/|/Volumes/|/private/|\b[A-Za-z]:\\|\\\\[A-Za-z]|~/)")),
    ("an e-mail address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("a GPU or other UUID", re.compile(r"\b(?:GPU-|MIG-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                                        r"[0-9a-fA-F]{12}\b")),
    ("a MAC address", re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")),
    # (not a version: 6.6.87.2-microsoft-standard-WSL2 is a kernel's release, and Salad's hosts run WSL2)
    ("an IP address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.-]|[A-Za-z])")),
    ("a PCI bus address", re.compile(r"\b[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b")),
    ("an API key", re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")),
]


def privacy_problems(text: str, private: list[str] | None = None) -> list[str]:
    """What in `text` names a person or a PC: a list of findings, empty when there is none."""
    out = []
    for what, rx in PRIVACY:
        m = rx.search(text)
        if m:
            out.append(f"{what}: {m.group(0)!r}")
    for word in private or []:
        if word and len(word) >= 3 and re.search(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", text):
            out.append(f"the word {word!r}")
    return out


def load(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(r, dict) or r.get("kind") not in ("strata-bench-report", "strata-session-profile"):
        raise ValueError(f"{path.name}: not a strata-poly bench report")
    return r


def _d(v) -> dict:
    return v if isinstance(v, dict) else {}


def _s(v, default="?") -> str:
    return v if isinstance(v, str) and v else default


def _n(v):
    """A number of a report, or None: a posted file is anyone's, so every field is checked for its type."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _i(v) -> int:
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else 0


def run_of(r: dict, name: str) -> dict | None:
    runs = r.get("runs")
    return next((x for x in runs if isinstance(x, dict) and x.get("name") == name), None) if isinstance(runs, list) else None


def figures(r: dict) -> dict:
    """The numbers a table row shows, by column."""
    hw = _d(r.get("hardware"))
    gpu, cpu = _d(hw.get("gpu")), _d(hw.get("cpu"))
    pr, model = _d(r.get("probes")), _d(r.get("model"))
    dots = pr.get("cpu_dot") if isinstance(pr.get("cpu_dot"), list) else []
    gen, width = _i(gpu.get("pcie_gen")), _i(gpu.get("pcie_width"))
    f = {
        "card": _s(gpu.get("name")),
        "model": f"{_s(model.get('name'))} ({_s(model.get('expert_types'))})",
        "cpu": _s(cpu.get("model")),
        "pcie": f"gen{gen} x{width}" if gen > 0 else "?",
        "vram_experts": f"{_i(r.get('cached_experts'))}/{_i(r.get('total_experts'))}",
        "h2d_pinned_gbs": _n(pr.get("h2d_pinned_gbs")),
        "ram_read_gbs": _n(pr.get("ram_read_gbs")),
        "gpu_mem_gbs": _n(pr.get("gpu_mem_gbs")),
        "cpu_dot_gbs": _n(_d(dots[0]).get("gbs_all_threads")) if dots else None,
    }
    for n in (302, 2048, 8192):
        x = run_of(r, f"prompt {n}")
        f[f"prompt_{n}"] = _n(x.get("prompt_tok_s")) if x else None
        f[f"decode_{n}"] = _n(x.get("decode_tok_s")) if x else None
    for x in r.get("long") if isinstance(r.get("long"), list) else []:
        if isinstance(x, dict):
            f[f"long_decode_{_i(x.get('context_end')) // 1024}k"] = _n(x.get("decode_tok_s"))
    return f


def md(v) -> str:
    """A table cell: no line or cell breaks, no HTML, headings, images or links from a posted file."""
    t = re.sub(r"[\r\n]+", " ", str(v))
    t = t.replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([|#!\[\]`*_\\])", r"\\\1", t)


SPEED = ["prompt_2048", "prompt_8192", "decode_302", "decode_2048", "decode_8192"]
PROBES = ["h2d_pinned_gbs", "ram_read_gbs", "gpu_mem_gbs", "cpu_dot_gbs"]


def flag_outliers(rows: list[dict]) -> None:
    """Each row's `flags`: its numbers under OUTLIER of the median of its peers' (the same card's other rows; for the
    speeds, of the same model too)."""
    for i, row in enumerate(rows):
        row.setdefault("flags", [])
        for cols, same in ((SPEED, ("card", "model")), (PROBES, ("card",))):
            peers = [o for j, o in enumerate(rows) if j != i and all(o[k] == row[k] for k in same)]
            for c in cols:
                vals = [o[c] for o in peers if isinstance(o.get(c), (int, float)) and o[c] > 0]
                v = row.get(c)
                if len(vals) >= 2 and isinstance(v, (int, float)) and v > 0:
                    med = statistics.median(vals)
                    if v < OUTLIER * med:
                        row["flags"].append(f"{c} {v:g} is {v / med:.0%} of its peers' median {med:g}")


def table(rows: list[dict]) -> str:
    cols = [("source", "Report"), ("model", "Model"), ("cpu", "CPU"), ("pcie", "PCIe"),
            ("vram_experts", "Experts in VRAM"), ("h2d_pinned_gbs", "H2D GB/s"), ("ram_read_gbs", "RAM GB/s"),
            ("cpu_dot_gbs", "CPU dot GB/s"), ("prompt_2048", "Prompt 2K tok/s"), ("prompt_8192", "Prompt 8K tok/s"),
            ("decode_302", "Decode tok/s after 302"), ("decode_8192", "Decode tok/s after 8K")]
    out = []
    for card in sorted({r["card"] for r in rows}):
        mine = [r for r in rows if r["card"] == card]
        out.append(f"### {md(card)} ({len(mine)} report{'s' if len(mine) != 1 else ''})\n")
        out.append("| " + " | ".join(t for _, t in cols) + " |")
        out.append("| " + " | ".join("---" for _ in cols) + " |")
        for r in mine:
            cells = []
            for k, _ in cols:
                v = r.get(k)
                cells.append("-" if v is None else f"{v:g}" if isinstance(v, float) else md(v))
            out.append("| " + " | ".join(cells) + " |")
        flagged = [(r["source"], f) for r in mine for f in r["flags"]]
        if flagged:
            out.append("\nFar from the same card's other reports:\n")
            out.extend(f"- {md(s)}: {md(f)}" for s, f in flagged)
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="the rows and their flags as JSON")
    ap.add_argument("--check-privacy", action="store_true", help="only check the files for private details")
    ap.add_argument("--private", default="", help="with --check-privacy: words that must not appear, comma separated")
    a = ap.parse_args(argv)
    if a.check_privacy:
        bad = 0
        words = [w.strip() for w in a.private.split(",") if w.strip()]
        for p in a.reports:
            problems = privacy_problems(p.read_text(encoding="utf-8", errors="replace"), words)
            for x in problems:
                print(f"{p.name}: {x}")
            bad += bool(problems)
        if not bad:
            print(f"{len(a.reports)} file(s): nothing private found")
        return 1 if bad else 0
    rows = []
    for p in a.reports:
        try:
            r = load(p)
            f = figures(r)
        except Exception as e:   # one bad file leaves the others' table
            print(f"skipped {p.name}: {e}", file=sys.stderr)
            continue
        f["source"] = p.stem
        rows.append(f)
    flag_outliers(rows)
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
