"""`mpd results`: read a campaign's `evaluate/summary.json` and say where the trained adapters are. Standard library only."""

from __future__ import annotations

import json
import math
from pathlib import Path

# The evaluate stage reports both readouts for every run. Which one is "primary" is the maintainer's assumption
# (delta for direct-logit runs, dual for score runs), not something this repository has confirmed.
PRIMARY_NOTE = ("* primary readout by this tool's convention (delta for direct-logit runs, dual for score runs). That "
                "convention is an assumption of the maintainer, not a confirmed standard; both readouts are always reported.")
MAX_UNITS = 12


def load_summary(campaign_dir: Path) -> dict | None:
    """The parsed `evaluate/summary.json`, or None when the evaluate stage has not written it."""
    path = Path(campaign_dir) / "evaluate" / "summary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def selected_adapters(campaign_dir: Path) -> list[tuple[str, Path]]:
    """(unit, checkpoint folder) for every trained unit whose selected checkpoint is still on disk."""
    found = []
    train = Path(campaign_dir) / "train"
    for unit_dir in sorted(p for p in train.glob("*") if p.is_dir() and ".incomplete-" not in p.name and not p.name.endswith(".partial")):
        try:
            step = json.loads((unit_dir / "SELECTED.json").read_text(encoding="utf-8"))["step"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        ckpt = unit_dir / f"checkpoint_{int(step):05d}"
        if ckpt.is_dir():
            found.append((unit_dir.name, ckpt))
    return found


def _num(x, signed: bool = False) -> str:
    if not isinstance(x, (int, float)) or math.isnan(x):
        return "nan"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def table(header: list[str], rows: list[list[str]], left: tuple[int, ...] = (0,)) -> list[str]:
    """Aligned text table: columns in `left` are left-aligned, the others right-aligned."""
    widths = [max([len(h), *(len(r[i]) for r in rows)]) for i, h in enumerate(header)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) if i in left else c.rjust(widths[i]) for i, c in enumerate(cells)).rstrip()

    return [line(header), *(line(r) for r in rows)]


def _mark(entry: dict, primary: str | None) -> str:
    return " *" if entry.get("primary") or (primary and entry.get("readout") == primary) else "  "


def _mean(group: list[dict], key: str) -> float:
    return sum(g.get(key, math.nan) for g in group) / len(group)


def format_results(summary: dict, campaign_dir: Path, *, max_units: int = MAX_UNITS) -> str:
    entries = [e for e in summary.get("results", []) if isinstance(e, dict)]
    primary = summary.get("primary_readout")
    lines = [f"campaign   {summary.get('campaign', Path(campaign_dir).name)}",
             f"objective  {summary.get('objective', '?')}   readouts: delta and dual, both always reported"]

    base = [e for e in entries if e.get("unit") == "base"]
    if base:
        rows = [[e.get("dataset", "?"), f"{e.get('readout', '?')}{_mark(e, primary)}", str(e.get("assays", "")),
                 _num(e.get("macro_spearman"))] for e in base]
        lines += ["", "frozen student (before training), macro Spearman over every assay of the dataset"]
        lines += table(["dataset", "readout", "assays", "macro Spearman"], rows, left=(0, 1))

    groups: dict[tuple, list[dict]] = {}
    for e in entries:
        if e.get("unit") != "base":
            groups.setdefault((e.get("dataset", "?"), e.get("readout", "?")), []).append(e)
    if groups:
        rows = []
        for (dataset, readout), group in sorted(groups.items()):
            tag = f"{readout}{_mark(group[0], primary)}"
            if len(group) > max_units:
                improved = sum(1 for g in group if g.get("delta_vs_base", 0) > 0)
                rows.append([dataset, tag, f"mean of {len(group)} units ({improved} improved)",
                             str(sum(int(g.get("assays", 0)) for g in group)), _num(_mean(group, "base_macro_same_assays")),
                             _num(_mean(group, "macro_spearman")), _num(_mean(group, "delta_vs_base"), signed=True)])
                continue
            rows += [[dataset, tag, str(g.get("unit", "?")), str(g.get("assays", "")), _num(g.get("base_macro_same_assays")),
                      _num(g.get("macro_spearman")), _num(g.get("delta_vs_base"), signed=True)] for g in group]
        lines += ["", "distilled student against the same student before training, on the same assays"]
        lines += table(["dataset", "readout", "unit", "assays", "base", "distilled", "delta_vs_base"], rows, left=(0, 1, 2))
    if not entries:
        lines += ["", "summary.json has no results"]
    return "\n".join([*lines, "", PRIMARY_NOTE])


def files_and_adapters(campaign_dir: Path, *, max_listed: int = 5) -> str:
    root = Path(campaign_dir)
    lines = ["", "files", f"  {root / 'evaluate' / 'summary.json'}",
             f"  {root / 'evaluate' / 'per_assay.csv'}   (one row per unit, assay and readout)"]
    adapters = selected_adapters(root)
    if adapters:
        lines.append("adapters (PEFT LoRA: adapter_config.json + adapter_model.safetensors; docs/RESULTS.md shows how to load one)")
        lines += [f"  {unit}: {path}" for unit, path in adapters[:max_listed]]
        if len(adapters) > max_listed:
            lines.append(f"  ... and {len(adapters) - max_listed} more under {root / 'train'}")
    else:
        lines.append(f"adapters: none found under {root / 'train'} (training not finished, or the folder was moved)")
    lines += ["", "This is scratch storage and it is not backed up. Copy what you want to keep, for example from your own computer:",
              f"  rsync -av <netid>@grace.hprc.tamu.edu:{root}/{{evaluate,spec.json,resolved.json}} ./{root.name}/"]
    return "\n".join(lines)


def render(campaign_dir: Path) -> tuple[str, bool]:
    """(text for `mpd results`, whether the evaluate stage had written a summary)."""
    root = Path(campaign_dir)
    summary = load_summary(root)
    if summary is None:
        return (f"no results yet: {root / 'evaluate' / 'summary.json'} does not exist, so the evaluate stage has not "
                f"finished.\nSee where the campaign stands with: mpd status {root.name}"), False
    return format_results(summary, root) + "\n" + files_and_adapters(root), True
