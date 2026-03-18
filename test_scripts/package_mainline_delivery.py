#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _collect_cases(run_json_path: Path, k: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    obj = _load_json(run_json_path)
    rows = obj.get("results", []) or []
    applicable = [r for r in rows if bool(r.get("revision_applicable_gt", False))]
    for r in applicable:
        r["_gain"] = _safe_float((r.get("metrics") or {}).get("revision_gain", 0.0))
    applicable.sort(key=lambda r: r["_gain"], reverse=True)
    success = applicable[:k]
    boundary = list(reversed(applicable[-k:])) if applicable else []
    return {"success": success, "boundary": boundary}


def _copy_case_images(
    *,
    dataset: str,
    cases: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [f"## {dataset}"]
    for bucket in ("success", "boundary"):
        lines.append(f"### {bucket}")
        picked = cases.get(bucket, [])
        if not picked:
            lines.append("- (none)")
            continue
        for idx, row in enumerate(picked, start=1):
            vis = row.get("visualization_path")
            if not vis:
                lines.append(f"- case {idx}: no visualization_path")
                continue
            src = Path(vis)
            if not src.exists():
                lines.append(f"- case {idx}: missing file `{src}`")
                continue
            sample_id = str(row.get("sample_id", f"{bucket}_{idx}"))
            gain = _safe_float((row.get("metrics") or {}).get("revision_gain", 0.0))
            shape = str(((row.get("edit_intent_gt") or {}).get("shape")) or "none")
            dst = out_dir / f"{dataset}_{bucket}_{idx}_{sample_id}_{shape}.png"
            shutil.copy2(src, dst)
            lines.append(f"- {bucket}-{idx}: sample={sample_id}, shape={shape}, gain={gain:.4f}, file=`{dst.name}`")
    return lines


def _write_tables_extract(report_md: Path, out_md: Path) -> None:
    text = report_md.read_text(encoding="utf-8")
    out_md.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package mainline delivery artifacts (tables + cases).")
    parser.add_argument("--consolidation-dir", required=True, help="Path to tmp/mainline_consolidation_vX")
    parser.add_argument("--output-dir", default="results_v2", help="Output root folder")
    parser.add_argument("--topk", type=int, default=2, help="Top/bottom cases per dataset")
    args = parser.parse_args()

    src = Path(args.consolidation_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Full raw package copy.
    raw_dst = out / "mainline_consolidation"
    if raw_dst.exists():
        shutil.rmtree(raw_dst)
    shutil.copytree(src, raw_dst)

    report_md = src / "consolidation_report.md"
    report_json = src / "consolidation_report.json"
    if not report_md.exists() or not report_json.exists():
        raise FileNotFoundError("Missing consolidation_report.md/json in consolidation dir")

    _write_tables_extract(report_md, out / "TABLES.md")

    # Case studies from repeat-1 localized outputs (where visualizations are generated).
    datasets = {
        "weather": src / "revision_runs" / "weather" / "r1" / "localized_full_revision.json",
        "mtbench": src / "revision_runs" / "mtbench" / "r1" / "localized_full_revision.json",
        "xtraffic_speed": src / "revision_runs" / "xtraffic_speed" / "r1" / "localized_full_revision.json",
        "xtraffic_flow": src / "revision_runs" / "xtraffic_flow" / "r1" / "localized_full_revision.json",
    }
    case_root = out / "cases"
    case_root.mkdir(parents=True, exist_ok=True)
    index_lines: List[str] = [
        "# Cases Index",
        "",
        f"- source consolidation dir: `{src}`",
        f"- report: `{(out / 'TABLES.md').name}`",
        f"- selection: top-{args.topk} success + top-{args.topk} boundary per dataset (applicable-only, r1 localized)",
        "",
    ]

    for name, run_json in datasets.items():
        if not run_json.exists():
            index_lines.append(f"## {name}\n- missing run json: `{run_json}`\n")
            continue
        subset = _collect_cases(run_json, k=args.topk)
        lines = _copy_case_images(dataset=name, cases=subset, out_dir=case_root)
        index_lines.extend(lines)
        index_lines.append("")

    (out / "CASES.md").write_text("\n".join(index_lines), encoding="utf-8")

    readme = out / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Results v2",
                "",
                "This package is the deliverable bundle for the latest mainline consolidation run.",
                "",
                "Contents:",
                "- `mainline_consolidation/`: full raw run outputs (tables, repeats, flow diagnosis, agent smoke)",
                "- `TABLES.md`: main report with Table 1/2/3 + one-page findings",
                "- `CASES.md`: indexed case list (success + boundary) for each dataset",
                "- `cases/`: copied case visualization images referenced by `CASES.md`",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(out),
                "tables": str(out / "TABLES.md"),
                "cases_index": str(out / "CASES.md"),
                "raw_report": str(out / "mainline_consolidation" / "consolidation_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

