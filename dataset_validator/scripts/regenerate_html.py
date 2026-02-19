"""Regenerate summary.html from existing results.json and dataset directories."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset_validator.report.generator import (
    generate_html_report,
    generate_html_scroll_report,
)


def main():
    parser = argparse.ArgumentParser(description="Regenerate HTML report from results.json")
    parser.add_argument("--results-json", type=str, required=True, help="Path to results.json")
    parser.add_argument("--input-dir", type=str, required=True, help="Input images directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output images directory")
    parser.add_argument("--reference-dir", type=str, required=True, help="Reference images directory")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (default: same dir as results.json)")
    parser.add_argument("--mode", choices=["scroll", "base64"], default="scroll",
                        help="scroll: lightweight infinite scroll (default), base64: embedded images")
    parser.add_argument("--max-passed", type=int, default=20, help="Max passed samples to display (base64 mode only)")
    args = parser.parse_args()

    results_json = Path(args.results_json)
    if not results_json.exists():
        print(f"Error: {results_json} not found")
        sys.exit(1)

    with open(results_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    results = report["results"]
    metadata = report["metadata"]
    statistics = report.get("statistics", {})
    out_path = Path(args.out) if args.out else results_json.parent / "summary.html"

    if args.mode == "scroll":
        image_dirs = {
            "input": str(Path(args.input_dir).resolve()),
            "reference": str(Path(args.reference_dir).resolve()),
            "output": str(Path(args.output_dir).resolve()),
        }
        print(f"Loaded {len(results)} results (scroll mode)")
        html_path = generate_html_scroll_report(
            results, metadata, statistics, image_dirs, out_path,
        )
    else:
        from dataset_validator.core.image_loader import scan_dataset
        matched, _ = scan_dataset(
            Path(args.input_dir), Path(args.reference_dir), Path(args.output_dir)
        )
        entries_map = {e["stem"]: e for e in matched}
        print(f"Loaded {len(results)} results, {len(entries_map)} matched entries (base64 mode)")
        html_path = generate_html_report(
            results, metadata, statistics, entries_map, out_path,
            max_passed_display=args.max_passed,
        )

    print(f"HTML report saved: {html_path}")


if __name__ == "__main__":
    main()
