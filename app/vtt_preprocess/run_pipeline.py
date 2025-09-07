# run_pipeline.py
from .__future__ import annotations
import argparse, os, sys

# 패키지 경로를 PYTHONPATH 맨 앞에 추가
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from .pipeline import run_preprocess


def main():
    ap = argparse.ArgumentParser(description="VTT preprocessing (no -m)")
    ap.add_argument("--vtt", required=True, help="input VTT file")
    ap.add_argument("--curriculum", required=True, help="curriculum Excel (.xlsx)")
    ap.add_argument("--outdir", default="./outputs", help="output directory")
    ap.add_argument(
        "--llm",
        default="disabled",
        help='"disabled" or OpenAI model (e.g., "gpt-4o-mini")',
    )
    ap.add_argument(
        "--remove-non-topic",
        action="store_true",
        help="drop non-topic blocks from final VTT",
    )
    ap.add_argument(
        "--report-md",
        default=None,
        help="path for markdown report (default: <outdir>/report.md)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_vtt = os.path.join(args.outdir, "cleaned.vtt")
    out_blocks = os.path.join(args.outdir, "non_topic_blocks.csv")
    out_corr = os.path.join(args.outdir, "corrections.csv")
    out_metrics = os.path.join(args.outdir, "metrics.json")
    report_md = args.report_md or os.path.join(args.outdir, "report.md")

    outputs = run_preprocess(
        vtt_path=args.vtt,
        curriculum_xlsx_path=args.curriculum,
        out_vtt_path=out_vtt,
        out_blocks_path=out_blocks,
        out_corrections_csv=out_corr,
        out_metrics_json=out_metrics,
        out_report_md=report_md,
        llm_model=args.llm,
        remove_non_topic_from_output=bool(args.remove_non_topic),
    )

    print("== Done ==")
    print("VTT:", out_vtt)
    print("Non-topic CSV:", out_blocks)
    print("Corrections CSV:", out_corr)
    print("Report MD:", report_md)
    print("Metrics JSON:", out_metrics)


if __name__ == "__main__":
    main()
