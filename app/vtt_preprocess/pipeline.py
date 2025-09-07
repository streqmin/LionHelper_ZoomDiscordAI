# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import os

from .config import Config
from .schema import Outputs, Segment, Span, CorrectionRecord
from .io_vtt import read_vtt  # write_vtt는 report.save_vtt 사용
from .normalize import normalize_segments
from .segment import merge_segments
from .topic import build_topic_index, TopicIndex
from .non_topic import detect_non_topic_blocks
from .detect import detect_error_spans
from .correct import correct_segments
from .llm_client import LLMClient
from .report import (
    save_vtt,
    save_corrections_csv,
    save_non_topic_blocks_csv_with_segments,
    build_markdown_report,
    save_markdown_report,
)

cfg = Config()


def _write_metrics_json(path: str, metrics: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(metrics, fw, ensure_ascii=False, indent=2)
    return path


def _filter_segments_by_non_topic(segments: List[Segment], blocks) -> List[Segment]:
    """
    비주제 블록을 최종 산출에서 제거하고 싶을 때 사용.
    - 현재는 기본적으로 '제거하지 않음'을 권장(리포트만 저장).
    - 제거가 필요하면 Config에 스위치 추가 후 이 함수를 호출하세요.
    """
    if not blocks:
        return segments
    # 블록 범위를 sid 기준으로 지운다고 가정
    drop_sids = set()
    for b in blocks:
        for sid in range(int(b.start_sid), int(b.end_sid) + 1):
            drop_sids.add(sid)
    out = [s for s in segments if s.sid not in drop_sids]
    return out if out else segments


def run_preprocess(
    vtt_path: str,
    curriculum_xlsx_path: str,
    out_vtt_path: str,
    out_blocks_path: str,
    out_corrections_csv: str,
    out_metrics_json: str,
    out_report_md: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",  # "disabled" 또는 "gpt-4o-mini" 등
    remove_non_topic_from_output: bool = False,  # 최종 VTT에서 비주제 블록 제거할지
) -> Outputs:
    """
    엔드투엔드 전처리 파이프라인.

    반환:
      Outputs(segments, non_topic_blocks, corrections, metrics)
      - segments: 최종 저장한 세그먼트(필터링/교정 반영)
    """
    # 1) 로드
    raw_segments: List[Segment] = read_vtt(vtt_path)

    # 2) 정규화
    norm_segments = normalize_segments(raw_segments)

    # 3) 병합
    merged_segments = merge_segments(norm_segments)

    # 4) 토픽 인덱스
    topic: TopicIndex = build_topic_index(curriculum_xlsx_path)

    # 5) 비주제 블록 감지 (토픽/큐/휴리스틱 사용)
    blocks = detect_non_topic_blocks(merged_segments, topic_index=topic, k_min=1)

    # 6) 오류 스팬 탐지 (원문/병합 기준)
    spans: List[Span] = detect_error_spans(merged_segments)

    # 7) 교정 (규칙 → 검증 → 실패 시 LLM, ΔTopic/길이비 검증)
    provider = "openai" if llm_model and llm_model.lower() != "disabled" else "disabled"
    llm = LLMClient(
        provider=provider, model=llm_model if provider == "openai" else "gpt-4o-mini"
    )

    new_segments, records = correct_segments(
        segments=merged_segments,
        spans=spans,
        topic=topic,
        llm=llm,
        curriculum_terms=list(getattr(topic, "terms", [])),
    )

    # (선택) 비주제 블록 제거
    final_segments = _filter_segments_by_non_topic(new_segments, blocks)

    # 8) 산출물 저장
    # 8-1) VTT
    save_vtt(final_segments, out_vtt_path)
    # 8-2) 비주제 CSV
    save_non_topic_blocks_csv_with_segments(blocks, merged_segments, out_blocks_path)
    # 8-3) 교정 CSV
    save_corrections_csv(records, out_corrections_csv)
    # 8-4) 리포트(Markdown, 옵션)
    if out_report_md:
        md = build_markdown_report(
            segments_before=merged_segments,
            segments_after=final_segments,
            corrections=records,
            non_topic_blocks=blocks,
            topic_threshold=cfg.topic_threshold,  # Config.topic_threshold 기본 사용
        )
        save_markdown_report(md, out_report_md)

    # 8-5) 메트릭(간단)
    metrics = {
        "accept_rate": (
            (sum(1 for r in records if r.orig != r.final) / len(records))
            if records
            else 0.0
        ),
        "applied_by_method": {
            "RULE": sum(1 for r in records if r.method == "RULE" and r.orig != r.final),
            "LLM_SELECT": sum(
                1 for r in records if r.method == "LLM_SELECT" and r.orig != r.final
            ),
            "LGV": sum(1 for r in records if r.method == "LGV" and r.orig != r.final),
        },
        "llm_calls": sum(1 for r in records if r.method == "LLM_SELECT"),
        "delta_topic_avg": (
            (
                sum(r.delta_topic for r in records if r.orig != r.final)
                / max(1, sum(1 for r in records if r.orig != r.final))
            )
            if records
            else 0.0
        ),
    }
    _write_metrics_json(out_metrics_json, metrics)

    return Outputs(
        segments=final_segments,
        non_topic_blocks=blocks,
        topic_segments=final_segments,
        corrections=records,
        metrics=metrics,
    )
