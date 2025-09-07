# -*- coding: utf-8 -*-
"""
report: Phase 6 - 결과 요약/리포트/내보내기

입력(소스 모듈에서 제공되는 구조 가정):
- segments_before: List[Segment]   # 교정/비주제 제거 전
- segments_after : List[Segment]   # 교정/비주제 제거 후 (또는 교정만 반영된 상태)
- corrections    : List[CorrectionRecord]
- non_topic_blocks: Optional[List[object]]  # non_topic.detect_non_topic_blocks 결과
- topic_index    : Optional[TopicIndex]     # 커리큘럼 토픽 스코어러(요약 계산시 사용)

출력(유틸 함수 제공):
- 요약 딕셔너리(통계)
- Markdown 리포트 문자열
- CSV 저장(교정/비주제 블록)
- VTT 저장(세그먼트 → VTT)

메모:
- NonTopicBlock 객체에 start_sec/end_sec가 없어도 CSV 저장 시
  segments를 이용해 (sid 범위 → 초 범위)로 환산합니다.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

from .config import Config
from .schema import Segment, CorrectionRecord
from .utils import timecode

_cfg = Config()


# -------------------------------------
# 요약 집계
# -------------------------------------
def summarize_corrections(records: List[CorrectionRecord]) -> Dict[str, object]:
    """
    교정 기록을 카테고리/방법/승인 여부로 집계.
    CorrectionRecord 스키마(사용자 지정):
        timecode: str
        sid: int
        offset: str
        category: str
        method: Literal["RULE", "LLM_SELECT", "LGV"]
        orig: str
        final: str
        score_or_fit: float
        delta_topic: float
        evidence: str
    """
    out: Dict[str, object] = {}
    total = len(records)
    accepted = sum(1 for r in records if r.orig != r.final)
    rule_uses = sum(1 for r in records if r.method == "RULE" and r.orig != r.final)
    llm_uses = sum(1 for r in records if r.method == "LLM_SELECT" and r.orig != r.final)
    lgv_uses = sum(1 for r in records if r.method == "LGV" and r.orig != r.final)

    by_cat: Dict[str, int] = {}
    for r in records:
        k = (r.category or "").upper()
        if r.orig != r.final:  # 적용된 케이스만
            by_cat[k] = by_cat.get(k, 0) + 1

    out["total_records"] = total
    out["accepted_changes"] = accepted
    out["accept_rate"] = (accepted / total) if total else 0.0
    out["applied_by_method"] = {
        "RULE": rule_uses,
        "LLM_SELECT": llm_uses,
        "LGV": lgv_uses,
    }
    out["applied_by_category"] = by_cat
    # ΔTopic 분포(간단 요약)
    deltas = [r.delta_topic for r in records if r.orig != r.final]
    if deltas:
        out["delta_topic_avg"] = sum(deltas) / len(deltas)
        out["delta_topic_min"] = min(deltas)
        out["delta_topic_max"] = max(deltas)
    else:
        out["delta_topic_avg"] = 0.0
        out["delta_topic_min"] = 0.0
        out["delta_topic_max"] = 0.0
    return out


def summarize_non_topic(blocks: Optional[List[object]]) -> Dict[str, object]:
    """
    NonTopicBlock 요약. (여기서는 start_sec/end_sec가 없으면 누적 시간 0으로 처리)
    CSV 저장 시에는 별도의 함수에서 segments를 사용해 sec 환산을 수행합니다.
    """
    out: Dict[str, object] = {}
    if not blocks:
        return {"count": 0, "total_secs": 0.0, "by_label": {}}
    cnt = len(blocks)
    total_secs = 0.0
    by_label: Dict[str, int] = {}
    for b in blocks:
        label = getattr(b, "label", "NON_TOPIC")
        by_label[label] = by_label.get(label, 0) + 1
        if hasattr(b, "start_sec") and hasattr(b, "end_sec"):
            try:
                total_secs += max(
                    0.0, float(getattr(b, "end_sec")) - float(getattr(b, "start_sec"))
                )
            except Exception:
                pass
    out["count"] = cnt
    out["total_secs"] = total_secs
    out["by_label"] = by_label
    return out


def summarize_topic_coverage(
    segments_after: List[Segment], threshold: Optional[float] = None
) -> Dict[str, object]:
    """
    세그먼트 메타에 'topic' 점수가 들어있다는 전제 하에 커버리지 개략 계산.
    - threshold 이상인 세그먼트 비율
    - 가중 평균(길이 가중치) 등 간단 수치
    - topic 점수가 0인 세그먼트는 계산에서 제외
    """
    thr = _cfg.topic_threshold if threshold is None else float(threshold)
    n = len(segments_after)
    if n == 0:
        return {"count": 0, "ratio": 0.0, "avg": 0.0, "wavg": 0.0, "threshold": thr}

    # topic 점수가 0이 아닌 세그먼트만 필터링
    valid_segments = []
    for seg in segments_after:
        topic_score = float(getattr(seg, "meta", {}).get("topic", 0.0))
        if topic_score > 0.0:  # 0이 아닌 경우만 포함
            valid_segments.append((seg, topic_score))
    
    n_valid = len(valid_segments)
    if n_valid == 0:
        return {"count": 0, "ratio": 0.0, "avg": 0.0, "wavg": 0.0, "threshold": thr}

    # 유효한 세그먼트들의 점수만 추출
    scores = [score for _, score in valid_segments]
    hits = sum(1 for s in scores if s >= thr)
    avg = sum(scores) / n_valid

    # 길이 가중치(초 단위) - 유효한 세그먼트만
    weights = []
    for seg, _ in valid_segments:
        dur = max(0.0, float(seg.end_sec) - float(seg.start_sec))
        weights.append(dur if dur > 0 else 1.0)
    wsum = sum(weights)
    wavg = sum(s * w for s, w in zip(scores, weights)) / wsum if wsum > 0 else avg

    return {
        "count": hits,
        "ratio": hits / n_valid,  # 유효한 세그먼트 수 기준으로 비율 계산
        "avg": avg,
        "wavg": wavg,
        "threshold": thr,
    }


# -------------------------------------
# 렌더링(텍스트/마크다운)
# -------------------------------------
def _md_title(s: str, level: int = 2) -> str:
    return f"{'#' * level} {s}\n\n"


def build_markdown_report(
    segments_before: List[Segment],
    segments_after: List[Segment],
    corrections: List[CorrectionRecord],
    non_topic_blocks: Optional[List[object]] = None,
    topic_threshold: Optional[float] = None,
) -> str:
    """
    간단한 Markdown 리포트 문자열 생성.
    """
    sb = len(segments_before)
    sa = len(segments_after)

    corr_sum = summarize_corrections(corrections)
    nt_sum = summarize_non_topic(non_topic_blocks)
    cov = summarize_topic_coverage(segments_after, threshold=topic_threshold)

    md: List[str] = []
    md.append(_md_title("VTT 전처리 리포트 (Phase 6)", 1))

    # 개요
    md.append(_md_title("개요", 2))
    md.append(f"- 원본 세그먼트 수: **{sb}**\n")
    md.append(f"- 교정 후 세그먼트 수: **{sa}**\n")
    md.append("\n")
    
    for seg in segments_before:
        md.append(f"sid {seg.sid} {seg.meta}\n")
    md.append("\n")
    
    for seg in segments_after:
        md.append(f"sid {seg.tags} {seg.sid} {seg.meta}\n")
    md.append("\n")

    # 비주제 블록
    md.append(_md_title("비주제(Non-Topic) 블록 요약", 2))
    md.append(f"- 블록 수: **{nt_sum.get('count', 0)}**\n")
    md.append(
        f"- 누적 시간(가용 정보 기준): **{nt_sum.get('total_secs', 0.0):.1f}초**\n"
    )
    by_label = nt_sum.get("by_label", {}) or {}
    if by_label:
        md.append("\n| 라벨 | 개수 |\n|---|---:|\n")
        for k, v in sorted(by_label.items(), key=lambda x: (-x[1], x[0])):
            md.append(f"| {k} | {v} |\n")
    md.append("\n")

    # 교정 요약
    md.append(_md_title("교정 요약", 2))
    md.append(f"- 전체 스팬 기록: **{corr_sum['total_records']}**\n")
    md.append(
        f"- 승인/적용된 교정 수: **{corr_sum['accepted_changes']}**  "
        f"(비율 **{corr_sum['accept_rate']:.1%}**)\n"
    )
    by_m = corr_sum["applied_by_method"]  # type: ignore[index]
    md.append(
        f"- 적용 수(방법별): RULE={by_m.get('RULE',0)}, LLM={by_m.get('LLM_SELECT',0)}, LGV={by_m.get('LGV',0)}\n"
    )
    by_c = corr_sum["applied_by_category"]  # type: ignore[index]
    if by_c:
        md.append("\n| 카테고리 | 적용 수 |\n|---|---:|\n")
        for k, v in sorted(by_c.items(), key=lambda x: (-x[1], x[0])):
            md.append(f"| {k} | {v} |\n")
    md.append(
        f"\n- ΔTopic: avg={corr_sum.get('delta_topic_avg',0.0):.3f}, "
        f"min={corr_sum.get('delta_topic_min',0.0):.3f}, "
        f"max={corr_sum.get('delta_topic_max',0.0):.3f}\n\n"
    )

    # 토픽 커버리지
    md.append(_md_title("커리큘럼 커버리지(간이)", 2))
    md.append(f"- 임계값(threshold): **{cov.get('threshold')}**\n")
    
    # 유효한 세그먼트 수 계산 (topic 점수 > 0)
    valid_count = sum(1 for seg in segments_after 
                     if float(getattr(seg, "meta", {}).get("topic", 0.0)) > 0.0)
    
    md.append(
        f"- 세그먼트 기준 커버리지: **{cov.get('ratio',0.0):.1%}**  "
        f"(임계 이상 {cov.get('count',0)} / 유효 세그먼트 {valid_count} / 전체 {sa})\n"
    )
    md.append(
        f"- 토픽 점수 평균: avg={cov.get('avg',0.0):.3f}, length-weighted avg={cov.get('wavg',0.0):.3f}\n"
    )

    return "".join(md)


# -------------------------------------
# 파일로 내보내기
# -------------------------------------
def save_corrections_csv(records: List[CorrectionRecord], fpath: str) -> str:
    """
    교정 기록을 CSV로 저장. (문자열 내 개행/쉼표는 CSV 규칙으로 이스케이프)
    """
    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    with open(fpath, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(
            [
                "timecode",
                "sid",
                "offset",
                "category",
                "method",
                "orig",
                "final",
                "score_or_fit",
                "delta_topic",
                "evidence",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.timecode,
                    r.sid,
                    r.offset,
                    r.category,
                    r.method,
                    r.orig,
                    r.final,
                    f"{r.score_or_fit:.4f}",
                    f"{r.delta_topic:.4f}",
                    r.evidence,
                ]
            )
    return fpath


def segments_to_vtt(segments: List[Segment]) -> str:
    """
    세그먼트 리스트를 단일 VTT 문자열로 변환.
    Segment(sid, start_sec, end_sec, text) + utils.timecode 사용
    """
    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        tc = timecode(seg.start_sec, seg.end_sec)
        lines.append(str(seg.sid))
        lines.append(str(round(seg.meta.get("topic", 0.0), 2)))
        lines.append(tc)
        lines.append(seg.text or "")
        lines.append("")  # blank line between cues
    return "\n".join(lines)


def save_vtt(segments: List[Segment], fpath: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as fw:
        fw.write(segments_to_vtt(segments))
    return fpath


def save_markdown_report(md: str, fpath: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as fw:
        fw.write(md)
    return fpath


# -------------------------------------
# sid → sec 환산 유틸 & Non-Topic CSV (옵션 A)
# -------------------------------------
def _sid_range_to_secs(
    segments: List[Segment], sid_lo: int, sid_hi: int
) -> Tuple[float, float]:
    """
    sid 범위를 시간(초) 범위로 근사 계산.
    - 범위에 걸친 세그먼트들의 start_sec 최소, end_sec 최대를 사용
    - 매칭이 없으면 (0.0, 0.0) 반환
    """
    if not segments:
        return (0.0, 0.0)
    start = float("inf")
    end = float("-inf")
    for s in segments:
        sid = int(s.sid)
        if sid_lo <= sid <= sid_hi:
            start = min(start, float(s.start_sec))
            end = max(end, float(s.end_sec))
    if start is float("inf") or end is float("-inf"):
        return (0.0, 0.0)
    return (start, end)


def save_non_topic_blocks_csv_with_segments(
    blocks: Optional[List[object]],
    segments: List[Segment],
    fpath: str,
) -> str:
    """
    NonTopicBlock에 start_sec/end_sec 필드가 없어도,
    segments에서 sid 범위를 시간으로 환산해 CSV로 저장.
    """
    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    with open(fpath, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(
            [
                "label",
                "start_sid",
                "end_sid",
                "confidence",
                "start_sec",
                "end_sec",
                "duration",
                "sample_text",
            ]
        )

        if not blocks:
            return fpath

        for b in blocks:
            label = getattr(b, "label", "NON_TOPIC")
            start_sid = int(getattr(b, "start_sid", -1))
            end_sid = int(getattr(b, "end_sid", -1))
            conf = float(getattr(b, "confidence", 0.0))

            # 먼저 객체에 시간이 있으면 그대로 사용
            if hasattr(b, "start_sec") and hasattr(b, "end_sec"):
                try:
                    start_sec = float(getattr(b, "start_sec"))
                    end_sec = float(getattr(b, "end_sec"))
                except Exception:
                    start_sec, end_sec = _sid_range_to_secs(
                        segments, start_sid, end_sid
                    )
            else:
                # 없으면 세그먼트에서 근사
                start_sec, end_sec = _sid_range_to_secs(segments, start_sid, end_sid)

            dur = max(0.0, end_sec - start_sec)
            sample = (getattr(b, "sample_text", "") or "").replace("\n", " ")[:160]
            w.writerow(
                [
                    label,
                    start_sid,
                    end_sid,
                    f"{conf:.3f}",
                    f"{start_sec:.3f}",
                    f"{end_sec:.3f}",
                    f"{dur:.3f}",
                    sample,
                ]
            )
    return fpath
