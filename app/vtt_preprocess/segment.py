# -*- coding: utf-8 -*-
"""
segment: 세그먼트 병합/유사 중복 제거

정책(확정판):
- 병합 단위: 연속 cue들을 모아 "30–60초" 또는 "200–400자" 기준을 만족하는 블록으로 합침.
- 종료 조건:
  1) 현재 블록이 최소조건(≥30초 또는 ≥200자)을 이미 만족했고,
  2) 다음 cue를 추가하면 최대조건(>60초 또는 >400자)을 넘길 경우 → 블록 종료.
- 유사 중복 제거: rapidfuzz.ratio(prev_text, curr_text) ≥ 0.90 이면 현재 블록을 폐기.

주의: 입력 Segment는 start_sec 기준 정렬되어 있다고 가정(일반적 webvtt).
"""
from __future__ import annotations

from typing import List, Dict
from dataclasses import replace

from rapidfuzz import fuzz

from .schema import Segment
from .config import Config

_cfg = Config()


def _should_close_block(
    cur_chars: int, cur_secs: float, next_chars: int, next_secs: float
) -> bool:
    """
    현재 블록(길이 cur_*)에 다음 cue(길이 next_*)를 더하면 최대 한계를 넘기는지 검사.
    단, 현재 블록이 '최소 조건'은 이미 충족해야 종료할 수 있음.
    """
    has_min = (cur_secs >= _cfg.merge_min_secs) or (cur_chars >= _cfg.merge_min_chars)
    would_exceed = ((cur_secs + next_secs) > _cfg.merge_max_secs) or (
        (cur_chars + next_chars) > _cfg.merge_max_chars
    )
    return has_min and would_exceed


def merge_segments(raw: List[Segment]) -> List[Segment]:
    """
    입력 cue 리스트를 정책에 따라 병합하고, 인접 유사 중복을 제거하여 반환.
    - 반환 세그먼트의 sid는 0부터 재할당.
    - 비어있는 텍스트 블록은 제외.
    - 기존 meta 정보는 병합하여 보존.
    """
    merged: List[Segment] = []

    if not raw:
        return merged

    # 누적 버퍼
    buf_text_parts: List[str] = []
    buf_start = raw[0].start_sec
    buf_end = raw[0].end_sec
    buf_chars = 0
    buf_secs = 0.0
    buf_meta: Dict[str, float] = {}  # meta 정보 누적용

    def flush():
        nonlocal buf_text_parts, buf_start, buf_end, buf_chars, buf_secs, buf_meta
        text = "\n".join([t for t in buf_text_parts if t.strip()]).strip()
        if text:
            merged.append(
                Segment(
                    sid=len(merged),
                    start_sec=buf_start,
                    end_sec=buf_end,
                    text=text,
                    tags=set(),
                    meta=buf_meta.copy(),  # 누적된 meta 정보 사용
                )
            )
        # reset
        buf_text_parts = []
        buf_start = 0.0
        buf_end = 0.0
        buf_chars = 0
        buf_secs = 0.0
        buf_meta = {}

    # 초기화
    buf_text_parts.append(raw[0].text or "")
    buf_chars = len(raw[0].text or "")
    buf_secs = max(0.0, (raw[0].end_sec - raw[0].start_sec))
    buf_start = raw[0].start_sec
    buf_end = raw[0].end_sec
    # 첫 번째 세그먼트의 meta 정보로 초기화
    buf_meta = raw[0].meta.copy() if raw[0].meta else {}

    # iterate
    for i in range(1, len(raw)):
        cur = raw[i]
        cur_text = cur.text or ""
        cur_chars = len(cur_text)
        cur_secs = max(0.0, (cur.end_sec - cur.start_sec))

        # 블록 종료 여부 판단
        if _should_close_block(buf_chars, buf_secs, cur_chars, cur_secs):
            flush()
            # 새 버퍼 시작
            buf_text_parts = [cur_text]
            buf_start = cur.start_sec
            buf_end = cur.end_sec
            buf_chars = cur_chars
            buf_secs = cur_secs
            # 새 버퍼의 meta 정보로 초기화
            buf_meta = cur.meta.copy() if cur.meta else {}
        else:
            # 같은 블록으로 누적
            buf_text_parts.append(cur_text)
            buf_end = cur.end_sec
            buf_chars += cur_chars
            buf_secs += cur_secs
            # meta 정보 병합 (MIXED_TOKENS는 누적, 다른 것은 첫 번째 값 유지)
            if cur.meta:
                for key, value in cur.meta.items():
                    if key == "MIXED_TOKENS":
                        # MIXED_TOKENS는 누적
                        buf_meta[key] = buf_meta.get(key, 0) + value
                    elif key not in buf_meta:
                        # 다른 meta는 첫 번째 값 사용
                        buf_meta[key] = value

    # 마지막 flush
    flush()

    # 유사 중복 제거(인접 블록 비교)
    if not merged:
        return merged

    deduped: List[Segment] = []
    prev_text = None
    for seg in merged:
        if prev_text is None:
            deduped.append(seg)
            prev_text = seg.text
            continue
        sim = fuzz.ratio(prev_text, seg.text) / 100.0
        if sim >= _cfg.dedup_similarity:
            # skip this segment (중복으로 간주)
            continue
        deduped.append(replace(seg, sid=len(deduped)))  # sid 재할당
        prev_text = seg.text

    return deduped
