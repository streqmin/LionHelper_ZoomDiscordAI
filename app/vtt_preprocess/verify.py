# -*- coding: utf-8 -*-
"""
verify: 교정안 승인(accept) 여부 판단 유틸

- ΔTopic 개선(TopicIndex 기반)
- 길이 비율, 괄호/따옴표 밸런스, 편집 비율(Levenshtein) 제한
- 카테고리별 관대한/보수적 정책
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .topic import TopicIndex
from .config import Config

_cfg = Config()


# -----------------------
# 간단 Levenshtein
# -----------------------
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # O(min(la,lb)) 메모리
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def _len_ratio(a: str, b: str) -> float:
    if len(a) == 0:
        return 1.0 if len(b) == 0 else float("inf")
    return len(b) / len(a)


def _balanced(s: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ("{", "}"), ('"', '"'), ("'", "'")]
    for l, r in pairs:
        if s.count(l) != s.count(r):
            return False
    return True


@dataclass
class VerifyResult:
    ok: bool
    reason: str
    delta_topic: float


# -----------------------
# Public API
# -----------------------
def approve_change(
    topic: TopicIndex,
    before: str,
    after: str,
    category: str,
) -> VerifyResult:
    """
    교정 승인 여부 판단.
    기준(요약):
      - ΔTopic >= cfg.delta_topic_min (기본 0.0~0.02로 권장)
      - 길이 비율 0.5..1.6 (카테고리별 완화/강화)
      - 괄호/따옴표 밸런스 유지
      - 편집율 제한(Levenshtein / max(len(before),1)) <= 0.55 (일반)
    """
    cat = (category or "").upper()
    if before == after:
        return VerifyResult(False, "no_change", 0.0)

    # ΔTopic
    dt = float(topic.delta_topic(before, after))

    # 길이 비율
    r = _len_ratio(before, after)
    lo, hi = 0.5, 1.6
    if cat in {"MIXED", "DIGIT_SUFFIX", "DIGIT_BOUNDARY", "PHONO"}:
        lo, hi = 0.4, 1.8  # 조금 관대
    elif cat in {"CONTEXT_ANOM", "NGRAM_OUTLIER"}:
        lo, hi = 0.6, 1.6  # 기본

    if not (lo <= r <= hi):
        return VerifyResult(False, f"length_ratio_out_of_bounds({r:.2f})", dt)

    # 괄호/따옴표
    if not _balanced(after):
        return VerifyResult(False, "unbalanced_after", dt)

    # 편집율
    dist = _levenshtein(before, after)
    denom = max(len(before), 1)
    edit_rate = dist / denom

    max_rate = 0.55
    if cat in {"MIXED", "DIGIT_SUFFIX", "DIGIT_BOUNDARY", "PHONO"}:
        max_rate = 0.75  # 약간 더 허용
    if edit_rate > max_rate:
        return VerifyResult(False, f"edit_rate_high({edit_rate:.2f})", dt)

    # ΔTopic 최종 조건
    if dt < _cfg.delta_topic_min:
        return VerifyResult(False, f"delta_topic_low({dt:.3f})", dt)

    return VerifyResult(True, "ok", dt)
