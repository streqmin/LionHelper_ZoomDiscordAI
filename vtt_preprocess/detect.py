# -*- coding: utf-8 -*-
"""
detect: 원문 기준 오류 스팬 탐지

카테고리:
- MIXED          : 숫자/라틴/기호+한글 혼용 토큰
- DIGIT_SUFFIX   : '42에는' 같은 숫자+조사 결합(단일 토큰)
- DIGIT_BOUNDARY : '42' '에는' 처럼 인접 두 토큰 결합 후보
- PHONO          : 발음 혼동(간단 셋) ex) 같이/가치, 국물/궁물, 학과/학꽈 ...
- MORPH_ODD      : 형태소 비정상(숫자 앞+조사, 영문 앞+조사 등 간단 휴리스틱)
- NGRAM_OUTLIER  : 문자 3그램 놀람도 상위 p% 이상치(기본 10%)
- CONTEXT_ANOM   : ContextFit ≤ cfg.contextfit_low

주의:
- 이 모듈은 '탐지'만 수행하며, 교정은 correct.py에서 처리됨.
- CONTEXT_ANOM/NGRAM_OUTLIER는 VTT 특성(짧은 문장) 고려해 문장(여러 문장 묶음) 단위로 span을 집계한다.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Set, Tuple

import numpy as np

from schema import Segment, Span
from config import Config
from utils import is_mixed_token, looks_like_digit_suffix

_cfg = Config()

# PHONO 간단 후보(오식→정형 가능성 높음). 탐지에만 사용.
_PHONO_SET: Set[str] = {
    "가치",
    "궁물",
    "학꽈",
    "수학적귀납볍",
    "라면",  # '라면'은 산란/분광 맥락 조건부 처리
}

# 조사(간단)
_JOSA = (
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "에서",
    "으로",
    "에게",
    "까지",
    "부터",
    "의",
    "만",
    "도",
)

# 숫자 단독 토큰
_RE_NUM = re.compile(r"^\d{1,3}$")

# 문장 경계 추정용
_SENT_ENDERS = set(".?!…")
_NEWLINE = "\n"

# LLM 대상 카테고리(문장 집계/확장 적용)
_AGGREGATE_CATS = {"CONTEXT_ANOM", "NGRAM_OUTLIER"}


_LLM_DEDUPE_IOU = getattr(_cfg, "llm_dedupe_iou", 0.75)


def _iou(L1: int, R1: int, L2: int, R2: int) -> float:
    inter = max(0, min(R1, R2) - max(L1, L2))
    union = max(R1, R2) - min(L1, L2)
    return 0.0 if union <= 0 else inter / union


def _iter_tokens_with_offsets(text: str):
    """
    공백 기반 토큰화 + (start,end) 오프셋.
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        yield (i, j, text[i:j])
        i = j


def _char_trigrams(s: str) -> List[str]:
    return [s[i : i + 3] for i in range(0, max(0, len(s) - 2))]


def _ngram_surprisal_scores(
    text: str,
) -> Tuple[Dict[str, int], Dict[Tuple[int, int], float]]:
    """
    문자 3그램 빈도로 놀람도 측정.
    반환:
      freq: {tri: count}
      token_score: { (start,end): surprisal_mean }
    """
    tris = _char_trigrams(text)
    if not tris:
        return {}, {}
    freq: Dict[str, int] = {}
    for t in tris:
        freq[t] = freq.get(t, 0) + 1
    total = sum(freq.values())
    prob = {t: (c / total) for t, c in freq.items()}
    token_score: Dict[Tuple[int, int], float] = {}
    for s, e, tok in _iter_tokens_with_offsets(text):
        tri = _char_trigrams(tok)
        if not tri:
            token_score[(s, e)] = 0.0
            continue
        vals = [-math.log(max(prob.get(t, 1.0 / total), 1e-12)) for t in tri]
        token_score[(s, e)] = float(sum(vals) / len(vals))
    return freq, token_score


def _split_sentences(text: str) -> List[Tuple[int, int]]:
    """
    단순 문장 분할: 종결 부호/개행을 문장 경계로 사용.
    반환: [(start_idx, end_idx)] ; end_idx는 '포함 끝+1'
    """
    n = len(text)
    spans: List[Tuple[int, int]] = []
    i = 0
    start = 0
    while i < n:
        ch = text[i]
        if ch in _SENT_ENDERS:
            end = i + 1
            # 공백 정리
            while start < n and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                spans.append((start, end))
            start = i + 1
        i += 1
    # 마지막 꼬리
    if start < n:
        end = n
        while start < n and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans or [(0, n)]


def _find_sentence_index(
    sentences: List[Tuple[int, int]], s: int, e: int
) -> Tuple[int, int]:
    """
    토큰 s..e가 걸쳐있는 문장 인덱스 범위 반환 (최소-최대).
    하나 문장에만 속하면 (k,k).
    """
    k0 = k1 = -1
    for k, (L, R) in enumerate(sentences):
        if not (e <= L or s >= R):  # 겹치면
            if k0 == -1:
                k0 = k
            k1 = k
    if k0 == -1:
        # 삽입 위치 근사
        for k, (L, R) in enumerate(sentences):
            if s < R:
                return (k, k)
        return (len(sentences) - 1, len(sentences) - 1)
    return (k0, k1)


def _aggregate_to_target_window(
    sentences: List[Tuple[int, int]],
    k0: int,
    k1: int,
    target: int,
    min_chars: int,
    max_chars: int,
) -> Tuple[int, int]:
    """
    k0..k1 문장 범위를 시작으로, 목표 길이가 될 때까지
    좌/우 문장을 추가로 합쳐 (L,R) 윈도우 반환.
    """
    kL, kR = k0, k1
    L, R = sentences[kL][0], sentences[kR][1]
    # 짧으면 확장
    while (R - L) < max(min_chars, target):
        left_possible = kL > 0 and (R - sentences[kL - 1][0]) <= max_chars
        right_possible = (kR + 1) < len(sentences) and (
            sentences[kR + 1][1] - L
        ) <= max_chars

        # 더 짧아지는 쪽(문장 길이가 짧은 쪽)부터 확장 시도
        expand_left_first = False
        if left_possible and right_possible:
            left_gain = R - sentences[kL - 1][0]
            right_gain = sentences[kR + 1][1] - L
            expand_left_first = left_gain <= right_gain

        expanded = False
        if left_possible and (expand_left_first or not right_possible):
            kL -= 1
            L = sentences[kL][0]
            expanded = True
        elif right_possible:
            kR += 1
            R = sentences[kR][1]
            expanded = True

        if not expanded:
            break  # 상한 또는 확장 불가

    # 상한 초과시 중앙 기준 자르기
    if (R - L) > max_chars:
        mid = (L + R) // 2
        half = max_chars // 2
        L, R = max(L, mid - half), min(R, mid + half)
    return (L, R)


def _overlap_ratio(aL: int, aR: int, bL: int, bR: int) -> float:
    inter = max(0, min(aR, bR) - max(aL, bL))
    union = max(aR, bR) - min(aL, bL)
    return 0.0 if union == 0 else inter / union


# -----------------------------
# 간단 응집도
# -----------------------------
def _cohesion_score(prev_text: str, cur_text: str, next_text: str) -> float:
    """
    인접 세그먼트와의 간단 응집도(토큰 overlap Jaccard).
    """

    def toks(s: str):
        return set(re.findall(r"[A-Za-z가-힣0-9]+", s.lower()))

    a = toks(prev_text) if prev_text else set()
    b = toks(cur_text) if cur_text else set()
    c = toks(next_text) if next_text else set()
    if not b:
        return 0.0
    j1 = len(b & a) / max(1, len(b | a))
    j2 = len(b & c) / max(1, len(b | c))
    return float((j1 + j2) / 2.0)


# -----------------------------
# 메인: 오류 스팬 탐지
# -----------------------------
def detect_error_spans(segments: List[Segment]) -> List[Span]:
    out: List[Span] = []
    if not segments:
        return out

    # 설정값(없으면 기본 사용)
    p_top = getattr(_cfg, "char_ngram_outlier_top_p", 0.10)
    contextfit_low = getattr(_cfg, "contextfit_low", 0.45)
    target_chars = getattr(_cfg, "llm_span_target_chars", 80)
    min_chars = getattr(_cfg, "llm_span_min_chars", 50)
    max_chars = getattr(_cfg, "llm_span_max_chars", 260)

    # 미리 각 세그먼트의 n-gram 놀람도 산출
    seg_ngram_token_scores: List[Dict[Tuple[int, int], float]] = []
    for seg in segments:
        _, tok_score = _ngram_surprisal_scores(seg.text or "")
        seg_ngram_token_scores.append(tok_score)

    # 전체 놀람도 분포 기반 상위 p 백분위수 산출 (토큰 단위)
    all_scores = [v for d in seg_ngram_token_scores for v in d.values()]
    if all_scores:
        th_ngram = float(np.quantile(np.array(all_scores, dtype=float), 1.0 - p_top))
    else:
        th_ngram = float("inf")  # 비활성화

    # 문장 캐시/중복 억제 키
    sent_cache: Dict[int, List[Tuple[int, int]]] = {}  # sid -> sentence spans
    seen_llm_spans: Set[Tuple[int, int, int]] = set()

    def _append_span_with_aggregation(seg: Segment, s: int, e: int, category: str):
        cat = (category or "").upper()
        text = seg.text or ""
        if cat in _AGGREGATE_CATS:
            sents = sent_cache.get(seg.sid)
            if sents is None:
                sents = _split_sentences(text)
                sent_cache[seg.sid] = sents
            k0, k1 = _find_sentence_index(sents, s, e)
            L, R = _aggregate_to_target_window(
                sents, k0, k1, target_chars, min_chars, max_chars
            )

            # ✅ 카테고리 무시하고 IoU로 dedupe
            is_dup = any(
                (sid == seg.sid and _iou(L, R, aL, aR) >= _LLM_DEDUPE_IOU)
                for (sid, aL, aR) in seen_llm_spans
            )
            if not is_dup:
                seen_llm_spans.add((seg.sid, L, R))
                out.append(Span(seg.sid, (L, R), text[L:R], cat))
        else:
            out.append(Span(seg.sid, (s, e), text[s:e], cat))

    # 순회
    for idx, seg in enumerate(segments):
        text = seg.text or ""
        tokens = list(_iter_tokens_with_offsets(text))

        # --- 1) MIXED ---
        for s, e, tok in tokens:
            if is_mixed_token(tok):
                _append_span_with_aggregation(seg, s, e, "MIXED")

        # --- 2) DIGIT_SUFFIX ---
        for s, e, tok in tokens:
            if looks_like_digit_suffix(tok):
                _append_span_with_aggregation(seg, s, e, "DIGIT_SUFFIX")

        # --- 3) DIGIT_BOUNDARY ---
        for i in range(len(tokens) - 1):
            s1, e1, a = tokens[i]
            s2, e2, b = tokens[i + 1]
            if _RE_NUM.fullmatch(a) and any(b.endswith(j) for j in _JOSA):
                # 예: "42" "에는"
                _append_span_with_aggregation(seg, s1, e2, "DIGIT_BOUNDARY")

        # --- 4) PHONO ---
        for s, e, tok in tokens:
            if tok in _PHONO_SET:
                _append_span_with_aggregation(seg, s, e, "PHONO")
            elif tok == "라면" and ("산란" in text or "분광" in text):
                _append_span_with_aggregation(seg, s, e, "PHONO")

        # --- 5) MORPH_ODD ---
        for i in range(len(tokens) - 1):
            s1, e1, a = tokens[i]
            s2, e2, b = tokens[i + 1]
            # (영문|숫자)+조사
            if (re.fullmatch(r"[A-Za-z]+", a) or _RE_NUM.fullmatch(a)) and any(
                b == j or b.endswith(j) for j in _JOSA
            ):
                _append_span_with_aggregation(seg, s1, e2, "MORPH_ODD")

        # --- 6) NGRAM_OUTLIER ---
        tok_score_map = seg_ngram_token_scores[idx]
        for (s, e), sc in tok_score_map.items():
            if sc >= th_ngram and (e - s) >= 2:  # 너무 짧은 토큰 제외
                _append_span_with_aggregation(seg, s, e, "NGRAM_OUTLIER")

        # --- 7) CONTEXT_ANOM ---
        # ContextFit = 0.30*Topic + 0.20*Cohesion(±1) + 0.20*(1-CharSurprisalNorm)
        #            + 0.10*MorphPMI + 0.15*DomainPMI + 0.05*Consistency
        # 여기서는 MorphPMI/DomainPMI/Consistency를 상수 0.5로 근사
        prev_text = segments[idx - 1].text if idx - 1 >= 0 else ""
        next_text = segments[idx + 1].text if idx + 1 < len(segments) else ""
        cohesion = _cohesion_score(prev_text, text, next_text)

        # 문자 놀람도 정규화(세그먼트 내 min-max)
        vals = list(tok_score_map.values()) or [0.0]
        vmin, vmax = (min(vals), max(vals))

        def norm_surprisal(v: float) -> float:
            if vmax - vmin < 1e-9:
                return 0.0
            return (v - vmin) / (vmax - vmin)

        topic_score = (
            float(segments[idx].meta.get("topic", 0.0))
            if getattr(segments[idx], "meta", None)
            else 0.0
        )
        for (s, e), sc in tok_score_map.items():
            # 토큰별 ContextFit
            cs = norm_surprisal(sc)  # 0..1
            fit = (
                (0.30 * topic_score)
                + (0.20 * cohesion)
                + (0.20 * (1.0 - cs))
                + (0.10 * 0.5)
                + (0.15 * 0.5)
                + (0.05 * 0.5)
            )
            if fit <= contextfit_low:
                _append_span_with_aggregation(seg, s, e, "CONTEXT_ANOM")

    return out
