# -*- coding: utf-8 -*-
"""
correct: Phase 5 교정 오케스트레이션
- CorrectionRecord 스키마 준수:
  timecode, sid, offset(str), category, method("RULE"/"LLM_SELECT"/"LGV"),
  orig, final, score_or_fit, delta_topic, evidence
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple, Optional
import re

from .schema import Segment, Span, CorrectionRecord
from .topic import TopicIndex
from .llm_client import LLMClient
from .verify import approve_change
from .utils import timecode
from .config import Config

_cfg = Config()

# ---------------------------------
# 규칙 기반 교정 후보
# ---------------------------------
_PHONO_MAP = {
    "가치": "같이",
    "궁물": "국물",
    "학꽈": "학과",
    "수학적귀납볍": "수학적 귀납법",
}
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
    "과",
    "와",
)
_RE_NUM_ONLY = re.compile(r"^\d+$")
_RE_DIGIT_SUFFIX = re.compile(r"^(\d+)([가-힣]+)$")

# LLM 허용 카테고리(외부 LLM은 이들만 호출)
_LLM_ALLOWED_CATEGORIES = frozenset({"NGRAM_OUTLIER", "CONTEXT_ANOM"})

# 문장 경계(간단): 종결부호/개행
_SENT_ENDERS = set(".?!…")
_NEWLINE = "\n"


def _iou(L1: int, R1: int, L2: int, R2: int) -> float:
    inter = max(0, min(R1, R2) - max(L1, L2))
    union = max(R1, R2) - min(L1, L2)
    return 0.0 if union <= 0 else inter / union


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """종결부호/개행으로 문장 경계를 잡아 (start,end) 리스트 반환(end는 포함+1)."""
    n = len(text)
    spans: list[tuple[int, int]] = []
    i = 0
    start = 0
    while i < n:
        ch = text[i]
        if ch in _SENT_ENDERS:
            end = i + 1
            while start < n and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                spans.append((start, end))
            start = i + 1
        i += 1
    if start < n:
        end = n
        while start < n and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans or [(0, n)]


def _find_span_sentence_band(
    sents: list[tuple[int, int]], s: int, e: int
) -> tuple[int, int]:
    """스팬 s..e와 겹치는 문장 인덱스 최소~최대(kL..kR) 반환."""
    kL = kR = None
    for i, (L, R) in enumerate(sents):
        if e <= L:
            break
        if not (e <= L or s >= R):  # overlap
            if kL is None:
                kL = i
            kR = i
    if kL is None:  # 못 찾으면 s 위치 기준 근사
        for i, (L, R) in enumerate(sents):
            if s < R:
                return (i, i)
        return (len(sents) - 1, len(sents) - 1)
    return (kL, kR)


def _make_sentence_context_for_span(
    text: str, s: int, e: int, left_sents: int, right_sents: int
) -> tuple[str, str, str]:
    """
    detect.py가 준 스팬 s..e(여러 문장 포함 가능)를 그대로 frag로 쓰고,
    좌/우는 문장 단위로 각각 left_sents/right_sents개 붙여 반환.
    """
    sents = _split_sentences(text)
    kL, kR = _find_span_sentence_band(sents, s, e)
    frag = (text[s:e]).strip()

    if left_sents > 0 and kL > 0:
        ls = sents[max(0, kL - left_sents)][0]
        left = text[ls : sents[kL][0]].strip()
    else:
        left = ""

    if right_sents > 0 and (kR + 1) < len(sents):
        reidx = min(len(sents) - 1, kR + right_sents)
        re = sents[reidx][1]
        right = text[sents[kR][1] : re].strip()
    else:
        right = ""

    return left, frag, right


def _rule_for_span(
    full_text: str,
    s: int,
    e: int,
    category: str,
) -> Optional[str]:
    """간단 규칙 기반 치환안을 1개 반환(없으면 None)."""
    frag = full_text[s:e]
    cat = (category or "").upper()

    if cat == "MIXED":
        t = re.sub(r"([가-힣])\s*-\s*([가-힣])", r"\1\2", frag)
        t = re.sub(r"[ ]{2,}", " ", t)
        t = re.sub(r"\.{3,}", "…", t)
        return t if t != frag else None

    if cat == "DIGIT_SUFFIX":
        m = _RE_DIGIT_SUFFIX.match(frag)
        if m:
            # 단위/맥락 보수적 처리: 숫자 제거, 조사/어휘만 남김
            _, suf = m.group(1), m.group(2)
            return suf if suf else None
        return None

    if cat == "DIGIT_BOUNDARY":
        # '42 에는' 같은 결합 후보 → 숫자 제거
        t = re.sub(r"^\s*\d+\s*", "", frag)
        return t if t != frag else None

    if cat == "PHONO":
        return _PHONO_MAP.get(frag, None)

    if cat == "MORPH_ODD":
        # (영문|숫자)+조사 → 조사 앞 공백 삽입(보수적)
        m = re.match(r"^([A-Za-z0-9]+)([가-힣]+)$", frag)
        if m and any(m.group(2).endswith(j) for j in _JOSA):
            return f"{m.group(1)} {m.group(2)}"
        return None

    # NGRAM_OUTLIER / CONTEXT_ANOM 은 규칙으로 처리하지 않음
    return None


def _llm_for_span(
    llm: LLMClient, left: str, frag: str, right: str, category: str
) -> Optional[str]:
    """허용 카테고리 & 사용가능 상태에서만 외부 LLM 호출."""
    if (category or "").upper() not in _LLM_ALLOWED_CATEGORIES:
        return None
    if not (llm and isinstance(llm, LLMClient) and llm.available()):
        return None
    return llm.suggest_span_edit(left, frag, right, category)


def correct_segments(
    segments: List[Segment],
    spans: List[Span],
    topic: TopicIndex,
    llm: LLMClient,
    curriculum_terms: List[str],
) -> Tuple[List[Segment], List[CorrectionRecord]]:
    """
    스팬을 세그먼트별로 정렬해 좌→우로 적용.
    - 각 스팬마다: 규칙 후보 → 검증 → (필요 시) LLM 후보 → 검증
    - 승인된 경우만 텍스트 반영, offset 보정
    - 로그는 CorrectionRecord 스키마에 맞춰 기록
    """
    # 세그먼트별 스팬 묶기
    print(f"segments: {len(segments)}")
    print(f"spans: {len(spans)}")
    print(f"topic: {topic}")
    print(f"llm: {llm}")
    print(f"curriculum_terms: {len(curriculum_terms)}")
    print(f"--------------------------------")
    by_sid: Dict[int, List[Span]] = {}
    for sp in spans:
        by_sid.setdefault(sp.sid, []).append(sp)
    for sid in by_sid:
        by_sid[sid].sort(key=lambda x: (x.offset[0], x.offset[1]))

    new_segments: List[Segment] = []
    records: List[CorrectionRecord] = []

    for seg in segments:
        # correct_segments(...) 내부, 세그먼트별 루프 시작 직후에 준비
        llm_iou = getattr(_cfg, "llm_dedupe_iou", 0.80)
        seen_llm_windows: list[tuple[int, int]] = []  # (s2, e2)

        sps = by_sid.get(seg.sid, [])
        if not sps:
            new_segments.append(seg)
            continue

        cur_text = seg.text or ""
        shift = 0  # 누적 길이 변화

        for sp in sps:
            s, e = sp.offset
            s2, e2 = s + shift, e + shift
            tc = timecode(seg.start_sec, seg.end_sec)
            offset_str = f"{s}:{e}"

            # 오프셋 검증
            if s2 < 0 or e2 > len(cur_text) or s2 >= e2:
                # best-effort 원문 추출(범위가 틀어져도 안전)
                be = cur_text[
                    max(0, min(len(cur_text), s2)) : max(0, min(len(cur_text), e2))
                ]
                records.append(
                    CorrectionRecord(
                        timecode=tc,
                        sid=seg.sid,
                        offset=offset_str,
                        category=sp.category,
                        method="RULE",
                        orig=be,
                        final=be,
                        score_or_fit=0.0,
                        delta_topic=0.0,
                        evidence="offset_out_of_range",
                    )
                )
                continue

            left = cur_text[max(0, s2 - 60) : s2]
            frag = cur_text[s2:e2]
            right = cur_text[e2 : e2 + 60]

            applied = False

            # 1) 규칙 후보
            cand = _rule_for_span(cur_text, s2, e2, sp.category)
            if cand is not None and cand != frag:
                after_text = cur_text[:s2] + cand + cur_text[e2:]
                vr = approve_change(topic, cur_text, after_text, sp.category)
                if vr.ok:
                    # 규칙 적용 OK
                    records.append(
                        CorrectionRecord(
                            timecode=tc,
                            sid=seg.sid,
                            offset=offset_str,
                            category=sp.category,
                            method="RULE",
                            orig=frag,
                            final=cand,
                            score_or_fit=1.0,
                            delta_topic=vr.delta_topic,
                            evidence="ok",
                        )
                    )
                    cur_text = after_text
                    shift += len(cand) - (e2 - s2)
                    applied = True
                else:
                    # 규칙 제안 거절(미적용)
                    records.append(
                        CorrectionRecord(
                            timecode=tc,
                            sid=seg.sid,
                            offset=offset_str,
                            category=sp.category,
                            method="RULE",
                            orig=frag,
                            final=frag,
                            score_or_fit=0.0,
                            delta_topic=vr.delta_topic,
                            evidence=f"rule_rejected:{vr.reason}",
                        )
                    )

            if applied:
                continue

            # 2) LLM 후보 — 허용 카테고리 + LLM 가용 조건 체크
            cat_up = (sp.category or "").upper()
            if cat_up not in _LLM_ALLOWED_CATEGORIES:
                records.append(
                    CorrectionRecord(
                        timecode=tc,
                        sid=seg.sid,
                        offset=offset_str,
                        category=sp.category,
                        method="LLM_SELECT",
                        orig=frag,
                        final=frag,
                        score_or_fit=0.0,
                        delta_topic=0.0,
                        evidence="llm_skipped:category_not_allowed",
                    )
                )
                continue

            if not (llm and isinstance(llm, LLMClient) and llm.available()):
                records.append(
                    CorrectionRecord(
                        timecode=tc,
                        sid=seg.sid,
                        offset=offset_str,
                        category=sp.category,
                        method="LLM_SELECT",
                        orig=frag,
                        final=frag,
                        score_or_fit=0.0,
                        delta_topic=0.0,
                        evidence="llm_skipped:unavailable",
                    )
                )
                continue

            left_sents = getattr(_cfg, "llm_left_sentences", 1)
            right_sents = getattr(_cfg, "llm_right_sentences", 1)
            left, frag, right = _make_sentence_context_for_span(
                cur_text, s2, e2, left_sents, right_sents
            )

            # ✅ IoU NMS: 이미 본 창과 많이 겹치면 스킵
            if any(_iou(s2, e2, a, b) >= llm_iou for (a, b) in seen_llm_windows):
                records.append(
                    CorrectionRecord(
                        timecode=tc,
                        sid=seg.sid,
                        offset=f"{s}-{e}",
                        category=sp.category,
                        method="LLM_SELECT",
                        orig=frag,
                        final=frag,
                        score_or_fit=0.0,
                        delta_topic=0.0,
                        evidence="llm_skipped:overlap_with_prev",
                    )
                )
                continue

            # 실제 LLM 호출
            cand2 = _llm_for_span(llm, left, frag, right, sp.category)
            print(
                f"LLM 교정 수행 - sid: {seg.sid} [{sp.category}]\nleft: {left}\nfrag: {frag}\nright: {right}\ncand2: {cand2}\n==========="
            )
            if not cand2:
                records.append(
                    CorrectionRecord(
                        timecode=tc,
                        sid=seg.sid,
                        offset=offset_str,
                        category=sp.category,
                        method="LLM_SELECT",
                        orig=frag,
                        final=frag,
                        score_or_fit=0.0,
                        delta_topic=0.0,
                        evidence="llm_no_candidate",
                    )
                )
            else:
                after_text = cur_text[:s2] + cand2 + cur_text[e2:]
                vr2 = approve_change(topic, cur_text, after_text, sp.category)
                if vr2.ok:
                    records.append(
                        CorrectionRecord(
                            timecode=tc,
                            sid=seg.sid,
                            offset=offset_str,
                            category=sp.category,
                            method="LLM_SELECT",
                            orig=frag,
                            final=cand2,
                            score_or_fit=1.0,
                            delta_topic=vr2.delta_topic,
                            evidence="ok",
                        )
                    )
                    cur_text = after_text
                    shift += len(cand2) - (e2 - s2)
                    applied = True
                else:
                    records.append(
                        CorrectionRecord(
                            timecode=tc,
                            sid=seg.sid,
                            offset=offset_str,
                            category=sp.category,
                            method="LLM_SELECT",
                            orig=frag,
                            final=frag,
                            score_or_fit=0.0,
                            delta_topic=vr2.delta_topic,
                            evidence=f"llm_rejected:{vr2.reason}",
                        )
                    )
            # 둘 다 실패 시에도 기록은 남겼으므로 추가 처리 없음
            seen_llm_windows.append((s2, e2))

        new_segments.append(replace(seg, text=cur_text))

    return new_segments, records
