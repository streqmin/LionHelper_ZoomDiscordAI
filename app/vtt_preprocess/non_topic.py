# -*- coding: utf-8 -*-
"""
non_topic: 시작/끝/쉬는 시간 등 비주제 구간 감지 (RuleGate 방식)

입력: List[Segment]
출력: List[NonTopicBlock]
- 각 Segment.meta["topic"]가 없으면 topic_index가 주어진 경우 계산하여 채움
- 점수식:
    StartNonTopic = 0.5*(1-Topic) + 0.3*PosStart + 0.2*Cues
    EndNonTopic   = 0.5*(1-Topic) + 0.3*PosEnd   + 0.2*Cues
- 스무딩: 이동평균(window = Config.smooth_window)
- 임계: 분위수 q_non_topic (기본 0.80), topic_threshold (기본 0.60)
- 연속 k=2, 최소 지속시간 30–90초

라벨링:
- 시작쪽 블록 → GREETING (인사/소개 키워드가 있으면 가산)
- 끝쪽 블록   → CLOSING (마무리/정리/감사/수고 키워드가 있으면 가산)
- 중간 블록   → BREAK(쉬는/점심/저녁) 또는 HOUSEKEEPING(공지/안내/설문) 아니면 NON_TOPIC
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .schema import Segment, NonTopicBlock
from .config import Config
from .topic import TopicIndex  # optional 사용

_cfg = Config()

# non_topic.py 상단의 cue 사전과 스코어러 부분을 다음으로 교체

import re

# ── 1) 큐 사전: 데이터 기반 확장 ─────────────────────────────────────────────
# Greeting: 시작부에서 실제로 많이 보인 것
_CUE_GREETING = (
    # 기존
    "안녕",
    "반갑",
    "소개",
    "오리엔테이션",
    "ot",
    "오티",
    # 데이터 근거 추가
    "안녕하세요",
    "반갑습니다",
    "처음에",
    "처음에는",
    "인사말",
    "들리시나요",
    "보이시나요",
    "여러분들",
    "여러분",
    "시작하겠습니다",
    "오늘은",
)

# Closing: 끝부분에서 자주 관찰
_CUE_CLOSING = (
    # 기존
    "마무리",
    "정리",
    "여기까지",
    "감사",
    "수고",
    # 데이터 근거 추가
    "감사합니다",
    "수고하셨습니다",
    "다음 시간",
    "다음에",
    "끝내겠습니다",
    "마무리 하도록",
    "마무리 하겠습니다",
    "마치도록 하겠습니다",
    "정리 하도록",
    "정리 하고",
)

# Break: 쉬는 시간/중간 휴식 멘트
_CUE_BREAK = (
    # 기존
    "쉬는",
    "휴식",
    "break",
    "점심",
    "저녁",
    "휴게",
    # 데이터 근거 추가
    "잠시만요",
    "잠깐",
    "잠깐만요",
    "쉬었다가",
    "쉬는 시간",
)

# Housekeeping: 과제/자료/링크/공유/녹화 등
_CUE_HK = (
    # 기존
    "공지",
    "안내",
    "설문",
    "출석",
    "과제",
    "평가",
    "공지사항",
    # 데이터 근거 추가
    "링크",
    "자료",
    "파일",
    "공유",
    "녹화",
    "업로드",
    "다운로드",
    "채팅",
    "화면 공유",
    "마이크",
    "카메라",
    "레포",
    "깃허브",
    "github",
)

# ── 2) 변형/어미까지 잡는 가벼운 정규식(부분 매치) ──────────────────────────
_CUE_RE_GREETING = [
    re.compile(r"(안녕하세|반갑습니|오리엔테이션|인사말)"),
    re.compile(r"(들리시나요|보이시나요)"),
    re.compile(r"시작하겠(습니다|어요)"),
    re.compile(r"처음에(는)?\b"),
]

_CUE_RE_CLOSING = [
    re.compile(r"(감사합니|수고하셨습니)"),
    re.compile(r"(마무리)\s*(하겠|하도록|할게요)"),
    re.compile(r"(마치)(겠|도록 하겠)"),
    re.compile(r"(정리)\s*(하겠|하도록|하고)"),
    re.compile(r"다음\s*(시간|주|주에)"),
    re.compile(r"(끝내겠|마무리하겠)"),
]

_CUE_RE_BREAK = [
    re.compile(r"(잠시|잠깐)\s*(쉬|휴식)"),
    re.compile(r"(쉬는\s*시간)"),
    re.compile(r"(점심)\s*(시간|먹기|전까지)"),
    re.compile(r"(5|10)\s*분\s*(쉬|휴식)"),
]

_CUE_RE_HK = [
    re.compile(r"(링크|파일|자료)\s*(공유|드리겠|올려|업로드|다운로드)"),
    re.compile(r"녹화\s*(파일|본|종료)"),
    re.compile(r"(깃허브|github|레포)"),
    re.compile(r"(채팅|화면\s*공유|마이크|카메라)"),
]


def _exp_pos(t: float, T: float, start=True) -> float:
    if T <= 0:
        return 0.0
    beta = _cfg.pos_beta_ratio * T
    if beta <= 0:
        return 0.0
    return math.exp(-t / beta) if start else math.exp(-(T - t) / beta)


def _moving_avg(x: List[float], w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return np.array(x, dtype=float)
    arr = np.array(x, dtype=float)
    pad = w // 2
    ext = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(ext, kernel, mode="valid")


def _cue_score(text: str) -> float:
    """사전에 더해, 변화형을 잡는 정규식까지 반영한 큐 점수."""
    if not text:
        return 0.0
    s = text.lower()

    hits = 0
    # 1) 사전 기반 부분일치
    for term in _CUE_GREETING + _CUE_CLOSING + _CUE_BREAK + _CUE_HK:
        t = term.lower()
        if t and (t in s):
            hits += 1

    # 2) 정규식 기반
    for rx in _CUE_RE_GREETING + _CUE_RE_CLOSING + _CUE_RE_BREAK + _CUE_RE_HK:
        if rx.search(s):
            hits += 1

    # 길이/중복에 따른 과도 가중 방지
    return min(1.0, hits / 6.0)


def _quantile(vals: List[float], q: float) -> float:
    if not vals:
        return 1.0
    return float(np.quantile(np.array(vals, dtype=float), q))


def _block_label(
    seg_slice: List[Segment], is_start_zone: bool, is_end_zone: bool
) -> str:
    joined = "\n".join(s.text for s in seg_slice).lower()
    if any(k in joined for k in _CUE_BREAK):
        return "BREAK"
    if any(k in joined for k in _CUE_HK):
        return "HOUSEKEEPING"
    if is_start_zone:
        return "GREETING"
    if is_end_zone:
        return "CLOSING"
    return "NON_TOPIC"


def detect_non_topic_blocks(
    segments: List[Segment], topic_index: Optional[TopicIndex] = None, k_min: int = 1
) -> List[NonTopicBlock]:
    if not segments:
        return []

    # 총 길이 T와 각 세그먼트 중심 시각
    durations = [max(0.0, s.end_sec - s.start_sec) for s in segments]
    T = float(sum(durations))
    midpoints = []
    accum = 0.0
    for d, s in zip(durations, segments):
        mid = accum + d * 0.5
        midpoints.append(mid)
        accum += d

    # TopicScore / Cue / PosStart / PosEnd
    start_scores, end_scores, topics, cues = [], [], [], []
    for m, seg in zip(midpoints, segments):
        topic = seg.meta.get("topic", None)
        if topic is None:
            if topic_index is not None:
                topic = float(topic_index.topic_score(seg.text))
                seg.meta["topic"] = topic
            else:
                topic = 0.0  # 정보 없을 때 보수적
                seg.meta["topic"] = topic
        topics.append(topic)

        cue = _cue_score(seg.text)
        cues.append(cue)

        ps = _exp_pos(m, T, start=True)
        pe = _exp_pos(m, T, start=False)

        start_scores.append(0.5 * (1.0 - topic) + 0.3 * ps + 0.2 * cue)
        end_scores.append(0.5 * (1.0 - topic) + 0.3 * pe + 0.2 * cue)

    # 스무딩
    start_sm = _moving_avg(start_scores, _cfg.smooth_window).tolist()
    end_sm = _moving_avg(end_scores, _cfg.smooth_window).tolist()

    # 임계
    th_start = _quantile(start_sm, _cfg.q_non_topic)
    th_end = _quantile(end_sm, _cfg.q_non_topic)
    th_topic = _cfg.topic_threshold

    # 후보 마스크
    start_mask = [(s >= th_start) and (t <= th_topic) for s, t in zip(start_sm, topics)]
    end_mask = [(e >= th_end) and (t <= th_topic) for e, t in zip(end_sm, topics)]

    # 연속 k 구간 찾기
    def _runs(mask: List[bool]) -> List[Tuple[int, int]]:
        runs = []
        i = 0
        n = len(mask)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= k_min:
                runs.append((i, j - 1))  # inclusive
            i = j
        return runs

    start_runs = _runs(start_mask)
    end_runs = _runs(end_mask)

    # 시간 조건 충족하는 구간으로 필터 (30–90초)
    def _dur(i: int, j: int) -> float:
        return float(sum(durations[i : j + 1]))

    blocks: List[NonTopicBlock] = []
    for i, j in start_runs + end_runs:
        dur = _dur(i, j)
        if dur < _cfg.min_block_secs_lo:
            continue
        # 상한은 너무 빡빡하게 보지 않고, 있으면 참고만 (너무 길면 분할을 고려)
        # if dur > _cfg.min_block_secs_hi: continue

        # 라벨
        is_start_zone = i <= max(1, len(segments) // 5)  # 처음 20% 영역 근처면 시작
        is_end_zone = j >= len(segments) - max(
            2, len(segments) // 5
        )  # 끝 20% 영역 근처면 끝
        label = _block_label(segments[i : j + 1], is_start_zone, is_end_zone)

        # 자신감 = 해당 스코어 평균
        conf_vals = start_sm[i : j + 1] if (i, j) in start_runs else end_sm[i : j + 1]
        confidence = float(np.mean(conf_vals)) if conf_vals else 0.5

        # 핵심 도메인 키워드가 많은 경우 보존(소프트 제외 정책과 호환) → evidence에 남김
        evidence = []
        if (
            segments[i].meta.get("topic", 0.0) >= 0.7
            or segments[j].meta.get("topic", 0.0) >= 0.7
        ):
            evidence.append("topic_high_edge")

        blocks.append(
            NonTopicBlock(
                start_sid=segments[i].sid,
                end_sid=segments[j].sid,
                label=label,
                confidence=confidence,
                evidence=evidence,
            )
        )

    # 중복/겹침 정리(단순 병합)
    blocks_sorted = sorted(blocks, key=lambda b: (b.start_sid, b.end_sid))
    merged: List[NonTopicBlock] = []
    for b in blocks_sorted:
        if not merged:
            merged.append(b)
            continue
        prev = merged[-1]
        if b.start_sid <= prev.end_sid:
            # 병합
            new = NonTopicBlock(
                start_sid=prev.start_sid,
                end_sid=max(prev.end_sid, b.end_sid),
                label=prev.label if prev.label == b.label else "NON_TOPIC",
                confidence=float((prev.confidence + b.confidence) / 2.0),
                evidence=list(set(prev.evidence + b.evidence)),
            )
            merged[-1] = new
        else:
            merged.append(b)
    return merged
