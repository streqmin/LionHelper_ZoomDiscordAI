# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Tuple, Set

Tag = Literal["NON_TOPIC_START", "NON_TOPIC_END", "NON_TOPIC_BREAK"]


@dataclass
class Segment:
    sid: int
    start_sec: float
    end_sec: float
    text: str
    tags: Set[Tag] = field(default_factory=set)
    meta: Dict[str, float] = field(
        default_factory=dict
    )  # topic, greet, close, scores ...


@dataclass
class Span:
    sid: int
    offset: Tuple[int, int]  # (start_char_idx, end_char_idx) inclusive-exclusive
    orig: str
    category: Literal[
        "MIXED",
        "DIGIT_SUFFIX",
        "DIGIT_BOUNDARY",
        "PHONO",
        "MORPH_ODD",
        "NGRAM_OUTLIER",
        "CONTEXT_ANOM",
    ]


@dataclass
class Candidate:
    text: str
    scores: Dict[
        str, float
    ]  # {"PHON":..., "ΔTopic":..., "FREQ":..., "POS":..., "PUN":...}


@dataclass
class RankedCandidates:
    span: Span
    ranked: List[Candidate]  # 점수 내림차순
    tau: float  # 85% 분위수 임계


@dataclass
class LLMSelectItem:
    text: str
    fit: float
    evidence: List[str]


@dataclass
class LLMSelectResult:
    span: Span
    suggested: Optional[LLMSelectItem]  # None이면 유지


@dataclass
class LLMPatch:
    span: Span
    action: Literal["REPLACE", "KEEP"]
    replace_with: str
    confidence: float
    reason: Literal["OFF_TOPIC", "CONTRADICTION", "REGISTER", "AMBIGUOUS"]
    changed_chars: int


@dataclass
class VerificationResult:
    accepted: bool
    reasons: List[str]
    deltas: Dict[str, float]  # {"ΔTopic":..., "ΔCharNgram":..., "DomainPMI":...}


@dataclass
class CorrectionRecord:
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


@dataclass
class NonTopicBlock:
    start_sid: int
    end_sid: int
    label: Literal["GREETING", "HOUSEKEEPING", "BREAK", "CLOSING", "NON_TOPIC"]
    confidence: float
    evidence: List[str]


@dataclass
class Outputs:
    segments: List[Segment]
    non_topic_blocks: List[NonTopicBlock]
    topic_segments: List[Segment]
    corrections: List[CorrectionRecord]
    metrics: Dict[str, float]  # CER/WER 샘플, coverage Δ, coherence Δ, llm_calls 등
