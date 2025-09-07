# -*- coding: utf-8 -*-
from __future__ import annotations

import unicodedata as ud
import re
from typing import List

from .schema import Segment
from .utils import is_mixed_token


# ---- 정규식 패턴 (가벼운 수준만) ----
# 줄 시작 화자명: "홍길동:", "T:", "교수님:" 등 (한글/영문/숫자 최대 15자 + 콜론)
_RE_SPEAKER_COLON = re.compile(r"^(?:[A-Za-z0-9가-힣]{1,15})\s*:\s+")
# 대괄호/괄호 태그: [웃음], (박수), [잡음], (노이즈), [음..], (침묵)
_RE_BRACKET_NOISE = re.compile(r"^\s*[\[\(].{0,12}?[\]\)]\s*")
# Zoom/플랫폼 안내 키워드가 앞머리에 올 때(짧은 안내 멘트 제거용)
_RE_PLATFORM_PREFIX = re.compile(
    r"^(?:줌|Zoom|마이크|카메라|화면\s*공유)\s*[가-힣A-Za-z]*\s*[:\-]\s*", re.IGNORECASE
)


def _nfkc_basic(s: str) -> str:
    s = ud.normalize("NFKC", s)
    # 따옴표/줄임표/대시 통일
    s = s.replace("…", "...").replace("—", "-").replace("―", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # stray control 제거
    s = s.replace("\u200b", "")  # zero width space
    return s


def _strip_leading_noise(line: str) -> str:
    """
    각 줄(line)의 선두에서 화자명/노이즈 태그/플랫폼 안내 등을 소거.
    과도한 제거를 방지하기 위해 '줄 선두'만 대상으로 함.
    """
    changed = True
    s = line
    # 여러 패턴이 겹칠 수 있으므로 최대 2회 반복
    for _ in range(2):
        before = s
        s = _RE_SPEAKER_COLON.sub("", s)
        s = _RE_BRACKET_NOISE.sub("", s)
        s = _RE_PLATFORM_PREFIX.sub("", s)
        if s == before:
            changed = False
            break
    return s.strip()


def normalize_segments(segments: List[Segment]) -> List[Segment]:
    """
    - 텍스트를 NFKC 정규화하고,
    - 줄 단위로 선행 화자/노이즈 태그를 소거,
    - 혼용 토큰 개수를 meta에 기록합니다.
    """
    out: List[Segment] = []
    for seg in segments:
        text = _nfkc_basic(seg.text or "")
        # 줄 단위 정리
        lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
        cleaned_lines = [_strip_leading_noise(ln) for ln in lines]
        cleaned = "\n".join(cleaned_lines).strip("\n")

        seg.text = cleaned
        # 혼용 토큰 카운트(공백 토큰 기준)
        seg.meta["MIXED_TOKENS"] = sum(
            1 for tok in cleaned.split() if is_mixed_token(tok)
        )
        out.append(seg)
    return out
