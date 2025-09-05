# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Optional
from config import Config

_cfg = Config()

# 미리 컴파일된 정규식
_RE_HANGUL = re.compile(r"[가-힣]")
_RE_DIGIT = re.compile(r"\d")
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_SYMBOL = re.compile(r"[-_/\.]")  # 혼용 판정에 쓰는 기본 기호
_RE_DIGIT_SUFFIX = re.compile(r"^\d{1,3}" + _cfg.mixed_digit_hangul_regex_suffix)


def is_mixed_token(tok: str) -> bool:
    """
    한 토큰 내에 '한글'과 '숫자/라틴/기호'가 혼용되었는지 판단.
    공백 기준 토큰을 가정.
    """
    if not tok:
        return False
    has_hangul = bool(_RE_HANGUL.search(tok))
    if not has_hangul:
        return False
    return bool(
        _RE_DIGIT.search(tok) or _RE_LATIN.search(tok) or _RE_SYMBOL.search(tok)
    )


def looks_like_digit_suffix(tok: str) -> bool:
    """
    '42에는' 같은 숫자+조사 결합 패턴 탐지.
    """
    if not tok:
        return False
    return bool(_RE_DIGIT_SUFFIX.match(tok))


def is_unit_context(left_token: Optional[str], right_token: Optional[str]) -> bool:
    """
    숫자 바로 뒤에 단위/기호가 붙는 수량 문맥 여부 판단.
    - left_token: 숫자 토큰
    - right_token: 숫자 바로 오른쪽 토큰(단위/기호 예상)
    """
    right = (right_token or "").strip()
    if not right:
        return False
    if right in _cfg.unit_tokens:
        return True
    # 추가적인 간단 기호 예외
    if re.fullmatch(r"[%℃]|MB|GB|TB|km|cm|mm", right):
        return True
    return False


def timecode(start_sec: float, end_sec: float) -> str:
    """
    VTT/WEBVTT 포맷의 시간코드 문자열 생성.
    예) 00:00:01.000 --> 00:00:02.500
    """

    def fmt(x: float) -> str:
        if x < 0:
            x = 0.0
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = x - h * 3600 - m * 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    return f"{fmt(start_sec)} --> {fmt(end_sec)}"
