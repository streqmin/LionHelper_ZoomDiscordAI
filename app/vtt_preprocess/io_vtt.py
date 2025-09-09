# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

import webvtt

from .schema import Segment


def _ts_to_seconds(ts: str) -> float:
    """
    'HH:MM:SS.mmm' 또는 'MM:SS.mmm' → 초(float)
    """
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        # webvtt 포맷이 아니면 0초 처리
        return 0.0


def read_vtt(path: str) -> List[Segment]:
    """
    WEBVTT 파일을 읽어 1 cue -> 1 Segment로 변환합니다.
    - sid: 0부터 순차 부여
    - start_sec/end_sec: float
    - text: cue 텍스트(여러 줄은 '\n' 포함)

    필요 패키지: webvtt-py
    """

    vtt = webvtt.read(path)  # type: ignore
    segments: List[Segment] = []
    sid = 0
    for cue in vtt:
        # webvtt.Caption: start, end, text
        start = _ts_to_seconds(cue.start)
        end = _ts_to_seconds(cue.end)
        # cue.text는 내부적으로 줄바꿈을 합친 문자열
        text = (cue.text or "").replace("\r\n", "\n").strip("\n")
        segments.append(Segment(sid=sid, start_sec=start, end_sec=end, text=text))
        sid += 1
    return segments
