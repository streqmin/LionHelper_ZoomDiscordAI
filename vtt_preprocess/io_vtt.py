# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

import webvtt

from schema import Segment


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


def _seconds_to_ts(sec: float) -> str:
    """
    초(float) → 'HH:MM:SS.mmm'
    """
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


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


def write_vtt(
    path: str, segments: List[Segment], keep_original_comments: bool = True
) -> None:
    """
    세그먼트 리스트를 WEBVTT 텍스트로 직렬화하여 저장합니다.
    - NOTE 블록(원문 보존)이 필요한 경우, seg.meta['original_text']가 존재하고
      seg.text와 다르면 NOTE로 기록합니다.
    - webvtt-py 없이도 동작하도록 수동으로 파일을 작성합니다.
    """
    lines: List[str] = ["WEBVTT", ""]

    for seg in segments:
        # NOTE(원문 보존)
        if keep_original_comments:
            orig = seg.meta.get("original_text")
            if (
                isinstance(orig, str)
                and orig
                and orig.strip()
                and orig.strip() != seg.text.strip()
            ):
                # NOTE 블록은 cue와 독립적으로 둘 수 있음
                note_body = orig.replace("\r\n", "\n").split("\n")
                lines.append("NOTE Original")
                lines.extend(note_body)
                lines.append("")  # NOTE 끝 공백 줄

        # Cue 블록
        start_ts = _seconds_to_ts(seg.start_sec)
        end_ts = _seconds_to_ts(seg.end_sec)
        lines.append(f"{start_ts} --> {end_ts}")
        # Cue 텍스트(여러 줄 허용)
        text_lines = (seg.text or "").replace("\r\n", "\n").split("\n")
        lines.extend(text_lines)
        lines.append("")  # cue 사이 공백 줄

    # 최종 저장
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
