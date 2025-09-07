# -*- coding: utf-8 -*-
"""
llm_client: 외부 LLM API(선택) 또는 로컬 폴백으로 교정안 제안

- 기본은 "disabled" (외부 호출 안함, 로컬 휴리스틱 폴백)
- OPENAI_API_KEY 가 있고 provider='openai'면 OpenAI API 사용
- suggest_span_edit(left, span, right, category, terms) -> Optional[str]
"""
from __future__ import annotations
import os, re
from typing import Optional, Sequence
from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        provider: str = "disabled",
        model: str = "gpt-4o-mini",
        max_output_tokens: int = 128,
        temperature: float = 0.2,
        timeout: int = 30,
    ) -> None:
        self.provider = (provider or "disabled").lower().strip()
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.timeout = int(timeout)

        self._openai_ready = False
        self._client: Optional[OpenAI] = None

        if self.provider == "openai":
            api_key = os.environ.get(
                "OPENAI_API_KEY",
                "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",
            ).strip()
            if api_key:
                try:
                    # 선택: 조직/프로젝트 ID를 환경변수로 받으면 지정
                    org = os.environ.get(
                        "OPENAI_ORG_ID", "org-an43vv4z5GbUjozNcy6g2zOp"
                    )
                    project = os.environ.get(
                        "OPENAI_PROJECT_ID", "proj_044fSsYSnjbfVJy5NcZRuAZl"
                    )

                    client = OpenAI(api_key=api_key, organization=org, project=project)
                    client = client.with_options(timeout=self.timeout)
                    self._client = client
                    self._openai_ready = True
                except Exception:
                    self._client = None
                    self._openai_ready = False

    def available(self) -> bool:
        return (
            self.provider == "openai"
            and self._openai_ready
            and self._client is not None
        )

    def suggest_span_edit(
        self,
        context_left: str,
        span_text: str,
        context_right: str,
        category: str,
    ) -> Optional[str]:
        if not span_text:
            return None
        if self.available():
            return self._call_openai_span(context_left, span_text, context_right)
        return self._fallback_edit(span_text, category)

    def _call_openai_span(self, left: str, span: str, right: str) -> Optional[str]:
        try:
            left_ctx = left or ""
            right_ctx = right or ""

            sys_prompt = (
                "You are a Korean transcript fixer for ASR-generated VTT lectures.\n"
                "Use [LEFT] and [RIGHT] only to understand context; never copy or edit them.\n"
                "Edit only the text inside [SPAN]. Make the smallest possible change while preserving meaning.\n"
                "Prefer standard Korean and natural spacing/orthography.\n"
                "Do not paraphrase, summarize, reorder, or add/remove information.\n"
                "Keep technical/identifier-like tokens (e.g., code names, APIs, letter–digit mixes), numbers, and casing unchanged.\n"
                "Preserve punctuation and line breaks unless they are obvious ASR artifacts (e.g., stray hyphens, repeated periods/spaces).\n"
                "Fix common ASR typos and particle spacing; correct clear misrecognitions only when confident from context.\n"
                "If [SPAN] is already fine, return it unchanged.\n"
                "Return ONLY the corrected SPAN text—no quotes, explanations, or formatting."
            )
            user_prompt = (
                f"[LEFT]\n{left_ctx}\n\n"
                f"[SPAN]\n{span}\n\n"
                f"[RIGHT]\n{right_ctx}\n\n"
                "Output: corrected SPAN only (no quotes, no code fences)."
            )

            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            content = self._postprocess_answer(content)
            if not content or content == span:
                return None
            return content

        except Exception as e:
            # 401 & 스코프 부족 시 가이던스 로그
            msg = str(e)
            if (
                "Missing scopes" in msg
                or "insufficient permissions" in msg
                or "status code: 401" in msg
            ):
                print(
                    "[LLMClient] 401/권한 오류: 현재 API 키에 'model.request' 권한이 없습니다. "
                    "Project → API Keys에서 권한을 'All'로 설정하거나, Restricted 키라면 모델 추론 권한을 추가하세요. "
                    "또한 Project → Limits → Model usage에 gpt-4o-mini 허용이 필요한지 확인하세요."
                )
            else:
                print("[LLMClient] OpenAI 호출 실패:", msg)
            return None

    def _postprocess_answer(self, s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            m = re.match(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", s)
            if m:
                s = m.group(1).strip()
        s = re.sub(r"^Output\s*:\s*", "", s, flags=re.IGNORECASE).strip()
        if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")
        ):
            s = s[1:-1].strip()
        s = re.sub(r"[ \t]{2,}", " ", s)
        s = re.sub(r"\s+\n", "\n", s).strip()
        return s

    def _fallback_edit(self, span: str, category: str) -> Optional[str]:
        cat = (category or "").upper()
        if cat in {"CONTEXT_ANOM", "NGRAM_OUTLIER"}:
            return None
        s = span
        s = re.sub(r"([가-힣])\s*-\s*([가-힣])", r"\1\2", s)
        s = re.sub(r"[ ]{2,}", " ", s)
        s = re.sub(r"\.{3,}", "…", s)
        return s if s != span else None
