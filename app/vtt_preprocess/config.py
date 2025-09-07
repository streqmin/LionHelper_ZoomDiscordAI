# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class Config:
    # 리소스/LLM
    llm_temperature: float = 0.0
    llm_max_tokens: int = 180
    llm_timeout_sec: int = 12
    llm_retry: int = 1
    llm_batch_max_spans: int = 6
    llm_concurrency: int = 1  # Render Free: 동시 1요청

    # LLM 컨텍스트/스팬 길이
    llm_left_ctx_chars: int = 120
    llm_right_ctx_chars: int = 120

    # LLM에 보낼 스팬(문장 묶음) 길이 목표/하한/상한
    llm_span_target_chars: int = 80  # ✅ 목표치
    llm_span_min_chars: int = 50  # 너무 짧으면 이웃 문장 합치기
    llm_span_max_chars: int = 260  # 너무 길어지지 않게 상한

    # 병합
    merge_min_secs: int = 30
    merge_max_secs: int = 60
    merge_min_chars: int = 200
    merge_max_chars: int = 400
    dedup_similarity: float = 0.90
    use_spacing_fix: bool = False  # 과교정 방지: 기본 미사용

    # 비주제 감지
    pos_beta_ratio: float = 0.04  # exp(-t/(β*T))
    smooth_window: int = 3
    q_non_topic: float = 0.80
    topic_threshold: float = 0.50
    min_block_secs_lo: int = 30
    min_block_secs_hi: int = 90
    greet_close_terms: Set[str] = frozenset(
        {
            "안녕하세요",
            "반갑",
            "소개",
            "오티",
            "OT",
            "공지",
            "출석",
            "설문",
            "다음 시간",
            "다음주",
            "마무리",
            "정리",
            "여기까지",
            "감사",
            "수고",
            "줌",
            "Zoom",
            "마이크",
            "카메라",
            "화면 공유",
            "좋아요",
            "구독",
        }
    )
    domain_keep_if_terms: int = 2  # 핵심 용어가 2개 이상이면 보존

    # 오류 스팬 탐지
    contextfit_low: float = 0.45
    char_ngram_outlier_top_p: float = 0.10  # 상위 10% 놀람도
    mixed_digit_hangul_regex_suffix = r"(에는|에서|으로|에게|에|의|을|를|만|까지|부터)$"

    # 교정 임계/규칙
    candidate_quantile_tau: float = 0.85
    llm_selector_fit_max: float = 0.50
    llm_selector_near_tau_band: float = 0.05
    llm_selector_top2_delta: float = 0.08
    lgv_conf_min: float = 0.60
    lgv_changed_chars_max: int = 4
    lgv_abs_len_delta_max: int = 3
    max_replacements_per_segment: int = 3
    delta_topic_min: float = 0.01  # ✅ 교정 전후 Topic 점수 최소 개선폭(ΔTopic)

    # 숫자→한글 음가
    digit_to_hangul = {
        "0": ["영"],
        "1": ["일"],
        "2": ["이"],
        "3": ["삼"],
        "4": ["사"],
        "5": ["오"],
        "6": ["육"],
        "7": ["칠"],
        "8": ["팔"],
        "9": ["구"],
    }
    # 수량/단위 가드
    unit_tokens: Set[str] = frozenset(
        {
            "개",
            "명",
            "%",
            "원",
            "초",
            "분",
            "시",
            "년",
            "월",
            "일",
            "회",
            "MB",
            "GB",
            "TB",
            "℃",
            "km",
            "cm",
            "mm",
        }
    )
    # 혼용 화이트리스트
    mixed_whitelist: Set[str] = frozenset({"C", "C#", ".NET", "A/B", "CNN", "AI", "ML"})

    # 캐시/로그
    lru_cache_size: int = 2000
