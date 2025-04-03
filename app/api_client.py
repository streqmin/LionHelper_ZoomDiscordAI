import os
import logging
import time
import requests
from typing import List, Optional
import re

logger = logging.getLogger(__name__)

class ClaudeAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/complete"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    def split_text(self, text: str, max_length: int = 800) -> List[str]:
        """텍스트를 청크로 분할"""
        if not text:
            return []
        
        # 정규식으로 문장 분할
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 현재 청크에 문장을 추가할 수 있는 경우
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # 현재 청크가 비어있지 않으면 저장
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                # 새로운 청크 시작
                current_chunk = [sentence]
                current_length = sentence_length
        
        # 마지막 청크 처리
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def call_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Claude API 호출"""
        data = {
            "prompt": prompt,
            "model": "claude-instant-1.2",
            "max_tokens_to_sample": 1500,
            "temperature": 0.7,
            "stop_sequences": ["\n\nHuman:"]
        }
        
        for attempt in range(max_retries):
            try:
                # 매 시도마다 새로운 세션 생성
                session = requests.Session()
                response = session.post(
                    self.base_url,
                    headers=self.headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()['completion']
            except requests.exceptions.RequestException as e:
                error_detail = f"HTTP 에러: {str(e)}"
                if hasattr(e, 'response') and e.response is not None:
                    error_detail += f" - 응답: {e.response.text}"
                logger.error(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {error_detail}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)
                    logger.info(f"재시도 대기 중... {wait_time}초")
                    time.sleep(wait_time)
                else:
                    return None
            except Exception as e:
                logger.error(f"예상치 못한 오류 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)
                    logger.info(f"재시도 대기 중... {wait_time}초")
                    time.sleep(wait_time)
                else:
                    return None

    def analyze_content(self, content: str, analysis_type: str = 'chat') -> str:
        """콘텐츠 분석"""
        try:
            chunks = self.split_text(content)
            total_chunks = len(chunks)
            logger.info(f"Split content into {total_chunks} chunks")
            
            results = []
            failed_chunks = []
            
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{total_chunks}")
                
                # 프롬프트 생성
                prompt_parts = [
                    "\n\nHuman: 당신은 ",
                    "줌 회의록" if analysis_type == 'vtt' else "채팅 로그",
                    " 분석 전문가입니다. ",
                    f"다음은 전체 {'회의록' if analysis_type == 'vtt' else '채팅'}의 {i}/{total_chunks} 부분입니다.\n\n",
                    f"{chunk}\n\n",
                    "다음 형식으로 분석 결과를 제공해주세요:\n\n",
                    "# 이 부분의 주요 내용\n[핵심 내용 요약]\n\n",
                    "# 주요 키워드\n[이 부분의 주요 키워드들]\n\n",
                    "# 중요 포인트" if analysis_type == 'vtt' else "# 대화 분위기",
                    "\n[",
                    "이 부분에서 특별히 주목할 만한 내용" if analysis_type == 'vtt' else "이 부분의 대화 톤과 분위기",
                    "]\n\n",
                    "Assistant:"
                ]
                
                prompt = ''.join(prompt_parts)
                result = self.call_api(prompt)
                
                if result:
                    results.append(result)
                    time.sleep(8)
                else:
                    failed_chunks.append(i)
                    results.append(f"[청크 {i} 처리 실패]")
                    time.sleep(15)
            
            if failed_chunks:
                logger.warning(f"다음 청크들의 처리에 실패했습니다: {failed_chunks}")
            
            return "\n\n".join(results) if results else "분석에 실패했습니다."
            
        except Exception as e:
            logger.error(f"분석 중 오류 발생: {str(e)}")
            return f"분석 중 오류가 발생했습니다: {str(e)}" 