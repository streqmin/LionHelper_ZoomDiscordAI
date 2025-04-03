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
        
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            # 마지막 문장이면 마침표를 추가하지 않음
            is_last_sentence = sentence == sentences[-1]
            sentence_with_dot = sentence + ('. ' if not is_last_sentence else '')
            
            if len(current_chunk) + len(sentence_with_dot) < max_length:
                current_chunk += sentence_with_dot
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence_with_dot
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
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
                if hasattr(e.response, 'text'):
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
                
                prompt = f"\n\nHuman: 당신은 {'줌 회의록' if analysis_type == 'vtt' else '채팅 로그'} 분석 전문가입니다. "
                prompt += f"다음은 전체 {'회의록' if analysis_type == 'vtt' else '채팅'}의 {i}/{total_chunks} 부분입니다.\n\n"
                prompt += f"{chunk}\n\n"
                prompt += "다음 형식으로 분석 결과를 제공해주세요:\n\n"
                prompt += "# 이 부분의 주요 내용\n[핵심 내용 요약]\n\n"
                prompt += "# 주요 키워드\n[이 부분의 주요 키워드들]\n\n"
                prompt += f"{'# 중요 포인트' if analysis_type == 'vtt' else '# 대화 분위기'}\n"
                prompt += f"[{'이 부분에서 특별히 주목할 만한 내용' if analysis_type == 'vtt' else '이 부분의 대화 톤과 분위기'}]\n\n"
                prompt += "Assistant:"
                
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