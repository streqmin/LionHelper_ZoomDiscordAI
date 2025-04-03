import os
import logging
import time
import json
import urllib3
from typing import List, Optional

logger = logging.getLogger(__name__)

class SimpleAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/complete"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        self.http = urllib3.PoolManager(retries=False)
        logger.info("SimpleAPIClient 초기화 완료")

    def make_request(self, prompt: str) -> Optional[str]:
        """단순한 API 요청"""
        data = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "model": "claude-instant-1.2",
            "max_tokens_to_sample": 1500,
            "temperature": 0.7,
            "stop_sequences": ["\n\nHuman:"]
        }
        
        try:
            logger.info("API 요청 시작")
            encoded_data = json.dumps(data).encode('utf-8')
            
            response = self.http.request(
                'POST',
                self.base_url,
                body=encoded_data,
                headers=self.headers,
                timeout=30.0
            )
            
            if response.status != 200:
                logger.error(f"API 요청 실패: HTTP {response.status}")
                return None
                
            response_data = json.loads(response.data.decode('utf-8'))
            result = response_data.get('completion', '')
            
            if result:
                logger.info(f"API 요청 성공: {len(result)} 문자 응답")
                return result
            else:
                logger.error("API 응답에 completion 필드가 없음")
                return None
                
        except urllib3.exceptions.HTTPError as e:
            logger.error(f"API 요청 실패 (HTTP 오류): {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"API 응답 JSON 파싱 실패: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"API 요청 실패 (예상치 못한 오류): {str(e)}")
            return None

    def analyze_text(self, text: str, analysis_type: str = 'chat') -> str:
        """텍스트 분석"""
        if not text:
            return "분석할 내용이 없습니다."

        # 청크 크기 계산 (토큰 제한을 고려)
        max_chunk_size = 1500
        chunks = []
        
        # 텍스트를 문장 단위로 분할
        sentences = text.replace('\r', '').split('\n')
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence)
            if current_size + sentence_size > max_chunk_size:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        logger.info(f"텍스트를 {len(chunks)}개의 청크로 분할")
        
        results = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"청크 {i}/{len(chunks)} 처리 중")
            
            prompt = (
                f"다음은 {analysis_type} 내용의 {i}/{len(chunks)} 부분입니다.\n\n"
                f"{chunk}\n\n"
                "다음 형식으로 분석해주세요:\n"
                "# 주요 내용\n[핵심 내용 요약]\n\n"
                "# 키워드\n[주요 키워드들]\n\n"
                "# 분석\n[상세 분석 내용]"
            )
            
            # 최대 3번 재시도
            for attempt in range(3):
                result = self.make_request(prompt)
                if result:
                    results.append(result)
                    break
                else:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"청크 {i} 처리 실패 (시도 {attempt + 1}/3). {wait_time}초 후 재시도")
                    time.sleep(wait_time)
            
            if not result:
                results.append(f"[청크 {i} 분석 실패]")
                
            # API 호출 간격
            if i < len(chunks):
                time.sleep(3)
                
        return "\n\n---\n\n".join(results) if results else "분석 실패" 