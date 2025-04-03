import os
import logging
import time
import json
import httpx
from typing import List, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# 로깅 설정
logger = logging.getLogger(__name__)

class GPTAPIClient:
    def __init__(self, api_key):
        """GPT API 클라이언트 초기화"""
        if not api_key:
            raise ValueError("API 키가 제공되지 않았습니다.")
            
        self.logger = logging.getLogger(__name__)
        self.model = "gpt-3.5-turbo"
        
        # httpx 클라이언트 설정
        http_client = httpx.Client()
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(
            api_key=api_key,
            http_client=http_client
        )
        
        self.logger.info(f"GPTAPIClient 초기화 완료 (모델: {self.model})")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def make_request(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        """GPT API 요청 수행"""
        self.logger.info(f"API 요청 시작 (프롬프트 길이: {len(prompt)} 문자)")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=max_tokens
            )
            
            if response and response.choices:
                result = response.choices[0].message.content
                self.logger.info("API 요청 성공")
                return result
            else:
                self.logger.error("API 응답이 비어있음")
                raise Exception("API 응답이 비어있습니다")
                
        except Exception as e:
            self.logger.error(f"API 요청 실패: {str(e)}")
            raise

    def split_text(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """텍스트를 청크로 분할"""
        if not text:
            logger.warning("분할할 텍스트가 비어있음")
            return []
            
        logger.info(f"텍스트 분할 시작 (전체 길이: {len(text)} 문자)")
        chunks = []
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
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(chunk_text)
                    logger.debug(f"청크 생성: {len(chunk_text)} 문자")
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
            logger.debug(f"마지막 청크 생성: {len(chunk_text)} 문자")
            
        logger.info(f"텍스트 분할 완료 (총 {len(chunks)}개 청크)")
        return chunks

    def analyze_text(self, text: str, analysis_type: str = 'chat') -> str:
        """텍스트 분석 수행"""
        try:
            if not text:
                return "분석할 내용이 없습니다."
                
            logger.info(f"텍스트 분석 시작 (유형: {analysis_type})")
            chunks = self.split_text(text)
            
            if not chunks:
                logger.error("분석할 청크가 없음")
                return "분석할 내용이 없습니다."
            
            results = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"청크 {i}/{total_chunks} 분석 중")
                
                prompt = (
                    f"다음은 {analysis_type} 내용의 {i}/{total_chunks} 부분입니다.\n\n"
                    f"{chunk}\n\n"
                    "다음 형식으로 분석해주세요:\n"
                    "# 주요 내용\n[핵심 내용 요약]\n\n"
                    "# 키워드\n[주요 키워드들]\n\n"
                    "# 분석\n[상세 분석 내용]"
                )
                
                try:
                    result = self.make_request(prompt)
                    if result:
                        results.append(result)
                    else:
                        results.append(f"[청크 {i} 분석 실패]")
                except Exception as e:
                    logger.error(f"청크 {i} 분석 중 오류 발생: {str(e)}")
                    results.append(f"[청크 {i} 분석 오류: {str(e)}]")
                
                # 마지막 청크가 아닌 경우 API 호출 간격 유지
                if i < total_chunks:
                    time.sleep(2)
            
            final_result = "\n\n---\n\n".join(results)
            logger.info("텍스트 분석 완료")
            return final_result
            
        except Exception as e:
            logger.error(f"분석 중 예상치 못한 오류 발생: {str(e)}")
            return f"분석 중 오류 발생: {str(e)}"

    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            logger.info("API 연결 테스트 시작")
            result = self.make_request("안녕하세요", max_tokens=10)
            return bool(result)
        except Exception as e:
            logger.error(f"API 연결 테스트 실패: {str(e)}")
            return False 