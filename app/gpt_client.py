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

    def analyze_text(self, text: str, analysis_type: str = 'vtt') -> str:
        """텍스트 분석을 수행"""
        try:
            # 텍스트를 청크로 분할
            logger.info(f"텍스트 분석 시작 (유형: {analysis_type})")
            chunks = self.split_text(text)
            
            # 각 청크별로 분석 수행
            results = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"청크 {i}/{len(chunks)} 분석 중")
                
                # 분석 유형에 따른 프롬프트 설정
                if analysis_type == 'vtt':
                    prompt = f"""
다음은 강의 내용을 텍스트로 변환한 것입니다. 강의 내용을 분석하여 다음 형식으로 응답해주세요:

[강의 내용]
{chunk}

다음 형식으로 응답해주세요:
# 주요 내용
(이 부분의 주요 내용을 2-3문장으로 요약)

# 키워드
(주요 키워드를 쉼표로 구분하여 나열)

# 분석
(강의 내용에 대한 전반적인 분석을 3-4문장으로 작성)

# 위험 발언
(차별적 발언, 부적절한 표현, 민감한 주제 등이 있다면 구체적으로 명시. 없다면 "위험 발언이 없습니다." 라고 표시)
"""
                elif analysis_type == 'chat':
                    prompt = f"""다음 채팅 내용을 분석하여 아래 형식으로 응답해주세요.

# 주요 대화 주제
- 채팅에서 다뤄진 주요 주제와 내용을 요약하여 나열

# 수강생 감정/태도 분석
1. 긍정적 반응
- 수업 내용에 대한 이해와 만족을 표현한 내용
- 적극적인 참여와 긍정적인 피드백

2. 부정적 반응
- 수업 내용이나 진행에 대한 불만이나 어려움 표현
- 부정적인 감정이나 태도가 드러난 내용

3. 질문/요청사항
- 수업 내용에 대한 질문
- 수업 진행 방식에 대한 요청사항

# 어려움/불만 상세 분석
1. 학습적 어려움
- 수업 내용의 난이도나 이해 문제
- 학습 진도나 과제 관련 어려움

2. 수업 진행 관련 문제
- 수업 속도나 시간 배분 문제
- 강의 방식이나 상호작용 관련 문제

3. 기술적 문제
- 온라인 플랫폼 사용의 어려움
- 음질, 화질 등 기술적 문제

# 개선 제안
1. 학습 내용 개선
- 수업 내용의 난이도 조정 제안
- 추가 학습 자료나 예제 요청

2. 수업 방식 개선
- 수업 진행 방식 개선 제안
- 상호작용 방식 개선 제안

3. 기술적 지원 강화
- 온라인 플랫폼 개선 제안
- 기술적 문제 해결을 위한 제안

# 위험 발언 및 주의사항
- 부적절한 언어 사용이나 태도
- 수업 분위기를 해치는 발언
- 개인정보 노출 위험

# 종합 제언
- 전반적인 개선점과 권장사항
- 향후 수업 운영을 위한 제안사항

채팅 내용:
{chunk}"""
                else:
                    prompt = f"""
다음 텍스트를 분석하여 주요 내용을 요약해주세요:

[텍스트 내용]
{chunk}

다음 형식으로 응답해주세요:
# 요약
(주요 내용을 3-4문장으로 요약)
"""
                
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
                if i < len(chunks):
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