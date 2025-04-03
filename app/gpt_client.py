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
                    prompt = f"""
당신은 수업 중 학생들의 채팅을 분석하여 수업 개선에 도움이 되는 인사이트를 도출하는 교육 분석가입니다.
다음 채팅 내용을 분석하여 수강생들의 어려움과 불만사항을 파악하고, 구체적인 개선점을 제시해주세요.

[채팅 내용]
{chunk}

다음 형식으로 상세히 분석해주세요:

# 주요 대화 주제
(이 채팅에서 다루어진 주요 주제나 이슈를 2-3문장으로 요약)

# 수강생 감정/태도 분석
1. 긍정적 반응
- 이해했다는 표현이나 긍정적인 피드백
- 적극적인 참여 의지를 보이는 발언
- 다른 학생을 돕거나 협력하는 모습

2. 부정적 반응
- 이해가 안 된다는 표현
- 좌절감이나 혼란을 나타내는 발언
- 불만이나 개선 요청 사항

3. 질문/요청사항
- 수업 내용에 대한 구체적인 질문들
- 추가 설명이나 자료 요청
- 기술적인 도움 요청

# 어려움/불만 상세 분석
1. 학습적 어려움
- 특히 어려워하는 개념이나 주제
- 이해가 부족한 부분
- 추가 설명이 필요한 부분

2. 수업 진행 관련 문제
- 수업 속도나 난이도 관련 의견
- 설명 방식이나 자료에 대한 불만
- 과제나 실습 관련 어려움

3. 기술적 문제
- 시스템이나 도구 사용의 어려움
- 접속 문제나 오류 상황
- 기술 지원 필요 사항

# 개선 제안
1. 학습 내용 개선
- 추가 설명이나 예제가 필요한 부분
- 보충 자료나 참고 리소스 제안
- 난이도 조정이 필요한 부분

2. 수업 방식 개선
- 수업 진행 방식 조정 제안
- 실습/과제 방식 개선점
- 상호작용 강화 방안

3. 기술적 지원 강화
- 필요한 기술 지원이나 가이드
- 시스템 개선 필요 사항
- 접근성 향상을 위한 제안

# 위험 발언 및 주의사항
- 부적절한 언어나 태도
- 공격적이거나 비하하는 표현
- 과도한 불만이나 부정적 태도
(없을 경우 "특별한 주의사항이 없습니다." 라고 표시)

# 종합 제언
(분석된 내용을 바탕으로 가장 시급하게 개선이 필요한 1-2가지 사항을 구체적으로 제시)
"""
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