# -*- coding: utf-8 -*-
"""
enhanced_gpt_client: VTT 전처리 모듈 기반 강화된 GPT 클라이언트

VTT 전처리 모듈의 결과를 활용하여 3가지 기능을 강화:
1. 비주제 블록 제거된 VTT로 강의 내용 요약
2. 교정된 VTT로 위험 발언 정확한 파악
3. 토픽 유사도 기반 커리큘럼 매칭
"""
import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from vtt_preprocess.topic import build_topic_index
from vtt_preprocess.schema import Segment

from vtt_preprocessor import VTTPreprocessor

logger = logging.getLogger(__name__)

class EnhancedGPTClient:
    def __init__(self, api_key: str):
        """강화된 GPT 클라이언트 초기화"""
        if not api_key:
            raise ValueError("API 키가 제공되지 않았습니다.")
            
        self.logger = logging.getLogger(__name__)
        self.model = "gpt-4o-mini"
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=api_key)
        
        # VTT 전처리기 초기화
        self.vtt_preprocessor = VTTPreprocessor()
        
        self.logger.info(f"EnhancedGPTClient 초기화 완료 (모델: {self.model})")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_request(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
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

    def preprocess_vtt(self, vtt_path: str, curriculum_path: str, output_dir: str) -> Dict[str, Any]:
        """VTT 파일 전처리 실행 (VTTPreprocessor 위임)"""
        return self.vtt_preprocessor.preprocess_vtt(vtt_path, curriculum_path, output_dir)

    def analyze_chat(self, text: str) -> str:
        """채팅 텍스트 분석을 수행"""
        try:
            self.logger.info(f"채팅 텍스트 분석 시작")
            chunks = self._split_text(text)
        
            results = []
            for i, chunk in enumerate(chunks, 1):
                self.logger.info(f"청크 {i}/{len(chunks)} 분석 중")

                # chat 분석만 지원
                prompt = f"""Please analyze the following chat content and respond in Korean using the format below:

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

                Chat Content:
                {chunk}"""

            result = self._make_request(prompt)
            if result:
                    results.append(result)

            return "\n\n---\n\n".join(results)

        except Exception as e:
            self.logger.error(f"분석 중 예상치 못한 오류 발생: {str(e)}")
            return f"분석 중 오류 발생: {str(e)}"

    def analyze_lecture_content(self, segments: List[Segment]) -> Dict[str, Any]:
        """1. 비주제 블록 제거된 VTT로 강의 내용 요약 (세그먼트 정보 포함)"""
        try:
            # 청크로 분할 (세그먼트 정보 포함)
            chunks = self._create_chunks_from_segments(segments, max_chunk_size=3000)
            
            # 각 청크별로 분석
            lecture_results = []
            for i, chunk in enumerate(chunks, 1):
                self.logger.info(f"강의 내용 청크 {i}/{len(chunks)} 분석 중")
                
                prompt = f"""
The following is the core content of a lecture. Non-topic blocks (greetings, conclusions, breaks, etc.) have been removed, containing only pure lecture content.

[Lecture Content]
{chunk['text']}

Please respond in the following format:

# Main Content
(Summarize the key content of this section in 2-3 sentences)

# Keywords
(List the main keywords separated by commas)

# Analysis
(Provide a general analysis of the lecture content in 3-4 sentences)

# Learning Points
(List important points that learners should pay attention to)
"""
                
                result = self._make_request(prompt)
                if result:
                    # 세그먼트 정보와 함께 결과 저장
                    lecture_results.append({
                        'chunk_id': i,
                        'chunk_text': chunk['text'],
                        'analysis': result,
                        'segments': [
                            {
                                'sid': seg.sid,
                                'start_sec': seg.start_sec,
                                'end_sec': seg.end_sec,
                                'text': seg.text,
                                'time_range': f"{self._format_time(seg.start_sec)} - {self._format_time(seg.end_sec)}"
                            }
                            for seg in chunk['segments']
                        ]
                    })
            
            return {
                'total_chunks': len(chunks),
                'lecture_results': lecture_results
            }
            
        except Exception as e:
            self.logger.error(f"강의 내용 분석 실패: {str(e)}")
            return {
                'error': f"강의 내용 분석 중 오류 발생: {str(e)}",
                'total_chunks': 0,
                'lecture_results': [],
                'summary': ''
            }

    def analyze_risk_content(self, segments: List[Segment]) -> Dict[str, Any]:
        """2. 교정된 VTT로 위험 발언 정확한 파악 (청크 기반 + 세그먼트 매핑)"""
        try:
            # segments는 이미 correct_segments에서 교정된 상태
            # 청크로 분할 (세그먼트 정보 포함)
            chunks = self._create_chunks_from_segments(segments, max_chunk_size=3000)
            
            # 각 청크별로 위험 발언 분석
            risk_results = []
            for i, chunk in enumerate(chunks, 1):
                self.logger.info(f"위험 발언 청크 {i}/{len(chunks)} 분석 중")
                
                # 청크 분석
                risk_analysis = self._analyze_chunk_for_risk(chunk['text'])
                
                # 위험 발언이 발견되면 세그먼트 매핑
                if risk_analysis['has_risk']:
                    mapped_segments = self._map_risk_to_segments(
                        risk_analysis['risk_texts'], 
                        chunk['segments']
                    )
                    risk_results.append({
                        'chunk_id': i,
                        'chunk_text': chunk['text'],
                        'risk_analysis': risk_analysis,
                        'affected_segments': mapped_segments
                    })
            
            return {
                'total_chunks': len(chunks),
                'risk_found': len(risk_results) > 0,
                'risk_results': risk_results,
                'summary': self._generate_risk_summary(risk_results)
            }
            
        except Exception as e:
            self.logger.error(f"위험 발언 분석 실패: {str(e)}")
            return {
                'error': f"위험 발언 분석 중 오류 발생: {str(e)}",
                'total_chunks': 0,
                'risk_found': False,
                'risk_results': []
            }

    def analyze_curriculum_matching(self, segments: List[Segment], curriculum_path: str) -> Dict[str, Any]:
        """3. 토픽 유사도 기반 커리큘럼 매칭"""
        try:
            # 토픽 인덱스 구축
            topic_index = build_topic_index(curriculum_path)
            
            # 각 세그먼트별 토픽 점수 계산
            segment_scores = []
            for seg in segments:
                if not seg.text.strip():
                    continue
                    
                # 토픽 점수 계산
                topic_score = topic_index.score_text(seg.text)
                
                # 가장 관련성 높은 토픽 찾기
                best_topic = None
                best_score = 0
                for term in getattr(topic_index, 'terms', []):
                    term_score = topic_index.score_text(seg.text, term)
                    if term_score > best_score:
                        best_score = term_score
                        best_topic = term
                
                segment_scores.append({
                    'sid': seg.sid,
                    'text': seg.text,
                    'topic_score': topic_score,
                    'best_topic': best_topic,
                    'best_score': best_score,
                    'start_sec': seg.start_sec,
                    'end_sec': seg.end_sec
                })
            
            # 토픽별 그룹화
            topic_groups = {}
            for seg_score in segment_scores:
                topic = seg_score['best_topic'] or '기타'
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(seg_score)
            
            # 각 토픽별 통계 계산
            topic_stats = {}
            for topic, segs in topic_groups.items():
                if not segs:
                    continue
                    
                total_score = sum(seg['best_score'] for seg in segs)
                avg_score = total_score / len(segs)
                total_duration = sum(seg['end_sec'] - seg['start_sec'] for seg in segs)
                
                topic_stats[topic] = {
                    'segment_count': len(segs),
                    'total_score': total_score,
                    'average_score': avg_score,
                    'total_duration_sec': total_duration,
                    'coverage_percentage': (total_duration / sum(seg['end_sec'] - seg['start_sec'] for seg in segment_scores)) * 100
                }
            
            return {
                'segment_scores': segment_scores,
                'topic_groups': topic_groups,
                'topic_stats': topic_stats,
                'curriculum_terms': getattr(topic_index, 'terms', [])
            }
            
        except Exception as e:
            self.logger.error(f"커리큘럼 매칭 분석 실패: {str(e)}")
            return {
                'error': f"커리큘럼 매칭 분석 중 오류 발생: {str(e)}",
                'segment_scores': [],
                'topic_groups': {},
                'topic_stats': {},
                'curriculum_terms': []
            }

    def _split_text(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """텍스트를 청크로 분할"""
        if not text:
            return []
            
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # 공백 포함
            if current_size + word_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            self.logger.info("API 연결 테스트 시작")
            result = self._make_request("안녕하세요", max_tokens=10)
            return bool(result)
        except Exception as e:
            self.logger.error(f"API 연결 테스트 실패: {str(e)}")
            return False


    def _create_chunks_from_segments(self, segments: List[Segment], max_chunk_size: int = 3000) -> List[Dict]:
        """세그먼트를 청크로 분할 (세그먼트 정보 포함)"""
        chunks = []
        current_chunk = {
            'text': '',
            'segments': [],
            'char_count': 0
        }
        
        for segment in segments:
            segment_text = segment.text or ""
            segment_length = len(segment_text)
            
            # 현재 청크에 추가할 수 있는지 확인
            if current_chunk['char_count'] + segment_length > max_chunk_size and current_chunk['segments']:
                # 현재 청크를 완료하고 새 청크 시작
                chunks.append({
                    'text': current_chunk['text'].strip(),
                    'segments': current_chunk['segments'].copy()
                })
                
                current_chunk = {
                    'text': segment_text,
                    'segments': [segment],
                    'char_count': segment_length
                }
            else:
                # 현재 청크에 추가
                if current_chunk['text']:
                    current_chunk['text'] += "\n" + segment_text
                else:
                    current_chunk['text'] = segment_text
                
                current_chunk['segments'].append(segment)
                current_chunk['char_count'] += segment_length
        
        # 마지막 청크 추가
        if current_chunk['segments']:
            chunks.append({
                'text': current_chunk['text'].strip(),
                'segments': current_chunk['segments']
            })
        
        return chunks

    def _analyze_chunk_for_risk(self, chunk_text: str) -> Dict[str, Any]:
        """청크에서 위험 발언 분석"""
        prompt = f"""
The following is corrected lecture content. ASR errors have been fixed to provide more accurate text.

[Corrected Lecture Content]
{chunk_text}

Please analyze this content for risk speech and respond in the following JSON format:

{{
    "has_risk": true/false,
    "risk_types": ["discriminatory", "inappropriate", "sensitive_topic", etc.],
    "risk_texts": ["exact text that contains risk speech"],
    "risk_analysis": "detailed analysis of the risk speech",
    "precautions": ["list of parts that require attention"],
    "improvement_suggestions": ["suggestions for improvement if risk found"]
}}

If no risk speech is found, set "has_risk" to false and provide empty arrays for risk-related fields.
"""
        
        try:
            result = self._make_request(prompt)
            if result:
                # JSON 파싱 시도
                import json
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 기본 구조 반환
                    return {
                        "has_risk": "risk" in result.lower() and "no risk" not in result.lower(),
                        "risk_types": [],
                        "risk_texts": [],
                        "risk_analysis": result,
                        "precautions": [],
                        "improvement_suggestions": []
                    }
            else:
                return {
                    "has_risk": False,
                    "risk_types": [],
                    "risk_texts": [],
                    "risk_analysis": "Analysis failed",
                    "precautions": [],
                    "improvement_suggestions": []
                }
        except Exception as e:
            self.logger.error(f"청크 위험 발언 분석 실패: {str(e)}")
            return {
                "has_risk": False,
                "risk_types": [],
                "risk_texts": [],
                "risk_analysis": f"Analysis error: {str(e)}",
                "precautions": [],
                "improvement_suggestions": []
            }

    def _map_risk_to_segments(self, risk_texts: List[str], segments: List[Segment]) -> List[Dict]:
        """위험 발언 텍스트를 세그먼트와 매칭"""
        affected_segments = []
        
        for segment in segments:
            segment_text = segment.text or ""
            matched_risks = []
            
            # 각 위험 발언 텍스트와 세그먼트 매칭
            for risk_text in risk_texts:
                if self._text_contains_risk(segment_text, risk_text):
                    risk_portion = self._extract_risk_portion(segment_text, risk_text)
                    matched_risks.append({
                        'risk_text': risk_text,
                        'risk_portion': risk_portion,
                        'confidence': self._calculate_match_confidence(segment_text, risk_text)
                    })
            
            # 매칭된 위험 발언이 있으면 세그먼트 정보 추가
            if matched_risks:
                affected_segments.append({
                    'sid': segment.sid,
                    'start_sec': segment.start_sec,
                    'end_sec': segment.end_sec,
                    'text': segment_text,
                    'matched_risks': matched_risks,
                    'time_range': f"{self._format_time(segment.start_sec)} - {self._format_time(segment.end_sec)}"
                })
        
        return affected_segments

    def _text_contains_risk(self, segment_text: str, risk_text: str) -> bool:
        """세그먼트 텍스트가 위험 발언을 포함하는지 확인"""
        if not risk_text or not segment_text:
            return False
        
        # 대소문자 무시하고 검색
        segment_lower = segment_text.lower()
        risk_lower = risk_text.lower()
        
        # 정확한 매칭
        if risk_lower in segment_lower:
            return True
        
        # 부분 매칭 (위험 발언의 핵심 키워드 추출)
        risk_keywords = self._extract_keywords(risk_lower)
        if risk_keywords:
            # 핵심 키워드가 모두 포함되어 있는지 확인
            return all(keyword in segment_lower for keyword in risk_keywords)
        
        return False

    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 핵심 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        import re
        
        # 불용어 제거
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        # 단어 추출 (2글자 이상)
        words = re.findall(r'\b\w{2,}\b', text.lower())
        
        # 불용어 제거하고 빈도가 높은 단어 선택
        filtered_words = [word for word in words if word not in stop_words]
        
        # 상위 3개 키워드 반환
        from collections import Counter
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(3)]

    def _extract_risk_portion(self, segment_text: str, risk_text: str) -> str:
        """세그먼트에서 위험 발언 부분 추출"""
        if not risk_text or not segment_text:
            return ""
        
        # 정확한 매칭이 있으면 해당 부분 반환
        if risk_text.lower() in segment_text.lower():
            start_idx = segment_text.lower().find(risk_text.lower())
            end_idx = start_idx + len(risk_text)
            return segment_text[start_idx:end_idx]
        
        # 부분 매칭의 경우 문장 단위로 추출
        sentences = segment_text.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in self._extract_keywords(risk_text.lower())):
                return sentence.strip()
        
        # 매칭이 없으면 전체 세그먼트 반환
        return segment_text

    def _calculate_match_confidence(self, segment_text: str, risk_text: str) -> float:
        """매칭 신뢰도 계산 (0.0 ~ 1.0)"""
        if not risk_text or not segment_text:
            return 0.0
        
        # 정확한 매칭
        if risk_text.lower() in segment_text.lower():
            return 1.0
        
        # 키워드 기반 매칭
        risk_keywords = self._extract_keywords(risk_text.lower())
        if not risk_keywords:
            return 0.0
        
        segment_lower = segment_text.lower()
        matched_keywords = sum(1 for keyword in risk_keywords if keyword in segment_lower)
        
        return matched_keywords / len(risk_keywords)

    def _format_time(self, seconds: float) -> str:
        """초를 MM:SS 형식으로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _generate_risk_summary(self, risk_results: List[Dict]) -> str:
        """위험 발언 분석 결과 요약 생성"""
        if not risk_results:
            return "No risk speech found in the lecture content."
        
        total_risks = sum(len(result['affected_segments']) for result in risk_results)
        risk_types = set()
        
        for result in risk_results:
            for risk_type in result['risk_analysis'].get('risk_types', []):
                risk_types.add(risk_type)
        
        summary = f"Found {total_risks} segments with risk speech across {len(risk_results)} chunks.\n"
        summary += f"Risk types detected: {', '.join(risk_types) if risk_types else 'Various'}\n\n"
        
        for i, result in enumerate(risk_results, 1):
            summary += f"Chunk {result['chunk_id']}: {len(result['affected_segments'])} segments with risk speech\n"
            for segment in result['affected_segments']:
                summary += f"  - Segment {segment['sid']} ({segment['time_range']}): {len(segment['matched_risks'])} risk(s)\n"
        
        return summary
