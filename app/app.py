import os
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from werkzeug.utils import secure_filename
from app.config import Config
from anthropic import Anthropic
from datetime import datetime
import json
import logging
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import socket
import asyncio
import anthropic
import re

# 글로벌 타임아웃 설정
socket.setdefaulttimeout(60)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
app.config['ALLOWED_EXTENSIONS'] = {'vtt', 'txt'}  # 허용된 파일 확장자

# 업로드 폴더가 없으면 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Anthropic 클라이언트 초기화
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    logger.error("ANTHROPIC_API_KEY not found in environment variables")
    raise ValueError("ANTHROPIC_API_KEY is required")

# httpx 클라이언트 설정
http_client = httpx.Client(
    timeout=httpx.Timeout(30.0, connect=15.0),  # 타임아웃 감소
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    verify=True
)

# Anthropic 클라이언트 초기화
client = anthropic.Anthropic(
    api_key=api_key,
    http_client=http_client
)
logger.info("Anthropic client initialized successfully")

def split_content(content, max_length=1000):  # 청크 크기를 1000자로 감소
    """콘텐츠를 작은 청크로 분할"""
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_length = len(line) + 1  # +1 for newline
        if current_length + line_length > max_length and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_claude_api(prompt):
    """Claude API 호출 함수 with 재시도 로직"""
    try:
        completion = client.completions.create(
            prompt=prompt,
            model="claude-instant-1.2",  # Claude Instant 모델로 변경
            max_tokens_to_sample=1500,  # 토큰 수 감소
            stop_sequences=["\n\nHuman:"],
            temperature=0.7
        )
        return completion
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        raise

def analyze_content_in_chunks(content, analysis_type='vtt'):
    """청크 단위로 콘텐츠 분석"""
    chunks = split_content(content)
    total_chunks = len(chunks)
    logger.info(f"Split content into {total_chunks} chunks")
    
    all_results = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{total_chunks}")
        
        if analysis_type == 'vtt':
            prompt = f"\n\nHuman: 당신은 줌 회의록 분석 전문가입니다. 다음은 전체 회의록의 {i}/{total_chunks} 부분입니다. 이 부분을 분석해주세요:\n\n{chunk}\n\n다음 형식으로 분석 결과를 제공해주세요:\n\n# 이 부분의 주요 내용\n[핵심 내용 요약]\n\n# 주요 키워드\n[이 부분의 주요 키워드들]\n\n# 중요 포인트\n[이 부분에서 특별히 주목할 만한 내용]\n\nAssistant:"
        else:
            prompt = f"\n\nHuman: 당신은 채팅 로그 분석 전문가입니다. 다음은 전체 채팅의 {i}/{total_chunks} 부분입니다. 이 부분을 분석해주세요:\n\n{chunk}\n\n다음 형식으로 분석 결과를 제공해주세요:\n\n# 이 부분의 주요 내용\n[핵심 내용 요약]\n\n# 주요 키워드\n[이 부분의 주요 키워드들]\n\n# 대화 분위기\n[이 부분의 대화 톤과 분위기]\n\nAssistant:"
        
        try:
            completion = call_claude_api(prompt)
            all_results.append(completion.completion)
            time.sleep(2)  # API 호출 간 간격 증가
        except Exception as e:
            logger.error(f"Failed to process chunk {i}: {str(e)}")
            all_results.append(f"[이 부분 처리 중 오류 발생: {str(e)}]")
            time.sleep(5)  # 오류 발생 시 더 긴 대기 시간
    
    # 최종 요약 생성
    try:
        summary_chunks = split_content(' '.join(all_results), 1000)  # 결과도 청크로 분할
        final_summaries = []
        
        for i, summary_chunk in enumerate(summary_chunks, 1):
            if analysis_type == 'vtt':
                final_prompt = f"\n\nHuman: 다음은 회의록 분석 결과의 {i}/{len(summary_chunks)} 부분입니다. 이 내용을 요약해주세요:\n\n{summary_chunk}\n\nAssistant:"
            else:
                final_prompt = f"\n\nHuman: 다음은 채팅 분석 결과의 {i}/{len(summary_chunks)} 부분입니다. 이 내용을 요약해주세요:\n\n{summary_chunk}\n\nAssistant:"
            
            completion = call_claude_api(final_prompt)
            final_summaries.append(completion.completion)
            time.sleep(2)
        
        # 최종 통합 요약
        if analysis_type == 'vtt':
            final_integration_prompt = f"\n\nHuman: 다음 요약들을 하나의 완성된 회의록 분석으로 통합해주세요:\n\n{' '.join(final_summaries)}\n\n다음 형식으로 작성해주세요:\n\n# 회의 요약\n- 회의 주제:\n- 주요 참석자:\n- 회의 시간:\n- 핵심 논의 사항:\n- 결정사항:\n- 후속 조치사항:\n\n# 상세 내용\n[시간대별 주요 내용]\n\n# 주요 키워드\n[회의에서 언급된 주요 키워드들]\n\n# 액션 아이템\n[구체적인 할 일과 담당자]\n\n# 추가 참고사항\n[기타 중요한 정보나 맥락]\n\nAssistant:"
        else:
            final_integration_prompt = f"\n\nHuman: 다음 요약들을 하나의 완성된 채팅 분석으로 통합해주세요:\n\n{' '.join(final_summaries)}\n\n다음 형식으로 작성해주세요:\n\n# 채팅 요약\n- 대화 주제:\n- 주요 참여자:\n- 핵심 논의 사항:\n- 결정사항:\n- 후속 조치사항:\n\n# 주요 키워드\n[대화에서 언급된 주요 키워드들]\n\n# 감정/태도 분석\n[대화의 전반적인 톤과 참여자들의 태도]\n\n# 추가 참고사항\n[기타 중요한 정보나 맥락]\n\nAssistant:"
        
        final_completion = call_claude_api(final_integration_prompt)
        return final_completion.completion
        
    except Exception as e:
        logger.error(f"Failed to generate final summary: {str(e)}")
        return "\n".join(all_results)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vtt_analysis')
def vtt_analysis():
    return render_template('vtt_analysis.html')

@app.route('/chat_analysis')
def chat_analysis():
    return render_template('chat_analysis.html')

@app.route('/analyze_vtt', methods=['POST'])
def analyze_vtt():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400
    
    if not file.filename.endswith('.vtt'):
        return jsonify({'error': 'VTT 파일만 업로드 가능합니다.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # VTT 파일 분석
        result = analyze_content_in_chunks(content, 'vtt')
        
        # 분석 결과 저장
        result_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        return jsonify({
            'message': '분석이 완료되었습니다.',
            'result': result,
            'download_url': f'/download/{result_filename}'
        })
        
    except Exception as e:
        logger.error(f"Error processing VTT file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_chat', methods=['POST'])
def analyze_chat():
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': '채팅 내용이 없습니다.'}), 400
        
        content = data['content']
        logger.info("Starting chat analysis with Claude API")
        
        # 채팅 내용 분석
        result = analyze_content_in_chunks(content, 'chat')
        
        # 분석 결과 저장
        result_filename = f"chat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        return jsonify({
            'message': '분석이 완료되었습니다.',
            'result': result,
            'download_url': f'/download/{result_filename}'
        })
        
    except Exception as e:
        logger.error(f"Error processing chat content: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/result/<filename>')
def get_result(filename):
    result_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(result_file):
        return send_file(result_file, as_attachment=True)
    return jsonify({'error': '결과 파일을 찾을 수 없습니다.'}), 404

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 