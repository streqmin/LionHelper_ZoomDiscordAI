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
    timeout=httpx.Timeout(60.0, connect=30.0),  # 타임아웃 증가
    limits=httpx.Limits(max_keepalive_connections=3, max_connections=6),  # 연결 수 제한
    verify=True,
    http2=True  # HTTP/2 활성화
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
    stop=stop_after_attempt(5),  # 재시도 횟수 증가
    wait=wait_exponential(multiplier=2, min=10, max=30)  # 대기 시간 증가
)
def call_claude_api(prompt):
    """Claude API 호출 함수 with 재시도 로직"""
    try:
        completion = client.completions.create(
            prompt=prompt,
            model="claude-instant-1.2",
            max_tokens_to_sample=1500,
            stop_sequences=["\n\nHuman:"],
            temperature=0.7
        )
        return completion
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        time.sleep(5)  # 오류 발생 시 추가 대기
        raise

def analyze_content_in_chunks(content, analysis_type='vtt'):
    """청크 단위로 콘텐츠 분석"""
    chunks = split_content(content)
    total_chunks = len(chunks)
    logger.info(f"Split content into {total_chunks} chunks")
    
    all_results = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{total_chunks}")
        
        try:
            for attempt in range(3):  # 청크별 최대 3번 시도
                try:
                    if analysis_type == 'vtt':
                        prompt = f"\n\nHuman: 당신은 줌 회의록 분석 전문가입니다. 다음은 전체 회의록의 {i}/{total_chunks} 부분입니다. 이 부분을 분석해주세요:\n\n{chunk}\n\n다음 형식으로 분석 결과를 제공해주세요:\n\n# 이 부분의 주요 내용\n[핵심 내용 요약]\n\n# 주요 키워드\n[이 부분의 주요 키워드들]\n\n# 중요 포인트\n[이 부분에서 특별히 주목할 만한 내용]\n\nAssistant:"
                    else:
                        prompt = f"\n\nHuman: 당신은 채팅 로그 분석 전문가입니다. 다음은 전체 채팅의 {i}/{total_chunks} 부분입니다. 이 부분을 분석해주세요:\n\n{chunk}\n\n다음 형식으로 분석 결과를 제공해주세요:\n\n# 이 부분의 주요 내용\n[핵심 내용 요약]\n\n# 주요 키워드\n[이 부분의 주요 키워드들]\n\n# 대화 분위기\n[이 부분의 대화 톤과 분위기]\n\nAssistant:"
                    
                    completion = call_claude_api(prompt)
                    all_results.append(completion.completion)
                    time.sleep(5)  # API 호출 간 간격 증가
                    break
                except Exception as e:
                    if attempt == 2:  # 마지막 시도에서 실패
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed for chunk {i}: {str(e)}")
                    time.sleep(10)  # 재시도 전 대기
        except Exception as e:
            logger.error(f"Failed to process chunk {i}: {str(e)}")
            all_results.append(f"[이 부분 처리 중 오류 발생: {str(e)}]")
            time.sleep(15)  # 오류 발생 시 더 긴 대기 시간
    
    if not all_results:
        return "분석에 실패했습니다. 네트워크 연결을 확인해주세요."
    
    return "\n\n".join(all_results)

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
        content = None
        logger.info("Received chat analysis request")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Form data: {request.form}")
        logger.info(f"Files: {request.files}")
        
        # JSON 형식 처리
        if request.is_json:
            logger.info("Processing JSON request")
            data = request.get_json()
            logger.info(f"Received JSON data: {data}")
            if data and 'content' in data:
                content = data['content']
                logger.info("Successfully extracted content from JSON")
        
        # form-data 형식 처리
        elif 'content' in request.form:
            logger.info("Processing form-data request")
            content = request.form['content']
            logger.info("Successfully extracted content from form")
        
        # 파일 업로드 처리
        elif 'file' in request.files:
            logger.info("Processing file upload")
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"File saved to {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.info("Successfully read file content")
                except Exception as e:
                    logger.error(f"Error reading file: {str(e)}")
                    return jsonify({'error': f'파일 읽기 오류: {str(e)}'}), 400
        
        if not content:
            logger.error("No content found in request")
            return jsonify({'error': '채팅 내용이 없습니다. content 필드를 확인해주세요.'}), 400
        
        if not isinstance(content, str):
            logger.error(f"Invalid content type: {type(content)}")
            return jsonify({'error': '채팅 내용은 문자열이어야 합니다.'}), 400
        
        if len(content.strip()) == 0:
            logger.error("Empty content received")
            return jsonify({'error': '채팅 내용이 비어있습니다.'}), 400
        
        logger.info("Starting chat analysis with Claude API")
        logger.info(f"Content length: {len(content)} characters")
        
        # 채팅 내용 분석
        result = analyze_content_in_chunks(content, 'chat')
        
        # 분석 결과 저장
        result_filename = f"chat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        logger.info("Analysis completed successfully")
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