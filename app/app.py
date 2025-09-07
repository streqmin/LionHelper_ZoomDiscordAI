import os
import logging
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from .enhanced_gpt_client import EnhancedGPTClient
import json
import queue

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'vtt'}  # 허용된 파일 확장자

# 업로드 폴더가 없으면 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# API 클라이언트 초기화
try:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    # 강화된 GPT 클라이언트 (VTT 분석용)
    enhanced_api_client = EnhancedGPTClient(api_key)
    
    if not enhanced_api_client.test_connection():
        raise ConnectionError("강화된 API 클라이언트 연결 테스트 실패")
    
    logger.info("API 클라이언트 초기화 및 연결 테스트 성공")
except Exception as e:
    logger.error(f"API 클라이언트 초기화 실패: {str(e)}")
    raise

# 분석 진행 상황을 저장할 전역 큐
progress_queue = queue.Queue()

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

@app.route('/analysis-progress')
def analysis_progress():
    def generate():
        while True:
            try:
                progress = progress_queue.get(timeout=30)  # 30초 타임아웃
                yield f"data: {json.dumps(progress)}\n\n"
            except queue.Empty:
                break
    return Response(generate(), mimetype='text/event-stream')

@app.route('/analyze_chat', methods=['POST'])
def analyze_chat():
    try:
        logger.info("채팅 분석 요청 수신")
        logger.info(f"요청 URL: {request.url}")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Form data: {request.form}")
        logger.info(f"Files: {request.files}")
        
        # 채팅 파일 확인
        if 'file' not in request.files:
            logger.error("채팅 파일이 요청에 포함되지 않음")
            return jsonify({'error': '채팅 파일이 없습니다'}), 400
            
        chat_file = request.files['file']
        if chat_file.filename == '':
            logger.error("채팅 파일명이 비어있음")
            return jsonify({'error': '채팅 파일이 선택되지 않았습니다'}), 400

        # 채팅 파일 처리
        chat_filename = secure_filename(chat_file.filename)
        chat_filepath = os.path.join(app.config['UPLOAD_FOLDER'], chat_filename)
        chat_file.save(chat_filepath)
        logger.info(f"채팅 파일 저장 완료: {chat_filepath}")
        
        try:
            # 채팅 파일 내용 읽기
            with open(chat_filepath, 'r', encoding='utf-8') as f:
                chat_content = f.read()
            logger.info(f"채팅 파일 내용 읽기 성공 (길이: {len(chat_content)} 문자)")
            
            # API를 통한 분석
            chat_result = enhanced_api_client.analyze_chat(chat_content)
            logger.info("채팅 분석 완료")
            
            # 결과를 HTML 형식으로 변환
            chat_html = format_analysis_result(chat_result, 'chat')
            return jsonify({
                'chat_result': chat_html
            })
            
        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # 임시 파일 삭제
            try:
                os.remove(chat_filepath)
                logger.info("임시 파일 삭제 완료")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {str(e)}")
                
    except Exception as e:
        logger.error(f"요청 처리 중 예상치 못한 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_vtt', methods=['POST'])
def analyze_vtt():
    try:
        logger.info("VTT 분석 요청 수신")
        
        # 파일 처리 및 검증
        if 'vtt_file' not in request.files or 'curriculum_file' not in request.files:
            return jsonify({'error': '필요한 파일이 누락되었습니다.'}), 400
            
        vtt_file = request.files['vtt_file']
        curriculum_file = request.files['curriculum_file']
        
        if vtt_file.filename == '' or curriculum_file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        # 파일 저장
        vtt_filename = secure_filename(vtt_file.filename)
        curriculum_filename = secure_filename(curriculum_file.filename)
        
        vtt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], vtt_filename)
        curriculum_filepath = os.path.join(app.config['UPLOAD_FOLDER'], curriculum_filename)
        
        vtt_file.save(vtt_filepath)
        curriculum_file.save(curriculum_filepath)
        
        # 전처리 결과 저장을 위한 임시 디렉토리 생성
        temp_output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_output')
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            update_progress("VTT 파일 전처리 중...")
            
            # 1. VTT 전처리 파이프라인 실행
            preprocess_result = enhanced_api_client.preprocess_vtt(
                vtt_path=vtt_filepath,
                curriculum_path=curriculum_filepath,
                output_dir=temp_output_dir
            )
            
            update_progress("강의 내용 분석 중...")
            
            # 2. 비주제 블록 제거된 VTT로 강의 내용 요약
            lecture_analysis = enhanced_api_client.analyze_lecture_content(
                segments=preprocess_result['topic_segments']
            )
            
            update_progress("위험 발언 분석 중...")
            
            # 3. 교정된 VTT로 위험 발언 정확한 파악
            risk_analysis = enhanced_api_client.analyze_risk_content(
                segments=preprocess_result['segments']
            )
            
            update_progress("커리큘럼 매칭 분석 중...")
            
            # 4. 토픽 유사도 기반 커리큘럼 매칭
            curriculum_matching = enhanced_api_client.analyze_curriculum_matching(
                segments=preprocess_result['segments'],
                curriculum_path=curriculum_filepath
            )
            
            update_progress("결과 포맷팅 중...")
            
            # 결과를 HTML 형식으로 변환
            lecture_html = format_enhanced_analysis_result(lecture_analysis, 'lecture')
            risk_html = format_enhanced_analysis_result(risk_analysis, 'risk')
            curriculum_html = format_curriculum_matching_result(curriculum_matching)
            
            # 전처리 메트릭 정보 추가
            preprocess_metrics = preprocess_result['metrics']
            
            return jsonify({
                'lecture_result': lecture_html,
                'risk_result': risk_html,
                'curriculum_result': curriculum_html,
                'preprocess_metrics': preprocess_metrics,
                'non_topic_blocks_count': len(preprocess_result['non_topic_blocks']),
                'corrections_count': len(preprocess_result['corrections'])
            })
            
        finally:
            # 임시 파일 삭제
            try:
                os.remove(vtt_filepath)
                os.remove(curriculum_filepath)
                # 전처리 결과 파일들도 삭제
                import shutil
                if os.path.exists(temp_output_dir):
                    shutil.rmtree(temp_output_dir)
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {str(e)}")
                
    except Exception as e:
        logger.error(f"강화된 VTT 분석 중 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_curriculum_file(filepath):
    """커리큘럼 파일(엑셀 또는 JSON)을 처리하여 내용을 반환"""
    ext = filepath.rsplit('.', 1)[1].lower()
    try:
        if ext in ['xlsx', 'xls']:
            import pandas as pd
            
            # 엑셀 파일의 모든 셀 데이터를 읽기
            df = pd.read_excel(filepath, header=None)
            
            # 결과를 저장할 리스트
            result = []
            current_subject = None
            current_details = []
            
            # 첫 번째 행이 헤더인지 확인하고 제외할 키워드 목록
            header_keywords = ['교과목명', '과목명', '교과목', '과목', 'subject']
            
            # 모든 행을 순회하면서 과목명과 세부내용 추출
            for _, row in df.iterrows():
                # 각 행의 모든 셀을 문자열로 변환하고 빈 셀 제거
                row_values = [str(cell).strip() for cell in row if str(cell).strip() != 'nan']
                if not row_values:  # 빈 행 무시
                    continue
                
                # 첫 번째 열이 비어있지 않은 경우, 새로운 과목으로 간주
                first_cell = str(row[0]).strip()
                if first_cell != 'nan' and first_cell:
                    # 헤더로 의심되는 행은 건너뛰기
                    if any(keyword.lower() in first_cell.lower() for keyword in header_keywords):
                        continue
                        
                    # 이전 과목의 정보가 있으면 저장
                    if current_subject and current_details:
                        result.append({
                            '과목명': current_subject,
                            '세부내용': current_details
                        })
                    # 새로운 과목 시작
                    current_subject = first_cell
                    current_details = []
                    # 같은 행에 세부내용이 있는 경우
                    if len(row_values) > 1:
                        current_details.extend(row_values[1:])
                else:
                    # 첫 번째 열이 비어있는 경우, 현재 과목의 세부내용으로 추가
                    if current_subject and row_values:
                        current_details.extend(row_values)
            
            # 마지막 과목 정보 추가
            if current_subject and current_details:
                result.append({
                    '과목명': current_subject,
                    '세부내용': current_details
                })
            
            if not result:
                raise ValueError('엑셀 파일에서 과목명과 세부내용을 추출할 수 없습니다.')
                
            return result
            
        elif ext == 'json':
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON 형식 검증
                if not isinstance(data, list):
                    raise ValueError('JSON 파일은 객체의 배열이어야 합니다.')
                
                result = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    
                    subject = item.get('subject') or item.get('과목명')
                    details = item.get('details') or item.get('세부내용')
                    
                    if subject and details:
                        if isinstance(details, str):
                            details = [details]
                        elif not isinstance(details, list):
                            continue
                            
                        result.append({
                            '과목명': subject,
                            '세부내용': [d for d in details if d]
                        })
                return result
        else:
            raise ValueError('지원하지 않는 파일 형식입니다')
            
    except Exception as e:
        logger.error(f"커리큘럼 파일 처리 중 오류 발생: {str(e)}")
        raise ValueError(f"커리큘럼 파일 처리 중 오류가 발생했습니다: {str(e)}")



def format_vtt_analysis(content):
    """VTT 분석 결과를 HTML 형식으로 변환"""
    logger.info(f"VTT 분석 결과 변환 시작: {content}")
    
    # 섹션을 분리 (--- 구분자 기준)
    sections = content.split('---')
    logger.info(f"VTT 섹션 분할 결과: {sections}")
    
    # HTML 생성
    html_content = ['<div class="analysis-result">']
    
    # 각 섹션의 내용을 저장할 딕셔너리
    vtt_sections = {
        '주요 내용': [],
        '키워드': [],
        '분석': [],
        '위험 발언': []
    }
    
    # 각 섹션 처리
    for section in sections:
        if not section.strip():
            continue
            
        logger.info(f"처리 중인 VTT 섹션: {section}")
        
        # 각 섹션의 내용을 파싱
        lines = section.strip().split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# '):
                current_category = line[2:].strip()  # '#' 제거
                continue
            
            # VTT 분석 결과 처리
            if current_category in vtt_sections:
                if line.startswith('- '):
                    vtt_sections[current_category].append(line[2:].strip())
                else:
                    vtt_sections[current_category].append(line.strip())
    
    # 주요 내용 섹션
    if vtt_sections['주요 내용']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">주요 내용</h2>',
            '    <div class="main-topics">',
            f'        <p>{". ".join(vtt_sections["주요 내용"])}</p>',
            '    </div>',
            '</div>'
        ])
    
    # 키워드 섹션
    if vtt_sections['키워드']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">키워드</h2>',
            '    <div class="main-topics">',
            '        <ul class="keyword-list">'
        ])
        for keyword in vtt_sections['키워드']:
            html_content.append(f'            <li>{keyword}</li>')
        html_content.extend([
            '        </ul>',
            '    </div>',
            '</div>'
        ])
    
    # 분석 섹션
    if vtt_sections['분석']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">분석</h2>',
            '    <div class="main-topics">',
            f'        <p>{". ".join(vtt_sections["분석"])}</p>',
            '    </div>',
            '</div>'
        ])
    
    # 위험 발언 섹션
    has_real_risks = False
    risk_items = []
    
    for risk in vtt_sections['위험 발언']:
        # 위험 발언이 없다는 내용의 텍스트는 제외
        if (risk and 
            not risk.endswith('없습니다.') and 
            not risk.startswith('특별한 주의사항이 없') and
            not '발견되지 않' in risk and
            not '확인되지 않' in risk and
            not '포함되어 있지 않' in risk and
            not '위험한 내용이 없' in risk and
            not '특별한 위험' in risk and
            not '부적절한 내용이 없' in risk):
            risk_items.append(risk)
            has_real_risks = True
    
    html_content.extend([
        '<div class="category-section risk-section' + (' has-risks' if has_real_risks else ' no-risks') + '">',
        '    <h2 class="category-title">위험 발언</h2>',
        '    <div class="risk-summary">',
        '        <div class="risk-icon">' + ('⚠️' if has_real_risks else '✅') + '</div>',
        '        <p>' + ('다음과 같은 위험 발언이 감지되었습니다.' if has_real_risks else '위험 발언이 감지되지 않았습니다.') + '</p>',
        '    </div>'
    ])
    
    if has_real_risks:
        html_content.extend([
            '    <ul class="risk-list">'
        ])
        for risk in risk_items:
            html_content.append(f'        <li>{risk}</li>')
        html_content.append('    </ul>')
    
    html_content.append('</div>')
    html_content.append('</div>')
    return '\n'.join(html_content)

def format_chat_analysis(content):
    """채팅 분석 결과를 HTML 형식으로 변환"""
    logger.info(f"채팅 분석 결과 변환 시작: {content}")
    
    # 섹션을 분리 (--- 구분자 기준)
    sections = content.split('---')
    logger.info(f"채팅 섹션 분할 결과: {sections}")
    
    # HTML 생성
    html_content = ['<div class="analysis-result">']
    
    # 카테고리별로 내용을 저장할 딕셔너리
    categories = {
        '주요 대화 주제': [],
        '수강생 감정/태도 분석': {
            '1. 긍정적 반응': [],
            '2. 부정적 반응': [],
            '3. 질문/요청사항': []
        },
        '어려움/불만 상세 분석': {
            '1. 학습적 어려움': [],
            '2. 수업 진행 관련 문제': [],
            '3. 기술적 문제': []
        },
        '개선 제안': {
            '1. 학습 내용 개선': [],
            '2. 수업 방식 개선': [],
            '3. 기술적 지원 강화': []
        },
        '위험 발언 및 주의사항': [],
        '종합 제언': []
    }
    
    # 모든 섹션의 내용을 카테고리별로 분류
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        current_category = None
        current_subcategory = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# '):
                current_category = line[2:].strip()  # '#' 제거
                current_subcategory = None
                continue
                
            # 하위 카테고리가 있는 섹션 처리
            if current_category in ['수강생 감정/태도 분석', '어려움/불만 상세 분석', '개선 제안']:
                subcategories = categories[current_category]
                for subcat in subcategories.keys():
                    if line.startswith(subcat):
                        current_subcategory = subcat
                        break
                if current_subcategory and line.startswith('- '):
                    # 중복 제거를 위해 이미 있는 항목은 추가하지 않음
                    if line not in categories[current_category][current_subcategory]:
                        categories[current_category][current_subcategory].append(line)
                continue
            
            # 다른 카테고리 처리
            if current_category in categories and isinstance(categories[current_category], list):
                if line not in categories[current_category]:  # 중복 제거
                    categories[current_category].append(line)
    
    # 주요 대화 주제 섹션
    if categories['주요 대화 주제']:
        # 주요 대화 주제를 하나의 문단으로 합치기
        main_topics = []
        for item in categories['주요 대화 주제']:
            if item.startswith('- '):
                main_topics.append(item[2:].strip())
            else:
                main_topics.append(item.strip())
        
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">주요 대화 주제</h2>',
            '    <div class="main-topics">',
            f'        <p>{". ".join(main_topics)}</p>',
            '    </div>',
            '</div>'
        ])
    
    # 수강생 감정/태도 분석 섹션
    if any(categories['수강생 감정/태도 분석'].values()):
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">수강생 감정/태도 분석</h2>'
        ])
        
        for subcategory, items in categories['수강생 감정/태도 분석'].items():
            if items:  # 해당 하위 카테고리에 내용이 있는 경우에만 표시
                html_content.extend([
                    f'    <div class="subsection">',
                    f'        <h3 class="subsection-title">{subcategory}</h3>',
                    f'        <ul class="analysis-list">'
                ])
                for item in items:
                    html_content.append(f'            <li>{item[2:]}</li>')  # '- ' 제거
                html_content.extend([
                    '        </ul>',
                    '    </div>'
                ])
        
        html_content.append('</div>')
    
    # 어려움/불만 상세 분석 섹션
    if any(categories['어려움/불만 상세 분석'].values()):
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">어려움/불만 상세 분석</h2>'
        ])
        
        for subcategory, items in categories['어려움/불만 상세 분석'].items():
            if items:  # 해당 하위 카테고리에 내용이 있는 경우에만 표시
                html_content.extend([
                    f'    <div class="subsection">',
                    f'        <h3 class="subsection-title">{subcategory}</h3>',
                    f'        <ul class="analysis-list">'
                ])
                for item in items:
                    html_content.append(f'            <li>{item[2:]}</li>')  # '- ' 제거
                html_content.extend([
                    '        </ul>',
                    '    </div>'
                ])
        
        html_content.append('</div>')
    
    # 개선 제안 섹션
    if any(categories['개선 제안'].values()):
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">개선 제안</h2>'
        ])
        
        for subcategory, items in categories['개선 제안'].items():
            if items:  # 해당 하위 카테고리에 내용이 있는 경우에만 표시
                html_content.extend([
                    f'    <div class="subsection">',
                    f'        <h3 class="subsection-title">{subcategory}</h3>',
                    f'        <ul class="analysis-list">'
                ])
                for item in items:
                    html_content.append(f'            <li>{item[2:]}</li>')  # '- ' 제거
                html_content.extend([
                    '        </ul>',
                    '    </div>'
                ])
        
        html_content.append('</div>')
    
    # 위험 발언 및 주의사항 섹션
    risk_items = []
    has_real_risks = False
    
    for item in categories['위험 발언 및 주의사항']:
        item = item.strip()
        # 위험 발언이 없다는 내용의 텍스트는 제외
        if (item and 
            not item.endswith('없습니다.') and 
            not item.startswith('특별한 주의사항이 없') and
            not '발견되지 않' in item and
            not '확인되지 않' in item and
            not '포함되어 있지 않' in item and
            not '위험한 내용이 없' in item and
            not '특별한 위험' in item and
            not '부적절한 내용이 없' in item):
            risk_items.append(item)
            has_real_risks = True
    
    if has_real_risks and risk_items:  # 실제 위험 발언이 있는 경우에만
        html_content.extend([
            '<div class="category-section risk-section">',
            '    <h2 class="category-title">위험 발언 및 주의사항</h2>',
            '    <div class="risk-summary">',
            '        <div class="risk-icon">⚠️</div>',
            '        <p>채팅에서 다음과 같은 위험 발언이 감지되었습니다.</p>',
            '    </div>',
            '    <ul class="risk-list">'
        ])
        for item in risk_items:
            html_content.append(f'        <li>{item}</li>')
        html_content.extend([
            '    </ul>',
            '</div>'
        ])
    elif any(categories.values()):  # 다른 카테고리에 내용이 있는 경우에만
        # 위험 발언이 없는 경우
        html_content.extend([
            '<div class="category-section risk-section safe">',
            '    <h2 class="category-title">위험 발언 및 주의사항</h2>',
            '    <div class="risk-summary">',
            '        <div class="risk-icon">✅</div>',
            '        <p>채팅에서 특별한 위험 발언이 감지되지 않았습니다.</p>',
            '    </div>',
            '</div>'
        ])
    
    # 종합 제언 섹션
    if categories['종합 제언']:
        # 종합 제언을 하나의 문단으로 합치기
        recommendations = []
        for item in categories['종합 제언']:
            if item.startswith('- '):
                recommendations.append(item[2:].strip())
            else:
                recommendations.append(item.strip())
        
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">종합 제언</h2>',
            '    <div class="main-topics">',
            f'        <p>{". ".join(recommendations)}</p>',
            '    </div>',
            '</div>'
        ])
    
    html_content.append('</div>')
    return '\n'.join(html_content)

def format_analysis_result(content, analysis_type='chat'):
    """분석 결과를 HTML 형식으로 변환"""
    if analysis_type == 'vtt':
        return format_vtt_analysis(content)
    else:
        return format_chat_analysis(content)

def format_enhanced_analysis_result(content, analysis_type='lecture'):
    """강화된 분석 결과를 HTML 형식으로 변환"""
    if analysis_type == 'lecture':
        return format_lecture_analysis(content)
    elif analysis_type == 'risk':
        return format_risk_analysis(content)
    else:
        return format_chat_analysis(content)

def format_lecture_analysis(content):
    """강의 내용 분석 결과를 HTML 형식으로 변환"""
    logger.info(f"강의 분석 결과 변환 시작: {content}")
    
    # content가 딕셔너리인 경우 처리 (enhanced_gpt_client.py에서 반환하는 구조)
    if isinstance(content, dict):
        if 'error' in content:
            return f'<div class="error-message"><p>{content["error"]}</p></div>'
        
        # 딕셔너리 구조에서 lecture_results 추출
        lecture_results = content.get('lecture_results', [])
        total_chunks = content.get('total_chunks', 0)
        
        html_content = ['<div class="analysis-result">']
        
        # 각 청크별 결과 처리
        for i, result in enumerate(lecture_results, 1):
            chunk_id = result.get('chunk_id', i)
            analysis = result.get('analysis', '')
            segments = result.get('segments', [])
            chunk_time_range = result.get('chunk_time_range', '')
            
            # 분석 결과를 파싱 (GPT가 반환한 문자열을 파싱)
            sections = analysis.split('---') if analysis else []
            lecture_sections = {
                '주요 내용': [],
                '키워드': [],
                '분석': [],
                '학습 포인트': []
            }
            
            # 각 섹션 처리
            for section in sections:
                if not section.strip():
                    continue
                    
                lines = section.strip().split('\n')
                current_category = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith('# '):
                        current_category = line[2:].strip()
                        continue
                    
                    if current_category in lecture_sections:
                        if line.startswith('- '):
                            lecture_sections[current_category].append(line[2:].strip())
                        else:
                            lecture_sections[current_category].append(line.strip())
            
            # 청크별 결과 HTML 생성
            html_content.extend([
                f'<div class="category-section">',
                f'    <h2 class="category-title">청크 {chunk_id}'
            ])
            
            # 시간대 정보 추가
            if chunk_time_range:
                html_content.append(f'        <span class="time-range">({chunk_time_range})</span>')
            
            # 키워드
            if lecture_sections['키워드']:
                html_content.extend([
                    '    <div class="keyword-section">',
                    '        <h3 class="keyword-title">키워드</h3>',
                    '        <ul class="keyword-list">'
                ])
                for keyword in lecture_sections['키워드']:
                    html_content.append(f'            <li>{keyword}</li>')
                html_content.extend([
                    '        </ul>',
                    '    </div>'
                ])
            
            html_content.append('    </h2>')
            
            # 주요 내용
            if lecture_sections['주요 내용']:
                html_content.extend([
                    '    <div class="subsection">',
                    '        <h3 class="subsection-title">주요 내용</h3>',
                    '        <div class="main-topics">',
                    f'        <p>{". ".join(lecture_sections["주요 내용"])}</p>',
                    '        </div>',
                    '    </div>'
                ])
            
            # 분석
            if lecture_sections['분석']:
                html_content.extend([
                    '    <div class="subsection">',
                    '        <h3 class="subsection-title">분석</h3>',
                    '        <div class="main-topics">',
                    f'        <p>{". ".join(lecture_sections["분석"])}</p>',
                    '        </div>',
                    '    </div>'
                ])
            
            # 학습 포인트
            if lecture_sections['학습 포인트']:
                html_content.extend([
                    '    <div class="subsection">',
                    '        <h3 class="subsection-title">학습 포인트</h3>',
                    '        <div class="main-topics">',
                    '        <ul class="learning-points">'
                ])
                for point in lecture_sections['학습 포인트']:
                    html_content.append(f'            <li>{point}</li>')
                html_content.extend([
                    '        </ul>',
                    '        </div>',
                    '    </div>'
                ])
            
            # 세그먼트 정보
            if segments:
                html_content.extend([
                    '    <div class="subsection">',
                    '        <h3 class="subsection-title">시간 정보</h3>',
                    '        <div class="main-topics">',
                    '        <ul class="time-info">'
                ])
                for segment in segments[:5]:  # 최대 5개만 표시
                    time_range = segment.get('time_range', '')
                    text = segment.get('text', '')[:100] + '...' if len(segment.get('text', '')) > 100 else segment.get('text', '')
                    html_content.append(f'            <li><strong>{time_range}:</strong> {text}</li>')
                html_content.extend([
                    '        </ul>',
                    '        </div>',
                    '    </div>'
                ])
            
            html_content.append('</div>')
        
        html_content.append('</div>')
        return '\n'.join(html_content)
    
    # 기존 문자열 처리 로직 (하위 호환성)
    sections = content.split('---')
    logger.info(f"강의 섹션 분할 결과: {sections}")
    
    # HTML 생성
    html_content = ['<div class="analysis-result">']
    
    # 각 섹션의 내용을 저장할 딕셔너리
    lecture_sections = {
        '주요 내용': [],
        '키워드': [],
        '분석': [],
        '학습 포인트': []
    }
    
    # 각 섹션 처리
    for section in sections:
        if not section.strip():
            continue
            
        logger.info(f"처리 중인 강의 섹션: {section}")
        
        # 각 섹션의 내용을 파싱
        lines = section.strip().split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# '):
                current_category = line[2:].strip()  # '#' 제거
                continue
            
            # 강의 분석 결과 처리
            if current_category in lecture_sections:
                if line.startswith('- '):
                    lecture_sections[current_category].append(line[2:].strip())
                else:
                    lecture_sections[current_category].append(line.strip())
    
    # 주요 내용 섹션
    if lecture_sections['주요 내용']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">주요 내용</h2>',
            '    <div class="main-topics">',
            f'        <p>{". ".join(lecture_sections["주요 내용"])}</p>',
            '    </div>',
            '</div>'
        ])
    
    # 키워드 섹션
    if lecture_sections['키워드']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">키워드</h2>',
            '    <div class="main-topics">',
            '        <ul class="keyword-list">'
        ])
        for keyword in lecture_sections['키워드']:
            html_content.append(f'            <li>{keyword}</li>')
        html_content.extend([
            '        </ul>',
            '    </div>',
            '</div>'
        ])
    
    # 분석 섹션
    if lecture_sections['분석']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">분석</h2>',
            '    <div class="main-topics">',
            f'        <p>{". ".join(lecture_sections["분석"])}</p>',
            '    </div>',
            '</div>'
        ])
    
    # 학습 포인트 섹션
    if lecture_sections['학습 포인트']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">학습 포인트</h2>',
            '    <div class="main-topics">',
            '        <ul class="learning-points">'
        ])
        for point in lecture_sections['학습 포인트']:
            html_content.append(f'            <li>{point}</li>')
        html_content.extend([
            '        </ul>',
            '    </div>',
            '</div>'
        ])
    
    html_content.append('</div>')
    return '\n'.join(html_content)

def format_risk_analysis(content):
    """위험 발언 분석 결과를 HTML 형식으로 변환"""
    logger.info(f"위험 발언 분석 결과 변환 시작: {content}")
    
    # content가 딕셔너리인 경우 처리 (enhanced_gpt_client.py에서 반환하는 구조)
    if isinstance(content, dict):
        if 'error' in content:
            return f'<div class="error-message"><p>{content["error"]}</p></div>'
        
        # 딕셔너리 구조에서 risk_results 추출
        risk_results = content.get('risk_results', [])
        total_chunks = content.get('total_chunks', 0)
        risk_found = content.get('risk_found', False)
        summary = content.get('summary', '')
        
        html_content = ['<div class="analysis-result">']
        
        if summary:
            html_content.extend([
                '<div class="category-section">',
                '    <h2 class="category-title">분석 요약</h2>',
                '    <div class="main-topics">',
                f'        <p>{summary}</p>',
                '    </div>',
                '</div>'
            ])
        
        # 위험 발언이 발견된 청크만 처리
        if not risk_results:
            # 위험 발언이 전혀 없는 경우
            html_content.extend([
                '<div class="category-section">',
                '    <h2 class="category-title">위험 발언 분석 결과</h2>',
                '    <div class="main-topics">',
                '        <p>전체 강의 내용에서 위험 발언이 발견되지 않았습니다.</p>',
                '    </div>',
                '</div>'
            ])
        else:
            # 각 청크별 결과 처리 (위험 발언이 있는 청크만)
            for i, result in enumerate(risk_results, 1):
                chunk_id = result.get('chunk_id', i)
                chunk_text = result.get('chunk_text', '')
                risk_analysis = result.get('risk_analysis', {})
                affected_segments = result.get('affected_segments', [])
                chunk_time_range = result.get('chunk_time_range', '')
                
                # 위험 발언 분석 결과 처리
                has_risk = risk_analysis.get('has_risk', False)
                
                print(f"chunk_id: {chunk_id}, has_risk: {has_risk}")
                # has_risk가 false인 경우 해당 청크는 HTML 생성하지 않음
                if not has_risk:
                    continue
                
                risk_types = risk_analysis.get('risk_types', [])
                risk_texts = risk_analysis.get('risk_texts', [])
                risk_analysis_text = risk_analysis.get('risk_analysis', '')
                precautions = risk_analysis.get('precautions', [])
                improvement_suggestions = risk_analysis.get('improvement_suggestions', [])
                
                # 청크별 결과 HTML 생성
                html_content.extend([
                    f'<div class="category-section">',
                    f'    <h2 class="category-title">청크 {chunk_id}'
                ])
                
                # 시간대 정보 추가
                if chunk_time_range:
                    html_content.append(f'        <span class="time-range">({chunk_time_range})</span>')
                
                html_content.append('    </h2>')
                
                # 위험 발언 상태
                html_content.extend([
                    '    <div class="subsection">',
                    '        <h3 class="subsection-title">위험 발언 상태</h3>',
                    '        <div class="main-topics">',
                    f'        <p>위험 발언 발견: 예</p>',
                    '    </div>',
                    '    </div>'
                ])
                
                # 위험 유형
                if risk_types:
                    html_content.extend([
                        '    <div class="subsection">',
                        '        <h3 class="subsection-title">위험 유형</h3>',
                        '        <div class="main-topics">',
                        '        <ul class="risk-types">'
                    ])
                    for risk_type in risk_types:
                        html_content.append(f'            <li>{risk_type}</li>')
                    html_content.extend([
                        '        </ul>',
                        '        </div>',
                        '    </div>'
                    ])
                
                # 위험 발언 텍스트
                if risk_texts:
                    html_content.extend([
                        '    <div class="subsection">',
                        '        <h3 class="subsection-title">위험 발언 텍스트</h3>',
                        '        <div class="main-topics">',
                        '        <ul class="risk-texts">'
                    ])
                    for risk_text in risk_texts:
                        html_content.append(f'            <li>{risk_text}</li>')
                    html_content.extend([
                        '        </ul>',
                        '        </div>',
                        '    </div>'
                    ])
                
                # 위험 분석
                if risk_analysis_text:
                    html_content.extend([
                        '    <div class="subsection">',
                        '        <h3 class="subsection-title">위험 분석</h3>',
                        '        <div class="main-topics">',
                        f'        <p>{risk_analysis_text}</p>',
                        '        </div>',
                        '    </div>'
                    ])
                
                # 주의사항
                if precautions:
                    html_content.extend([
                        '    <div class="subsection">',
                        '        <h3 class="subsection-title">주의사항</h3>',
                        '        <div class="main-topics">',
                        '        <ul class="precautions">'
                    ])
                    for precaution in precautions:
                        html_content.append(f'            <li>{precaution}</li>')
                    html_content.extend([
                        '        </ul>',
                        '        </div>',
                        '    </div>'
                    ])
                
                # 개선 제안
                if improvement_suggestions:
                    html_content.extend([
                        '    <div class="subsection">',
                        '        <h3 class="subsection-title">개선 제안</h3>',
                        '        <div class="main-topics">',
                        '        <ul class="improvement-suggestions">'
                    ])
                    for suggestion in improvement_suggestions:
                        html_content.append(f'            <li>{suggestion}</li>')
                    html_content.extend([
                        '        </ul>',
                        '        </div>',
                        '    </div>'
                    ])
                
                # 영향받은 세그먼트
                if affected_segments:
                    html_content.extend([
                        '    <div class="subsection">',
                        '        <h3 class="subsection-title">영향받은 세그먼트</h3>',
                        '        <div class="main-topics">',
                        '        <ul class="affected-segments">'
                    ])
                    for segment in affected_segments:
                        time_range = segment.get('time_range', '')
                        text = segment.get('text', '')[:100] + '...' if len(segment.get('text', '')) > 100 else segment.get('text', '')
                        matched_risks = segment.get('matched_risks', [])
                        html_content.append(f'            <li><strong>{time_range}:</strong> {text}')
                        if matched_risks:
                            html_content.append('                <ul>')
                            for risk in matched_risks:
                                risk_text = risk.get('risk_text', '')
                                confidence = risk.get('confidence', 0)
                                html_content.append(f'                    <li>위험: {risk_text} (신뢰도: {confidence:.2f})</li>')
                            html_content.append('                </ul>')
                        html_content.append('            </li>')
                    html_content.extend([
                        '        </ul>',
                        '        </div>',
                        '    </div>'
                    ])
                
                html_content.append('</div>')
        
        html_content.append('</div>')
        return '\n'.join(html_content)
    
    # 기존 문자열 처리 로직 (하위 호환성)
    sections = content.split('---')
    logger.info(f"위험 발언 섹션 분할 결과: {sections}")
    
    # HTML 생성
    html_content = ['<div class="analysis-result">']
    
    # 각 섹션의 내용을 저장할 딕셔너리
    risk_sections = {
        '위험 발언 분석': [],
        '주의사항': [],
        '개선 제안': []
    }
    
    # 각 섹션 처리
    for section in sections:
        if not section.strip():
            continue
            
        logger.info(f"처리 중인 위험 발언 섹션: {section}")
        
        # 각 섹션의 내용을 파싱
        lines = section.strip().split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# '):
                current_category = line[2:].strip()  # '#' 제거
                continue
            
            # 위험 발언 분석 결과 처리
            if current_category in risk_sections:
                if line.startswith('- '):
                    risk_sections[current_category].append(line[2:].strip())
                else:
                    risk_sections[current_category].append(line.strip())
    
    # 위험 발언 분석 섹션
    has_real_risks = False
    risk_items = []
    
    for risk in risk_sections['위험 발언 분석']:
        # 위험 발언이 없다는 내용의 텍스트는 제외
        if (risk and 
            not risk.endswith('없습니다.') and 
            not risk.startswith('특별한 주의사항이 없') and
            not '발견되지 않' in risk and
            not '확인되지 않' in risk and
            not '포함되어 있지 않' in risk and
            not '위험한 내용이 없' in risk and
            not '특별한 위험' in risk and
            not '부적절한 내용이 없' in risk):
            risk_items.append(risk)
            has_real_risks = True
    
    html_content.extend([
        '<div class="category-section risk-section' + (' has-risks' if has_real_risks else ' no-risks') + '">',
        '    <h2 class="category-title">위험 발언 분석</h2>',
        '    <div class="risk-summary">',
        '        <div class="risk-icon">' + ('⚠️' if has_real_risks else '✅') + '</div>',
        '        <p>' + ('다음과 같은 위험 발언이 감지되었습니다.' if has_real_risks else '위험 발언이 감지되지 않았습니다.') + '</p>',
        '    </div>'
    ])
    
    if has_real_risks:
        html_content.extend([
            '    <ul class="risk-list">'
        ])
        for risk in risk_items:
            html_content.append(f'        <li>{risk}</li>')
        html_content.append('    </ul>')
    
    html_content.append('</div>')
    
    # 주의사항 섹션
    if risk_sections['주의사항']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">주의사항</h2>',
            '    <div class="main-topics">',
            '        <ul class="warning-list">'
        ])
        for warning in risk_sections['주의사항']:
            html_content.append(f'            <li>{warning}</li>')
        html_content.extend([
            '        </ul>',
            '    </div>',
            '</div>'
        ])
    
    # 개선 제안 섹션
    if risk_sections['개선 제안']:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">개선 제안</h2>',
            '    <div class="main-topics">',
            '        <ul class="improvement-list">'
        ])
        for improvement in risk_sections['개선 제안']:
            html_content.append(f'            <li>{improvement}</li>')
        html_content.extend([
            '        </ul>',
            '    </div>',
            '</div>'
        ])
    
    html_content.append('</div>')
    return '\n'.join(html_content)

def format_curriculum_matching_result(curriculum_matching):
    """커리큘럼 매칭 결과를 HTML 형식으로 변환"""
    if 'error' in curriculum_matching:
        return f'<div class="error-message"><p>{curriculum_matching["error"]}</p></div>'
    
    html_content = ['<div class="curriculum-matching-result">']
    
    # 토픽별 통계
    topic_stats = curriculum_matching.get('topic_stats', {})
    if topic_stats:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">토픽별 커버리지</h2>',
            '    <div class="topic-stats">'
        ])
        
        for topic, stats in topic_stats.items():
            coverage = stats.get('coverage_percentage', 0)
            avg_score = stats.get('average_score', 0)
            segment_count = stats.get('segment_count', 0)
            
            html_content.extend([
                f'        <div class="topic-item">',
                f'            <div class="topic-name">{topic}</div>',
                f'            <div class="topic-metrics">',
                f'                <span class="coverage">커버리지: {coverage:.1f}%</span>',
                f'                <span class="score">평균 점수: {avg_score:.2f}</span>',
                f'                <span class="segments">세그먼트: {segment_count}개</span>',
                f'            </div>',
                f'        </div>'
            ])
        
        html_content.extend([
            '    </div>',
            '</div>'
        ])
    
    # 상세 매칭 결과
    segment_scores = curriculum_matching.get('segment_scores', [])
    if segment_scores:
        html_content.extend([
            '<div class="category-section">',
            '    <h2 class="category-title">세그먼트별 매칭 결과</h2>',
            '    <div class="segment-matching">',
            '        <table class="matching-table">',
            '            <thead>',
            '                <tr>',
            '                    <th>시간</th>',
            '                    <th>토픽</th>',
            '                    <th>점수</th>',
            '                    <th>내용 미리보기</th>',
            '                </tr>',
            '            </thead>',
            '            <tbody>'
        ])
        
        for seg_score in segment_scores[:20]:  # 상위 20개만 표시
            start_time = f"{int(seg_score['start_sec']//60):02d}:{int(seg_score['start_sec']%60):02d}"
            topic = seg_score.get('best_topic', '기타')
            score = seg_score.get('best_score', 0)
            preview = seg_score.get('text', '')[:100] + '...' if len(seg_score.get('text', '')) > 100 else seg_score.get('text', '')
            
            html_content.extend([
                '                <tr>',
                f'                    <td>{start_time}</td>',
                f'                    <td>{topic}</td>',
                f'                    <td>{score:.2f}</td>',
                f'                    <td>{preview}</td>',
                '                </tr>'
            ])
        
        html_content.extend([
            '            </tbody>',
            '        </table>',
            '    </div>',
            '</div>'
        ])
    
    html_content.append('</div>')
    return '\n'.join(html_content)

def format_list_items(content):
    """목록 항목을 HTML 형식으로 변환"""
    items = []
    for line in content.split(chr(10)):  # chr(10)은 '\n'과 동일
        line = line.strip()
        if line.startswith('- '):
            items.append(f'<li>{line[2:].strip()}</li>')
        elif line.startswith('• '):
            items.append(f'<li>{line[2:].strip()}</li>')
        elif line:  # 일반 텍스트인 경우
            items.append(f'<li>{line}</li>')
    return '\n'.join(items)

def update_progress(message):
    """분석 진행 상황을 큐에 추가"""
    progress_queue.put({'message': message})

def split_vtt_content(content, chunk_size=5000):
    """VTT 내용을 청크로 분할"""
    words = content.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # 공백 포함
        if current_size + word_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def combine_analysis_results(results):
    """여러 청크의 분석 결과를 하나로 통합"""
    combined = {
        '주요 내용': [],
        '키워드': set(),
        '분석': [],
        '위험 발언': []
    }
    
    for result in results:
        sections = result.split('---')
        current_category = None
        
        for section in sections:
            lines = section.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('# '):
                    current_category = line[2:].strip()
                    continue
                if line and current_category in combined:
                    if current_category == '키워드':
                        combined[current_category].update(line.split(', '))
                    else:
                        combined[current_category].append(line)
    
    # 키워드를 리스트로 변환하고 정렬
    combined['키워드'] = sorted(list(combined['키워드']))
    
    # 결과를 문자열로 변환
    return '\n---\n'.join([
        f"# {category}\n" + '\n'.join(items if isinstance(items, list) else [items])
        for category, items in combined.items()
    ])

if __name__ == '__main__':
    app.run(debug=True) 