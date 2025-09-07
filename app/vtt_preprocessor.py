# -*- coding: utf-8 -*-
"""
vtt_preprocessor: VTT 파일 전처리 전용 모듈

VTT 전처리 파이프라인을 실행하고 결과를 반환하는 기능을 제공합니다.
"""
import os
import logging
from typing import Dict, Any

import sys
import os

# 현재 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from vtt_preprocess.pipeline import run_preprocess
logger = logging.getLogger(__name__)


class VTTPreprocessor:
    """VTT 파일 전처리 전용 클래스"""
    
    def __init__(self):
        """VTT 전처리기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("VTTPreprocessor 초기화 완료")

    def preprocess_vtt(self, vtt_path: str, curriculum_path: str, output_dir: str) -> Dict[str, Any]:
        """VTT 파일 전처리 실행"""
        try:
            self.logger.info(f"VTT 전처리 시작: {vtt_path}")
            
            # 출력 파일 경로 설정
            out_vtt_path = os.path.join(output_dir, "preprocessed.vtt")
            out_blocks_path = os.path.join(output_dir, "non_topic_blocks.csv")
            out_corrections_csv = os.path.join(output_dir, "corrections.csv")
            out_metrics_json = os.path.join(output_dir, "metrics.json")
            out_report_md = os.path.join(output_dir, "report.md")
            
            # 전처리 파이프라인 실행
            outputs = run_preprocess(
                vtt_path=vtt_path,
                curriculum_xlsx_path=curriculum_path,
                out_vtt_path=out_vtt_path,
                out_blocks_path=out_blocks_path,
                out_corrections_csv=out_corrections_csv,
                out_metrics_json=out_metrics_json,
                out_report_md=out_report_md,
                llm_model="gpt-4o-mini",
                remove_non_topic_from_output=False
            )
            
            self.logger.info("VTT 전처리 완료")
            
            return {
                'segments': outputs.segments,
                'non_topic_blocks': outputs.non_topic_blocks,
                'corrections': outputs.corrections,
                'topic_segments': outputs.topic_segments,
                'metrics': outputs.metrics,
                'output_files': {
                    'vtt': out_vtt_path,
                    'blocks': out_blocks_path,
                    'corrections': out_corrections_csv,
                    'metrics': out_metrics_json,
                    'report': out_report_md
                }
            }
            
        except Exception as e:
            self.logger.error(f"VTT 전처리 실패: {str(e)}")
            raise


# 편의를 위한 함수형 인터페이스
def preprocess_vtt(vtt_path: str, curriculum_path: str, output_dir: str) -> Dict[str, Any]:
    """VTT 파일 전처리 실행 (함수형 인터페이스)"""
    preprocessor = VTTPreprocessor()
    return preprocessor.preprocess_vtt(vtt_path, curriculum_path, output_dir)
