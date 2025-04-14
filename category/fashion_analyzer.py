import os
import base64
from openai import OpenAI
from typing import Dict, Any, List
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fashion-analyzer")

class FashionAnalyzer:
    def __init__(self):
        """OpenAI API 키를 환경 변수에서 가져와 초기화"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # 의류 카테고리 정의
        self.categories = {
            'top': ['티셔츠', '셔츠', '블라우스', '니트웨어', '후드', '스웨터'],
            'bottom': ['청바지', '슬랙스', '스커트', '반바지', '레깅스'],
            'outer': ['자켓', '코트', '패딩', '가디건', '블레이저'],
            'dress': ['원피스', '점프수트'],
            'etc': ['액세서리', '신발', '가방']
        }

    def _encode_image(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_fashion(self, image_path: str) -> Dict[str, Any]:
        """의류 이미지 분석
        
        Args:
            image_path (str): 분석할 이미지 파일 경로
            
        Returns:
            Dict[str, Any]: 분석 결과를 담은 딕셔너리
        """
        try:
            # 이미지 인코딩
            base64_image = self._encode_image(image_path)
            
            # GPT-4 Vision API 요청 내용 구성
            prompt = """
            이 의류 이미지를 분석해주세요. 다음 정보를 포함해주세요:
            1. 의류의 카테고리 (상의/하의/아우터/원피스/기타)
            2. 구체적인 아이템 종류 (예: 티셔츠, 청바지 등)
            3. 주요 색상
            4. 특징적인 디자인 요소
            5. 스타일 (캐주얼/포멀/스포티 등)
            
            JSON 형식으로 응답해주세요.
            """
            
            # API 요청
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # 응답 파싱 및 반환
            result = response.choices[0].message.content
            logger.info(f"분석 완료: {result}")
            
            return {
                "status": "success",
                "analysis": result
            }
            
        except Exception as e:
            logger.error(f"의류 분석 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def get_style_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """분석 결과를 바탕으로 스타일 추천
        
        Args:
            analysis_result (Dict[str, Any]): 분석 결과
            
        Returns:
            List[str]: 스타일 추천 목록
        """
        # TODO: 스타일 추천 로직 구현
        pass 