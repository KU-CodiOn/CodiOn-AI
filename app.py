from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import uvicorn
import os
import uuid
import logging
import base64
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
import json
import boto3
from botocore.exceptions import ClientError

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fashion-api")

# FastAPI 앱 생성
app = FastAPI(
    title="Fashion Analysis API",
    description="의류 분석을 위한 API 서버",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙을 위한 디렉토리 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# S3 클라이언트 초기화
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# 이미지 업로드 디렉토리 생성
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def upload_to_s3(file_content: bytes, file_name: str) -> str:
    """S3에 파일 업로드하고 URL 반환"""
    try:
        # 고유한 파일명 생성
        unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}_{file_name}"
        
        # S3에 업로드
        s3_client.put_object(
            Bucket=os.getenv("S3_BUCKET_NAME"),
            Key=unique_filename,
            Body=file_content,
            ContentType='image/jpeg'
        )
        
        # URL 생성
        url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{unique_filename}"
        
        logger.info(f"S3 업로드 완료: {url}")
        return url
    except ClientError as e:
        logger.error(f"S3 업로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="S3 업로드 중 오류가 발생했습니다.")

async def analyze_fashion(image_url: str) -> dict:
    """GPT Vision API를 사용하여 의류 이미지 분석"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이 의류 이미지를 분석해주세요. 다음 정보를 JSON 형식으로 제공해주세요: 의류 카테고리, 아이템 종류, 주요 색상, 디자인 요소, 스타일"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        # 응답에서 JSON 부분 추출
        content = response.choices[0].message.content
        json_str = content.split("```json")[1].split("```")[0].strip()
        result = json.loads(json_str)
        
        logger.info(f"분석 완료: {content}")
        return result
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/fashion/upload")
async def analyze_uploaded_fashion(file: UploadFile = File(...)):
    """업로드된 이미지 분석"""
    try:
        # 파일 유효성 검사
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
        
        # 파일 크기 제한 (10MB)
        MAX_SIZE = 10 * 1024 * 1024  # 10MB
        content = await file.read()
        if len(content) > MAX_SIZE:
            raise HTTPException(status_code=400, detail="파일 크기는 10MB를 초과할 수 없습니다.")
        
        # S3에 업로드
        image_url = upload_to_s3(content, file.filename)
        
        # 이미지 분석
        result = await analyze_fashion(image_url)
        
        # 결과에 이미지 URL 추가
        result['image_url'] = image_url
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"이미지 업로드 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fashion", response_class=HTMLResponse)
async def read_fashion():
    """의류 분석 테스트 페이지"""
    return FileResponse("static/fashion.html")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 