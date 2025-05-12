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
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any

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
    title="퍼스널컬러 및 의류 분석 API",
    description="퍼스널컬러 분석 및 의류 분석을 위한 API 서버",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

# 모델 로드
MODEL_PATH = "models/face_model_mobilenet_20250405_121601"
model_file = os.path.join(MODEL_PATH, "final_model.h5")
indices_file = os.path.join(MODEL_PATH, "class_indices.txt")

# 모델 파일이 없으면 다운로드 시도
if not os.path.exists(model_file) or not os.path.exists(indices_file):
    logger.info("모델 파일이 없습니다. S3에서 다운로드를 시도합니다.")
    from download_models import download_models_from_s3
    download_models_from_s3()

# 모델과 클래스 인덱스 로드
model = load_model(model_file)

# 클래스 인덱스 로드
with open(indices_file, "r") as f:
    class_indices = {i: line.strip() for i, line in enumerate(f.readlines())}

def preprocess_image(img):
    """이미지 전처리 함수"""
    # 이미지 크기 조정
    img = cv2.resize(img, (224, 224))
    # BGR에서 RGB로 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 정규화
    img = img.astype(np.float32) / 255.0
    # 배치 차원 추가
    img = np.expand_dims(img, axis=0)
    return img

def predict_personal_color(img):
    """퍼스널컬러 예측 함수"""
    # 이미지 전처리
    processed_img = preprocess_image(img)
    
    # 예측
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    # 클래스 이름 가져오기
    personal_color = class_indices[predicted_class]
    
    # 정확도 계산 (70-100% 사이로 조정)
    accuracy = int(70 + (confidence * 30))
    
    return personal_color, accuracy

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
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이미지의 의류를 분석해주세요. 다음 형식의 JSON으로만 응답해주세요:\n{\n  \"카테고리\": \"상의/아우터/바지/원피스/스커트 중 하나\",\n  \"퍼스널컬러\": \"봄웜/여름쿨/가을웜/겨울쿨 중 하나\",\n  \"주요색상\": \"하나의 색상명\"\n}\n\n주의사항:\n1. 카테고리는 주어진 5개 중 하나만 선택\n2. 퍼스널컬러는 주어진 4개 중 하나만 선택\n3. 주요색상은 하나의 색상만 선택\n4. 다른 설명이나 추가 정보는 포함하지 마세요"
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

# Pydantic 모델 정의
class ImageUrlInput(BaseModel):
    image_url: HttpUrl = Field(..., description="분석할 이미지의 URL")
    
class PersonalColorResponse(BaseModel):
    result: str = Field(..., description="퍼스널컬러 분석 결과 (JSON 문자열)")
    
class FashionAnalysisResponse(BaseModel):
    result: str = Field(..., description="의류 분석 결과 (JSON 문자열)")

@app.post("/analyze/fashion/upload", response_model=FashionAnalysisResponse, tags=["의류 분석"], summary="파일 업로드를 통한 의류 분석")
async def analyze_fashion_upload(file: UploadFile = File(...)):
    """
    업로드된 이미지 파일을 분석하여 의류 정보를 반환합니다.
    
    - **file**: 분석할 의류 이미지 파일
    
    **반환값**:
    - 의류 카테고리, 퍼스널컬러, 주요 색상 정보
    """
    try:
        # 이미지를 메모리에 저장
        contents = await file.read()
        
        # 이미지를 base64로 인코딩
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # GPT Vision API로 분석
        response = client.chat.completions.create(
            model="gpt-4-turbo-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이미지의 의류를 분석해주세요. 다음 형식의 JSON으로만 응답해주세요:\n{\n  \"카테고리\": \"상의/아우터/바지/원피스/스커트 중 하나\",\n  \"퍼스널컬러\": \"봄웜/여름쿨/가을웜/겨울쿨 중 하나\",\n  \"주요색상\": \"하나의 색상명\"\n}\n\n주의사항:\n1. 카테고리는 주어진 5개 중 하나만 선택\n2. 퍼스널컬러는 주어진 4개 중 하나만 선택\n3. 주요색상은 하나의 색상만 선택\n4. 다른 설명이나 추가 정보는 포함하지 마세요"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        # 분석 결과 반환
        return {"result": response.choices[0].message.content}
        
    except Exception as e:
        error_detail = str(e)
        if "model_not_found" in error_detail:
            error_detail = "GPT-4 Vision 모델이 더 이상 사용되지 않습니다. 최신 모델로 업데이트가 필요합니다."
        logger.error(f"이미지 업로드 분석 중 오류 발생: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )

@app.get("/fashion", response_class=HTMLResponse, tags=["의류 분석"], summary="의류 분석 테스트 페이지")
async def read_fashion():
    """의류 분석 테스트를 위한 HTML 페이지를 제공합니다."""
    return FileResponse("static/fashion.html")

@app.post("/analyze/personal-color/upload", response_model=PersonalColorResponse, tags=["퍼스널컬러 분석"], summary="파일 업로드를 통한 퍼스널컬러 분석")
async def analyze_personal_color_upload(file: UploadFile = File(...)):
    """
    업로드된 얼굴 이미지 파일을 분석하여 퍼스널컬러 정보를 반환합니다.
    
    - **file**: 분석할 얼굴 이미지 파일
    
    **반환값**:
    - 퍼스널컬러 유형, 설명, 정확도
    """
    try:
        # 이미지를 메모리에 저장
        contents = await file.read()
        
        # 이미지를 numpy 배열로 변환
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 퍼스널컬러 예측
        personal_color, accuracy = predict_personal_color(img)
        
        # 결과 생성
        result = {
            "퍼스널컬러": personal_color,
            "설명": f"피부톤과 얼굴 특성을 분석한 결과 {personal_color}로 판단됩니다.",
            "정확도": accuracy
        }
        
        return {"result": json.dumps(result, ensure_ascii=False)}
        
    except Exception as e:
        error_detail = str(e)
        logger.error(f"퍼스널컬러 분석 중 오류 발생: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )

@app.get("/personal-color", response_class=HTMLResponse, tags=["퍼스널컬러 분석"], summary="퍼스널컬러 분석 테스트 페이지")
async def read_personal_color():
    """퍼스널컬러 분석 테스트를 위한 HTML 페이지를 제공합니다."""
    return FileResponse("static/personal-color.html")

@app.post("/analyze/personal-color/url", response_model=PersonalColorResponse, tags=["퍼스널컬러 분석"], summary="URL을 통한 퍼스널컬러 분석")
async def analyze_personal_color_url(image_url_input: ImageUrlInput):
    """
    URL로 제공된 얼굴 이미지를 분석하여 퍼스널컬러 정보를 반환합니다.
    
    - **image_url_input**: 분석할 이미지의 URL
    
    **반환값**:
    - 퍼스널컬러 유형, 설명, 정확도
    """
    try:
        # URL을 문자열로 변환
        image_url_str = str(image_url_input.image_url)
        
        # URL에서 이미지 다운로드
        response = requests.get(image_url_str)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="이미지를 다운로드할 수 없습니다.")
        
        # 이미지를 numpy 배열로 변환
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 처리할 수 없습니다.")
        
        # 퍼스널컬러 예측
        personal_color, accuracy = predict_personal_color(img)
        
        # 결과 생성
        result = {
            "퍼스널컬러": personal_color,
            "설명": f"피부톤과 얼굴 특성을 분석한 결과 {personal_color}로 판단됩니다.",
            "정확도": accuracy
        }
        
        return {"result": json.dumps(result, ensure_ascii=False)}
        
    except Exception as e:
        error_detail = str(e)
        logger.error(f"이미지 URL 분석 중 오류 발생: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )

@app.post("/analyze/fashion/url", response_model=FashionAnalysisResponse, tags=["의류 분석"], summary="URL을 통한 의류 분석")
async def analyze_fashion_url(image_url_input: ImageUrlInput):
    """
    URL로 제공된 의류 이미지를 분석하여 의류 정보를 반환합니다.
    
    - **image_url_input**: 분석할 이미지의 URL
    
    **반환값**:
    - 의류 카테고리, 퍼스널컬러, 주요 색상 정보
    """
    try:
        # URL을 문자열로 변환
        image_url_str = str(image_url_input.image_url)
        
        # GPT Vision API로 분석
        response = client.chat.completions.create(
            model="gpt-4-turbo-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이미지의 의류를 분석해주세요. 다음 형식의 JSON으로만 응답해주세요:\n{\n  \"카테고리\": \"상의/아우터/바지/원피스/스커트 중 하나\",\n  \"퍼스널컬러\": \"봄웜/여름쿨/가을웜/겨울쿨 중 하나\",\n  \"주요색상\": \"하나의 색상명\"\n}\n\n주의사항:\n1. 카테고리는 주어진 5개 중 하나만 선택\n2. 퍼스널컬러는 주어진 4개 중 하나만 선택\n3. 주요색상은 하나의 색상만 선택\n4. 다른 설명이나 추가 정보는 포함하지 마세요"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url_str
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        # 응답 내용을 문자열로 변환하여 반환
        content = response.choices[0].message.content
        return {"result": content}
        
    except Exception as e:
        error_detail = str(e)
        if "model_not_found" in error_detail:
            error_detail = "GPT-4 Vision 모델이 더 이상 사용되지 않습니다. 최신 모델로 업데이트가 필요합니다."
        logger.error(f"이미지 URL 분석 중 오류 발생: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 