import os
import boto3
import logging
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model-downloader")

def download_models_from_s3():
    """S3에서 필요한 모델 파일을 다운로드합니다."""
    try:
        # 환경 변수 확인
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "ap-northeast-2")
        bucket_name = os.getenv("S3_BUCKET_NAME")
        
        if not all([aws_access_key, aws_secret_key, aws_region, bucket_name]):
            logger.warning("AWS 환경 변수가 완전히 설정되지 않았습니다. S3에서 모델을 다운로드할 수 없습니다.")
            return False
        
        # S3 클라이언트 초기화
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # 모델 디렉토리 생성
        model_dir = "models/face_model_mobilenet_20250405_121601"
        os.makedirs(model_dir, exist_ok=True)
        
        # 다운로드할 파일 목록
        files_to_download = [
            "final_model.h5",
            "class_indices.txt"
        ]
        
        # 이미 모든 파일이 존재하는지 확인
        all_files_exist = True
        for file_name in files_to_download:
            if not os.path.exists(os.path.join(model_dir, file_name)):
                all_files_exist = False
                break
                
        if all_files_exist:
            logger.info("모든 모델 파일이 이미 존재합니다. 다운로드를 건너뜁니다.")
            return True
            
        # 파일 다운로드
        for file_name in files_to_download:
            local_path = os.path.join(model_dir, file_name)
            s3_path = f"{model_dir}/{file_name}"
            
            try:
                logger.info(f"S3에서 파일 다운로드 중: {s3_path}")
                s3_client.download_file(bucket_name, s3_path, local_path)
                logger.info(f"다운로드 완료: {local_path}")
            except Exception as e:
                logger.error(f"파일 다운로드 중 오류 발생: {str(e)}")
                # 다운로드 실패하더라도 계속 진행
                continue
        
        return True
        
    except Exception as e:
        logger.error(f"모델 다운로드 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("S3에서 모델 파일 다운로드 시작...")
    success = download_models_from_s3()
    if success:
        logger.info("모델 다운로드 완료 또는 이미 존재함")
    else:
        logger.warning("모델 다운로드에 실패했습니다. 로컬 모델이 존재하는지 확인하세요.") 