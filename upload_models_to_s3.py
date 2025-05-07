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
logger = logging.getLogger("model-uploader")

def upload_models_to_s3():
    """모델 파일을 S3에 업로드합니다."""
    try:
        # 환경 변수 확인
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "ap-northeast-2")
        bucket_name = os.getenv("S3_BUCKET_NAME")
        
        if not all([aws_access_key, aws_secret_key, aws_region, bucket_name]):
            logger.error("AWS 환경 변수가 완전히 설정되지 않았습니다. S3에 모델을 업로드할 수 없습니다.")
            return False
        
        # S3 클라이언트 초기화
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # 모델 디렉토리 확인
        model_dir = "models/face_model_mobilenet_20250405_121601"
        if not os.path.exists(model_dir):
            logger.error(f"모델 디렉토리를 찾을 수 없습니다: {model_dir}")
            return False
        
        # 업로드할 파일 목록
        files_to_upload = [
            "final_model.h5",
            "class_indices.txt"
        ]
        
        # 파일 업로드
        for file_name in files_to_upload:
            local_path = os.path.join(model_dir, file_name)
            s3_path = f"{model_dir}/{file_name}"
            
            if not os.path.exists(local_path):
                logger.error(f"파일을 찾을 수 없습니다: {local_path}")
                continue
                
            try:
                logger.info(f"S3에 파일 업로드 중: {local_path} -> s3://{bucket_name}/{s3_path}")
                s3_client.upload_file(local_path, bucket_name, s3_path)
                logger.info(f"업로드 완료: s3://{bucket_name}/{s3_path}")
            except Exception as e:
                logger.error(f"파일 업로드 중 오류 발생: {str(e)}")
                return False
        
        logger.info("모든 모델 파일이 성공적으로 업로드되었습니다.")
        return True
        
    except Exception as e:
        logger.error(f"모델 업로드 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("S3에 모델 파일 업로드 시작...")
    success = upload_models_to_s3()
    if success:
        logger.info("모델 업로드가 완료되었습니다.")
    else:
        logger.error("모델 업로드에 실패했습니다.") 