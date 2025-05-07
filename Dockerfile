FROM python:3.9-slim

WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# 애플리케이션 코드 복사
COPY . .

# 실행 권한 설정
RUN chmod +x /app/entrypoint.sh

# 서버 실행
EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"] 