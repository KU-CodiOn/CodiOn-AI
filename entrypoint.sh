#!/bin/bash

# AWS S3에서 모델 파일 다운로드
python /app/download_models.py

# Gunicorn으로 서버 시작
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000 