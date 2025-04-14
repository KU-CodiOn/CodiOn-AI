# 퍼스널 컬러 분류 시스템

이 프로젝트는 딥러닝을 활용하여 사람의 얼굴 이미지로부터 퍼스널 컬러(봄, 여름, 가을, 겨울)를 분류하는 시스템입니다.

## 주요 기능

- VGG16 기반 퍼스널 컬러 분류 모델
- 이미지 전처리(화이트 밸런싱, 얼굴 추출)
- 데이터 증강을 통한 모델 성능 향상
- 상세한 모델 평가 및 분석

## 설치 방법

1. 저장소 클론

```bash
git clone <repository-url>
cd personalColor
```

2. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 모델 학습

기본 설정으로 모델 학습:

```bash
python run_training.py
```

추가 옵션을 사용한 학습:

```bash
python run_training.py --epochs 150 --batch_size 32 --extract_face --evaluate
```

### 모델 평가

학습된 모델 평가:

```bash
python evaluate_improved_model.py --model_dir models/personalcolor_vgg16_YYYYMMDD_HHMMSS
```

## 프로젝트 구조

```
personalColor/
│
├── train_model.py           # 기존 모델 학습 스크립트
├── improve_model.py         # 개선된 VGG16 모델 구현
├── run_training.py          # 학습 실행 스크립트
├── evaluate_improved_model.py # 모델 평가 스크립트
│
├── dataset/                 # 데이터셋 디렉토리
│   └── all_combined_split/
│       ├── train/          # 훈련 데이터
│       └── validation/     # 검증 데이터
│
├── models/                  # 학습된 모델 저장 디렉토리
│
└── requirements.txt         # 필요한 패키지 목록
```

## 모델 아키텍처

이 프로젝트는 다음 모델 아키텍처를 사용합니다:

- 백본 모델: VGG16 (ImageNet 사전 학습)
- 특성 추출: GlobalAveragePooling2D
- 분류 레이어: Dense(1024) → BatchNorm → Dropout → Dense(512) → BatchNorm → Dropout → Dense(4)
- 최적화: Adamax 옵티마이저 (학습률 0.0005)
- 손실 함수: 카테고리 교차 엔트로피

## 개선 사항

1. 화이트 밸런싱 및 얼굴 추출 전처리 적용
2. 확장된 데이터 증강 기법 활용
3. VGG16 모델 사용 (기존 EfficientNetB0 대체)
4. 단계적 학습률 감소 스케줄러 도입
5. 배치 정규화와 드롭아웃을 통한 일반화 개선
6. 모델 평가 및 오분류 분석 강화

## 참고 자료

- 퍼스널 컬러 진단 및 스타일링제품 추천 시스템 (부산대학교) 