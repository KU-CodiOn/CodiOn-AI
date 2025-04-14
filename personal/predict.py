import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# YCbCr 색공간에서 피부색 범위 정의
SKIN_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_UPPER = np.array([255, 173, 127], dtype=np.uint8)

# 시즌별 특성과 색상 정보 
SEASON_INFO = {
    'spring': {
        'tone': 'warm',
        'characteristic': '밝고 선명한 톤',
        'colors': ['노랑', '복숭아색', '황금색', '산호색', '밝은 옥색'],
        'color_codes': ['#F7E600', '#F9A781', '#FFD700', '#FF7F50', '#40E0D0']
    },
    'summer': {
        'tone': 'cool',
        'characteristic': '부드럽고 연한 톤',
        'colors': ['라벤더', '베이비블루', '파스텔 핑크', '민트', '하늘색'],
        'color_codes': ['#E6E6FA', '#89CFF0', '#FFD1DC', '#98FB98', '#87CEEB']
    },
    'autumn': {
        'tone': 'warm',
        'characteristic': '깊고 풍부한 톤',
        'colors': ['갈색', '올리브', '테라코타', '황토색', '다크 그린'],
        'color_codes': ['#A0522D', '#808000', '#E2725B', '#D2691E', '#006400']
    },
    'winter': {
        'tone': 'cool',
        'characteristic': '선명하고 차가운 톤',
        'colors': ['검정', '흰색', '로얄블루', '선명한 핑크', '보라'],
        'color_codes': ['#000000', '#FFFFFF', '#4169E1', '#FF1493', '#800080']
    }
}

def extract_skin_features(img, face_area=None):
    """이미지에서 피부색 특성 추출"""
    try:
        # 얼굴 영역이 주어진 경우 해당 영역만 추출
        if face_area is not None:
            x, y, w, h = face_area
            img = img[y:y+h, x:x+w]
        
        # YCrCb 변환
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        # 피부색 마스크 생성
        skin_mask = cv2.inRange(img_ycrcb, SKIN_LOWER, SKIN_UPPER)
        
        # 노이즈 제거를 위한 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # 피부 영역 비율 계산
        skin_ratio = np.count_nonzero(skin_mask) / (skin_mask.shape[0] * skin_mask.shape[1])
        
        # 피부 영역이 너무 적으면 None 반환
        if skin_ratio < 0.05:
            return None
            
        # 피부색 특성 추출
        skin = cv2.bitwise_and(img, img, mask=skin_mask)
        skin_pixels = skin[skin_mask > 0]
        
        if len(skin_pixels) == 0:
            return None
            
        # 피부색 평균, 표준편차 계산 (BGR)
        mean_skin_color = np.mean(skin_pixels, axis=0)
        std_skin_color = np.std(skin_pixels, axis=0)
        
        # HSV로 변환하여 특성 추출
        skin_hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
        hsv_pixels = skin_hsv[skin_mask > 0]
        mean_hsv = np.mean(hsv_pixels, axis=0)
        std_hsv = np.std(hsv_pixels, axis=0)
        
        # LAB로 변환하여 특성 추출
        skin_lab = cv2.cvtColor(skin, cv2.COLOR_BGR2LAB)
        lab_pixels = skin_lab[skin_mask > 0]
        mean_lab = np.mean(lab_pixels, axis=0)
        std_lab = np.std(lab_pixels, axis=0)
        
        # 추출된 특성들을 하나의 벡터로 합침
        features = np.concatenate([
            mean_skin_color, std_skin_color,
            mean_hsv, std_hsv,
            mean_lab, std_lab,
            [skin_ratio]
        ])
        
        return features
    
    except Exception as e:
        print(f"특성 추출 오류: {e}")
        return None

def detect_face(image):
    """이미지에서 얼굴 검출"""
    # 얼굴 감지기 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # 가장 큰 얼굴 반환 (여러 얼굴이 검출된 경우)
    if len(faces) > 1:
        max_area = 0
        max_face = None
        for face in faces:
            x, y, w, h = face
            area = w * h
            if area > max_area:
                max_area = area
                max_face = face
        return max_face
    
    return faces[0]

def load_latest_model():
    """최신 모델 파일 로드"""
    # 직접 모델 경로 지정
    model_path = '../models/face_model_mobilenet_20250405_121601/best_model.h5'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 기본 클래스 인덱스 설정
    class_indices = {
        0: 'autumn',
        1: 'spring',
        2: 'summer',
        3: 'winter'
    }
    
    is_dual_input = False  # MobileNetV2는 단일 입력 모델
    
    print(f"모델 로드: {model_path}")
    return load_model(model_path), class_indices, is_dual_input

def predict_personal_color(image_path):
    """이미지에서 퍼스널 컬러 예측"""
    # 모델 로드
    try:
        model, class_indices, is_dual_input = load_latest_model()
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return None
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # 얼굴 검출
    face_area = detect_face(image)
    
    # 이미지 리사이징
    img_resized = cv2.resize(image, (224, 224))
    img_array = img_to_array(img_resized) / 255.0
    
    if is_dual_input:
        # 피부색 특성 추출
        features = extract_skin_features(image, face_area)
        
        if features is None:
            print("피부색 특성을 추출할 수 없습니다. 기본 모델로 예측합니다.")
            is_dual_input = False
        else:
            # 듀얼 인풋 모델로 예측
            img_batch = np.expand_dims(img_array, axis=0)
            features_batch = np.expand_dims(features, axis=0)
            predictions = model.predict([img_batch, features_batch])[0]
    
    # 기본 모델 또는 듀얼 모델 사용 불가능한 경우
    if not is_dual_input:
        img_batch = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_batch)[0]
    
    # 가장 높은 확률의 클래스 
    predicted_class_idx = np.argmax(predictions)
    season = class_indices[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # 웜톤/쿨톤 판단
    is_warm = SEASON_INFO[season]['tone'] == 'warm'
    
    # 웜톤/쿨톤 신뢰도 계산
    warm_seasons = ['spring', 'autumn']
    cool_seasons = ['summer', 'winter']
    
    warm_confidence = sum(predictions[class_indices.index(s)] for s in warm_seasons if s in class_indices.values())
    cool_confidence = sum(predictions[class_indices.index(s)] for s in cool_seasons if s in class_indices.values())
    
    # 웜톤/쿨톤 중 더 높은 값의 신뢰도
    warm_cool_confidence = max(warm_confidence, cool_confidence)
    
    # 결과
    result = {
        'season': season,
        'season_confidence': confidence,
        'is_warm': is_warm,
        'warm_cool_confidence': warm_cool_confidence,
        'face_area': face_area,
        'all_probabilities': {class_indices[i]: prob for i, prob in enumerate(predictions)}
    }
    
    return result

def visualize_results(image_path, result, output_path=None):
    """분석 결과 시각화"""
    if result is None:
        print("시각화할 결과가 없습니다.")
        return
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"시각화를 위한 이미지를 로드할 수 없습니다: {image_path}")
        return
    
    # 복사본 생성
    viz_img = image.copy()
    viz_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
    
    # 얼굴 영역 강조 표시
    face_area = result.get('face_area')
    if face_area is not None:
        x, y, w, h = face_area
        cv2.rectangle(viz_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 시즌 정보
    season = result['season']
    season_info = SEASON_INFO[season]
    
    # 플롯 생성
    plt.figure(figsize=(15, 10))
    
    # 원본 이미지
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("원본 이미지")
    plt.axis('off')
    
    # 분석 결과
    plt.subplot(2, 2, 2)
    plt.imshow(viz_rgb)
    plt.title("분석 결과")
    plt.axis('off')
    
    # 결과 정보
    plt.figtext(0.5, 0.5, f"계절: {season.capitalize()} ({result['season_confidence']:.1%})\n"
                        f"톤: {'웜톤' if result['is_warm'] else '쿨톤'} ({result['warm_cool_confidence']:.1%})\n"
                        f"특징: {season_info['characteristic']}", 
                ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
    
    # 추천 색상 팔레트
    plt.subplot(2, 2, 3)
    color_codes = season_info['color_codes']
    plt.pie([1] * len(color_codes), colors=color_codes, startangle=90)
    plt.axis('equal')
    plt.title(f"{season.capitalize()} 추천 색상 팔레트")
    
    # 모든 계절 확률
    all_probs = result.get('all_probabilities', {})
    if all_probs:
        plt.subplot(2, 2, 4)
        seasons = list(all_probs.keys())
        probs = list(all_probs.values())
        plt.bar(seasons, probs, color=['#D2691E', '#F7E600', '#87CEEB', '#4169E1'])
        plt.ylim(0, 1)
        plt.title("계절별 확률")
        for i, v in enumerate(probs):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    # 출력 경로 설정
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"personal_color_result_{timestamp}.png"
    
    # 저장
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='퍼스널 컬러 분석')
    parser.add_argument('image_path', help='분석할 이미지 경로')
    parser.add_argument('--output', help='결과 이미지 저장 경로')
    
    args = parser.parse_args()
    
    try:
        # 이미지 분석
        print(f"이미지 분석 중: {args.image_path}")
        result = predict_personal_color(args.image_path)
        
        if result:
            # 결과 출력
            print("\n===== 퍼스널 컬러 분석 결과 =====")
            print(f"계절: {result['season'].capitalize()} (신뢰도: {result['season_confidence']:.1%})")
            print(f"톤: {'웜톤' if result['is_warm'] else '쿨톤'} (신뢰도: {result['warm_cool_confidence']:.1%})")
            print(f"특징: {SEASON_INFO[result['season']]['characteristic']}")
            print(f"추천 색상: {', '.join(SEASON_INFO[result['season']]['colors'])}")
            
            # 결과 시각화
            output_path = visualize_results(args.image_path, result, args.output)
            print(f"\n분석 결과가 저장되었습니다: {output_path}")
        else:
            print("분석 결과를 얻을 수 없습니다.")
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 