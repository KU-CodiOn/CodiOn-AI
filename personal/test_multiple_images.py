import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# predict.py에서 필요한 함수 가져오기
from predict import detect_face, extract_skin_features, SEASON_INFO

# 시즌 매핑
SEASON_MAPPING = {
    0: 'autumn',
    1: 'spring',
    2: 'summer',
    3: 'winter'
}

def load_model_for_test():
    """테스트용 모델 로드"""
    model_path = 'models/face_model_mobilenet_20250405_121601/best_model.h5'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    print(f"모델 로드: {model_path}")
    return load_model(model_path)

def preprocess_image(image_path):
    """이미지 전처리"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # 이미지 리사이징
    img_resized = cv2.resize(image, (224, 224))
    img_array = img_resized.astype(np.float32) / 255.0
    
    return image, img_array

def predict_image(model, image_path):
    """이미지 예측"""
    original_image, processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None
    
    # 예측을 위해 배치 차원 추가
    img_batch = np.expand_dims(processed_image, axis=0)
    
    # 예측
    predictions = model.predict(img_batch)[0]
    
    # 얼굴 검출
    face_area = detect_face(original_image)
    
    # 가장 높은 확률의 클래스
    predicted_class_idx = np.argmax(predictions)
    season = SEASON_MAPPING[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # 웜톤/쿨톤 판단
    is_warm = SEASON_INFO[season]['tone'] == 'warm'
    
    # 웜톤/쿨톤 신뢰도 계산
    warm_seasons = ['spring', 'autumn']
    cool_seasons = ['summer', 'winter']
    
    warm_confidence = sum(predictions[i] for i, s in SEASON_MAPPING.items() if s in warm_seasons)
    cool_confidence = sum(predictions[i] for i, s in SEASON_MAPPING.items() if s in cool_seasons)
    
    # 웜톤/쿨톤 중 더 높은 값의 신뢰도
    warm_cool_confidence = max(warm_confidence, cool_confidence)
    
    # 결과
    result = {
        'season': season,
        'season_confidence': confidence,
        'is_warm': is_warm,
        'warm_cool_confidence': warm_cool_confidence,
        'face_area': face_area,
        'all_probabilities': {SEASON_MAPPING[i]: prob for i, prob in enumerate(predictions)}
    }
    
    return result, original_image

def display_results(image_path, result, image):
    """결과 출력"""
    print(f"\n===== {os.path.basename(image_path)} 분석 결과 =====")
    print(f"계절: {result['season'].capitalize()} (신뢰도: {result['season_confidence']:.1%})")
    print(f"톤: {'웜톤' if result['is_warm'] else '쿨톤'} (신뢰도: {result['warm_cool_confidence']:.1%})")
    print(f"특징: {SEASON_INFO[result['season']]['characteristic']}")
    print(f"추천 색상: {', '.join(SEASON_INFO[result['season']]['colors'])}")
    print(f"각 계절별 확률: {', '.join([f'{season}: {prob:.1%}' for season, prob in result['all_probabilities'].items()])}")
    
    # 얼굴 영역 표시
    if result['face_area'] is not None:
        x, y, w, h = result['face_area']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 이미지 출력
    filename = os.path.basename(image_path)
    output_path = f"test_result_{result['season']}_{filename}"
    cv2.imwrite(output_path, image)
    print(f"결과 이미지가 저장되었습니다: {output_path}")
    
    return output_path

def main():
    if len(sys.argv) < 2:
        print("사용법: python test_multiple_images.py 이미지1 이미지2 ...")
        sys.exit(1)
    
    # 테스트할 이미지 목록
    image_paths = sys.argv[1:]
    
    # 와일드카드 확장
    expanded_paths = []
    for path in image_paths:
        if '*' in path:
            expanded_paths.extend(glob.glob(path))
        else:
            expanded_paths.append(path)
    
    if not expanded_paths:
        print("테스트할 이미지가 없습니다.")
        sys.exit(1)
    
    # 모델 로드
    try:
        model = load_model_for_test()
    except Exception as e:
        print(f"모델 로드 오류: {str(e)}")
        sys.exit(1)
    
    # 각 이미지 테스트
    results = []
    
    for image_path in expanded_paths:
        try:
            print(f"\n이미지 분석 중: {image_path}")
            result_data = predict_image(model, image_path)
            
            if result_data:
                result, image = result_data
                output_path = display_results(image_path, result, image)
                results.append((image_path, result, output_path))
            else:
                print(f"이미지 {image_path}를 분석할 수 없습니다.")
        except Exception as e:
            print(f"이미지 {image_path} 분석 중 오류 발생: {str(e)}")
    
    # 요약 출력
    print("\n===== 테스트 결과 요약 =====")
    print(f"총 테스트 이미지 수: {len(expanded_paths)}")
    print(f"성공적으로 분석된 이미지 수: {len(results)}")
    
    # 계절별 카운트
    season_counts = {season: 0 for season in ['spring', 'summer', 'autumn', 'winter']}
    tone_counts = {'warm': 0, 'cool': 0}
    
    for _, result, _ in results:
        season_counts[result['season']] += 1
        tone_counts['warm' if result['is_warm'] else 'cool'] += 1
    
    print("\n계절별 분류 결과:")
    for season, count in season_counts.items():
        percentage = count / len(results) * 100 if results else 0
        print(f"  {season.capitalize()}: {count} ({percentage:.1f}%)")
    
    print("\n톤별 분류 결과:")
    for tone, count in tone_counts.items():
        percentage = count / len(results) * 100 if results else 0
        print(f"  {tone.capitalize()}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 