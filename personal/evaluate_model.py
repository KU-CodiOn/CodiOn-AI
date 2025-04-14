import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import pandas as pd

def load_latest_model():
    """최신 모델 파일 로드"""
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {models_dir}")
    
    # 모델 디렉토리 목록
    model_dirs = [d for d in os.listdir(models_dir) 
                if os.path.isdir(os.path.join(models_dir, d))]
    
    if not model_dirs:
        raise FileNotFoundError("모델 디렉토리에 모델이 없습니다.")
    
    # 날짜별 정렬 (최신 모델 찾기)
    model_dirs.sort(reverse=True)
    latest_model_dir = os.path.join(models_dir, model_dirs[0])
    
    # 모델 파일 찾기 (.keras 또는 .h5 파일)
    model_files = ['best_model.keras', 'final_model.keras', 'best_model.h5', 'final_model.h5']
    model_path = None
    
    for model_file in model_files:
        if os.path.exists(os.path.join(latest_model_dir, model_file)):
            model_path = os.path.join(latest_model_dir, model_file)
            break
    
    if not model_path:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {latest_model_dir}")
    
    # 모델 로드
    model = load_model(model_path)
    
    # 클래스 인덱스 파일 로드
    class_indices_path = os.path.join(latest_model_dir, 'class_indices.txt')
    class_indices = {}
    
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            for line in f:
                if ':' in line:
                    name, idx = line.strip().split(':')
                    class_indices[int(idx)] = name
    else:
        # 기본 클래스 인덱스
        class_indices = {
            0: 'autumn',
            1: 'spring',
            2: 'summer',
            3: 'winter'
        }
    
    return model, class_indices, latest_model_dir

def evaluate_model(test_dir, model_dir=None, batch_size=32, input_shape=(224, 224)):
    """모델 평가"""
    # 모델 로드
    if model_dir:
        # 특정 모델 디렉토리 사용
        model_files = ['best_model.keras', 'final_model.keras', 'best_model.h5', 'final_model.h5']
        model_path = None
        for model_file in model_files:
            if os.path.exists(os.path.join(model_dir, model_file)):
                model_path = os.path.join(model_dir, model_file)
                break
        
        if not model_path:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_dir}")
        
        model = load_model(model_path)
        
        # 클래스 인덱스 파일 로드
        class_indices_path = os.path.join(model_dir, 'class_indices.txt')
        class_indices = {}
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        name, idx = line.strip().split(':')
                        class_indices[int(idx)] = name
        else:
            # 기본 클래스 인덱스
            class_indices = {
                0: 'autumn',
                1: 'spring',
                2: 'summer',
                3: 'winter'
            }
    else:
        # 최신 모델 사용
        model, class_indices, model_dir = load_latest_model()
    
    # 테스트 데이터 생성기 설정
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # 테스트 데이터 로드
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # 클래스 이름 매핑
    class_names = [class_indices[i] for i in range(len(class_indices))]
    
    # 모델 평가
    print("\n===== 모델 평가 =====")
    metrics = model.evaluate(test_generator)
    metric_names = model.metrics_names
    
    print(f"테스트 데이터셋: {test_generator.samples}개 이미지")
    for name, value in zip(metric_names, metrics):
        print(f"{name}: {value:.4f}")
    
    # 예측 수행
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('혼동 행렬 (정규화)')
    plt.ylabel('실제 클래스')
    plt.xlabel('예측 클래스')
    confusion_matrix_path = os.path.join(model_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # 분류 리포트
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # 분류 리포트 저장
    report_path = os.path.join(model_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    
    # 정확도, 정밀도, 재현율, F1 점수 시각화
    plt.figure(figsize=(12, 8))
    
    # 클래스별 메트릭 추출 (마지막 3행 제외 - avg/total 행)
    metrics_data = report_df.iloc[:-3]
    
    plt.subplot(2, 2, 1)
    metrics_data['precision'].plot(kind='bar')
    plt.title('정밀도 (Precision)')
    plt.ylim(0, 1)
    
    plt.subplot(2, 2, 2)
    metrics_data['recall'].plot(kind='bar')
    plt.title('재현율 (Recall)')
    plt.ylim(0, 1)
    
    plt.subplot(2, 2, 3)
    metrics_data['f1-score'].plot(kind='bar')
    plt.title('F1 점수')
    plt.ylim(0, 1)
    
    plt.subplot(2, 2, 4)
    metrics_data['support'].plot(kind='bar')
    plt.title('데이터 수 (Support)')
    
    plt.tight_layout()
    metrics_path = os.path.join(model_dir, 'metrics_visualization.png')
    plt.savefig(metrics_path)
    plt.close()
    
    # 클래스별 정확도 시각화 (각 클래스의 정확도)
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=class_accuracy)
    plt.title('클래스별 정확도')
    plt.ylim(0, 1)
    plt.ylabel('정확도')
    plt.xlabel('클래스')
    
    # 각 막대 위에 값 표시
    for i, acc in enumerate(class_accuracy):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')
    
    class_acc_path = os.path.join(model_dir, 'class_accuracy.png')
    plt.savefig(class_acc_path)
    plt.close()
    
    # 요약 출력
    print("\n===== 평가 결과 요약 =====")
    print(f"전체 정확도: {report['accuracy']:.4f}")
    print("\n클래스별 성능:")
    for cls in class_names:
        print(f"{cls}: 정밀도={report[cls]['precision']:.4f}, 재현율={report[cls]['recall']:.4f}, F1={report[cls]['f1-score']:.4f}")
    
    print(f"\n결과 파일이 저장되었습니다: {model_dir}")
    print(f"- 혼동 행렬: {confusion_matrix_path}")
    print(f"- 분류 리포트: {report_path}")
    print(f"- 메트릭 시각화: {metrics_path}")
    print(f"- 클래스별 정확도: {class_acc_path}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description='학습된 모델 평가')
    parser.add_argument('--test-dir', help='테스트 데이터 디렉토리', default='dataset/all_combined_split/validation')
    parser.add_argument('--model-dir', help='모델 디렉토리 (지정하지 않으면 최신 모델 사용)')
    parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    args = parser.parse_args()
    
    print(f"테스트 데이터 디렉토리: {args.test_dir}")
    
    evaluate_model(
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main() 