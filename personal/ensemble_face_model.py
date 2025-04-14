import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Input, Average, Concatenate
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import pandas as pd
import shutil
import traceback

# 데이터 경로 설정
FACE_DATA_DIR = 'dataset/faces_dlib'
PROCESSED_DATA_DIR = 'dataset/faces_processed_safe_ensemble'
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 150
NUM_CLASSES = 4
LEARNING_RATE = 0.00002  # 매우 낮은 학습률

def create_mobilenet_model(input_tensor=None, input_shape=(224, 224, 3)):
    """MobileNetV2 기반 모델 생성"""
    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = input_tensor
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs,
        input_shape=input_shape
    )
    
    # 레이어 동결 설정
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # 특성 추출
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(NUM_CLASSES, activation='softmax', name='mobilenet_output')(x)
    
    # 입력 텐서가 제공된 경우 레이어로 반환, 아니면 모델 생성
    if input_tensor is not None:
        return outputs
    else:
        model = Model(inputs=inputs, outputs=outputs)
        return model

def create_resnet_model(input_tensor=None, input_shape=(224, 224, 3)):
    """ResNet50 기반 모델 생성"""
    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = input_tensor
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs,
        input_shape=input_shape
    )
    
    # 레이어 동결 설정
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # 특성 추출
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(NUM_CLASSES, activation='softmax', name='resnet_output')(x)
    
    # 입력 텐서가 제공된 경우 레이어로 반환, 아니면 모델 생성
    if input_tensor is not None:
        return outputs
    else:
        model = Model(inputs=inputs, outputs=outputs)
        return model

def create_efficientnet_model(input_tensor=None, input_shape=(224, 224, 3)):
    """EfficientNetB0 기반 모델 생성"""
    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = input_tensor
    
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs,
        input_shape=input_shape
    )
    
    # 레이어 동결 설정
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # 특성 추출
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(NUM_CLASSES, activation='softmax', name='efficientnet_output')(x)
    
    # 입력 텐서가 제공된 경우 레이어로 반환, 아니면 모델 생성
    if input_tensor is not None:
        return outputs
    else:
        model = Model(inputs=inputs, outputs=outputs)
        return model

def create_ensemble_model():
    """앙상블 모델 생성"""
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    
    # 공통 입력
    inputs = Input(shape=input_shape)
    
    # 각 모델의 출력 레이어 생성 (공유 입력 사용)
    mobilenet_output = create_mobilenet_model(inputs, input_shape)
    resnet_output = create_resnet_model(inputs, input_shape)
    efficientnet_output = create_efficientnet_model(inputs, input_shape)
    
    # 앙상블 (평균)
    ensemble_output = Average(name='ensemble_average')([mobilenet_output, resnet_output, efficientnet_output])
    
    # 앙상블 모델 생성
    ensemble_model = Model(inputs=inputs, outputs=ensemble_output)
    
    # 모델 컴파일
    ensemble_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return ensemble_model

def safe_prepare_data_directory():
    """안전한 데이터 디렉토리 준비 - 모든 파일 검증 후 복사"""
    print("안전한 데이터셋 준비 중...")
    
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 원본 데이터 디렉토리가 존재하는지 확인
    if not os.path.exists(FACE_DATA_DIR):
        raise FileNotFoundError(f"원본 데이터 디렉토리를 찾을 수 없습니다: {FACE_DATA_DIR}")
    
    seasons = ['spring', 'summer', 'autumn', 'winter']
    count_per_season = {}
    problematic_files = []
    
    for season in seasons:
        src_dir = os.path.join(FACE_DATA_DIR, season)
        dst_dir = os.path.join(PROCESSED_DATA_DIR, season)
        
        # 시즌 디렉토리가 존재하는지 확인
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"시즌 디렉토리를 찾을 수 없습니다: {src_dir}")
        
        os.makedirs(dst_dir, exist_ok=True)
        count = 0
        
        # 이미지 파일 검증 및 안전한 파일만 복사
        for file in os.listdir(src_dir):
            if file.endswith('.jpg') and not file.endswith('_landmarks.jpg'):
                try:
                    src_file = os.path.join(src_dir, file)
                    dst_file = os.path.join(dst_dir, file)
                    
                    # 파일 존재 확인
                    if not os.path.exists(src_file):
                        print(f"파일이 존재하지 않음: {src_file}")
                        continue
                    
                    # 이미지 유효성 확인
                    try:
                        img = load_img(src_file, target_size=(IMG_SIZE, IMG_SIZE))
                        img_array = img_to_array(img)
                        
                        # 문제 없으면 파일 복사
                        shutil.copy2(src_file, dst_file)
                        count += 1
                    except Exception as img_err:
                        print(f"손상된/유효하지 않은 이미지 파일: {src_file}, 오류: {img_err}")
                        problematic_files.append((src_file, str(img_err)))
                        continue
                        
                except Exception as e:
                    print(f"파일 처리 중 오류: {src_file}, 오류: {e}")
                    problematic_files.append((src_file, str(e)))
                    continue
        
        count_per_season[season] = count
        print(f"{season}: {count}개 유효한 이미지 복사됨")
    
    # 문제 파일 리포트
    if problematic_files:
        print(f"\n문제가 있는 파일 총 {len(problematic_files)}개:")
        for file_path, error in problematic_files[:10]:
            print(f"  - {file_path}: {error}")
        if len(problematic_files) > 10:
            print(f"  ... 외 {len(problematic_files) - 10}개 파일")
    
    # 클래스별 이미지 수 확인
    print("\n클래스별 유효한 이미지 수:")
    for season, count in count_per_season.items():
        print(f"  {season}: {count}")
    
    # 최소 이미지 수 검사
    min_count = min(count_per_season.values())
    if min_count == 0:
        raise ValueError("하나 이상의 클래스에 이미지가 없습니다.")
    elif min_count < 100:
        print(f"\n경고: 일부 클래스의 이미지 수가 적습니다(최소: {min_count}개). 모델 학습에 영향을 줄 수 있습니다.")
    
    return PROCESSED_DATA_DIR, count_per_season

def create_data_generators(data_dir):
    """데이터 제너레이터 생성 - 색상 정보 보존에 초점"""
    # 데이터 디렉토리 확인
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
    
    # 각 시즌 디렉토리 파일 수 확인
    for season in ['spring', 'summer', 'autumn', 'winter']:
        season_dir = os.path.join(data_dir, season)
        if not os.path.exists(season_dir):
            raise FileNotFoundError(f"시즌 디렉토리가 존재하지 않습니다: {season_dir}")
        
        files = os.listdir(season_dir)
        if not files:
            raise ValueError(f"{season} 디렉토리에 이미지가 없습니다.")
        print(f"{season} 디렉토리에 {len(files)}개의 파일이 있습니다.")
    
    # 데이터 증강 설정
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,  # 회전 범위 제한
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        # 밝기 변화 매우 제한적으로 적용 (색상 정보 보존 중요)
        brightness_range=[0.95, 1.05],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # 검증 데이터 생성기
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    try:
        print("\n데이터 생성기 초기화 중...")
        
        # 훈련 데이터 생성기
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # 검증 데이터 생성기
        validation_generator = validation_datagen.flow_from_directory(
            data_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print("데이터 생성기 초기화 완료!")
        return train_generator, validation_generator
    except Exception as e:
        print(f"데이터 생성기 생성 중 오류 발생: {e}")
        traceback.print_exc()
        raise

def compute_class_weights(train_generator):
    """클래스 불균형 처리를 위한 가중치 계산"""
    # 클래스 인덱스 확인
    class_indices = train_generator.class_indices
    class_indices = {v: k for k, v in class_indices.items()}
    
    # 클래스별 가중치 계산
    classes = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    
    # 가중치 제한 (극단적인 가중치 방지)
    class_weights = np.clip(class_weights, 0.7, 1.5)
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("클래스 인덱스:", class_indices)
    print("클래스 가중치:", class_weight_dict)
    
    return class_weight_dict

def plot_training_history(history, model_dir):
    """훈련 과정 시각화 및 저장"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()
    
    # 추가 측정 지표 그래프 (AUC, Precision, Recall)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history.history['auc'], label='Training AUC')
    plt.plot(epochs_range, history.history['val_auc'], label='Validation AUC')
    plt.legend(loc='lower right')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history.history['precision'], label='Training Precision')
    plt.plot(epochs_range, history.history['val_precision'], label='Validation Precision')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history.history['recall'], label='Training Recall')
    plt.plot(epochs_range, history.history['val_recall'], label='Validation Recall')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    
    plt.savefig(os.path.join(model_dir, 'metrics_history.png'))
    plt.close()

def main():
    try:
        print("==== 앙상블 퍼스널 컬러 분류 모델 학습 시작 ====")
        
        # 현재 시간을 기반으로 모델 저장 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join('models', f'face_model_ensemble_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)
        print(f"모델 저장 디렉토리: {model_dir}")
        
        # 안전한 데이터 디렉토리 준비
        processed_data_dir, count_per_season = safe_prepare_data_directory()
        print(f"데이터 디렉토리 준비 완료: {processed_data_dir}")
        
        # 데이터 생성기 생성
        train_generator, validation_generator = create_data_generators(processed_data_dir)
        
        # 데이터 파이프라인 테스트
        print("\n데이터 파이프라인 테스트 중...")
        try:
            x_batch, y_batch = next(iter(train_generator))
            print(f"훈련 데이터 배치 형태: {x_batch.shape}, 레이블 형태: {y_batch.shape}")
            
            x_val_batch, y_val_batch = next(iter(validation_generator))
            print(f"검증 데이터 배치 형태: {x_val_batch.shape}, 레이블 형태: {y_val_batch.shape}")
            
            print("데이터 파이프라인 테스트 성공!")
        except Exception as e:
            print(f"데이터 테스트 중 오류 발생: {e}")
            print("데이터 준비에 실패했습니다. 모델 훈련을 중단합니다.")
            return
        
        # 클래스 가중치 계산
        class_weight_dict = compute_class_weights(train_generator)
        
        # 앙상블 모델 생성
        print("\n앙상블 모델 생성 중...")
        model = create_ensemble_model()
        print("앙상블 모델 생성 완료!")
        
        # 모델 구조 저장
        with open(os.path.join(model_dir, 'model_summary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # 콜백 설정
        callbacks = [
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=0.0000005,
                verbose=1
            ),
            CSVLogger(
                os.path.join(model_dir, 'training_log.csv'),
                append=True
            )
        ]
        
        # 훈련 시작 시간 기록
        start_time = datetime.datetime.now()
        print(f"훈련 시작 시간: {start_time}")
        
        # 훈련 실행
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # 훈련 종료 시간 및 소요 시간
        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        print(f"훈련 종료 시간: {end_time}")
        print(f"총 훈련 시간: {training_time}")
        
        # 학습 결과 시각화
        plot_training_history(history, model_dir)
        
        # 최종 모델 저장
        model.save(os.path.join(model_dir, 'final_model.h5'))
        print(f"앙상블 모델이 {model_dir}에 저장되었습니다.")
        
        # 모델 평가
        evaluation = model.evaluate(validation_generator)
        metrics = {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'auc': evaluation[2],
            'precision': evaluation[3],
            'recall': evaluation[4]
        }
        
        # 평가 결과 저장
        with open(os.path.join(model_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"훈련 시작: {start_time}\n")
            f.write(f"훈련 종료: {end_time}\n")
            f.write(f"훈련 시간: {training_time}\n\n")
            
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value}\n")
                print(f"{metric_name}: {metric_value}")
        
        print("==== 앙상블 모델 학습 완료 ====")
    
    except Exception as e:
        print(f"훈련 과정에서 예기치 않은 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 정보 저장
        if 'model_dir' in locals():
            error_log_path = os.path.join(model_dir, 'error_log.txt')
            with open(error_log_path, 'w') as f:
                f.write(f"오류 발생 시간: {datetime.datetime.now()}\n")
                f.write(f"오류 메시지: {str(e)}\n\n")
                traceback.print_exc(file=f)
            print(f"오류 정보가 {error_log_path}에 저장되었습니다.")
        
        print("오류가 발생하여 모델 훈련이 중단되었습니다.")

if __name__ == "__main__":
    main() 