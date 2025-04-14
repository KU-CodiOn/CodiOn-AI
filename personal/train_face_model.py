import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import pandas as pd
import shutil

# 데이터 경로 설정
FACE_DATA_DIR = '../dataset/faces_dlib'
PROCESSED_DATA_DIR = '../dataset/faces_processed'
IMG_SIZE = 224  # VGG16에 맞게 조정
BATCH_SIZE = 16
EPOCHS = 150
NUM_CLASSES = 4
LEARNING_RATE = 0.0001

def create_model():
    """VGG16 기반 분류 모델 생성"""
    # 기본 모델 (VGG16)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # 레이어 1-7은 동결, 8-16은 학습 가능하도록 설정
    # VGG16의 레이어 구조를 확인
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= 7  # 7번째 레이어부터 학습 가능하도록 설정
    
    # 분류 헤드 추가
    x = base_model.output
    x = Flatten()(x)  # VGG16 모델에는 Flatten 레이어 사용
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)  # 드롭아웃 비율 0.5로 설정 (논문 기준)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def prepare_data_directory():
    """랜드마크 이미지를 제외한 데이터 디렉토리 준비"""
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    seasons = ['spring', 'summer', 'autumn', 'winter']
    
    for season in seasons:
        src_dir = os.path.join(FACE_DATA_DIR, season)
        dst_dir = os.path.join(PROCESSED_DATA_DIR, season)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 랜드마크가 아닌 이미지만 복사
        for file in os.listdir(src_dir):
            if file.endswith('.jpg') and not file.endswith('_landmarks.jpg'):
                src_file = os.path.join(src_dir, file)
                dst_file = os.path.join(dst_dir, file)
                shutil.copy2(src_file, dst_file)
        
        print(f"{season}: {len(os.listdir(dst_dir))}개 이미지 복사됨")
    
    return PROCESSED_DATA_DIR

def create_data_generators(data_dir):
    """데이터 제너레이터 생성"""
    # 데이터 증강 설정
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # 검증 데이터 생성기는 증강 없이 리스케일만 적용
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
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
    
    return train_generator, validation_generator

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
    
    # 가중치 딕셔너리 생성
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
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()

def main():
    # 현재 시간을 기반으로 모델 저장 디렉토리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join('../models', f'face_model_vgg16_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    
    # 랜드마크가 없는 얼굴 이미지만 사용하기 위해 데이터 디렉토리 준비
    processed_data_dir = prepare_data_directory()
    
    # 데이터 생성기 생성
    train_generator, validation_generator = create_data_generators(processed_data_dir)
    
    # 클래스 가중치 계산
    class_weight_dict = compute_class_weights(train_generator)
    
    # 모델 생성
    model = create_model()
    print(model.summary())
    
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
            patience=20,  # 인내심 증가
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=0.000001,
            verbose=1
        ),
        CSVLogger(
            os.path.join(model_dir, 'training_log.csv'),
            append=True
        )
    ]
    
    # 훈련 실행
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # 학습 결과 시각화
    plot_training_history(history, model_dir)
    
    # 최종 모델 저장
    model.save(os.path.join(model_dir, 'final_model.h5'))
    print(f"모델이 {model_dir}에 저장되었습니다.")
    
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
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    print("평가 결과:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main() 