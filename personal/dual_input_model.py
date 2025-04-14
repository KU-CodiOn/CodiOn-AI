import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Input, Concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import csv
import json
import shutil
import traceback
from pathlib import Path

# 데이터 경로 설정
FACE_DATA_DIR = 'dataset/faces_dlib'
FEATURES_FILE = 'dataset/skin_features.json'  # JSON 피처 파일로 변경
PROCESSED_DATA_DIR = 'dataset/faces_processed_safe_dual'
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 150
NUM_CLASSES = 4
LEARNING_RATE = 0.00002  # 매우 낮은 학습률

# 피처 이름 및 수 설정
FEATURE_COLUMNS = [
    'face_r_mean', 'face_g_mean', 'face_b_mean',
    'face_r_std', 'face_g_std', 'face_b_std',
    'face_h_mean', 'face_s_mean', 'face_v_mean',
    'face_h_std', 'face_s_std', 'face_v_std',
    'face_l_mean', 'face_a_mean', 'face_b_mean_lab',
    'face_l_std', 'face_a_std', 'face_b_std_lab',
    'dominant_h', 'dominant_s', 'dominant_v'
]
NUM_FEATURES = len(FEATURE_COLUMNS)

# 피처 데이터 로드 함수
def load_feature_data(features_file):
    """JSON 파일에서 피처 데이터 로드"""
    all_features = {}
    
    # 파일이 존재하는지 확인
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"피처 파일을 찾을 수 없습니다: {features_file}")
    
    try:
        with open(features_file, 'r') as f:
            features_data = json.load(f)
            
            for filename, features_dict in features_data.items():
                # 피처 값을 추출
                features = []
                for col in FEATURE_COLUMNS:
                    # 데이터가 없는 경우 0으로 처리
                    if col in features_dict and features_dict[col] is not None:
                        try:
                            value = float(features_dict[col])
                            features.append(value)
                        except (ValueError, TypeError):
                            features.append(0.0)
                    else:
                        features.append(0.0)
                
                # 모든 피처가 누락된 경우 예외 처리
                if all(f == 0.0 for f in features):
                    print(f"경고: {filename}의 모든 피처가 0입니다.")
                
                all_features[filename] = features
        
        print(f"총 {len(all_features)}개의 피처 데이터를 로드했습니다.")
        return all_features
    except Exception as e:
        print(f"피처 데이터 로드 중 오류 발생: {e}")
        traceback.print_exc()
        raise

class DualInputGenerator(tf.keras.utils.Sequence):
    """이미지와 피처 데이터를 함께 제공하는 제너레이터"""
    def __init__(self, data_dir, features_data, batch_size=32, img_size=(224, 224),
                 shuffle=True, subset='training', validation_split=0.2,
                 augment=False, season_classes=None):
        self.data_dir = data_dir
        self.features_data = features_data
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.subset = subset
        self.validation_split = validation_split
        self.augment = augment
        self.season_classes = season_classes or ['spring', 'summer', 'autumn', 'winter']
        
        # 클래스 인덱스 생성
        self.class_indices = {season: i for i, season in enumerate(self.season_classes)}
        self.indices = []
        
        # 데이터 경로 및 레이블 준비
        self.samples = []
        
        for season in self.season_classes:
            season_dir = os.path.join(data_dir, season)
            if not os.path.exists(season_dir):
                print(f"경고: {season_dir} 디렉토리를 찾을 수 없습니다.")
                continue
                
            for filename in os.listdir(season_dir):
                if filename.endswith('.jpg') and not filename.endswith('_landmarks.jpg'):
                    img_path = os.path.join(season_dir, filename)
                    
                    # 이미지 경로와 레이블, 피처 데이터 저장
                    label = self.class_indices[season]
                    
                    # 피처 데이터가 있는지 확인 (파일명 기준으로 매칭)
                    has_feature = filename in self.features_data
                    if not has_feature:
                        # 기본 피처 생성 (모두 0)
                        print(f"경고: {filename}의 피처 데이터가 없습니다. 기본값을 사용합니다.")
                    
                    self.samples.append((img_path, label, filename))
        
        # 검증 세트와 훈련 세트로 분할
        np.random.seed(42)
        np.random.shuffle(self.samples)
        
        n_samples = len(self.samples)
        split_idx = int(n_samples * (1 - validation_split))
        
        if subset == 'training':
            self.samples = self.samples[:split_idx]
        else:  # validation
            self.samples = self.samples[split_idx:]
        
        self.n_samples = len(self.samples)
        self.indices = np.arange(self.n_samples)
        self.on_epoch_end()
        
        print(f"{subset} 세트에 {self.n_samples}개의 샘플이 있습니다.")
    
    def __len__(self):
        """배치 수 반환"""
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, index):
        """배치 데이터 반환"""
        # 배치에 해당하는 인덱스 가져오기
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # 이미지와 피처 데이터, 레이블 준비
        batch_imgs = np.zeros((len(batch_indices), *self.img_size, 3), dtype=np.float32)
        batch_features = np.zeros((len(batch_indices), NUM_FEATURES), dtype=np.float32)
        batch_labels = np.zeros((len(batch_indices), len(self.class_indices)), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            img_path, label, filename = self.samples[idx]
            
            # 이미지 로드 및 전처리
            try:
                img = load_img(img_path, target_size=self.img_size)
                img_array = img_to_array(img) / 255.0  # 정규화
                batch_imgs[i] = img_array
            except Exception as e:
                print(f"이미지 로드 중 오류: {img_path}, {e}")
                # 오류 발생 시 검은색 이미지로 대체
                batch_imgs[i] = np.zeros((*self.img_size, 3), dtype=np.float32)
            
            # 피처 데이터 할당
            if filename in self.features_data:
                batch_features[i] = self.features_data[filename]
            else:
                # 기본 피처 데이터 사용
                batch_features[i] = np.zeros(NUM_FEATURES, dtype=np.float32)
            
            # 원-핫 인코딩 레이블 생성
            batch_labels[i, label] = 1.0
        
        # 이미지 증강 (훈련 세트에만 적용)
        if self.augment and self.subset == 'training':
            # 여기에 데이터 증강 로직 추가 (향후 구현)
            pass
        
        # 입력 데이터와 레이블 반환
        return {'image_input': batch_imgs, 'feature_input': batch_features}, batch_labels
    
    def on_epoch_end(self):
        """에포크 종료 시 호출되는 메서드"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_features_shape(self):
        """피처 데이터 형태 반환"""
        return (NUM_FEATURES,)
    
    def get_classes(self):
        """클래스 목록 반환"""
        return np.array([sample[1] for sample in self.samples])  # 레이블 배열 반환

def create_tf_dataset(generator):
    """Sequence 제너레이터를 tf.data.Dataset으로 변환"""
    def gen_func():
        for i in range(len(generator)):
            inputs, labels = generator[i]
            yield inputs, labels
    
    output_signature = (
        {
            'image_input': tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            'feature_input': tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32)
        },
        tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
    )
    
    return tf.data.Dataset.from_generator(
        gen_func,
        output_signature=output_signature
    )

def create_dual_input_model():
    """이미지와 피처를 함께 사용하는 듀얼 인풋 모델 생성"""
    # 이미지 입력 분기
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    
    # EfficientNetB0 기반 모델
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=img_input
    )
    
    # 레이어 동결 설정
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # 이미지 특성 추출
    x1 = base_model.output
    x1 = GlobalAveragePooling2D()(x1)
    x1 = BatchNormalization()(x1)
    
    # 피처 입력 분기
    feature_input = Input(shape=(NUM_FEATURES,), name='feature_input')
    x2 = BatchNormalization()(feature_input)
    x2 = Dense(64, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # 두 분기 결합
    combined = Concatenate()([x1, x2])
    
    # 분류기
    x = Dense(512, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # 모델 생성
    model = Model(inputs=[img_input, feature_input], outputs=outputs)
    
    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

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

def compute_class_weights(dual_gen):
    """클래스 불균형 처리를 위한 가중치 계산"""
    # 클래스 인덱스 확인
    class_indices = dual_gen.class_indices
    class_indices_inv = {v: k for k, v in class_indices.items()}
    
    # 클래스별 가중치 계산
    samples = dual_gen.samples
    classes = [sample[1] for sample in samples]  # 레이블 추출
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    
    # 가중치 제한 (극단적인 가중치 방지)
    class_weights = np.clip(class_weights, 0.7, 1.5)
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("클래스 인덱스:", class_indices_inv)
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
        print("==== 듀얼 인풋 퍼스널 컬러 분류 모델 학습 시작 ====")
        
        # 현재 시간을 기반으로 모델 저장 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join('models', f'face_model_dual_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)
        print(f"모델 저장 디렉토리: {model_dir}")
        
        # 안전한 데이터 디렉토리 준비
        processed_data_dir, count_per_season = safe_prepare_data_directory()
        print(f"데이터 디렉토리 준비 완료: {processed_data_dir}")
        
        # 피처 데이터 로드
        print("\n피처 데이터 로드 중...")
        features_data = load_feature_data(FEATURES_FILE)
        
        # 듀얼 인풋 데이터 생성기 생성
        print("\n데이터 생성기 초기화 중...")
        train_generator = DualInputGenerator(
            processed_data_dir,
            features_data,
            batch_size=BATCH_SIZE,
            img_size=(IMG_SIZE, IMG_SIZE),
            shuffle=True,
            subset='training',
            validation_split=0.2,
            augment=True
        )
        
        validation_generator = DualInputGenerator(
            processed_data_dir,
            features_data,
            batch_size=BATCH_SIZE,
            img_size=(IMG_SIZE, IMG_SIZE),
            shuffle=False,
            subset='validation',
            validation_split=0.2,
            augment=False
        )
        
        # tf.data.Dataset으로 변환
        print("\ntf.data.Dataset으로 변환 중...")
        train_dataset = create_tf_dataset(train_generator)
        validation_dataset = create_tf_dataset(validation_generator)
        
        # 데이터셋 최적화
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
        
        # 데이터 파이프라인 테스트
        print("\n데이터 파이프라인 테스트 중...")
        try:
            # tf.data.Dataset에서 샘플 가져오기
            for inputs, labels in train_dataset.take(1):
                print(f"훈련 데이터 배치 형태 - 이미지: {inputs['image_input'].shape}, 피처: {inputs['feature_input'].shape}, 레이블: {labels.shape}")
            
            for inputs, labels in validation_dataset.take(1):
                print(f"검증 데이터 배치 형태 - 이미지: {inputs['image_input'].shape}, 피처: {inputs['feature_input'].shape}, 레이블: {labels.shape}")
            
            print("데이터 파이프라인 테스트 성공!")
        except Exception as e:
            print(f"데이터 테스트 중 오류 발생: {e}")
            print("데이터 준비에 실패했습니다. 모델 훈련을 중단합니다.")
            return
        
        # 클래스 가중치 계산
        class_weight_dict = compute_class_weights(train_generator)
        
        # 듀얼 인풋 모델 생성
        print("\n듀얼 인풋 모델 생성 중...")
        model = create_dual_input_model()
        print("듀얼 인풋 모델 생성 완료!")
        
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
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
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
            train_dataset,
            epochs=EPOCHS,
            validation_data=validation_dataset,
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
        print(f"듀얼 인풋 모델이 {model_dir}에 저장되었습니다.")
        
        # 모델 평가
        evaluation = model.evaluate(validation_dataset)
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
        
        print("==== 듀얼 인풋 모델 학습 완료 ====")
    
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