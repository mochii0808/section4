import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
tfds.disable_progress_bar()


# 데이터
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

#-----------------------------------------------------------------------------------------------------


# 전처리 함수


# 이미지 정규화
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask
     

# 훈련 데이터
# 이미지와 레이블 호출, 사이즈 조정, flip 및 정규화
@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    # 이미지 증강(크기 조정, 반전)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask
     

# 검증 데이터
# 이미지와 레이블 호출, 사이즈 조정, 정규화
def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    # 검증 데이터 증강 X

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

#==============================================================

# 전처리 적용

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
     
train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)


train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
                    # cache : preprocessing 시간을 줄이고 싶을 때
                    # shuffle : 숫자를 데이터 수만큼 설정하면 완전 랜덤 셔플
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                    # prefetch : 학습중일 때 로드 시간을 줄이기 위해 미리 메모리에 적재
test_dataset = test.batch(BATCH_SIZE)

#------------------------------------------------------------------------------------------------------


# masking 확인

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for image, mask in train.take(1): # 원하는 이미지 번호 지정
    sample_image, sample_mask = image, mask

display([sample_image, sample_mask])

#-------------------------------------------------------------------------------------------------------


# U-net


# down sampling : MobileNetV2 일부층 활용

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 출력층 변경
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

#==============================================================

# up sampling : Pix2Pix

from tensorflow_examples.models.pix2pix import pix2pix

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

#==============================================================

# U-Net 모델링

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # Downsampling
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # Downsampling 결과를 Concatenate 해줍니다.
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # 출력되는 이미지 크기를 동일하게 하기 위해 마지막 층을 구현
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(3) # output_channels = 3

#==============================================================

# 모델 플롯 확인

tf.keras.utils.plot_model(model, show_shapes=True)


#-------------------------------------------------------------------------------------------------------


# 모델 예측 확인

# 픽셀 레이블 확정
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
                        # ellipsis : 데이터 인덱싱
                        # [..., ~] : 열 선택
    return pred_mask[0]


# 예측한 레이블을 시각화
def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...]))])
        

show_predictions()

#-------------------------------------------------------------------------------------------------------


# 모델 학습

from IPython.display import clear_output # 출력을 지우는 라이브러리

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

# 매 에포크마다 이전 결과를 지우고 새로운 결과를 출력하는 callback 클래스
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\n에포크 이후 예측 예시 {}\n'.format(epoch+1))

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])