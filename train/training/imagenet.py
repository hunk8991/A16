import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)

base_model.trainable = False #凍結

data_augmentation = keras.Sequential( #循序型資料增補
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"), #水平和垂直隨機翻轉每個圖像
        layers.experimental.preprocessing.RandomRotation(0.1), #隨機旋轉每個圖像(0.1弧度值)
    ]
)

#建構模型
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
x = norm_layer(x)
norm_layer.set_weights([mean, var])

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x) #整體平均池化
x = keras.layers.Dropout(0.2)(x) #棄置調節
outputs = keras.layers.Dense(5)(x) #全連接層
model = keras.Model(inputs, outputs)


batch_size = 16 #批次
image_size=(150,150)
loss=tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0,
    reduction="auto",
    name="categorical_crossentropy",
)

train_ds = tf.keras.preprocessing.image_dataset_from_directory( #訓練資料集
    directory="../training",
    labels="inferred",
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory( #驗證資料集
    directory="../test",
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
)


model.compile(optimizer=keras.optimizers.Adam(), #compile 優化器
              loss=loss,metrics=['accuracy']) #損失函數、評價函數(正確率)

model.summary()
print(train_ds)

model.fit(train_ds, validation_data=val_ds, epochs=100) #模型訓練

model.save("vactor.h5")
model.save("model")
