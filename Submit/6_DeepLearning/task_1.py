# 課題1: データセット「dogs-vs-cats-redux-kernels-edition」を用いて画像から犬猫を判別するモデル（2値分類タスク）を作成せよ
#
# 以下の項目をJupyter Notebook上で出力した状態で提出すること
#
# - 前処理を実施した場合、その処理を加えた画像（数枚程度）
# - 学習曲線（横軸をEpoch、縦軸をLoss）
# - ハイパーパラメータごとのLossを記録し、最も高い性能を出すパラメータを出力
# - 予測結果（インプットした画像と予測ラベルを併記する）

# 以下よりコードを記入してください  ##############################

#まずマウントする
from google.colab import drive
drive.mount('/content/drive')

#必要なものをimportする
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from keras.applications.vgg16 import VGG16
import numpy as np
from keras.layers import Activation, Convolution2D, Dense, Flatten, MaxPooling2D, BatchNormalization,Dropout,Input
from keras.models import Sequential, load_model, Model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pickle
from keras.models import load_model

#trainデータとtestデータに分ける。
#データを全て使うと時間がかかりすぎる割に精度はあまり上がらなかったので4000ずつ使うことにした。
image_size=100
classes = ["cat","dog"]
num_classes = len(classes)


x = []
y = []


for index, class_name in enumerate(classes):
  img_dir =  "/content/drive/My Drive/task/dogs-vs-cats-redux-kernels-edition/dataset/" + class_name
  files = glob.glob(img_dir + "/*.jpg")
  for i, file in enumerate(files):
    if i >= 4000: break
  
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    
    x.append(data)
    y.append(index)

x = np.array(x)
y = np.array(y)

#numpyとしてセーブする

x_train, x_test, y_train, y_test = train_test_split(x,y)
xy = (x_train, x_test, y_train, y_test)
np.save("/content/drive/My Drive/task/dogs-vs-cats-redux-kernels-edition/dataset/cat_dog.npy", xy)

xy = np.load("/content/drive/My Drive/task/dogs-vs-cats-redux-kernels-edition/dataset/cat_dog.npy", allow_pickle=True)

(x_train, x_test, y_train, y_test) = xy

x_train = x_train.astype("float") / 255
x_test  = x_test.astype("float")  / 255

y_train = np_utils.to_categorical(y_train, num_classes)
y_test  = np_utils.to_categorical(y_test, num_classes)

#まずcnnのモデルを作る
cnn_model = Sequential()

cnn_model.add(Convolution2D(32,(3,3),padding="same", input_shape=(img_width,img_height,3), ))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Convolution2D(32,(3,3),padding="same", input_shape=(img_width,img_height,3)))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Dropout(0.25))


cnn_model.add(Flatten())
cnn_model.add(Dense(512))
cnn_model.add(Activation("relu"))

cnn_model.add(BatchNormalization())

cnn_model.add(Dense(256))
cnn_model.add(Activation("relu"))

cnn_model.add(Dense(128))
cnn_model.add(Activation("relu"))

cnn_model.add(BatchNormalization())
cnn_model.add(Dense(2))
cnn_model.add(Activation("softmax"))

cnn_model.compile(loss="binary_crossentropy",
             optimizer=optimizers.SGD(lr=0.001, momentum=0.9),
             metrics=["accuracy"])

cnn_history = cnn_model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data = (x_test, y_test))

#accとlossを可視化する
acc = cnn_history.history['accuracy']
val_acc = cnn_history.history['val_accuracy']
loss = cnn_history.history['loss']
val_loss = cnn_history.history['val_loss']

epochs = range(len(acc))


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('cnn_model')
plt.legend()


plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('cnn_model')
plt.legend()

#結果が微妙だったのでvgg16を使ってモデルを作る。
#ここからはハイパーパラメータを変えて調整していく。
nput_tensor = Input(shape=(100,100,3))
vgg16 = VGG16(include_top = False, weights="imagenet", input_tensor=input_tensor)

top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:19]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data = (x_test, y_test))

top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(512,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))

top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.8)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=2, epochs=8, validation_data = (x_test, y_test))


top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=16, epochs=13, validation_data = (x_test, y_test))


top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data = (x_test, y_test))


top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=2, epochs=5, validation_data = (x_test, y_test))

vgg16_acc = vgg16_history.history['accuracy']
vgg16_val_acc = vgg16_history.history['val_accuracy']
vgg16_loss = vgg16_history.history['loss']
vgg16_val_loss = vgg16_history.history['val_loss']

epochs = range(len(vgg16_acc))


plt.plot(epochs, vgg16_acc, 'bo', label='Training acc')
plt.plot(epochs, vgg16_val_acc, 'b', label='Validation acc')
plt.title('vgg16_model')
plt.legend()


plt.figure()

plt.plot(epochs, vgg16_loss, 'bo', label='Training loss')
plt.plot(epochs, vgg16_val_loss, 'b', label='Validation loss')
plt.title('vgg16_model')
plt.legend()


top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=8, epochs=5, validation_data = (x_test, y_test))


top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data = (x_test, y_test))


vgg16_acc = vgg16_history.history['accuracy']
vgg16_val_acc = vgg16_history.history['val_accuracy']
vgg16_loss = vgg16_history.history['loss']
vgg16_val_loss = vgg16_history.history['val_loss']

epochs = range(len(vgg16_acc))


plt.plot(epochs, vgg16_acc, 'bo', label='Training acc')
plt.plot(epochs, vgg16_val_acc, 'b', label='Validation acc')
plt.title('vgg16_model')
plt.legend()


plt.figure()

plt.plot(epochs, vgg16_loss, 'bo', label='Training loss')
plt.plot(epochs, vgg16_val_loss, 'b', label='Validation loss')
plt.title('vgg16_model')
plt.legend()

top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data = (x_test, y_test))


vgg16_acc = vgg16_history.history['accuracy']
vgg16_val_acc = vgg16_history.history['val_accuracy']
vgg16_loss = vgg16_history.history['loss']
vgg16_val_loss = vgg16_history.history['val_loss']

epochs = range(len(vgg16_acc))


plt.plot(epochs, vgg16_acc, 'bo', label='Training acc')
plt.plot(epochs, vgg16_val_acc, 'b', label='Validation acc')
plt.title('vgg16_model')
plt.legend()


plt.figure()

plt.plot(epochs, vgg16_loss, 'bo', label='Training loss')
plt.plot(epochs, vgg16_val_loss, 'b', label='Validation loss')
plt.title('vgg16_model')
plt.legend()


top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-8, decay=1e-4),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data = (x_test, y_test))


fine_top_model = vgg16.output
fine_top_model = Flatten(input_shape=vgg16.output_shape[1:])(fine_top_model)
fine_top_model = Dense(256,activation="sigmoid")(fine_top_model)
fine_top_model = Dropout(0.5)(fine_top_model)
fine_top_model = Dense(2, activation="softmax")(fine_top_model)

fine_model = Model(inputs=vgg16.input, outputs=fine_top_model)

for layer in vgg16.layers:
  if layer.name.startswith("block5_conv"):
    layer.trainable = True
  else:
    layer.trainable=False

fine_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
             metrics=["accuracy"])

fine_history = fine_model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data = (x_test, y_test))



fine_acc = fine_history.history['accuracy']
fine_val_acc = fine_history.history['val_accuracy']
fine_loss = fine_history.history['loss']
fine_val_loss = fine_history.history['val_loss']

fine_epochs = range(len(fine_acc))


plt.plot(fine_epochs, fine_acc, 'bo', label='Training acc')
plt.plot(fine_epochs, fine_val_acc, 'b', label='Validation acc')
plt.title('fine_model')
plt.legend()


plt.figure()

plt.plot(fine_epochs, fine_loss, 'bo', label='Training loss')
plt.plot(fine_epochs, fine_val_loss, 'b', label='Validation loss')
plt.title('fine_model')
plt.legend()



top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=["accuracy"])

vgg16_history = vgg16_model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data = (x_test, y_test))


vgg16_acc = vgg16_history.history['accuracy']
vgg16_val_acc = vgg16_history.history['val_accuracy']
vgg16_loss = vgg16_history.history['loss']
vgg16_val_loss = vgg16_history.history['val_loss']

epochs = range(len(vgg16_acc))


plt.plot(epochs, vgg16_acc, 'bo', label='Training acc')
plt.plot(epochs, vgg16_val_acc, 'b', label='Validation acc')
plt.title('vgg16_model')
plt.legend()


plt.figure()

plt.plot(epochs, vgg16_loss, 'bo', label='Training loss')
plt.plot(epochs, vgg16_val_loss, 'b', label='Validation loss')
plt.title('vgg16_model')
plt.legend()


vgg16_model.save("/content/drive/My Drive/task/dogs-vs-cats-redux-kernels-edition/dataset/vgg16_model.h5")
vgg16_model = load_model("/content/drive/My Drive/task/dogs-vs-cats-redux-kernels-edition/dataset/vgg16_model.h5")

#保存したモデルを使い予測して結果を表示する。

submmit_dir =  "/content/drive/My Drive/task/dogs-vs-cats-redux-kernels-edition/submmit/" 
submmit_imgs = glob.glob(submmit_dir + "*")

submmit = []

for i in range(1,10001):
  image = Image.open(submmit_imgs[i])
  image = image.convert("RGB")
  image = image.resize((image_size, image_size))
  data = np.asarray(image)

  submmit.append(data)

submmit = np.array(submmit)
np.save("/content/drive/My Drive/task/dogs-vs-cats-redux-kernels-edition/submmit/submmit.npy", submmit)


submmit = np.load("/content/drive/My Drive/task/dogs-vs-cats-redux-kernels-edition/submmit/submmit.npy")

pred = np.argmax(vgg16_model.predict(submmit[0:15]), axis=1)
pred_label = []

def to_label(x):
  if x == 0:
    return "cat"
  else:
    return "dog"
    
plt.figure(figsize=(15,15))
for i in range(15):
  plt.subplot(3,5,i+1)
  plt.imshow(submmit[i], "gray")
  plt.axis("off")
  plt.title(to_label(pred[i]))
plt.show




























