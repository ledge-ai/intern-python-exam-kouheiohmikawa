# 課題2: データセット「intel-image-classification-dataset」を用いて画像から建物や森などの風景を判別するモデル（マルチラベル分類）を作成せよ
#
# 以下の項目をJupyter Notebook上で出力した状態で提出すること
#
# - 前処理を実施した場合、その処理を加えた画像（数枚程度）
# - 学習曲線（横軸をEpoch、縦軸をLoss）
# - ハイパーパラメータごとのLossを記録し、最も高い性能を出すパラメータを出力
# - 予測結果（インプットした画像と予測ラベルを併記する）

# 以下よりコードを記入してください  ##############################

#画像サイズがバラバラだったので揃える。
#enumerateで1つのコードにしようと思ったらエラーが出てしまったので1つずつ書いた
#一応、画像の名前の数字がバラバラだったので順番にする

classes = ["buildings", "forest", "glacier", "sea", "street"]
num_classes = len(classes)




for index, class_name in enumerate(classes):
  img_dir =  "/content/drive/My Drive/task/dogs-vs-cats-redux-kernels-edition/dataset/" + class_name
  files = glob.glob(img_dir + "/*.jpg")
  for f in files:
    img = Image.open(f)
    img_resize = img.resize((image_size, image_size))
    img_resize.save(f)
    
path = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/buildings"
files = glob.glob(path + '/*')

for i, old_name in enumerate(files):
    
    new_name = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/buildings/{0:00d}.jpg".format(i)
    
    os.rename(old_name, new_name)
    
    
path = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/forest"
files = glob.glob(path + '/*')

for i, old_name in enumerate(files):
   
    new_name = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/forest/{0:00d}.jpg".format(i)
    
    os.rename(old_name, new_name)
    
path = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/glacier"
files = glob.glob(path + '/*')

for i, old_name in enumerate(files):
    
    new_name = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/glacier/{0:00d}.jpg".format(i)
    
    os.rename(old_name, new_name)
    
path = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/mountain"
files = glob.glob(path + '/*')

for i, old_name in enumerate(files):
   
    new_name = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/mountain/{0:00d}.jpg".format(i)
  
    os.rename(old_name, new_name)
    

path = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/sea"
files = glob.glob(path + '/*')

for i, old_name in enumerate(files):
    
    new_name = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/sea/{0:00d}.jpg".format(i + 1)
   
    os.rename(old_name, new_name)
    

path = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/street"
files = glob.glob(path + '/*')

for i, old_name in enumerate(files):
    
    new_name = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/street/{0:00d}.jpg".format(i)
 
    os.rename(old_name, new_name)
    
#2つずつ画像の水増しをして保存する。
def draw_images(generator, x, output_dir, index):
   
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=save_name, save_format='jpg')

    for i in range(2):
        bach = g.next()

generator = ImageDataGenerator(
                    rotation_range=45, 
                    width_shift_range=0.2, 
                    height_shift_range=0.3, 
                    channel_shift_range=150.0, 
                    shear_range=0.39, 
                    horizontal_flip=True, 
                    vertical_flip=True 
                    )
                    
                
img_dir =  "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/street" 
files = glob.glob(img_dir + "/*")

for i in range(len(files)):
  img = load_img(files[i])
       
  x = img_to_array(img)
  x = np.expand_dims(x, axis=0)
      
  draw_images(generator, x, img_dir, i)
  
 
train_data_dir = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset"

input_tensor = Input(shape=(100,100,3))
vgg16 = VGG16(include_top = False, weights="imagenet", input_tensor=input_tensor)

top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256,activation="sigmoid")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(6, activation="softmax")(top_model)

vgg16_model = Model(inputs=vgg16.input, outputs=top_model)

for layer in vgg16_model.layers[:15]:
  layer.trainable=False

vgg16_model.compile(loss="categorical_crossentropy",
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=["accuracy"])
             
             
             
train_datagen = ImageDataGenerator(
    rotation_range=45, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.1, 
    zoom_range=0.1, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rescale=1.0 / 255, 
    validation_split=0.2
    )

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') 

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, 
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') 
    

vgg16_history = vgg16_model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.samples // batch_size,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps = validation_generator.samples // batch_size)


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


submmit_dir =  "/content/drive/My Drive/task/intel-image-classification-dataset/submmit/" 
submmit_imgs = glob.glob(submmit_dir + "*")

submmit = []

len(submmit_imgs)

for i in range(0,7300):
  image = Image.open(submmit_imgs[i])
  image = image.convert("RGB")
  image = image.resize((image_size, image_size))
  data = np.asarray(image)

  submmit.append(data)

submmit = np.array(submmit)
np.save("/content/drive/My Drive/task/intel-image-classification-dataset/submmit/submmit.npy", submmit)
                    
                    


vgg16_model = load_model("/content/drive/My Drive/task/intel-image-classification-dataset/dataset/vgg16_model.h5")


pred = np.argmax(vgg16_model.predict(submmit[0:20]), axis=1)
pred_label = []

def to_label(x):
  if x == 0:
    return "street"
  elif x == 1:
    return "sea"
  elif x == 2:
    return "mountain"
  elif x == 3:
    return "glacier"
  elif x == 4:
    return "forest"
  elif x == 5:
    return "buildings"
    

plt.figure(figsize=(15,15))
for i in range(20):
  plt.subplot(4,5,i+1)
  plt.imshow(submmit[i], "gray")
  plt.axis("off")
  plt.title(to_label(pred[i]))
plt.show

#前処理した画像を表示する
path = "/content/drive/My Drive/task/intel-image-classification-dataset/dataset/buildings"
files = glob.glob(path + '/*')

plt.figure(figsize=(20,10))
for i in range(10):
  plt.subplot(2,5,i+1)
  im = Image.open(files[i])
  image = np.asarray(im)
  plt.imshow(image)
  plt.axis("off")
  plt.show
                
                
                
                
                
                
                
                
                

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    