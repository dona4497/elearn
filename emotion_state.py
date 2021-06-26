from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense,LSTM,TimeDistributed
import os
import pandas as pd
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

def read_from_folder(path):
    print("==================================================")
    print("Scanning Folders....")
    images=[]
    labels=[]
    subfolders= [f for f in os.scandir(path) if f.is_dir()]
    c=0
    for sf in subfolders:
        imgs1=[f for f in os.scandir(sf.path)]
        for img in imgs1:
            path1=path+'/'+sf.name+'/'+img.name
            img1=load_process_image(path1)
            img1=np.array(img1)
            images.append([img1,img1])
            labels.append(c)
        c=c+1              
    print('found ',len(images),' images belonging to',len(subfolders),' classes')
    print("converting to dataframe...")
    image_dict={'images':images,'labels':labels}
    image_df=pd.DataFrame(image_dict)
    image_df= image_df.sample(frac=1).reset_index(drop=True)
    print("dataframe ready")
    return image_df

def load_process_image(path):
    img=image.load_img(path,target_size=(244,244))
    # norm = np.linalg.norm(img)
    # img = img/norm
    # img/=255.0
    return img

df=read_from_folder('E:\Arjun\Emotion detection\small')
#training data
# df['images'] = df['images'].apply(lambda im: np.array(im))
x_train=np.stack(df['images'],axis=0)
x_train=x_train.astype('float32')
x_train=x_train/255
print(x_train.shape)
y_train=np.array(df['labels'])
test=df.sample(frac=0.5).reset_index(drop=True)
#testing data
x_test=np.stack(test['images'],axis=0)
x_test=x_test.astype('float32')
x_test=x_test/255
y_test=np.array(test['labels'])
# x_train = np.reshape(x_train,(len(df['images']),150,150, 3))
# x_test = np.reshape(x_test,(len(test['images']),150,150, 3))
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(x_train.shape)



vgg16_model =VGG16()
print(type(vgg16_model))

v16 = Sequential()
for layer in vgg16_model.layers[:-1]:
    v16.add(layer)

for layer in v16.layers:
    layer.trainable = False

model = Sequential()    
model.add(TimeDistributed(v16,input_shape=(2,244,244,3)))
model.add(LSTM(128))
model.add(Dense(units=5, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy')



model.fit(x_train, y_train, epochs=2,
        shuffle=True,
        batch_size=8, validation_data=(x_test, y_test),verbose=1)

# model_json = model.to_json()
# with open("emotion_model1.json", "w") as json_file:
#     json_file.write(model_json)

model.save('emotion_model.md')

