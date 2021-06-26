import os
import pandas as pd
from PIL import Image
from PIL import Image
from PIL import ImageFilter
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
            img1=edge_enhance(path1)
            images.append(img1)
            labels.append(c)
        c=c+1              
    print('found ',len(images),' images belonging to',len(subfolders),' classes')
    print("converting to dataframe...")
    image_dict={'images':images,'labels':labels}
    image_df=pd.DataFrame(image_dict)
    image_df= image_df.sample(frac=1).reset_index(drop=True)
    print("dataframe ready")
    return image_df
import cv2
def edge_enhance(path):
    img=cv2.imread(path)
    imageObject =Image.fromarray(img)
    edge = imageObject.filter(ImageFilter.EDGE_ENHANCE_MORE)
    edge=np.array(edge)
    # edge= np.stack((edge,)*3, axis=-1)
    edge=cv2.resize(edge,(128,128))
    edge=[edge,edge]
    edge=np.array(edge)
    return edge




edge=edge_enhance('anger.png')
print(edge.shape)

df=read_from_folder('ck+48')



from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
#training data
# df['images'] = df['images'].apply(lambda im: np.array(im))
x_train=np.vstack(train['images'])
y_train=np.array(train['labels'])

#testing data
x_test=np.vstack(test['images'])
y_test=np.array(test['labels'])
x_train = np.reshape(x_train,(len(train['images']),2,128,128, 3))
x_test = np.reshape(x_test,(len(test['images']),2,128,128, 3))
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(x_train.shape)


from model import main1

model=main1((2,128,128,3))
checkpointer = ModelCheckpoint(filepath='emotion_model1.h5', verbose=1, save_best_only=True ,monitor='val_accuracy')


model.fit(x_train, y_train, epochs=70,
        shuffle=True,
        batch_size=16, validation_data=(x_test, y_test),
        callbacks=[checkpointer], verbose=1)

model_json = model.to_json()
with open("emotion_model1.json", "w") as json_file:
    json_file.write(model_json)    



