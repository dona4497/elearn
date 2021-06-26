from keras.applications import VGG16
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD


def vgg161(l):
    baseModel = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(128, 128, 3)))
    # construct the head of the model that will be placed on top of the
    # the base model
    for layer in baseModel.layers[:]:
        layer.trainable = False

    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    # headModel = Dense(512, activation="relu")(headModel)
    # headModel = Dropout(0.5)(headModel)
    # headModel = Dense(l, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    return model




    
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input,Conv2D,MaxPool2D,GlobalAveragePooling2D,LSTM
from keras.layers import TimeDistributed, GRU, Dense, Dropout,Activation,GlobalMaxPool2D,Concatenate,BatchNormalization

opt = SGD(lr=0.01)
def main1(shape):
	#  with (112, 112, 3) input shape
	m1= vgg161(5)
	
	# then create our final model
	model1 =Sequential()
	# with (5, 112, 112, 3) shape
	model1.add(TimeDistributed(m1, input_shape=shape))
	
	model1.add(LSTM(64))
	

	x=Dense(512,activation='relu')(model1.output)
	x1=Dense(5,activation='softmax')(x)
    
    
	model = Model(inputs=model1.input, outputs=x1)
	model.compile(optimizer=opt, loss='categorical_crossentropy',
	               metrics=['accuracy'])
	model.summary()
	return model

#main1((2,128,128,3))    





