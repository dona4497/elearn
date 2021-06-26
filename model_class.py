from keras.models import model_from_json
import numpy as np
import cv2

class EmotionModel(object):
    emotion_list=[
        'Angry','Fear',
        'Happy','Sadness',
        'Surprise'
    ]
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
            

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)[0]
        print(self.preds)
        print(np.argmax(self.preds))
        return (EmotionModel.emotion_list[np.argmax(self.preds)],np.argmax(self.preds))
        # return self.preds

from PIL import Image
from PIL import ImageFilter
def edge_enhance(img):
    imageObject =Image.fromarray(img)
    edge = imageObject.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    edge=np.array(edge)
    print(edge.shape)
    # edge= np.stack((edge,)*3, axis=-1)
    edge=cv2.resize(edge,(128,128))
    edge=[edge,edge]
    edge=np.array(edge)
    return edge
if __name__ == '__main__':
    model=EmotionModel("emotion_model1.json", "emotion_model1.h5")
    img=cv2.imread('sadness.png')
    img=edge_enhance(img)
    img=np.reshape([img],(1,2,128,128,3))
    pred=model.predict_emotion(img)
    print(pred)

