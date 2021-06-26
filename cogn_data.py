import cv2
from model_class import EmotionModel
import numpy as np
import pyHook
import pythoncom
import time
import cv2
import time
import math
import pandas as pd
from audio import TapTester


def onclick(event):
    global x_loc,y_loc
    x,y=event.Position
    x_loc.append(str(x))
    y_loc.append(str(y))
    return True
# video capture

print("capture")
# face detector
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def __get_data__(fr):
    # st,fr = rgb.read()
    #gray scale conversion
    print(fr)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # face detection
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, fr, gray

def start_app(cnn):
    rgb = cv2.VideoCapture(0)
    EMOTIONS_LIST = ["Angry", "Disgust","Fear", "Happy",
                     "Sad"]
    global x_loc,y_loc,time_series,amplitudes,e_nums
    x_loc=[]
    y_loc=[]    
    time_series=[]  
    amplitudes=[]  
    e_nums=[]        
    lis=[]
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    hm = pyHook.HookManager()
    hm.SubscribeMouseAll(onclick)
    hm.HookMouse()
    tt=TapTester()
    # contineous frame reading
    count=0
    while True:
        
        amplitudes.append(str(tt.listen()))
        ret, frame1 = rgb.read()
        frame=frame1
        cv2.line(frame, (320,0), (320,480), (0,200,0), 2)
        cv2.line(frame, (0,200), (640,200), (0,200,0), 2)
        if ret==True:
            col=frame
            
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            pupilFrame=frame
            clahe=frame
            blur=frame
            edges=frame
            eyes = cv2.CascadeClassifier('haarcascade_eye.xml')
            detected = eyes.detectMultiScale(frame, 1.3, 5)
            for (x,y,w,h) in detected: #similar to face detection
                cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255),1)	 #draw rectangle around eyes
                cv2.line(frame, (x,y), ((x+w,y+h)), (0,0,255),1)   #draw cross
                cv2.line(frame, (x+w,y), ((x,y+h)), (0,0,255),1)
                pupilFrame = cv2.equalizeHist(frame[math.ceil(y+(h*.25)):(y+h), x:(x+w)]) #using histogram equalization of better image. 
                cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #set grid size
                clahe = cl1.apply(pupilFrame)  #clahe
                blur = cv2.medianBlur(clahe, 7)  #median blur
                circles = cv2.HoughCircles(blur ,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=25,minRadius=4,maxRadius=25) #houghcircles
                if circles is not None: #if atleast 1 is detected
                    circles = np.round(circles[0, :]).astype("int") #change float to integer
                    print (circles)
                    for (x,y,r) in circles:
                        cv2.circle(pupilFrame, (x, y), r, (0, 255, 255), 2)
                        cv2.rectangle(pupilFrame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                        #set thresholds
                        #thresholding(x)
                        time_series.append(str(x))  
                        count+=1        
        ix += 1
        pythoncom.PumpWaitingMessages()
        faces, fr, gray_fr = __get_data__(frame1)
        for (x, y, w, h) in faces:
            fc = fr[y:y+h, x:x+w]
            
            im =fc
            im=np.array(im)
            im=cv2.resize(im,(128,128))
            imgs=[]
            imgs.append([im,im])
            #imgs.append(im)
            imgs=np.stack(imgs,axis=0)
            # emotion prediction from face
            emotion,e_num = cnn.predict_emotion(imgs)
            e_nums.append(str(e_num))
            # add text in the window
            cv2.putText(fr, emotion, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            lis.append(emotion)
        # if cv2.waitKey(1) == 27:
        #     break
        print("count--------",str(count))
        if count>10:
            return 0
        #cv2.imshow('Out', fr)
    #cv2.destroyAllWindows()
    dic={}
    for i in EMOTIONS_LIST:
        dic[i]=lis.count(i)
    print("dictionary")
    return dic

def main2(c):
   
    # load trained models
    global x_loc,y_loc,time_series,amplitudes,e_nums
    model = EmotionModel("emotion_model1.json", "emotion_model1.h5")
    values=start_app(model)
    x_loc=' '.join(x_loc)
    y_loc=' '.join(y_loc)
    time_series=' '.join(time_series)
    amplitudes=' '.join(amplitudes)
    e_nums=' '.join(e_nums)
    if len(time_series)<1:
        time_series='2 4'
    if len(x_loc)<1:
        x_loc='2 4'
    if len(y_loc)<1:
        y_loc='2 4' 
    if len(e_nums)<1:
        e_nums='2 4' 
    if len(amplitudes)<1:
        amplitudes='2 4'               
    
    #try:

    df1=pd.read_csv('cognitive.csv')
    dic={}
    dic['m_x']=[x_loc]
    dic['m_y']=[y_loc]
    dic['eye']=[time_series]
    dic['amplitudes']=[y_loc]
    dic['emot']=[e_nums]
    dic['c_state']=[c]
    df=pd.DataFrame(dic)
    df1=pd.concat([df1[['m_x','m_y','eye','amplitudes','emot','c_state']],df[['m_x','m_y','eye','amplitudes','emot','c_state']]]).sample(frac=1).reset_index(drop=True)
    df1.to_csv('cognitive.csv')
    print('good==================================================')
    print('done')
        
##    except:
##        dic={}
##        dic['m_x']=[x_loc]
##        dic['m_y']=[y_loc]
##        dic['eye']=[time_series]
##        dic['amplitudes']=[y_loc]
##        dic['emot']=[e_nums]
##        dic['c_state']=[c]
##        df=pd.DataFrame(dic)
##        df.to_csv('cognitive.csv')
##        print('done')
    x_loc=[]
    y_loc=[]    
    time_series=[]  
    amplitudes=[]  
    e_nums=[]        
    lis=[]
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    
    # contineous frame reading
    count=0    
    return values
#main(0.1)
