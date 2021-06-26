import tkinter as tk
from PIL import ImageTk, Image
import sqlite3,csv
from tkinter import messagebox
#from camera2 import main
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox,DISABLED,NORMAL
# import pymysql
import datetime
from functools import partial
from PIL import Image, ImageTk
# from testing import process
import time
title="E-learning engagement"
path1="sample.jpg" 
path2="sample1.jpg"
import PIL.Image, PIL.ImageTk


def logcheck():
     global username_var,pass_var
     uname=username_var.get()
     pass1=pass_var.get()
     if uname=="admin" and pass1=="admin":
        pass
     else:
          pass
          messagebox.showinfo("alert","Wrong Credentials") 
          return redirect('gui.py')
         
     showcheck()     

def showeye():
    top.config(menu=menubar)
    global f
    f.pack_forget()
    f=Frame(top)
    f.config(bg="#41ddff")
    f.pack(side="top", fill="both", expand=True,padx=10,pady=10)
    l=Label(f,text="Eye data collection",font = "Verdana 40 bold",fg="white",bg="#41ddff")
    l.place(x=50,y=50)
    l2=Label(f,text="Value (1 for watching 0 for reading):",font="Verdana 10 bold",bg="#41ddff")
    l2.place(x=300,y=300)
    global label_var
    label_var=StringVar()
    e1=Entry(f,textvariable=label_var,font="Verdana 10 bold")
    e1.place(x=700,y=300)
    #showcheck()
    b1=Button(f,text="Create", command=eyecreate,font="Verdana 10 bold")
    b1.place(x=750,y=360)

def showcog():
    top.config(menu=menubar)
    global f
    f.pack_forget()
    f=Frame(top)
    f.config(bg="#41ddff")
    f.pack(side="top", fill="both", expand=True,padx=10,pady=10)
    l=Label(f,text="Cognitive data collection",font = "Verdana 40 bold",fg="white",bg="#41ddff")
    l.place(x=50,y=50)
    l2=Label(f,text="cognitive value:",font="Verdana 10 bold",bg="#41ddff")
    l2.place(x=550,y=300)
    global cabel_var
    cabel_var=StringVar()
    e1=Entry(f,textvariable=cabel_var,font="Verdana 10 bold")
    e1.place(x=700,y=300)
    #showcheck()
    b1=Button(f,text="Create", command=cogcreate,font="Verdana 10 bold")
    b1.place(x=750,y=360)    

from eye_state_data import create_eye
from cogn_data import main2

def eyecreate():
    global label_var
    l=label_var.get()
    try:
        create_eye(l)
        messagebox.showinfo('Success','saved')
    except Exception as e:
        messagebox.showinfo('error','No camera detected or %s'%e)

def cogcreate():
    global cabel_var
    c=cabel_var.get()
    #try:
    main2(c)
    messagebox.showinfo('Success','saved')
##    except Exception as e:
##        messagebox.showinfo('error','No camera detected or %s'%e)




# show home page
def showhome():
    top.config(menu=menubar)
    global f
    f.pack_forget()
    f=Frame(top)
    f.config(bg="#41ddff")
    f.pack(side="top", fill="both", expand=True,padx=10,pady=10)
    image = Image.open("leaf.jpg")
    photo = ImageTk.PhotoImage(image.resize((top.winfo_width(), top.winfo_height()), Image.ANTIALIAS))
    label = Label(f, image=photo, bg='#41ddff')
    label.image = photo
    label.pack()

    l=Label(f,text="Welcome",font = "Verdana 60 bold",fg="White",bg="#41ddff")
    l.place(x=500,y=300)

def showcheck():
    top.title(title)
    top.config(menu=menubar)
    global f,f1,f_bottom,f_top,f_b1

    f.pack_forget()
    f=Frame(top)
    f.config(bg="#41ddff")
    f.pack(side="top", fill="both", expand=True,padx=10,pady=10)
    
    f_bottom=Frame(f)
    f_bottom.config(bg="#41ddff",width=1500,height=130)
    f_bottom.pack_propagate(False)
    f_bottom.pack(side='bottom',fill='both')

    f_b1=Frame(f_bottom)
    f_b1.config(bg="#41ddff",width=500,height=50)
    f_b1.pack_propagate(False)
    f_b1.pack(side='top')


    f_top=Frame(f)
    f_top.config(bg="#41ddff",height=800,width=1500)
    f_top.pack_propagate(False)
    f_top.pack(side='bottom',fill='both')


    f1=Frame(f_top)
    f1.pack_propagate(False)
    f1.config(bg="#41ddff",width=350)
    f1.pack(side="left",fill="both")

    global f2
    f2=Frame(f_top)
    f2.pack_propagate(False)
    f2.config(bg="#41ddff",width=700)
    f2.pack(side="right",fill="both")
    l4=Label(f2,text="Results",font="Helvetica 13 bold")
    l4.pack(fill='both',pady=5,padx=5,side='top')
    f2a=Frame(f2)
    f2a.config(bg="#41ddff")
    f2a.pack(side="top",fill="both",padx=5)


    f3=Frame(f_top)
    f3.pack_propagate(False)
    f3.config(bg="#41ddff",width=700)
    f3.pack(side="right",fill="both")

    


    
    global f4
    f4=Frame(f1)
    f4.pack_propagate(False)
    f4.config(bg="#41ddff",height=200)
    f4.pack(side="bottom",fill="both")

    f7=Frame(f1)
    f7.pack_propagate(False)
    f7.config(height=20)
    f7.pack(side="top",fill="both",padx="3")

    l2=Label(f7,text="Process",font="Helvetica 13 bold")
    l2.pack()

    global lb1,cn1,cn2

    lb1=Listbox(f1,width=400,height=400,font="Helvetica 13 bold")
    lb1.pack(pady=10,padx=5)
    scrollbar = Scrollbar(f1) 
    scrollbar.config(command = lb1.yview) 
    scrollbar.pack(side = RIGHT, fill = BOTH) 
    lb1.config(yscrollcommand = scrollbar.set)

    # for x in range(100):
    #     lb1.insert(END, str(x))
    b2=Button(f4,text="Start",font="Verdana 10 bold",command=detect)
    b2.pack(pady=2)
    b2=Button(f4,text="Stop process",font="Verdana 10 bold",command=stop)
    b2.pack(pady=2)
    
    l3=Label(f3,text="Output",font="Helvetica 13 bold")
    l3.pack(fill='both',pady=5)
    
    global cn1,c211,c212,c221,c222,c231,c232,f3a,lb232
    cn1 =Canvas(f3, width = 300, height = 400)
    cn1.pack()

    f3a=Frame(f3)
    f3a.config(bg="#41ddff",width=350,height=200)
    f3a.pack_propagate(False)
    f3a.pack(side='top',padx=3)

    
    f21=Frame(f2a)
    f21.config(bg="#41ddff")
    f21.pack_propagate(False)
    f21.config(width=200,height=800)
    f21.pack(side='left',padx=3)

    c211 =Canvas(f21, width = 200, height = 200)
    c211.pack(side="top",pady=2)
    l211=Label(f21,text='Grey scale image')
    l211.pack(side="top",pady=2)
    c212 =Canvas(f21, width = 200, height = 200)
    c212.pack(side="top",pady=2)
    l212=Label(f21,text='Result')
    l212.pack(side="top",pady=2)

    f22=Frame(f2a)
    f22.config(bg="#41ddff")
    f22.pack_propagate(False)
    f22.config(width=200,height=800)
    f22.pack(side='left',padx=3)

    c221 =Canvas(f22, width = 200, height = 200)
    c221.pack(side="top",pady=2)
    l221=Label(f22,text='Eye marked')
    l221.pack(side="top",pady=2)
    c222 =Canvas(f22, width = 200, height = 200)
    c222.pack(side="top",pady=2)
    l222=Label(f22,text='Result')
    l222.pack(side="top",pady=2)

    f23=Frame(f2a)
    f23.pack_propagate(False)
    f23.config(bg="#41ddff")
    f23.config(width=200,height=800)
    f23.pack(side='left',padx=3)

    c231 =Canvas(f23, width = 200, height = 200)
    c231.pack(side="top",pady=2)
    l231=Label(f23,text='Cropped face')
    l231.pack(side="top",pady=2)

    lb232=Listbox(f23,width=200,height=10)
    lb232.pack(side="top",pady=2)
    l232=Label(f23,text='Cognitive value')
    l232.pack(side="top",pady=2)
    global s_clicked,d_clicked
    s_clicked=0
    d_clicked=0
    
    
    
def Prev():
    global cn1,c211,c212,c221,c222,c231,c232,f3a,c1
    folder='results/'
    if c1!=1:
        c1-=1
    im1 = Image.open(folder+str(c1)+'.1.jpg')
    im1=im1.resize((300, 400), Image.ANTIALIAS)
    photo1 = PIL.ImageTk.PhotoImage(im1)
    top.photo1=photo1
    cn1.create_image(0, 0, image = photo1, anchor = NW)
    im2 = Image.open(folder+str(c1)+'.2.jpg')
    im2=im2.resize((200, 200), Image.ANTIALIAS)
    photo2 = PIL.ImageTk.PhotoImage(im2)
    top.photo2=photo2
    c211.create_image(0, 0, image = photo2, anchor = NW)
    im3 = Image.open(folder+str(c1)+'.3.jpg')
    im3=im3.resize((200, 200), Image.ANTIALIAS)
    photo3 = PIL.ImageTk.PhotoImage(im3)
    top.photo3=photo3
    c221.create_image(0, 0, image = photo3, anchor = NW)
    im4 = Image.open(folder+str(c1)+'.4.jpg')
    im4=im4.resize((200, 200), Image.ANTIALIAS)
    photo4 = PIL.ImageTk.PhotoImage(im4)
    top.photo4=photo4
    c231.create_image(0, 0, image = photo4, anchor = NW)
    im5 = Image.open(folder+str(c1)+'.5.jpg')
    im5=im5.resize((200, 200), Image.ANTIALIAS)
    photo5 = PIL.ImageTk.PhotoImage(im5)
    top.photo5=photo5
    c212.create_image(0, 0, image = photo5, anchor = NW)
    c212.create_image(0, 0, image = photo5, anchor = NW)
    im6 = Image.open(folder+str(c1)+'.5.jpg')
    im6=im6.resize((200, 200), Image.ANTIALIAS)
    photo6 = PIL.ImageTk.PhotoImage(im6)
    top.photo6=photo6
    c222.create_image(0, 0, image = photo6, anchor = NW)
    
        
def Next():
    global cn1,c211,c212,c221,c222,c231,c232,f3a,c1
    folder='results/'
    c1+=1
    im1 = Image.open(folder+str(c1)+'.1.jpg')
    im1=im1.resize((300, 400), Image.ANTIALIAS)
    photo1 = PIL.ImageTk.PhotoImage(im1)
    top.photo1=photo1
    cn1.create_image(0, 0, image = photo1, anchor = NW)
    im2 = Image.open(folder+str(c1)+'.2.jpg')
    im2=im2.resize((200, 200), Image.ANTIALIAS)
    photo2 = PIL.ImageTk.PhotoImage(im2)
    top.photo2=photo2
    c211.create_image(0, 0, image = photo2, anchor = NW)
    im3 = Image.open(folder+str(c1)+'.3.jpg')
    im3=im3.resize((200, 200), Image.ANTIALIAS)
    photo3 = PIL.ImageTk.PhotoImage(im3)
    top.photo3=photo3
    c221.create_image(0, 0, image = photo3, anchor = NW)
    im4 = Image.open(folder+str(c1)+'.4.jpg')
    im4=im4.resize((200, 200), Image.ANTIALIAS)
    photo4 = PIL.ImageTk.PhotoImage(im4)
    top.photo4=photo4
    c231.create_image(0, 0, image = photo4, anchor = NW)
    im5 = Image.open(folder+str(c1)+'.5.jpg')
    im5=im5.resize((200, 200), Image.ANTIALIAS)
    photo5 = PIL.ImageTk.PhotoImage(im5)
    top.photo5=photo5
    c212.create_image(0, 0, image = photo5, anchor = NW)
    im6 = Image.open(folder+str(c1)+'.5.jpg')
    im6=im6.resize((200, 200), Image.ANTIALIAS)
    photo6 = PIL.ImageTk.PhotoImage(im6)
    top.photo6=photo6
    c222.create_image(0, 0, image = photo6, anchor = NW)
    
               
    
def showres():
    global cn1,c211,c212,c221,c222,c231,c232,f3a,c1,s_clicked,f_b1
    if s_clicked==0:
        b4=Button(f_b1,text="Prev",font="Verdana 10 bold",command=Prev)
        b4.pack(pady=5,side='left')
        b5=Button(f_b1,text="Next",font="Verdana 10 bold",command=Next)
        b5.pack(pady=5,side='right')
        s_clicked=1
    
    folder='results/'
    c1=1
    
    
    im1 = Image.open(folder+str(c1)+'.1.jpg')
    im1=im1.resize((300, 400), Image.ANTIALIAS)
    photo1 = PIL.ImageTk.PhotoImage(im1)
    top.photo1=photo1
    cn1.create_image(0, 0, image = photo1, anchor = NW)
    im2 = Image.open(folder+str(c1)+'.2.jpg')
    im2=im2.resize((200, 200), Image.ANTIALIAS)
    photo2 = PIL.ImageTk.PhotoImage(im2)
    top.photo2=photo2
    c211.create_image(0, 0, image = photo2, anchor = NW)
    im3 = Image.open(folder+str(c1)+'.3.jpg')
    im3=im3.resize((200, 200), Image.ANTIALIAS)
    photo3 = PIL.ImageTk.PhotoImage(im3)
    top.photo3=photo3
    c221.create_image(0, 0, image = photo3, anchor = NW)
    im4 = Image.open(folder+str(c1)+'.4.jpg')
    im4=im4.resize((200, 200), Image.ANTIALIAS)
    photo4 = PIL.ImageTk.PhotoImage(im4)
    top.photo4=photo4
    c231.create_image(0, 0, image = photo4, anchor = NW)
    im5 = Image.open(folder+str(c1)+'.5.jpg')
    im5=im5.resize((200, 200), Image.ANTIALIAS)
    photo5 = PIL.ImageTk.PhotoImage(im5)
    top.photo5=photo5
    c212.create_image(0, 0, image = photo5, anchor = NW)
    im6 = Image.open(folder+str(c1)+'.5.jpg')
    im6=im6.resize((200, 200), Image.ANTIALIAS)
    photo6 = PIL.ImageTk.PhotoImage(im6)
    top.photo6=photo6
    c222.create_image(0, 0, image = photo6, anchor = NW)
    
def stop():
    global sflag,lb1,f4
    sflag=0
        
    

import threading
def detect():
    global lb1,sflag,cn1,top,cn2,f4,b2,f_bottom,d_clicked
    
    lb1.delete(0,END)
    sflag=1
    # sflag=1
    
    cog=main1(lb1,cn1,top)
    global lb232
    lb232.insert(0,cog)
    lb232.insert(1,str(int(cog*100))+" percentage : Engagement")
    if d_clicked==1:
        b2.destroy()
        b2=Button(f_bottom,text="Show results",font="Verdana 10 bold",command=showres)
        b2.pack(pady=2)
    else:
        b2=Button(f_bottom,text="Show results",font="Verdana 10 bold",command=showres) 
        b2.pack(pady=2)   
    d_clicked=1
    
    # t1.start() 
    

def delayed_insert(label,index,message):
    label.insert(0,message)  



import threading
def insert1(label,msg):
    label.insert(0,message) 
    

def delayed_insert(label,index,message):
    # t1=threading.Thread(target=insert1,args=(label,message))
    # t1.start()
    label.insert(0,message) 

    
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
#from LBP import lbp
import os,shutils
import statistics
import pickle

def onclick(event):
    global x_loc,y_loc
    x,y=event.Position
    x_loc.append(str(x))
    y_loc.append(str(y))
    return True
# video capture
def __get_data__(fr):
    facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # st,fr = rgb.read()
    #gray scale conversion
    print(fr)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # face detection
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, fr, gray
    print("capture")

def start_app(cnn):
    font = cv2.FONT_HERSHEY_SIMPLEX
    rgb = cv2.VideoCapture(0)
    EMOTIONS_LIST = ["Angry", "Disgust","Fear", "Happy",
                     "Sad"]
    global x_loc,y_loc,time_series,amplitudes,e_nums
    with open("eye_model.sav", "rb") as f:
        eye_model = pickle.load(f)
    eye=[]
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
    lt=0
    lp=0
    eye_act=''
    e_count=0
    saving_count=1
    folder='results/'
    while True:
        try:
            if sflag==0:
                break
            
            amplitudes.append(str(tt.listen()))
            ret, frame1 = rgb.read()
            frame=frame1
            cv2.line(frame, (320,0), (320,480), (0,200,0), 2)
            cv2.line(frame, (0,200), (640,200), (0,200,0), 2)

            cv2.imwrite(folder+str(saving_count)+'.1.jpg',frame)
            lb1.after(lt,delayed_insert,lb1,lp,"Detecting eye...")
            lb1.update()
            lt+=1
            lp+=1
            
            if ret==True:
                col=frame
                
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                pupilFrame=frame
                clahe=frame
                blur=frame
                edges=frame
                cv2.imwrite(folder+str(saving_count)+'.2.jpg',frame)
                eyes = cv2.CascadeClassifier('haarcascade_eye.xml')
                detected = eyes.detectMultiScale(frame, 1.3, 5)
                for (x,y,w,h) in detected: #similar to face detection
                    
                    cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255),1)	 #draw rectangle around eyes
                    
                    cv2.imwrite(folder+str(saving_count)+'.3.jpg',frame)
                        
                    cv2.line(frame, (x,y), ((x+w,y+h)), (0,0,255),1)   #draw cross
                    cv2.line(frame, (x+w,y), ((x,y+h)), (0,0,255),1)
                    pupilFrame = cv2.equalizeHist(frame[math.ceil(y+(h*.25)):(y+h), x:(x+w)]) #using histogram equalization of better image. 
                    cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #set grid size
                    clahe = cl1.apply(pupilFrame)  #clahe
                    blur = cv2.medianBlur(clahe, 7)  #median blur
                    circles = cv2.HoughCircles(blur ,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=15,minRadius=4,maxRadius=25) #houghcircles
                    if circles is not None: #if atleast 1 is detected
                        circles = np.round(circles[0, :]).astype("int") #change float to integer
                        print (circles)
                        for (x,y,r) in circles:
                            cv2.circle(pupilFrame, (x, y), r, (0, 255, 255), 2)
                            cv2.rectangle(pupilFrame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                            
                            lb1.after(lt,delayed_insert,lb1,lp,"Detecting eye movement...")
                            lb1.update()
                            lt+=1
                            lp+=1
                            
                            #set thresholds
                            #thresholding(x)
                            time_series.append(str(x)) 
                            eye.append(str(x))
                            e_count+=1
                            if e_count>20:
                                lb1.after(lt,delayed_insert,lb1,lp,"Detecting Activity...")
                                lb1.update()
                                lt+=1
                                lp+=1
                                dict1={} 
                                dict1['eye']=[eye]
                                df1=pd.DataFrame(dict1)
                                
                                df1['eye']=[list(map(int,i)) for i in df1.eye]

                                df1['sum']=[sum(i) for i in df1.eye]
                                df1['psd']=[statistics.pstdev(i)  for i in df1.eye]
                                df1['sd']=[statistics.stdev(i)  for i in df1.eye]
                                df1['var']=[statistics.variance(i)  for i in df1.eye]
                                df1['pvar']=[statistics.pvariance(i)  for i in df1.eye]
                                X=df1[['sum', 'psd', 'sd', 'var','pvar']] 
                                
                                pred=eye_model.predict(X)
                                if pred[0]==1:
                                    eye_act="Watching" 
                                else:
                                    eye_act="Reading"    
                                lb1.after(lt,delayed_insert,lb1,lp,eye_act+'=====================')
                                lb1.update()
                                lt+=1
                                lp+=1 
                                eye=[]
                                e_count=0 

                            count+=1        
            ix += 1
            pythoncom.PumpWaitingMessages()
            lb1.after(lt,delayed_insert,lb1,lp,"Detecting mouse movement...")
            lb1.update()
            lt+=1
            lp+=1
            lb1.after(lt,delayed_insert,lb1,lp,"Detecting audio...")
            lb1.update()
            lt+=1
            lp+=1
            faces, fr, gray_fr = __get_data__(frame1)
            lb1.after(lt,delayed_insert,lb1,lp,"Detecting face...")
            lb1.update()
            lt+=1
            lp+=1
            for (x, y, w, h) in faces:
                fc = fr[y:y+h, x:x+w]
                
                im =fc
                cv2.imwrite(folder+str(saving_count)+'.4.jpg',fc)
                im=np.array(im)
                im=cv2.resize(im,(128,128))
                imgs=[]
                imgs.append([im,im])
                #imgs.append(im)
                imgs=np.stack(imgs,axis=0)
                # emotion prediction from face
                lb1.after(lt,delayed_insert,lb1,lp,"Detecting emotion...")
                lb1.update()
                lt+=1
                lp+=1
                emotion,e_num = cnn.predict_emotion(imgs)
                lb1.after(lt,delayed_insert,lb1,lp,emotion)
                lb1.update()
                lt+=1
                lp+=1
                e_nums.append(str(e_num))
                # add text in the window
                cv2.putText(fr, emotion, (x, y+h+10), font, 1, (255, 255, 0), 2)
                
                cv2.putText(fr, eye_act, (20, 50), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.imwrite(folder+str(saving_count)+'.5.jpg',fr)
                lis.append(emotion)
                eye_act=''
                saving_count+=1
            # if cv2.waitKey(1) == 27:
            #     break
            if count>200:
                break
            fr=cv2.resize(fr,(400,400))    
            photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(fr))
            cn1.create_image(0, 0, image = photo, anchor = NW)
            top.update()
        except Exception as e:
            print(e)
            continue        
    #cv2.destroyAllWindows()
    dic={}
    for i in EMOTIONS_LIST:
        dic[i]=lis.count(i)
    print("dictionary")
    return 1

# from age_gender import detect_age_gender
# from counter import count_people

def main1(lb1,cn1,top):
    
    # load trained models
    global x_loc,y_loc,time_series,amplitudes,e_nums
    model = EmotionModel("emotion_model1.json", "emotion_model1.h5")
    values=start_app(model)
    if values:
        x_loc1=' '.join(x_loc)
        y_loc1=' '.join(y_loc)
        time_series1=' '.join(time_series)
        amplitudes1=' '.join(amplitudes)
        e_nums1=' '.join(e_nums)
        if len(time_series1)<1:
            time_series1='2 4'
        if len(x_loc1)<1:
            x_loc1='2 4'
        if len(y_loc1)<1:
            y_loc1='2 4' 
        if len(e_nums1)<1:
            e_nums1='2 4' 
        if len(amplitudes1)<1:
            amplitudes1='2 4'               
        
        dic={}
        dic['m_x']=[x_loc1]
        dic['m_y']=[y_loc1]
        dic['eye']=[time_series1]
        dic['amplitudes']=[y_loc1]
        dic['emot']=[e_nums1]
        df=pd.DataFrame(dic)
        df1=df[['m_x','m_y','eye','amplitudes','emot']]
        df1['eye']=[i.split(' ') for i in df1.eye]
        df1['eye']=[list(map(int,i)) for i in df1.eye]

        df1['e_sum']=[sum(i) for i in df1.eye]
        df1['e_psd']=[statistics.pstdev(i)  for i in df1.eye]
        df1['e_sd']=[statistics.stdev(i)  for i in df1.eye]
        df1['e_var']=[statistics.variance(i)  for i in df1.eye]
        df1['e_pvar']=[statistics.pvariance(i)  for i in df1.eye]

        df1['m_x']=[i.split(' ') for i in df1.m_x]
        df1['m_x']=[list(map(int,i)) for i in df1.m_x]

        df1['mx_sum']=[sum(i) for i in df1.m_x]
        df1['mx_psd']=[statistics.pstdev(i)  for i in df1.m_x]
        df1['mx_sd']=[statistics.stdev(i)  for i in df1.m_x]
        df1['mx_var']=[statistics.variance(i)  for i in df1.m_x]
        df1['mx_pvar']=[statistics.pvariance(i)  for i in df1.m_x]

        df1['m_y']=[i.split(' ') for i in df1.m_y]
        df1['m_y']=[list(map(int,i)) for i in df1.m_y]

        df1['my_sum']=[sum(i) for i in df1.m_y]
        df1['my_psd']=[statistics.pstdev(i)  for i in df1.m_y]
        df1['my_sd']=[statistics.stdev(i)  for i in df1.m_y]
        df1['my_var']=[statistics.variance(i)  for i in df1.m_y]
        df1['my_pvar']=[statistics.pvariance(i)  for i in df1.m_y]

        df1['emot']=[i.split(' ') for i in df1.emot]
        df1['emot']=[list(map(int,i)) for i in df1.emot]

        df1['em_sum']=[sum(i) for i in df1.emot]
        df1['em_psd']=[statistics.pstdev(i)  for i in df1.emot]
        df1['em_sd']=[statistics.stdev(i)  for i in df1.emot]
        df1['em_var']=[statistics.variance(i)  for i in df1.emot]
        df1['em_pvar']=[statistics.pvariance(i)  for i in df1.emot]

        df1['amplitudes']=[i.split(' ') for i in df1.amplitudes]
        df1['amplitudes']=[list(map(float,i)) for i in df1.amplitudes]

        df1['a_sum']=[sum(i) for i in df1.amplitudes]
        df1['a_psd']=[statistics.pstdev(i)  for i in df1.amplitudes]
        df1['a_sd']=[statistics.stdev(i)  for i in df1.amplitudes]
        df1['a_var']=[statistics.variance(i)  for i in df1.amplitudes]
        df1['a_pvar']=[statistics.pvariance(i)  for i in df1.amplitudes]

        x=df1.drop(['eye','m_x','m_y','amplitudes','emot'], axis = 1) 
        with open("cognitive_model.sav", "rb") as f:
            regressor = pickle.load(f)
        pred=regressor.predict(x)    

        return pred[0]


#main()


if __name__=="__main__":

    top = Tk()  
    top.title("Login")
    top.geometry("1900x700")
    footer = Frame(top, bg='grey', height=30)
    footer.pack(fill='both', side='bottom')

    lab1=Label(footer,text="Developed by MCA",font = "Verdana 8 bold",fg="white",bg="grey")
    lab1.pack()

    menubar = Menu(top)  
    #menubar.add_command(label="Home",command=showhome)  
    menubar.add_command(label="Detection",command=showcheck)
    menubar.add_command(label="Eye movement data",command=showeye)
    menubar.add_command(label="Cognitive data",command=showcog)

    top.config(bg="#41ddff",relief=RAISED)  
    f=Frame(top)
    f.config(bg="#41ddff")
    f.pack(side="top", fill="both", expand=True,padx=10,pady=10)
    l=Label(f,text=title,font = "Verdana 40 bold",fg="white",bg="#41ddff")
    l.place(x=150,y=50)
    l2=Label(f,text="Username:",font="Verdana 10 bold",bg="#41ddff")
    l2.place(x=550,y=300)
    global username_var
    username_var=StringVar()
    e1=Entry(f,textvariable=username_var,font="Verdana 10 bold")
    e1.place(x=700,y=300)

    l3=Label(f,text="Password:",font="Verdana 10 bold",bg="#41ddff")
    l3.place(x=550,y=330)
    global pass_var
    pass_var=StringVar()
    e2=Entry(f,textvariable=pass_var,font="Verdana 10 bold",show="*")
    e2.place(x=700,y=330)
    #showcheck()

    b1=Button(f,text="Login", command=logcheck,font="Verdana 10 bold")
    b1.place(x=750,y=360)

    top.mainloop() 

