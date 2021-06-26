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
title="Customer Monitoring"
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
from cogn_data import main1

def eyecreate():
    global label_var
    l=label_var.get()
    create_eye(l)

def cogcreate():
    global cabel_var
    c=cabel_var.get()
    main1(c)




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
    f3.config(bg="#41ddff",width=400)
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
    
    global cn1,c211,c212,c221,c222,c231,c232,f3a
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
    l221=Label(f22,text='Cropped face')
    l221.pack(side="top",pady=2)
    c222 =Canvas(f22, width = 200, height = 200)
    c222.pack(side="top",pady=2)
    l222=Label(f22,text='Customer traffic')
    l222.pack(side="top",pady=2)

    f23=Frame(f2a)
    f23.pack_propagate(False)
    f23.config(bg="#41ddff")
    f23.config(width=200,height=800)
    f23.pack(side='left',padx=3)

    c231 =Canvas(f23, width = 200, height = 200)
    c231.pack(side="top",pady=2)
    l231=Label(f23,text='RGB ')
    l231.pack(side="top",pady=2)
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
    im6 = Image.open(folder+str(c1)+'.6.jpg')
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
    im6 = Image.open(folder+str(c1)+'.6.jpg')
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
    im6 = Image.open(folder+str(c1)+'.6.jpg')
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
    
    main(lb1,cn1,top)
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
from model import FacialExpressionModel
import numpy as np
#from LBP import lbp
import os,shutils


print("capture")

def start_app(cnn,lb1,cn1,top):
    
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    global facec,font,rgb
    facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    rgb = cv2.VideoCapture('a.mp4')
    folder='results/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    saving_count=1
    lt=0
    lp=8
    
    lis=[]
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    # contineous frame reading
    frs=0
    while True:
        frs+=1
        if frs!=10:
            continue
        frs=0
        try:
            if sflag==0:
                cv2.destroyAllWindows()
                rgb.release()
                del rgb
                os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
                break
            ix += 1
            lb1.after(lt,delayed_insert,lb1,lp,"Reading frame...")
            lb1.update()
            lt+=1
            lp+=1
            st,fr = rgb.read()
        
            #gray scale conversion
            lb1.after(lt,delayed_insert,lb1,lp,"Grayscale conversion...")
            lb1.update()
            lt+=1
            lp+=1
            print(fr)
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            p_count=count_people(fr)
            print(p_count)
            lb1.after(lt,delayed_insert,lb1,lp,"Detecting face...")
            lb1.update()
            lt+=1
            lp+=1

            # face detection
            faces = facec.detectMultiScale(gray, 1.3, 5)
            
            
            for (x, y, w, h) in faces:
                if saving_count<50:
                    cv2.imwrite(folder+str(saving_count)+'.1.jpg',fr)
                    cv2.imwrite(folder+str(saving_count)+'.2.jpg',gray)
                lb1.after(lt,delayed_insert,lb1,lp,"Cropping face...")
                lb1.update()
                lt+=1
                lp+=1
                fc_gray = gray[y:y+h, x:x+w]
                fc_gray=cv2.resize(fc_gray,(48,48))
                fc_rgb=fr[y:y+h, x:x+w]
                if saving_count<50:
                    cv2.imwrite(folder+str(saving_count)+'.3.jpg',fc_gray)
                

                
                age,gend=detect_age_gender(fc_rgb)
                print(age,gend)

                
                # emotion prediction from face
                roi=np.reshape(fc_gray,(1,48,48,1))
                if saving_count<50:
                    cv2.imwrite(folder+str(saving_count)+'.4.jpg',fc_rgb)
                lb1.after(lt,delayed_insert,lb1,lp,"prediction...")
                lb1.update()
                lt+=1
                lp+=1
                pred,sat,act,eng,vio = cnn.predict_emotion(roi)
                lb1.after(lt,delayed_insert,lb1,lp,pred)
                lb1.update()
                lt+=1
                lp+=1
                # add text in the window
                
                cv2.putText(fr, age, (x+w+10, y+10), font, 1.2, (0, 255, 0), 2)
                cv2.putText(fr, gend, (x+w+10, y+50), font, 1.2, (0, 0, 255), 2)
                cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 5)
                cv2.putText(fr, "Satisfaction:%.2f"%(float(sat))+'%', (x, y+h+15), font, 1, (255, 200, 0), 2)
                cv2.putText(fr, "Activation:%.2f "%(act)+'%', (x, y+h+40), font, 1, (255, 150, 0), 2)
                cv2.putText(fr, "Engagement:%.2f "%(eng)+'%', (x, y+h+65), font, 1, (255, 100, 0), 2)
                cv2.putText(fr, "Violence:%.2f "%(vio)+'%', (x, y+h+90), font, 1, (255, 50, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
                res1=fr[y-50:y+h+200, x-50:x+w+200]
                if saving_count<50:
                    cv2.imwrite(folder+str(saving_count)+'.5.jpg',res1)
                cv2.putText(fr, "Number of customers : %s"%(str(p_count)), (50, 50), font, 1.5, (100, 255,100), 5)
                p_res=fr[0:100,600:700]
                if saving_count<50:
                    cv2.imwrite(folder+str(saving_count)+'.6.jpg',p_res)
                saving_count+=1
            #lis.append(pred)
            if cv2.waitKey(1) == 27:
                break
            cv2.resizeWindow('image', 600,600)
            fr1=cv2.resize(fr,(400,400))
            # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(fr1))
            cn1.create_image(0, 0, image = photo, anchor = NW)
            top.update()
        except Exception as e:
            print(e)
            continue    
        #cv2.imshow("Output", fr)
    cv2.destroyAllWindows()
    dic={}
    
    return None

# from age_gender import detect_age_gender
# from counter import count_people

def main(lb1,cn1,top):

    global video,audio,step,duration,c,aem,facec,font
    print("capture")
    # load trained models
    md=FacialExpressionModel('face_model.json','face_model.h5')
    values=start_app(md,lb1,cn1,top)
    return values

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
    menubar.add_command(label="Home",command=showhome)  
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

