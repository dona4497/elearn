import pyHook
import pythoncom
import time

x_loc=[]
y_loc=[]
def onclick(event):
    x,y=event.Position
    x_loc.append(x)
    y_loc.append(y)
    return True

hm = pyHook.HookManager()
hm.SubscribeMouseAll(onclick)
hm.HookMouse()
# pythoncom.PumpMessages()
# hm.UnhookMouse()
while time.clock() < 5:
    pythoncom.PumpWaitingMessages()
print(x_loc)