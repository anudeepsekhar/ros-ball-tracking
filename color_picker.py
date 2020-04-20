import cv2
import numpy as np
def nothing(x):
    pass
# Create a black image, a window
img = np.zeros((640,480,3), np.uint8)
cv2.namedWindow('image')
# create trackbars for color change
cv2.createTrackbar('H High','image',0,180,nothing)
cv2.createTrackbar('H Low','image',0,180,nothing)
cv2.createTrackbar('S High','image',0,255,nothing)
cv2.createTrackbar('S Low','image',0,255,nothing)
cv2.createTrackbar('V High','image',0,255,nothing)
cv2.createTrackbar('V Low','image',0,255,nothing)
# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ONapt'
cv2.createTrackbar(switch, 'image',0,1,nothing)
cap = cv2.VideoCapture(1)
while(1):
    ret, frame = cap.read()
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of four trackbars
    hH = cv2.getTrackbarPos('H High','image')
    hL = cv2.getTrackbarPos('H Low','image')
    sH = cv2.getTrackbarPos('S High','image')
    sL = cv2.getTrackbarPos('S Low','image')
    vH = cv2.getTrackbarPos('V High','image')
    vL = cv2.getTrackbarPos('V Low','image')
    s = cv2.getTrackbarPos(switch,'image')
    if s == 0:
        if ret == True:
            img = frame
        else:
            img[:] = 0
    else:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower = np.array([hL,sL,vL])
        upper = np.array([hH,sH,vH])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower, upper)
        img = mask

cap.release()
cv2.destroyAllWindows()
