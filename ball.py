import numpy as np
import cv2

cap = cv2.VideoCapture(0)

MAX = np.sqrt((640/2)**2 + (480/2)**2)

kp = 255/MAX

def calcDist(center):
    Cx = 640/2
    Cy = 480/2

    x = center[0]
    y = center[1]

    dist = np.sqrt((x-Cx)**2 + (y-Cy)**2)

    return dist



while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        height = frame.shape[0]
        width = frame.shape[1]
        cimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_orange = np.array([16,164,157])
        upper_orange = np.array([34,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(mask,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= dilation)

        im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            cv2.drawContours(frame, [box], 0, (0, 0, 255))

            # finally, get the min enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(c)
            # convert all values to int
            center = (int(x), int(y))
            radius = int(radius)
            print('center: ', center)
            print('Radius; ' , radius)
            # and draw the circle in blue
            img = cv2.circle(frame, center, radius, (255, 0, 0), 2)
            error = calcDist(center)
            R = error*kp
            G = 255 - error*kp
            color = []
            cv2.arrowedLine(frame,center,(width/2,height/2),(0,G,R),5)
            
        
        #image height
        height = frame.shape[0]
        width = frame.shape[1]
        # Draw a diagonal blue line with thickness of 5 px
        cv2.line(frame,(0,height/2),(width,height/2),(0,255,0),5)
        cv2.line(frame,(width/2,0),(width/2,height),(0,255,0),5)
        cv2.imshow('frame',frame)
        # cv2.imshow('mask',dilation)
        # cv2.imshow('res',res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

