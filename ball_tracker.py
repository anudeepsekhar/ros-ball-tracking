#!/usr/bin/env python
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""
__author__ =  'Anudeepsekhar Bolimera'
__version__=  '0.1'
__license__ = 'MIT'
# Python libs
import sys, time

# numpy and scipy
import numpy as np

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False

class ball_tracker:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage)

        self.ball_pos_pub = rospy.Publisher('ball_pos', Point, queue_size=10)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1)
        self.center = [0,0]
        self.MAX = np.sqrt((640/2)**2 + (480/2)**2)
        self.kp = 255/self.MAX

        self.pos_msg = Point()
        if VERBOSE :
            print "subscribed to /usb_cam/image_raw/compressed"

    def calcDist(self,center):
        Cx = 640/2
        Cy = 480/2

        x = center[0]
        y = center[1]

        dist = np.sqrt((x-Cx)**2 + (y-Cy)**2)

        return dist


    def find_ball(self,frame):
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
            # print('center: ', center)
            # print('Radius; ' , radius)
            # and draw the circle in blue
            cv2.circle(frame, center, radius, (255, 0, 0), 2)
            error = self.calcDist(center)
            R = error*self.kp
            G = 255 - error*self.kp
            color = []
            cv2.arrowedLine(frame,center,(640/2,480/2),(0,G,R),5)

            if center is not None:
                self.center=center

        return frame


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        frame = image_np

    
        frame = self.find_ball(frame)
        print self.center
 
    
        cv2.imshow('cv_img', frame)
        cv2.waitKey(2)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.pos_msg.x = self.center[0]
        self.pos_msg.y = self.center[1]
        self.pos_msg.z = 0
        self.image_pub.publish(msg)
        self.ball_pos_pub.publish(self.pos_msg)
        
        #self.subscriber.unregister()

def main(args):
    '''Initializes and cleanup ros node'''
    ic = ball_tracker()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
