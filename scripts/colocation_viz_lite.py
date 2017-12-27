#!/usr/bin/python

""" Subscribes to NapVisualEdgeMsg and publishes the image pair

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 11th July, 2017
"""
import rospy
import rospkg

import collections
import numpy as np
import code
import cv2
cv2.ocl.setUseOpenCL(False)

from cv_bridge import CvBridge, CvBridgeError


from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


from nap.msg import NapMsg
from nap.msg import NapNodeMsg
from nap.msg import NapVisualEdgeMsg

PKG_PATH = rospkg.RosPack().get_path('nap')
# PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'



def visual_edge_callback( data ):
    curr_image = CvBridge().imgmsg_to_cv2( data.curr_image, 'bgr8' )
    prev_image = CvBridge().imgmsg_to_cv2( data.prev_image, 'bgr8' )

    # xtra_string = ""
    # #---DEBUG - Geometric Verification with essential matrix
    # print curr_image.shape, prev_image.shape
    # code.interact( local=locals())

    # kp1, des1 = orb.detectAndCompute(np.array(prev_image), None)
    # kp2, des2 = orb.detectAndCompute(np.array(curr_image), None)
    # print 'len(kp1) : ', len(kp1), '    des1.shape : ', des1.shape
    # print 'len(kp2) : ', len(kp2), '    des2.shape : ', des2.shape
    # matches_org = flann.knnMatch(des1.astype('float32'),des2.astype('float32'),k=2) #find 2 nearest neighbours
    # __pt1, __pt2 = lowe_ratio_test( kp1, kp2, matches_org )
    # E, mask = cv2.findEssentialMat( __pt1, __pt2 )
    # nMatches = __pt1.shape[0]
    # nInliers = 0
    # if mask is not None:
    #     nInliers = mask.sum()
    # xtra_string = '::%d,%d' %(nInliers,nMatches)
    # #---END


    curr_label = data.curr_label
    prev_label = data.prev_label
    sc = curr_image.shape[0]/180. #scale of font and thickness
    cv2.putText( curr_image, curr_label, (10,curr_image.shape[0]/2), cv2.FONT_HERSHEY_SIMPLEX, sc, (71,55,40), int(4*sc) )
    cv2.putText( prev_image, prev_label, (10,curr_image.shape[0]/2), cv2.FONT_HERSHEY_SIMPLEX, sc, (34,34,178), int(4*sc) )

    catted = np.concatenate( (curr_image,prev_image), axis=1)
    msg_frame = CvBridge().cv2_to_imgmsg( catted, "bgr8" )
    pub_raw_edge_image_pair.publish( msg_frame )
    # cv2.imshow( 'curr_image', curr_image )
    # cv2.imshow( 'prev_image', prev_image )
    # cv2.imshow( 'cated', np.concatenate( (curr_image,prev_image), axis=1 ) )
    # cv2.waitKey(1)


rospy.init_node( 'colocation_viz', log_level=rospy.INFO )
rospy.Subscriber( '/raw_graph_visual_edge', NapVisualEdgeMsg, visual_edge_callback, queue_size=10000 )
rospy.loginfo( 'Subscribed to Raw Graph : /raw_graph_visual_edge')

rospy.loginfo( 'Publish : /colocation_viz/raw/image_pair')
pub_raw_edge_image_pair = rospy.Publisher( '/mish/image_pair', Image, queue_size=1000 )


rate = rospy.Rate(30)
while not rospy.is_shutdown():
    # print 'Length :', len(_d)
    rate.sleep()

rospy.loginfo( 'Done..!' )
