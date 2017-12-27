#!/usr/bin/python
""" Testing rviz_satelite plugin
        Came across this interesting plugins from KumarRobotics
        https://github.com/KumarRobotics/rviz_satellite

        Use dji_sdk/GlobalPosition. The above plugin needs `sensor_msgs/NavSatFix`
"""


import rospy
from dji_sdk.msg import GlobalPosition
from sensor_msgs.msg import NavSatFix

def callback_gps(data):
    rospy.loginfo( 'Rcvd')
    n = NavSatFix()
    n.header = data.header
    n.latitude = data.latitude
    n.longitude = data.longitude
    n.altitude = data.altitude
    pub_satmsg.publish( n )


rospy.init_node( 'gps_test', anonymous=True)
rospy.Subscriber( "/dji_sdk/global_position", GlobalPosition, callback_gps )

pub_satmsg = rospy.Publisher( 'chatter', NavSatFix, queue_size=10 )
rospy.spin()
