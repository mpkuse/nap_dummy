#!/usr/bin/python
"""
        Use dji_sdk/GlobalPosition. The above plugin needs `sensor_msgs/NavSatFix`
        use Rviz plugin : https://github.com/KumarRobotics/rviz_satellite for
        cool visualization of GPS

        gps.lat and gps.long converted to xyz. Published as odometry msg.
"""
# PKG_PATH = rospkg.RosPack().get_path('nap')
PKG_PATH = '/home/mpkuse/catkin_ws/src/nap/'

import rospy
from dji_sdk.msg import GlobalPosition
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
import numpy as np


# Standard GPS util
def geodedic_to_ecef( lati, longi, alti ):
    """ lati in degrees, longi in degrees. alti in meters (mean sea level) """
    # Adopted from https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    phi = lati / 180. * np.pi
    lambada = longi / 180. * np.pi
    h = alti

    #N = 6371000 #in meters
    e = 0.081819191 #earth ecentricity
    q = np.sin( phi )
    N = 6378137.0 / np.sqrt( 1 - e*e * q*q )
    X = (N + h) * np.cos( phi ) * np.cos( lambada )
    Y = (N + h) * np.cos( phi ) * np.sin( lambada )
    Z = (N*(1-e*e) + h) * np.sin( phi )

    return X,Y,Z

def compute_ecef_to_enu_transform( lati_r, longi_r ):
    """ Computes a matrix_3x3 which transforms a ecef diff-point to ENU (East-North-Up)
        Needs as input the latitude and longitude (in degrees) of the reference point
    """
    # Adopted from https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

    phi = lati_r / 180. * np.pi
    lambada = longi_r / 180. * np.pi

    cp = np.cos( phi ) #cos(phi)
    sp = np.sin( phi ) #cos(phi)
    cl = np.cos( lambada )
    sl = np.sin( lambada )

    T = np.zeros( (3,3), dtype='float64' )
    T[0,0] = -sl
    T[0,1] = cl
    T[0,2] = 0

    T[1,0] = -sp * cl
    T[1,1] = -sp * sl
    T[1,2] = cp

    T[2,0] = cp * cl
    T[2,1] = cp * sl
    T[2,2] = sp

    T_enu_ecef = T
    return T_enu_ecef

seq_odom = 0
gps_home_set = False
radar_lat = 0.0
radar_long = 0.0
radar_alti = 0.0
def callback_gps(data):
    global gps_home_set, radar_lat, radar_long, radar_alti
    if gps_home_set is False:
        radar_lat = data.latitude
        radar_long = data.longitude
        radar_alti = data.altitude
        gps_home_set = True


    # rospy.logdebug( 'Rcvd')
    #################
    # Publish Satellitle Msg
    #################
    n = NavSatFix()
    n.header = data.header
    n.header.frame_id = 'world'
    n.latitude = data.latitude
    n.longitude = data.longitude
    n.altitude = data.altitude
    pub_satmsg.publish( n )

    ###############
    # publish xyz as markers
    ###############22.334500, 114.263082
    # GPS (geodedic to Earth-center cords, ie. ecef )
    # Xr, Yr, Zr = geodedic_to_ecef( 22.334500, 114.263082, 173.073608398 ) #of radar -hkust
    # T_enu_ecef = compute_ecef_to_enu_transform( 22.334500,114.263082 )

    Xr, Yr, Zr = geodedic_to_ecef( radar_lat, radar_long, radar_alti ) #of radar
    T_enu_ecef = compute_ecef_to_enu_transform(radar_lat, radar_long )

    Xp, Yp, Zp = geodedic_to_ecef( data.latitude, data.longitude, data.altitude ) #curr pos of drone

    #
    # ECEF to ENU (East-North-Up)
    delta = np.array( [Xp-Xr, Yp-Yr, Zp-Zr] )
    p = np.dot( T_enu_ecef, delta )


    # publish Marker
    global seq_odom
    marker = Marker()
    marker.header = data.header
    marker.id = seq_odom
    seq_odom += 1
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = p[0];
    marker.pose.position.y = p[1];
    marker.pose.position.z = p[2];
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    pub_odom_marker.publish( marker )

    odom = Odometry()
    odom.header = data.header
    odom.pose.pose.position.x = p[0]
    odom.pose.pose.position.y = p[1]
    odom.pose.pose.position.z = p[2]
    odom.pose.pose.orientation.x = 0.0
    odom.pose.pose.orientation.y = 0.0
    odom.pose.pose.orientation.z = 0.0
    odom.pose.pose.orientation.w = 1.0
    # odom.child_frame_id = 'drone'
    pub_odom.publish( odom )





rospy.init_node( 'gps_test', anonymous=True)
rospy.Subscriber( "/dji_sdk/global_position", GlobalPosition, callback_gps )
print 'Subscribed to /dji_sdk/global_position'

pub_satmsg = rospy.Publisher( 'chatter', NavSatFix, queue_size=10 )
print 'Publishing /chatter of type sensor_msgs.NavSatFix'

pub_odom_marker = rospy.Publisher( '/gps_odom_marker', Marker, queue_size=10 )
print 'Publishing gps_odom_marker of type visualization_msgs.Marker'
pub_odom = rospy.Publisher( '/gps_odom', Odometry, queue_size=10 )
print 'Publishing gps_odom of type nav_msgs.Odometry'


rospy.spin()
