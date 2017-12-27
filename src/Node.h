#pragma once

/**
    This class hold info on a pose-graph node.

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 3rd Oct, 2017
*/

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <ostream>


#include <thread>
#include <mutex>
#include <atomic>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>


#include <ros/ros.h>
#include <ros/package.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <nap/NapMsg.h>

#include "cnpy.h"


#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

using namespace std;

class Node {

public:
  Node( ros::Time time_stamp, geometry_msgs::Pose pose );

  ros::Time time_stamp; //this is timestamp of the pose
  geometry_msgs::Pose pose;

  ros::Time time_pcl, time_pose, time_image, time_feat2d, time_nap_clustermap;

  //optimization variables
  // double *opt_position; //translation component ^wT_1
  // double *opt_quat;     //rotation component ^wR_1

  Vector3d e_p;
  Quaterniond e_q;

  Vector3d org_p;
  Quaterniond org_q;

  void getCurrTransform(Matrix4d& M);
  void getOriginalTransform(Matrix4d& M);

  // 3d point cloud
  Matrix<double,3,Dynamic> ptCld; //TODO: Consider making this private
  bool m_3dpts;//TODO: Consider making this private
  void setPointCloud( ros::Time time, const vector<geometry_msgs::Point32> & points );
  void setPointCloud( ros::Time time, const Matrix<double,3,Dynamic>& e );
  const Matrix<double,3,Dynamic>& getPointCloud( );
  void getPointCloudHomogeneous( MatrixXd& M );


  // 2d features (tracked)
  Matrix<double,3,Dynamic> feat2d; //TODO: Consider making this private
  bool m_2dfeats; //TODO: Consider making this private
  void setFeatures2dHomogeneous( ros::Time time, const vector<geometry_msgs::Point32> & points );
  void setFeatures2dHomogeneous( ros::Time time, const Matrix<double,3,Dynamic>& e );
  void getFeatures2dHomogeneous( MatrixXd& M ); //< M will be 3xN matrix


  int getn3dpts()      { return ptCld.cols(); }
  int getn2dfeat()     { return feat2d.cols(); }
  bool valid_3dpts()   { return m_3dpts; }
  bool valid_2dfeats() { return m_2dfeats; }





  //image
  void setImage( ros::Time time, const cv::Mat& im );
  const cv::Mat& getImageRef();
  bool valid_image()   { return (image.data!=NULL); }

  // nap clusters
  void setNapClusterMap( ros::Time time, const cv::Mat& im );
  const cv::Mat& getNapClusterMap();
  bool valid_clustermap()   { return (nap_clusters.data!=NULL); }


  // Write to file XML
  void write_debug_xml( char *fname  );

private:
  cv::Mat image;

  cv::Mat nap_clusters;

};
