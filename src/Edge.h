#pragma once

/** Edge.h
    Class to handle edges of Pose Graph

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


#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

using namespace std;


#include "Node.h"

// TYPE DECLARATIONS
#define EDGE_TYPE_ODOMETRY 0
#define EDGE_TYPE_LOOP_CLOSURE 1

#define EDGE_TYPE_LOOP_SUBTYPE_BASIC 10 //Has enough sparse matches
#define EDGE_TYPE_LOOP_SUBTYPE_GUIDED 67 //Guided matching in the msg
#define EDGE_TYPE_LOOP_SUBTYPE_3WAY 11 //need 3 way matching, not enough sparse-feature matches



// CLass
// Edge represents connection between 2 nodes.
class Edge {
public:
  Edge( const Node *a, int a_id, const Node * b, int b_id, int type );
  void setEdgeTimeStamps( ros::Time time_a, ros::Time time_b );
  void setLoopEdgeSubtype( int sub_type );

  int getEdgeType();
  int getEdgeSubType();

  // Given the pose in frame of b. In other words, relative pose of a in frame
  // of reference of b. This is a 4x4 matrix, top-left 3x3 represents rotation part.
  // 4th col represents translation.
  void setEdgeRelPose( const Matrix4d& b_T_a );

  // Convert the stored pose into matrix and return. Note that the
  // stored pose is b_T_a, ie. pose of a in reference frame b.
  void getEdgeRelPose( Matrix4d& M );


  const Node *a, *b; //nodes
  int type; //TODO Consider making this private and provide an access function
  int sub_type; //TODO COnsider making this private and give an access function
  int a_id, b_id;
  ros::Time a_timestamp, b_timestamp;

  Vector3d e_p;
  Quaterniond e_q;

};
