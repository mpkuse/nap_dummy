#pragma once
/** DataManager.h


      (in DataManager_core.cpp)
      This class handles (and syncs) the subscribers for image, odometry,
      point-cloud, loop closure msg. Internally it builds up a pose-graph
      with class Edge and class Node.

      (in DataManager_rviz_visualization.cpp)
      pose-graph as Marker msg for visualization.

      (in DataManager_looppose_computation.cpp)
      Another critically important part of this is the computation of relative
      pose of loop closure msg. Thus, it also has the camera-instrinsic class PinholeCamera

      (in DataManager_ceres.cpp)
      TODO Yet Another important part is going to be call to solve the
      pose graph optimization problem with CERES.


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



#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>


#include <ceres/ceres.h>

using namespace std;


// CLasses In this Node
#include "Node.h"
#include "Edge.h"
#include "PinholeCamera.h"



class DataManager
{
public:
  DataManager( ros::NodeHandle &nh );
  DataManager(const DataManager &obj);

  void setCamera( PinholeCamera& camera );
  void setVisualizationTopic( string rviz_topic );

  ~DataManager();  //< Writes pose graph to file and deallocation

  // //////////////// //
  //    Callbacks     //
  // //////////////// //

  /// These should be same images you feed to place-recognition system
  /// Subscribes to images and associates these with the pose-graph.
  /// This is not required for actual computation, but is used for debugging
  void image_callback( const sensor_msgs::ImageConstPtr& msg );

  /// Nap Cluster assignment in raw format. mono8 type image basically a 60x80 array of numbers with intensity as cluster ID
  /// This is used for daisy matching
  void raw_nap_cluster_assgn_callback( const sensor_msgs::ImageConstPtr& msg );


  /// Subscribes to pointcloud. pointcloud messages are sublish by the
  /// visual-innertial odometry system from Qin Tong. These are used for
  /// loop-closure relative pose computation
  void point_cloud_callback( const sensor_msgs::PointCloudConstPtr& msg ); //< 3d
  void tracked_features_callback( const sensor_msgs::PointCloudConstPtr& msg ); //< 2d tracked features


  /// Subscribes to odometry messages. These are used to make the nodes.
  /// Everytime a new odometry is received, a new node is created.
  /// The newly created node has a timestamp and a pose (wrt to world ie. ^{w}T_c )
  /// make sure to subscribe to camera_pose without loop,
  void camera_pose_callback( const nav_msgs::Odometry::ConstPtr msg );


  /// Subscribes to loop-closure messages
  void place_recog_callback( const nap::NapMsg::ConstPtr& msg  );


  // ////////////////   //
  //  Visualization     //
  // ////////////////   //
  // All 3 publish with handle `pub_pgraph`
  void publish_once(); //< Calls the next 2 functions successively
  void publish_pose_graph_nodes(); //< Publishes all nNodes
  void publish_pose_graph_nodes_original_poses(); // Publish into pub_pgraph_org
  void publish_pose_graph_edges( const std::vector<Edge*>& x_edges ); //< publishes the given edge set


  // ////////////////////////////   //
  //  Relative Pose Computation     //
  // ////////////////////////////   //

  void pose_from_2way_matching( const nap::NapMsg::ConstPtr& msg, Matrix4d& p_T_c );
  void pose_from_3way_matching( const nap::NapMsg::ConstPtr& msg, Matrix4d& p_T_c );



  // //////////////////////////////////////////// //
  //  Ceres Main                                  //
  //  - Indented to be called by new thread.      //
  //  - This function wont return                 //
  // //////////////////////////////////////////// //
  void ceres_main();
  void doOptimization();


  bool enable_ceres;
  mutex lock_enable_ceres;



private:

    // /////////////////////////////////////////////// //
    // Republish                                       //
    // /////////////////////////////////////////////// //
    ros::Publisher pub_chatter_colocation;
    void republish_nap( const ros::Time& t_c, const ros::Time& t_p, const Matrix4d& p_T_c, int32_t op_mode );
    void republish_nap( const nap::NapMsg::ConstPtr& msg );

  //
  // Core Data variables
  //
  vector<Node*> nNodes; //list of notes
  vector<Edge*> odometryEdges; //list of odometry edges
  vector<Edge*> loopClosureEdges; //List of closure edges

  //
  // Buffer Utilities
  //
  int find_indexof_node( ros::Time stamp );

  std::queue<cv::Mat> unclaimed_im;
  std::queue<ros::Time> unclaimed_im_time;
  void flush_unclaimed_im();

  std::queue<cv::Mat> unclaimed_napmap;
  std::queue<ros::Time> unclaimed_napmap_time;
  void flush_unclaimed_napmap();

  std::queue<Matrix<double,3,Dynamic>> unclaimed_pt_cld;
  std::queue<ros::Time> unclaimed_pt_cld_time;
  void flush_unclaimed_pt_cld();

  std::queue<Matrix<double,3,Dynamic>> unclaimed_2d_feat;
  std::queue<ros::Time> unclaimed_2d_feat_time;
  void flush_unclaimed_2d_feat();


  void print_cvmat_info( string msg, const cv::Mat& A );
  string type2str( int );

  //
  // rel pose computation utils
  //

  // msg --> Received with callback
  // mat_ptr_curr, mat_pts_prev, mat_pts_curr_m --> 2xN outputs
  void extract_3way_matches_from_napmsg( const nap::NapMsg::ConstPtr& msg,
        cv::Mat&mat_pts_curr, cv::Mat& mat_pts_prev, cv::Mat& mat_pts_curr_m );

  // image, 2xN, image, 2xN, image 2xN, out_image
  // or
  // image, 1xN 2 channel, image 1xN 2 channel, image 1xN 2 channel
  void plot_3way_match( const cv::Mat& curr_im, const cv::Mat& mat_pts_curr,
                        const cv::Mat& prev_im, const cv::Mat& mat_pts_prev,
                        const cv::Mat& curr_m_im, const cv::Mat& mat_pts_curr_m,
                        cv::Mat& dst, const string& msg=string(""));
  void plot_3way_match_clean( const cv::Mat& curr_im, const cv::Mat& mat_pts_curr,
                        const cv::Mat& prev_im, const cv::Mat& mat_pts_prev,
                        const cv::Mat& curr_m_im, const cv::Mat& mat_pts_curr_m,
                        cv::Mat& dst, const string& msg=string(""));


  // image, 2xN.
  // If mat is more than 2d will only take 1st 2 dimensions as (x,y) ie (cols,rows)
  void plot_point_sets( const cv::Mat& im, const cv::Mat& pts_set, cv::Mat& dst, const cv::Scalar& color, const string& msg=string("") );
  void plot_point_sets( const cv::Mat& im, const MatrixXd& pts_set, cv::Mat& dst, const cv::Scalar& color, const string& msg=string("") );
  std::vector<std::string> split( std::string const& original, char separator );

  // Given the odometry poses of 2 cameras compute the fundamental matrix.
  // If the poses are known in canonical form, ie. [I|0] , [R,t] ; than
  // Fundamental matrix is given as F:= [t]_x * R
  void make_fundamentalmatrix_from_pose( const Matrix4d& w_T_c, const Matrix4d& w_T_cm,
                                          Matrix3d& F );


  // My wrapper for cv2.triangulatePoints()
  // [Input]
  // ix_curr        : index of node corresponding to curr
  // mat_pts_curr   : 2xN matrix representing point matches in curr
  // ix_curr_m      : index of node corresponding to curr-1
  // mat_pts_curr_m : 2xN matrix representing point matches in curr-1
  // [Output]
  // c_3dpts        : 3D points in co-ordinate frame of curr
  void triangulate_points( int ix_curr, const cv::Mat& mat_pts_curr,
                           int ix_curr_m, const cv::Mat& mat_pts_curr_m,
                           cv::Mat& c_3dpts );


  // My wrapper for cv2.solvePnP().
  // [Input]
  // c_3dpts : 3d points. 3xN, 1-channel. It is also ok to pass 4xN. Last row will be ignored
  // pts2d   : 2d Points 2xN 1-channel
  // [Output]
  // im_T_c  : Pose of model-cordinates (system in which 3d pts are specified) wrt to camera in which the 2d points are specified
  void estimatePnPPose( const cv::Mat& c_3dpts, const cv::Mat& pts2d,
                        Matrix4d& im_T_c  );
  void estimatePnPPose_withguess( const cv::Mat& c_3dpts, const cv::Mat& pts2d,
                        Matrix4d& im_T_c, const Matrix4d& odom_w_T_c, const Matrix4d& odom_w_T_p );
        //for pnp with guess, give as input odometry pose. Atleast pitch and roll are accurate. Will try and extract it.


  // Own implementation of PnP with ceres. Basically Setting up the pnp problem as non-linear least squares
  // c_3dpts_4N - 4xN. 3d points in homogeneous co-ordinates
  // pts2d - 2xN. Image points in undistorted-normalized-image co-ordinates.
  void estimatePnPPose_ceres( const cv::Mat& c_3dpts_4N, const cv::Mat& pts2d,
                        Matrix4d& im_T_c  );


  void _to_homogeneous( const cv::Mat& in, cv::Mat& out );
  void _from_homogeneous( const cv::Mat& in, cv::Mat& out );
  void _perspective_divide_inplace( cv::Mat& in );
  double _diff_2d( const cv::Mat&A, const cv::Mat&B ); //< 2xM, 2xM,  RMS of these 2 matrices
  void quaternion_to_T( double * opt_q, double * opt_t, Matrix4d& Tr ); //opt_q : [w,x,y,z]. Converts the 7 params to a SE(3) matrix


  void convert_rvec_eigen4f( const cv::Mat& rvec, const cv::Mat& tvec, Matrix4f& Tr );
  bool if_file_exist( string fname ); //in DataManager_rviz_visualization.cpp
  bool if_file_exist( char * fname );

  string matrix4f_to_string( const Matrix4f& M );
  string matrix4d_to_string( const Matrix4d& M );


  // Debug file in opencv format utils (in DataManager_core.cpp)
  // void open_debug_xml( const string& fname );
  // const cv::FileStorage& get_debug_file_ptr();
  // void close_debug_xml();
  // cv::FileStorage debug_fp;

  // END 'rel pose computation utils'


  ros::NodeHandle nh; //< Node Handle
  ros::Publisher pub_pgraph; //< Visualization Marker handle, nodes will have curr pose
  ros::Publisher pub_pgraph_org; //< Publishes Original (unoptimized pose graph)

  PinholeCamera camera; //< Camera Intrinsics. See corresponding class

};




#include "Node.h"
#include "Edge.h"

class Residue4DOF
{
public:
  Residue4DOF( const Matrix4d& obs_edge_pose )
  {
    this->obs_edge_pose = Matrix4d( obs_edge_pose );
  }

  template <typename T>
	bool operator()(const T* const q_a, const T* t_a,
                  const T* const q_b, const T* t_b,
                  T* residuals) const
  {
    //
    // Make w_T_a : State
    Matrix<T,3,3> w_R_a;// = Matrix<T,3,3>::Identity();
    Quaternion<T> ___q_a = Map<const Quaternion<T>>(q_a);
    w_R_a = ___q_a.toRotationMatrix();
    Matrix<T,3,1> w_t_a;
    w_t_a << t_a[0], t_a[1], t_a[2];

    Matrix<T,4,4> w_T_a;
    // w_T_a.block<3,3>(0,0) = w_R_a;
    // w_T_a.block<3,1>(3,0) = w_t_a;
    w_T_a << w_R_a(0,0), w_R_a(0,1), w_R_a(0,2), w_t_a(0,0),
              w_R_a(1,0), w_R_a(1,1), w_R_a(1,2), w_t_a(1,0),
              w_R_a(2,0), w_R_a(2,1), w_R_a(2,2), w_t_a(2,0),
              T(0.0), T(0.0), T(0.0), T(1.0);




    //
    // Make w_T_b : State
    Matrix<T,3,3> w_R_b;// = Matrix<T,3,3>::Identity();
    Quaternion<T> ___q_b = Map<const Quaternion<T>>(q_b);
    w_R_b = ___q_b.toRotationMatrix();
    Matrix<T,3,1> w_t_b;
    w_t_a << t_b[0], t_b[1], t_b[2];

    Matrix<T,4,4> w_T_b;
    // w_T_b.block<3,3>(0,0) = w_R_b;
    // w_T_b.block<3,1>(3,0) = w_t_b;
    w_T_b << w_R_b(0,0), w_R_b(0,1), w_R_b(0,2), w_t_b(0,0),
              w_R_b(1,0), w_R_b(1,1), w_R_b(1,2), w_t_b(1,0),
              w_R_b(2,0), w_R_b(2,1), w_R_b(2,2), w_t_b(2,0),
              T(0.0), T(0.0), T(0.0), T(1.0);


    //
    // b_T_cap_a : Observation (type cast to T)
    Matrix<T,4,4> b_T_cap_a;
    b_T_cap_a = obs_edge_pose.cast<T>();


    // Make difference Transform (between observed pose and poses of the corresponding nodes)
    // Now we have : w_T_a, w_T_b, and b_T_cap_a
    Matrix<T,4,4> diff_T ;
    diff_T = w_T_a.inverse() * w_T_b * b_T_cap_a;



    // Make residue
    residuals[0] = diff_T(0,3);
    residuals[1] = diff_T(1,3);
    residuals[2] = diff_T(2,3);
    residuals[3] = T(0.0); //yaw_of( diff_T );


    return true;

  }

  static ceres::CostFunction * Create( const Matrix4d& observed_edge_pose )
  {
    return ( new ceres::AutoDiffCostFunction<Residue4DOF, 4, 4,3, 4,3 > ( new Residue4DOF(observed_edge_pose) )) ;
    // return NULL;
  }

private:
  Matrix4d obs_edge_pose; //< b_T_a
};




class PnPReprojectionError {
public:
  PnPReprojectionError( const VectorXd& _3dpt, const VectorXd& _2dpt )
  {
    X << _3dpt(0), _3dpt(1), _3dpt(2);
    x << _2dpt(0), _2dpt(1);
  }


  template <typename T>
                        // 4-vector     3-vector        2-vector
  bool operator()(const T* const quat, const T* const tran, T* e) const
  {
    // Transform X with P1, P2 ==> X_out := P1*X + P2
    Quaternion<T> q( quat[0], quat[1], quat[2], quat[3] );//w,x,y,z

    Matrix<T,3,1> t;
    t<< tran[0], tran[1], tran[2];

    Matrix<T,3,1> c_X;
    c_X << T(X(0)), T(X(1)), T(X(2));

    Matrix<T,3,1> X_transformed;
    X_transformed = q.toRotationMatrix() * c_X + t;



    // error[0] := (X_out.x / X_out.z) - x.x
    e[0] = (X_transformed(0) / X_transformed(2) ) - T(x(0));

    // error[1] := (X_out.y / X_out.z) - x.y
    e[1] = (X_transformed(1) / X_transformed(2) ) - T(x(1));

    return true;

  }

  static ceres::CostFunction* Create(const Vector4d& _3dpt, const Vector2d& _2dpt){
    return (new ceres::AutoDiffCostFunction<PnPReprojectionError, 2, 4, 3>(
        new PnPReprojectionError(_3dpt, _2dpt)));
      }

private:
  Vector3d X;
  Vector2d x;
};
