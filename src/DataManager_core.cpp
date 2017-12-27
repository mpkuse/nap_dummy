#include "DataManager.h"

// Core functions to subscribe to messages and build the pose graph

DataManager::DataManager(ros::NodeHandle &nh )
{
    this->nh = nh;

    // init republish colocation topic
    pub_chatter_colocation = this->nh.advertise<nap::NapMsg>( "/colocation_chatter", 1000 );
}


DataManager::DataManager(const DataManager &obj) {
   cout << "Copy constructor allocating ptr." << endl;

}

void DataManager::setCamera( PinholeCamera& camera )
{
  this->camera = camera;

  cout << "--- Camera Params from DataManager ---\n";
  cout << "K\n" << this->camera.e_K << endl;
  cout << "D\n" << this->camera.e_D << endl;
  cout << "--- END\n";
}

void DataManager::setVisualizationTopic( string rviz_topic )
{
  // usually was "/mish/pose_nodes"
  pub_pgraph = nh.advertise<visualization_msgs::Marker>( rviz_topic.c_str(), 0 );
  pub_pgraph_org = nh.advertise<visualization_msgs::Marker>( (rviz_topic+string("_original")).c_str(), 0 );
}


DataManager::~DataManager()
{
  cout << "In ~DataManager\n";

  string base_path = string( "/home/mpkuse/Desktop/a/drag/" );
  // string base_path = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/";

  // string file_name = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/pose_graph.nodes.csv";
  string file_name = base_path + "/pose_graph.nodes.csv";
  ofstream fp_nodes;
  fp_nodes.open( file_name );
  cout << "Write file (" <<  file_name << ") with " << nNodes.size() << " entries\n";


  fp_nodes << "#i, t, x, y, z, q.x, q.y, q.z, q.w\n";
  for( int i=0 ; i<nNodes.size() ; i++ )
  {
    Node * n = nNodes[i];

    fp_nodes <<  i << ", " << n->time_stamp  << endl;
              // << e_p[0] << ", " << e_p[1] << ", "<< e_p[2] << ", "
              // << e_q.x() << ", "<< e_q.y() << ", "<< e_q.z() << ", "<< e_q.w() << endl;
  }
  fp_nodes.close();


  // Write Odometry Edges
  // file_name = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/pose_graph.odomedges.csv";
  file_name = base_path + "/pose_graph.odomedges.csv";
  ofstream fp_odom_edge;
  fp_odom_edge.open( file_name );
  cout << "Write file (" <<  file_name << ") with " << odometryEdges.size() << " entries\n";

  fp_odom_edge << "#i, i_c, i_p, t_c, t_p\n";
  for( int i=0 ; i<odometryEdges.size() ; i++ )
  {
    Edge * e = odometryEdges[i];
    fp_odom_edge << i << ", "<< e->a_id << ", "<< e->a_timestamp
                      << ", "<< e->b_id << ", "<< e->b_timestamp << endl;
  }
  fp_odom_edge.close();


  // Write Loop Closure Edges
  // file_name = ros::package::getPath( "nap" ) + "/DUMP_pose_graph/pose_graph.loopedges.csv";
  file_name = base_path + "/pose_graph.loopedges.csv";
  ofstream fp_loop_edge;
  fp_loop_edge.open( file_name );
  cout << "Write file (" <<  file_name << ") with " << loopClosureEdges.size() << " entries\n";

  fp_loop_edge << "#i, i_c, i_p, t_c, t_p\n";
  for( int i=0 ; i<loopClosureEdges.size() ; i++ )
  {
    Edge * e = loopClosureEdges[i];
    fp_loop_edge << i << ", "<< e->a_id << ", "<< e->a_timestamp
                      << ", "<< e->b_id << ", "<< e->b_timestamp << endl;
  }
  fp_loop_edge.close();


}

void DataManager::raw_nap_cluster_assgn_callback( const sensor_msgs::ImageConstPtr& msg )
{
  cout << "clu_assgn rcvd : " << msg->header.stamp << endl;
  int i_ = find_indexof_node(msg->header.stamp);
  cv::Mat clustermap;

  try {
    clustermap = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::MONO8 )->image;
  }
  catch( cv_bridge::Exception& e)
  {
    ROS_ERROR( "cv_bridge exception in raw_nap_cluster_assgn_callback(): %s", e.what() );
  }

  cout << "nap_clusters i_=" << i_ << "     napmap_buffer="<< unclaimed_napmap.size() << endl;
  // if this i_ is found in the pose-graph set()
  if( i_ < 0 )
  {
    unclaimed_napmap.push( clustermap.clone() );
    unclaimed_napmap_time.push( ros::Time(msg->header.stamp) );
    flush_unclaimed_napmap();

    //TODO: Code up the buffering part of nap clusters. For now, you don't need to as
    // this is garunteed to be a bit delayed due to needed computation time
  }
  else //if found than associated the node with this image
  {
    nNodes[i_]->setNapClusterMap( msg->header.stamp, clustermap );
  }


}


void DataManager::image_callback( const sensor_msgs::ImageConstPtr& msg )
{
  // Search for the timestamp in pose-graph
  int i_ = find_indexof_node(msg->header.stamp);
  ROS_DEBUG( "Received - Image - %d", i_ );

  cv::Mat image, image_resized;
  try{
    image = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 )->image;
    cv::resize( image, image_resized, cv::Size(320,240) );
  }
  catch( cv_bridge::Exception& e)
  {
    ROS_ERROR( "cv_bridge exception: %s", e.what() );
  }

  // if the timestamp was not found in pose-graph,
  // buffer this image in queue
  if( i_ < 0 )
  {
    // unclaimed_im.push( image.clone() );
    unclaimed_im.push( image_resized.clone() );
    unclaimed_im_time.push( ros::Time(msg->header.stamp) );
    flush_unclaimed_im();
  }
  else //if found than associated the node with this image
  {
    // nNodes[i_]->setImage( msg->header.stamp, image );
    nNodes[i_]->setImage( msg->header.stamp, image_resized );
  }

}

void DataManager::tracked_features_callback( const sensor_msgs::PointCloudConstPtr& msg )
{
  // ROS_INFO( 'Received2d:: Features: %d', (int)msg->points.size() );
  // ROS_INFO( "Received2d");
  int i_ = find_indexof_node(msg->header.stamp);
  cout << "stamp2d : " << msg->header.stamp << endl;
  cout << "Received2d:: Node:"<< i_ <<  " size=" << msg->points.size() << endl;

  // if i_ < 0 : Node not found for this timestamp. Buffer points
  if( i_ < 0 )
  {
    Matrix<double,3,Dynamic> tracked_2d_features;
    tracked_2d_features = Matrix<double,3,Dynamic>(3,msg->points.size()); //in homogeneous co-ords. Qin Tong publishes features points in homogeneous cords
    for( int i=0 ; i<msg->points.size() ; i++ )
    {
      tracked_2d_features(0,i) = msg->points[i].x; //x
      tracked_2d_features(1,i) = msg->points[i].y; //y
      tracked_2d_features(2,i) = msg->points[i].z; //1.0
    }

    //push to buffer
    unclaimed_2d_feat.push( tracked_2d_features );
    unclaimed_2d_feat_time.push( msg->header.stamp );
    flush_unclaimed_2d_feat();
  }
  else // if i_> 0 : Found node for this. Associate these points with a node
  {
    nNodes[i_]->setFeatures2dHomogeneous( msg->header.stamp, msg->points  );
  }


}


void DataManager::point_cloud_callback( const sensor_msgs::PointCloudConstPtr& msg )
{
  int i_ = find_indexof_node(msg->header.stamp);
  cout << "stamp3d : " << msg->header.stamp << endl;
  ROS_INFO( "Received3d:: PointCloud: %d. nUnclaimed: %d", i_, (int)unclaimed_pt_cld.size() );

  if( i_ < 0 )
  {
    // 1. msg->points to eigen matrix
    Matrix<double,3,Dynamic> ptCld;
    ptCld = Matrix<double,3,Dynamic>(3,msg->points.size());
    for( int i=0 ; i<msg->points.size() ; i++ )
    {
      ptCld(0,i) = msg->points[i].x;
      ptCld(1,i) = msg->points[i].y;
      ptCld(2,i) = msg->points[i].z;
    }

    // 2. Put this eigen matrix to queue
    unclaimed_pt_cld.push( ptCld );
    unclaimed_pt_cld_time.push( msg->header.stamp );
    flush_unclaimed_pt_cld();
  }
  else
  {
    // Corresponding node exist
    nNodes[i_]->setPointCloud( msg->header.stamp, msg->points );

  }

}


void DataManager::camera_pose_callback( const nav_msgs::Odometry::ConstPtr msg )
{
  Node * n = new Node(msg->header.stamp, msg->pose.pose);
  nNodes.push_back( n );
  ROS_DEBUG( "Recvd msg - camera_pose_callback");
  cout << "add-node : " << msg->header.stamp << endl;


  // ALSO add odometry edges to 1 previous.
  int N = nNodes.size();
  int prev_k = 1; //TODO: This could be a parameter.
  if( N <= prev_k )
    return;

  //add conenction from `current` to `current-1`.
  // Edge * e = new Edge( nNodes[N-1], N-1, nNodes[N-2], N-2 );
  // odometryEdges.push_back( e );

  for( int i=0 ; i<prev_k ; i++ )
  {
    Node * a_node = nNodes[N-1];
    Node * b_node = nNodes[N-2-i];
    Edge * e = new Edge( a_node, N-1, b_node, N-2-i, EDGE_TYPE_ODOMETRY );
    e->setEdgeTimeStamps(nNodes[N-1]->time_stamp, nNodes[N-2-i]->time_stamp);

    // add relative transform as edge-inferred (using poses from corresponding edges)
    // ^{w}T_a; a:= N-1
    Matrix4d w_T_a;
    a_node->getCurrTransform( w_T_a );


    // ^{w}T_b; b:= N-2-i
    Matrix4d w_T_b;
    b_node->getCurrTransform( w_T_b );


    // ^{b}T_a = inv[ ^{w}T_b ] * ^{w}T_a
    Matrix4d b_T_a = w_T_b.inverse() * w_T_a;

    // Set
    e->setEdgeRelPose(b_T_a);

    odometryEdges.push_back( e );
  }


}


void DataManager::place_recog_callback( const nap::NapMsg::ConstPtr& msg  )
{
  // if( loopClosureEdges.size() == 10 )
  // {
  //   lock_enable_ceres.lock();
  //   enable_ceres = true;
  //   lock_enable_ceres.unlock();
  // }
  // else
  // {
  //   lock_enable_ceres.lock();
  //   enable_ceres = false;
  //   lock_enable_ceres.unlock();
  // }


  ROS_INFO( "Received - NapMsg");
  // cout << msg->c_timestamp << " " << msg->prev_timestamp << endl;

  assert( this->camera.isValid() );

  //
  // Look it up in nodes list (iterate over nodelist)
  int i_curr = find_indexof_node(msg->c_timestamp);
  int i_prev = find_indexof_node(msg->prev_timestamp);

  cout << i_curr << "<-->" << i_prev << endl;
  cout <<  msg->c_timestamp-nNodes[0]->time_stamp << "<-->" << msg->prev_timestamp-nNodes[0]->time_stamp << endl;
  cout << "Last Node timestamp : "<< nNodes.back()->time_stamp - nNodes[0]->time_stamp << endl;
  if( i_curr < 0 || i_prev < 0 )
    return;

  //
  // make a loop closure edge
  Edge * e = new Edge( nNodes[i_curr], i_curr, nNodes[i_prev], i_prev, EDGE_TYPE_LOOP_CLOSURE );
  e->setEdgeTimeStamps(msg->c_timestamp, msg->prev_timestamp);

  ///////////////////////////////////
  // Relative Pose Computation     //
  ///////////////////////////////////
  // cout << "n_sparse_matches : " << msg->n_sparse_matches << endl;
  cout << "co-ordinate match sizes : " << msg->curr.size() << " " << msg->prev.size() << " " << msg->curr_m.size() << endl;

  // //////////////////
  //---- case-a : If 3way matching is empty : do ordinary way to compute relative pose. Borrow code from Qin. Basically using 3d points from odometry (wrt curr) and having known same points in prev do pnp
  // /////////////////
  // Use the point cloud (wrt curr) and do PnP using prev
  // if( msg->n_sparse_matches >= 200 )
  if( msg->op_mode == 10 )
  {
    ROS_INFO( "Set closure-edge-subtype : EDGE_TYPE_LOOP_SUBTYPE_BASIC");
    e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_BASIC);

    // TODO: Put Qin Tong's code here. ie. rel pose computation when we have sufficient number of matches
    Matrix4d p_T_c;
    this->pose_from_2way_matching(msg, p_T_c );


    // Set the computed pose into edge
    // e->setEdgeRelPose( p_T_c );

    loopClosureEdges.push_back( e );

    // Re-publish op_mode:= 10 (as is)
    Matrix4d __h;
    int32_t mode = 10;
    republish_nap( msg->c_timestamp, msg->prev_timestamp, __h, mode );


    return;
  }


  // //////////////////
  //---- case-b : If 3way matching is not empty : i) Triangulate curr-1 and curr. ii) pnp( 3d pts from (i) ,  prev )
  // //////////////////
  // Pose computation with 3way matching
  // if( msg->n_sparse_matches < 200 && msg->curr.size() > 0 && msg->curr.size() == msg->prev.size() && msg->curr.size() == msg->curr_m.size()  )
  if( msg->op_mode == 29 )
  {
    ROS_INFO( "Set closure-edge-subtype : EDGE_TYPE_LOOP_SUBTYPE_3WAY");
    e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_3WAY);

    // TODO: Relative pose from 3way matching
    Matrix4d p_T_c = Matrix4d::Identity();
    this->pose_from_3way_matching(msg, p_T_c );

    // Set the computed pose into edge
    e->setEdgeRelPose( p_T_c );

    loopClosureEdges.push_back( e );
    // doOptimization();

    // Re-publish with pose, op_mode:=30
    int32_t mode = 30;
    republish_nap( msg->c_timestamp, msg->prev_timestamp, p_T_c, mode );


    return;
  }


  if( msg->op_mode == 20 )
  {
    // This is when the expanded matches are present. basically need to just forward this. No geometry computation here.
    ROS_INFO( "Set closure-edge-subtype : EDGE_TYPE_LOOP_SUBTYPE_GUIDED");

    e->setLoopEdgeSubtype(EDGE_TYPE_LOOP_SUBTYPE_GUIDED);


    loopClosureEdges.push_back( e );

    // Re-publish op_mode:= 10 (as is)
    // Matrix4d __h;
    // int32_t mode = 10;
    // republish_nap( msg->c_timestamp, msg->prev_timestamp, __h, mode );
    republish_nap( msg );


    return;

  }

  ROS_ERROR( "in place_recog_callback: Error computing rel pose. Edge added without pose. This might be fatal!");


}

void DataManager::republish_nap( const nap::NapMsg::ConstPtr& msg )
{
  pub_chatter_colocation.publish( *msg );
}

void DataManager::republish_nap( const ros::Time& t_c, const ros::Time& t_p, const Matrix4d& p_T_c, int32_t op_mode )
{
  cout << "Not Implemented Republish\n";
  nap::NapMsg msg;

  msg.c_timestamp = t_c;
  msg.prev_timestamp = t_p;
  msg.op_mode = op_mode;

  // if op_mode is 30 means that pose p_T_c was computed from 3-way matching
  if( op_mode == 30 )
  {
    Matrix3d p_R_c;
    Vector4d p_t_c;

    p_R_c = p_T_c.topLeftCorner<3,3>();
    p_t_c = p_T_c.col(3);

    Quaterniond q = Quaterniond( p_R_c );
    msg.p_T_c.position.x = p_t_c[0];
    msg.p_T_c.position.y = p_t_c[1];
    msg.p_T_c.position.z = p_t_c[2];

    msg.p_T_c.orientation.x = q.x();
    msg.p_T_c.orientation.y = q.y();
    msg.p_T_c.orientation.z = q.z();
    msg.p_T_c.orientation.w = q.w();
  }
  else if( op_mode == 10 ) // contains no pose info.
  {
    ;
  }
  else
  {
    ROS_ERROR( "Cannot re-publish nap. Invalid op_mode" );
  }

  pub_chatter_colocation.publish( msg );
}



// ////////////////////////////////

// Loop over each node and return the index of the node which is clossest to the specified stamp
int DataManager::find_indexof_node( ros::Time stamp )
{
  ros::Duration diff;
  for( int i=0 ; i<nNodes.size() ; i++ )
  {
    diff = nNodes[i]->time_stamp - stamp;

    // cout << i << " "<< diff.sec << " " << diff.nsec << endl;

    // if( abs(diff.sec) <= int32_t(0) && abs(diff.nsec) < int32_t(1000000) ) {
    // if( abs(diff.sec) <= int32_t(0) && abs(diff.nsec) == int32_t(0) ) {
    if( diff < ros::Duration(0.0001) && diff > ros::Duration(-0.0001) ){
      return i;
    }
  }//TODO: the duration can be a fixed param. Basically it is used to compare node timestamps.
  // ROS_INFO( "Last Diff=%d:%d. Cannot find specified timestamp in nodelist. ", diff.sec,diff.nsec);
  return -1;
}


void DataManager::flush_unclaimed_napmap()
{
  ROS_WARN( "flush_unclaimed_napmapIM:%d, T:%d", (int)unclaimed_napmap.size(), (int)unclaimed_napmap_time.size() );


  // int N = max(20,(int)unclaimed_im.size() );
  int N = unclaimed_napmap.size() ;
  for( int i=0 ; i<N ; i++)
  {
    cv::Mat image = cv::Mat(unclaimed_napmap.front());
    ros::Time stamp = ros::Time(unclaimed_napmap_time.front());
    unclaimed_napmap.pop();
    unclaimed_napmap_time.pop();
    int i_ = find_indexof_node(stamp);
    if( i_ < 0 )
    {
      unclaimed_napmap.push( image.clone() );
      unclaimed_napmap_time.push( ros::Time(stamp) );
    }
    else
    {
      nNodes[i_]->setNapClusterMap( stamp, image );
    }
  }

}

void DataManager::flush_unclaimed_im()
{
  ROS_WARN( "IM:%d, T:%d", (int)unclaimed_im.size(), (int)unclaimed_im_time.size() );

  // std::queue<cv::Mat> X_im;
  // std::queue<ros::Time> X_tm;

  int N = max(20,(int)unclaimed_im.size() );
  // while( !unclaimed_im.empty() )
  for( int i=0 ; i<N ; i++)
  {
    cv::Mat image = cv::Mat(unclaimed_im.front());
    ros::Time stamp = ros::Time(unclaimed_im_time.front());
    unclaimed_im.pop();
    unclaimed_im_time.pop();
    int i_ = find_indexof_node(stamp);
    if( i_ < 0 )
    {
      unclaimed_im.push( image.clone() );
      unclaimed_im_time.push( ros::Time(stamp) );
    }
    else
    {
      nNodes[i_]->setImage( stamp, image );
    }
  }


  // // Put back unfound ones
  // while( !X_tm.empty() )
  // {
  //   unclaimed_im.push( cv::Mat(X_im.front()) );
  //   unclaimed_im_time.push( ros::Time(X_tm.front()) );
  //   X_im.pop();
  //   X_tm.pop();
  // }
}


void DataManager::flush_unclaimed_pt_cld()
{
  ROS_WARN( "PtCld %d, %d", (int)unclaimed_pt_cld.size(), (int)unclaimed_pt_cld_time.size() );
  int M = max(20,(int)unclaimed_pt_cld.size()); // Potential BUG. If not found, the ptcld is pushed at the end, where you will never get to as you see only first 20 elements!
  for( int i=0 ; i<M ; i++ )
  {
    Matrix<double,3,Dynamic> e;
    e = unclaimed_pt_cld.front();
    ros::Time t = ros::Time( unclaimed_pt_cld_time.front() );
    unclaimed_pt_cld.pop();
    unclaimed_pt_cld_time.pop();
    int i_ = find_indexof_node(t);
    if( i_ < 0 )
    {
      //still not found, push back again
      unclaimed_pt_cld.push( e );
      unclaimed_pt_cld_time.push( t );
    }
    else
    {
      nNodes[i_]->setPointCloud(t, e);
    }
  }

}
void DataManager::flush_unclaimed_2d_feat()
{
  ROS_WARN( "flush2dfeat %d, %d", (int)unclaimed_2d_feat.size(), (int)unclaimed_2d_feat_time.size() );
  // int M = max(20,(int)unclaimed_2d_feat.size());
  int M = unclaimed_2d_feat.size();
  cout << "flush_feat2d()\n";
  for( int i=0 ; i<M ; i++ )
  {
    Matrix<double,3,Dynamic> e;
    e = unclaimed_2d_feat.front();
    ros::Time t = ros::Time( unclaimed_2d_feat_time.front() );
    unclaimed_2d_feat.pop();
    unclaimed_2d_feat_time.pop();
    int i_ = find_indexof_node(t);
    if( i_ < 0 )
    {
      //still not found, push back again
      unclaimed_2d_feat.push( e );
      unclaimed_2d_feat_time.push( t );
    }
    else
    {
      cout << "found "<< t << "--> " << i_ << endl;
      nNodes[i_]->setFeatures2dHomogeneous(t, e); //this will be set2dFeatures()
      return;
    }
  }

}

// /// Debug file - Mark for removal. The debug txt file is now handled inside
// void DataManager::open_debug_xml( const string& fname)
// {
//   ROS_INFO( "Open DEBUG XML : %s", fname.c_str() );
//   (this->debug_fp).open( fname, cv::FileStorage::WRITE );
// }
//
// const cv::FileStorage& DataManager::get_debug_file_ptr()
// {
//   if( debug_fp.isOpened() == false ) {
//     ROS_ERROR( "get_debug_file_ptr(): debug xml file is not open. Call open_debug_xml() before this function" );
//     return NULL;
//   }
//
//   return debug_fp;
// }
//
// void DataManager::close_debug_xml()
// {
//   if( debug_fp.isOpened() == false )
//   {
//     ROS_ERROR( "close_debug_xml() : Attempting to close a file that is not open. COnsider calling open_debug_xml() before this function");
//   }
//   this->debug_fp.release();
// }
