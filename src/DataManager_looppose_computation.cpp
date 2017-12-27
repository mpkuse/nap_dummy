#include "DataManager.h"

// Functions for computation of relative pose

void DataManager::pose_from_2way_matching( const nap::NapMsg::ConstPtr& msg, Matrix4d& p_T_c )
{
  cout << "Do Qin Tong's Code here, for computation of relative pose\nNot Yet implemented";

}

#define _DEBUG_3WAY
#define _DEBUG_PNP
void DataManager::pose_from_3way_matching( const nap::NapMsg::ConstPtr& msg, Matrix4d& p_T_c )
{
  cout << "Code here for 3way match pose computation\n";



  // Get nodes from the timestamps
  int ix_curr = find_indexof_node(msg->t_curr);
  int ix_prev = find_indexof_node(msg->t_prev);
  int ix_curr_m = find_indexof_node(msg->t_curr_m);
  assert( ix_curr > 0 && ix_prev>0 && ix_curr_m>0 );


  // [ ^{c-1}T_c ] ==> [ inv( ^wT_{c-1} ) * ^wT_c ]
  Matrix4d w_T_cm;
  nNodes[ix_curr_m]->getCurrTransform(w_T_cm);//4x4
  Matrix4d w_T_c;
  nNodes[ix_curr]->getCurrTransform(w_T_c);//4x4
  Matrix4d w_T_p;
  nNodes[ix_prev]->getCurrTransform(w_T_p);//4x4

  Matrix4d Tr;
  Tr = w_T_cm.inverse() * w_T_c; //relative transform

  Matrix3d F_c_cm;
  cv::Mat mat_F_c_cm;
  make_fundamentalmatrix_from_pose( w_T_c, w_T_cm, F_c_cm );
  cv::eigen2cv( F_c_cm, mat_F_c_cm );
  cout << "F_c_cm:\n" << F_c_cm << endl;


  #ifdef _DEBUG_3WAY
    // Open DEBUG file
    char fname[200];
    sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d.opencv", ix_curr, ix_prev, ix_curr_m );
    cout << "Writing file : " << fname << endl;
    cv::FileStorage debug_fp( fname, cv::FileStorage::WRITE );
  #endif

  //
  // Step-1 : Get 3way matches from msg as matrix
  //
  cv::Mat mat_pts_curr, mat_pts_prev, mat_pts_curr_m; //2xN 1-channel
  extract_3way_matches_from_napmsg(msg, mat_pts_curr, mat_pts_prev, mat_pts_curr_m);


  // The above matches are in observed_image_space. For geometry, need to
  // convert these into undistorted_image_space. Qin Tong had told me it
  // publishes the keyframe original, however the pointcloud is published in
  // undistorted_image_space.

  cv::Mat undistorted_pts_curr, undistorted_pts_prev, undistorted_pts_curr_m; //2xN 1-channel
  // camera.jointUndistortPointSets( mat_F_c_cm,
                                  // mat_pts_curr, mat_pts_curr_m,
                                  // undistorted_pts_curr, undistorted_pts_curr_m );
  camera.undistortPointSet( mat_pts_curr, undistorted_pts_curr);
  camera.undistortPointSet( mat_pts_curr_m, undistorted_pts_curr_m);
  camera.undistortPointSet( mat_pts_prev, undistorted_pts_prev);



#ifdef _DEBUG_3WAY
  // Collect undistortedPoints in normalized cords for analysis
  cv::Mat undist_normed_curr, undist_normed_curr_m, undist_normed_prev; //TODO, these 3 normalized image co-ordinate are in use. Move them outside debug
  camera.getUndistortedNormalizedCords( mat_pts_curr,  undist_normed_curr );
  camera.getUndistortedNormalizedCords( mat_pts_curr_m,  undist_normed_curr_m );
  camera.getUndistortedNormalizedCords( mat_pts_prev,  undist_normed_prev );
  debug_fp << "undist_normed_curr" << undist_normed_curr;
  debug_fp << "undist_normed_curr_m" << undist_normed_curr_m;
  debug_fp << "undist_normed_prev" << undist_normed_prev;


  debug_fp << "K" << camera.m_K;
  debug_fp << "D" << camera.m_D;
  debug_fp << "mat_pts_curr" << mat_pts_curr ;
  debug_fp << "undistorted_pts_curr" << undistorted_pts_curr ;
  debug_fp << "mat_pts_prev" << mat_pts_prev ;
  debug_fp << "undistorted_pts_prev" << undistorted_pts_prev ;
  debug_fp << "mat_pts_curr_m" << mat_pts_curr_m ;
  debug_fp << "undistorted_pts_curr_m" << undistorted_pts_curr_m ;


  // Collect Images - for debug
  cv::Mat curr_im, prev_im, curr_m_im;
  // Remember to disable the subscriber for images when not using this debug. Will save up lot of memory here!
  curr_im = this->nNodes[ix_curr]->getImageRef();
  prev_im = this->nNodes[ix_prev]->getImageRef();
  curr_m_im = this->nNodes[ix_curr_m]->getImageRef();


  // Start images writing. Writing to FileStorage is too slow
  cv::Mat grey_curr_im, grey_curr_m_im, grey_prev_im;
  cv::cvtColor(curr_im, grey_curr_im, cv::COLOR_BGR2GRAY);
  cv::cvtColor(curr_m_im, grey_curr_m_im, cv::COLOR_BGR2GRAY);
  cv::cvtColor(prev_im, grey_prev_im, cv::COLOR_BGR2GRAY);
  // debug_fp << "curr_im" << grey_curr_im;
  // debug_fp << "curr_m_im" << grey_curr_m_im;
  // debug_fp << "prev_im" << grey_prev_im;

  sprintf( fname, "/home/mpkuse/Desktop/a/drag/%d.png", ix_curr );
  cout << "Writing : " << fname << endl;
  cv::imwrite( fname, curr_im );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/%d.png", ix_curr_m );
  cout << "Writing : " << fname << endl;
  cv::imwrite( fname, curr_m_im );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/%d.png", ix_prev );
  cout << "Writing : " << fname << endl;
  cv::imwrite( fname, prev_im );
  // End of Images Writing


  // Plot 3way match
  cv::Mat dst_plot_3way;
  this->plot_3way_match( curr_im, mat_pts_curr,
                        prev_im, mat_pts_prev,
                        curr_m_im, mat_pts_curr_m,
                        dst_plot_3way );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d.jpg", ix_curr, ix_prev, ix_curr_m );
  cv::imwrite( fname, dst_plot_3way );
  this->plot_3way_match_clean( curr_im, mat_pts_curr,
                        prev_im, mat_pts_prev,
                        curr_m_im, mat_pts_curr_m,
                        dst_plot_3way );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d.png", ix_curr, ix_prev, ix_curr_m );

  cout << "Writing file : " << fname << endl;
  cv::imwrite( fname, dst_plot_3way );


  // Plot points  on images
  cv::Mat dst;
  plot_point_sets( curr_im, mat_pts_curr, dst, cv::Scalar(255,0,0) );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d_curr_observed.png", ix_curr, ix_prev, ix_curr_m );
  cout << "Writing file : " << fname << endl;
  cv::imwrite( fname, dst );

  cv::Mat dst2;
  plot_point_sets( curr_m_im, mat_pts_curr_m, dst2, cv::Scalar(255,0,0) );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d_currm_observed.png", ix_curr, ix_prev, ix_curr_m );
  cout << "Writing file : " << fname << endl;
  cv::imwrite( fname, dst2 );

  cv::Mat dst2_5;
  plot_point_sets( prev_im, mat_pts_prev, dst2_5, cv::Scalar(255,0,0) );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d_prev_observed.png", ix_curr, ix_prev, ix_curr_m );
  cout << "Writing file : " << fname << endl;
  cv::imwrite( fname, dst2_5 );

#endif

  //
  // Step-2 : Triangulation using matched points from curr and curr-1
  //
  cv::Mat c_3dpts_4N;//4xN
  triangulate_points( ix_curr, undistorted_pts_curr, ix_curr_m, undistorted_pts_curr_m, c_3dpts_4N );
  _perspective_divide_inplace( c_3dpts_4N );


#ifdef _DEBUG_3WAY
  //
  // Reproject 3d Points - Analysis
  //

  // // write 3d pts to file
  print_cvmat_info( "c_3dpts_4N", c_3dpts_4N );
  debug_fp << "c_3dpts_4N" << c_3dpts_4N ;


  // // Reproject 3d pts on image-curr
  cv::Mat _reprojected_pts_into_c;
  camera.perspectiveProject3DPoints( c_3dpts_4N, _reprojected_pts_into_c );
  print_cvmat_info( "reprojected_pts_into_c" , _reprojected_pts_into_c );
  debug_fp << "reprojected_pts_into_c" << _reprojected_pts_into_c;


  // Reproject 3d pts on image-curr_m

  Matrix4f Tr_float = Tr.cast<float>();

  cv::Mat _reprojected_pts_into_cm;
  camera.perspectiveProject3DPoints( c_3dpts_4N, Tr_float, _reprojected_pts_into_cm );
  print_cvmat_info( "reprojected_pts_into_cm" , _reprojected_pts_into_cm );
  debug_fp << "reprojected_pts_into_cm" << _reprojected_pts_into_cm;

  cv::Mat mat_w_T_cm, mat_w_T_c, mat_cm_T_c;
  cv::eigen2cv( w_T_cm, mat_w_T_cm );
  cv::eigen2cv( w_T_c, mat_w_T_c );
  cv::eigen2cv( Tr, mat_cm_T_c );
  debug_fp << "w_T_cm" << mat_w_T_cm;
  debug_fp << "w_T_c" << mat_w_T_c;
  debug_fp << "cm_T_c" << mat_cm_T_c;


  debug_fp << "F_c_cm" << mat_F_c_cm;


  // Plot reprojected pts on image-curr
  cv::Mat dst3;
  plot_point_sets( curr_im, _reprojected_pts_into_c, dst3, cv::Scalar(0,0,255) );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d_curr_reproj.png", ix_curr, ix_prev, ix_curr_m );
  cout << "Writing file : " << fname << endl;
  cv::imwrite( fname, dst3 );

  // Plot reprojected pts on image-curr_m
  cv::Mat dst4;
  plot_point_sets( curr_m_im, _reprojected_pts_into_cm, dst4, cv::Scalar(0,0,255) );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d_currm_reproj.png", ix_curr, ix_prev, ix_curr_m );
  cout << "Writing file : " << fname << endl;
  cv::imwrite( fname, dst4 );




#endif

  //
  // Step-3 : PnP using a) 3d pts from step2 and b) corresponding matches from prev
  //
  // 3dpts (in frame of curr), 2dpts (in prev)
  // TODO: Consider naming these outputs of pnp with a prefix `pnp`
  // TODO: Try and use only pitch and roll (do not use yaw as initialization for pnp)
  Matrix4d prev_T_c = Matrix4d::Identity();
  // estimatePnPPose(  c_3dpts_4N, mat_pts_prev, prev_T_c ); //This is the real deal. Above 2 only for debug
  // estimatePnPPose(  c_3dpts_4N, undist_normed_prev, prev_T_c );
  // estimatePnPPose_withguess( c_3dpts_4N, undist_normed_prev, prev_T_c, w_T_c, w_T_p );
  estimatePnPPose_ceres( c_3dpts_4N, undist_normed_prev, prev_T_c );



#ifdef _DEBUG_PNP
  // 3dpts, curr-image
  Matrix4d curr_T_c = Matrix4d::Identity();
  //estimatePnPPose(  c_3dpts_4N, mat_pts_curr, curr_T_c ); // This should giveout identity, for debug/verification
  estimatePnPPose_ceres(  c_3dpts_4N, undist_normed_curr, curr_T_c );

  // 3dpts, curr_m-image
  Matrix4d currm_T_c = Matrix4d::Identity();
  //estimatePnPPose(  c_3dpts_4N, mat_pts_curr_m, currm_T_c ); // This should be same as cm_T_c, for debug/verification
  estimatePnPPose_ceres(  c_3dpts_4N, undist_normed_curr_m, currm_T_c );

  cv::Mat pnp_curr_T_c, pnp_currm_T_c, pnp_prev_T_c;
  cv::eigen2cv( curr_T_c, pnp_curr_T_c );
  cv::eigen2cv( currm_T_c, pnp_currm_T_c );
  cv::eigen2cv( prev_T_c, pnp_prev_T_c );
  debug_fp << "pnp_curr_T_c"<< pnp_curr_T_c;
  debug_fp << "pnp_currm_T_c"<< pnp_currm_T_c;
  debug_fp << "pnp_prev_T_c"<< pnp_prev_T_c;


  // Reproject 3dpoints on prev view using the pose estimated from pnp
  cv::Mat _reprojected_pts_into_prev;
  Matrix4f prev_Tr_c_float = prev_T_c.cast<float>();
  camera.perspectiveProject3DPoints( c_3dpts_4N, prev_Tr_c_float, _reprojected_pts_into_prev );
  print_cvmat_info( "_reprojected_pts_into_prev" , _reprojected_pts_into_prev );
  debug_fp << "_reprojected_pts_into_prev" << _reprojected_pts_into_prev;


  // Reproject 3dpoints on prev view using pose estimates from odometry
  cv::Mat _reprojected_pts_into_prev__pose_from_odom;
  Matrix4f prev_Tr_c_odom_float = (w_T_p.inverse() * w_T_c).cast<float>();
  camera.perspectiveProject3DPoints( c_3dpts_4N, prev_Tr_c_odom_float, _reprojected_pts_into_prev__pose_from_odom);
  print_cvmat_info( "_reprojected_pts_into_prev__pose_from_odom" , _reprojected_pts_into_prev__pose_from_odom );
  debug_fp << "_reprojected_pts_into_prev__pose_from_odom" << _reprojected_pts_into_prev__pose_from_odom;






  // Goodness of triangulation. Difference between reprojected and observed pts.
  double err_c = _diff_2d( mat_pts_curr, _reprojected_pts_into_c );
  double err_cm = _diff_2d( mat_pts_curr_m, _reprojected_pts_into_cm );
  double err_p = _diff_2d( mat_pts_prev, _reprojected_pts_into_prev );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d_reproj_err.txt", ix_curr, ix_prev, ix_curr_m );
  ofstream myf;
  myf.open( fname );
  myf << ix_curr << ", " << ix_prev << ", "<< ix_curr_m << ", " <<
            _reprojected_pts_into_c.cols << ", "
            << err_c << ", " << err_cm << "," << err_p << endl;
  myf.close();


  // Plot reprojected pts on prev
  cv::Mat dst5;
  plot_point_sets( prev_im, _reprojected_pts_into_prev, dst5, cv::Scalar(0,0,255), matrix4f_to_string(prev_Tr_c_float) );
  sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d_prev_reproj.png", ix_curr, ix_prev, ix_curr_m );
  cout << "Writing file : " << fname << endl;
  cv::imwrite( fname, dst5 );



// Plot reprojected pts on prev
// cv::Mat dst5_1;
// plot_point_sets( prev_im, _reprojected_pts_into_prev__pose_from_odom, dst5_1, cv::Scalar(0,0,255), matrix4f_to_string(prev_Tr_c_odom_float) );
// sprintf( fname, "/home/mpkuse/Desktop/a/drag/pg_%d_%d___%d_prev_reproj_pose_from_odom.png", ix_curr, ix_prev, ix_curr_m );
// cout << "Writing file : " << fname << endl;
// cv::imwrite( fname, dst5_1 );


#endif


  //
  // Step-4 : Put the relative pose from step3 into eigen matrix in required convention
  //       (p_T_c)
  p_T_c = prev_T_c;



#ifdef _DEBUG_3WAY
  debug_fp.release();
#endif


}





// //////////////// UTILS ////////////////////////// //

void DataManager::extract_3way_matches_from_napmsg( const nap::NapMsg::ConstPtr& msg,
      cv::Mat&mat_pts_curr, cv::Mat& mat_pts_prev, cv::Mat& mat_pts_curr_m )
{
  int N = msg->curr.size();
  assert( N>0 && msg->curr.size()==N && msg->prev.size()==N && msg->curr_m.size()==N );

  mat_pts_curr = cv::Mat(2,N,CV_32F);
  mat_pts_prev = cv::Mat(2,N,CV_32F);
  mat_pts_curr_m = cv::Mat(2,N,CV_32F);
  for( int kl=0 ; kl<N ; kl++ )
  {
    mat_pts_curr.at<float>(0,kl) = (float)msg->curr[kl].x;
    mat_pts_curr.at<float>(1,kl) = (float)msg->curr[kl].y;
    mat_pts_prev.at<float>(0,kl) = (float)msg->prev[kl].x;
    mat_pts_prev.at<float>(1,kl) = (float)msg->prev[kl].y;
    mat_pts_curr_m.at<float>(0,kl) = (float)msg->curr_m[kl].x;
    mat_pts_curr_m.at<float>(1,kl) = (float)msg->curr_m[kl].y;
  }
}




void DataManager::make_fundamentalmatrix_from_pose( const Matrix4d& w_T_c, const Matrix4d& w_T_cm,
                                        Matrix3d& F )
{
  // Convert the poses to Canonical Form
  Matrix4d Tr;
  Tr = w_T_cm.inverse() * w_T_c; //relative transform


  F = Matrix3d::Identity();
  Matrix3d e = Matrix3d::Zero(); // |_ t _|
  e(0,1) = -Tr(2,3); // -tz
  e(0,2) = Tr(1,3);    //  ty
  e(1,2) = -Tr(0,3);    // -tx
  e(1,0) = -e(0,1);
  e(2,0) = -e(0,2);
  e(2,1) = -e(1,2);

  F = e * Tr.topLeftCorner<3,3>();



}



void DataManager::triangulate_points( int ix_curr, const cv::Mat& mat_pts_curr,
                         int ix_curr_m, const cv::Mat& mat_pts_curr_m,
                         cv::Mat& c_3dpts )
  {
  //
  // Step-1 : Relative pose, Projection Matrix of ix_curr and ix_curr_m

  // K [ I | 0 ]
  MatrixXd I_0;
  I_0 = Matrix4d::Identity().topLeftCorner<3,4>();
  MatrixXd P1 = camera.e_K * I_0; //3x4

  // K [ ^{c-1}T_c ] ==> K [ inv( ^wT_{c-1} ) * ^wT_c ]
  Matrix4d w_T_cm;
  nNodes[ix_curr_m]->getCurrTransform(w_T_cm);//4x4
  Matrix4d w_T_c;
  nNodes[ix_curr]->getCurrTransform(w_T_c);//4x4

  MatrixXd Tr;
  Tr = w_T_cm.inverse() * w_T_c; //relative transform
  MatrixXd P2 = camera.e_K * Tr.topLeftCorner<3,4>(); //3x4

  cv::Mat xP1(3,4,CV_64F );
  cv::Mat xP2(3,4,CV_64F );
  cv::eigen2cv( P1, xP1 );
  cv::eigen2cv( P2, xP2 );


  //
  // Step-2 : OpenCV Triangulate
  cv::triangulatePoints( xP1, xP2,  mat_pts_curr, mat_pts_curr_m,   c_3dpts );


}


void DataManager::estimatePnPPose( const cv::Mat& c_3dpts, const cv::Mat& pts2d,
                      Matrix4d& im_T_c  )
{
  assert( (c_3dpts.cols == pts2d.cols) &&  pts2d.cols>0 );

  //cv::Mat to std::vector<cv::Point3f>
  //cv::Mat to std::vector<cv::Point2f>
  vector<cv::Point3f> _3d;
  vector<cv::Point2f> _2d;
  for( int i=0 ; i<pts2d.cols ; i++ )
  {
    cv::Point3f point3d_model = cv::Point3f( c_3dpts.at<float>(0,i), c_3dpts.at<float>(1,i), c_3dpts.at<float>(2,i) );
    cv::Point2f point2d_scene = cv::Point2f( pts2d.at<float>(0,i), pts2d.at<float>(1,i) );

    _3d.push_back( point3d_model );
    _2d.push_back( point2d_scene );
  }



  // solvePnP
  cv::Mat rvec, tvec; //CV_64FC1

  // Simple PnP
  // cv::solvePnP( _3d, _2d, camera.m_K, camera.m_D, rvec, tvec );


  // When 2d pts are in normalized undistorted image co-ordinates
  cv::Mat eye = cv::Mat::eye(3,3,CV_32FC1);
  cv::Mat zer = cv::Mat::zeros(1,4,CV_32FC1);

  // cv::solvePnP( _3d, _2d, eye, zer, rvec, tvec );
  // cout << "solvePnP returned : " << flag << endl;

  //Try out solvePnPRansac
  // cv::solvePnPRansac( _3d, _2d, eye, zer, rvec, tvec );
  cv::solvePnPRansac( _3d, _2d, eye, zer, rvec, tvec, false, 100, 8.0, .99, cv::noArray(), cv::SOLVEPNP_ITERATIVE  );


  // Convert rvec, tvec to Eigen::Matrix4d or Eigen::Matrix4f
  // #define ___JH_ROT_PRINT
#ifdef ___JH_ROT_PRINT
  print_cvmat_info( "rvec", rvec );cout << "rvec:"<< rvec << endl;
  print_cvmat_info( "tvec", tvec );cout << "tvec:"<< tvec << endl;
#endif

  cv::Mat rot;
  cv::Rodrigues( rvec, rot );

#ifdef ___JH_ROT_PRINT
  print_cvmat_info( "rot", rot );cout << "rot:\n" << rot << endl;
#endif

  Matrix3d eig_rot;
  cv::cv2eigen( rot, eig_rot );

  Vector3d eig_trans;
  cv::cv2eigen( tvec, eig_trans );


  im_T_c = Matrix4d::Identity();
  im_T_c.topLeftCorner<3,3>() = eig_rot;
  im_T_c.block<3,1>(0,3) = eig_trans;


}



void DataManager::estimatePnPPose_withguess( const cv::Mat& c_3dpts, const cv::Mat& pts2d,
                      Matrix4d& im_T_c,
                      const Matrix4d& odom_w_T_c, const Matrix4d& odom_w_T_p )
{
  //
  // Process the Guess to get to rvec, tvec
  //
  // cout << "NOT IMPLEMENTED> PNP WITH GUESS FROM ODOM\n";
  Matrix4d odom_p_T_c = odom_w_T_p.inverse() * odom_w_T_c;
  Matrix3d odom_p_R_c = odom_p_T_c.topLeftCorner<3,3>();
  Vector3d odom_p_t_c;
  // odom_p_t_c << odom_p_T_c(0,3), odom_p_T_c(1,3), odom_p_T_c(2,3);
  odom_p_t_c << 0.0, 0.0, 0.0;

  cv::Mat _rot, tvec, rvec;
  cv::eigen2cv( odom_p_R_c, _rot );
  cv::eigen2cv( odom_p_t_c, tvec );
  cv::Rodrigues( _rot, rvec );


  //
  // pts to vector<Point3f>, vector<Point2f>
  //
  assert( (c_3dpts.cols == pts2d.cols) &&  pts2d.cols>0 );

  //cv::Mat to std::vector<cv::Point3f>
  //cv::Mat to std::vector<cv::Point2f>
  vector<cv::Point3f> _3d;
  vector<cv::Point2f> _2d;
  for( int i=0 ; i<pts2d.cols ; i++ )
  {
    cv::Point3f point3d_model = cv::Point3f( c_3dpts.at<float>(0,i), c_3dpts.at<float>(1,i), c_3dpts.at<float>(2,i) );
    cv::Point2f point2d_scene = cv::Point2f( pts2d.at<float>(0,i), pts2d.at<float>(1,i) );

    _3d.push_back( point3d_model );
    _2d.push_back( point2d_scene );
  }


  // Do PnP
  // When 2d pts are in normalized undistorted image co-ordinates
  cv::Mat eye = cv::Mat::eye(3,3,CV_32FC1);
  cv::Mat zer = cv::Mat::zeros(1,4,CV_32FC1);

  cv::solvePnPRansac( _3d, _2d, eye, zer, rvec, tvec, true, 100, 8.0, .99, cv::noArray(), cv::SOLVEPNP_EPNP  );

  // Convert back to eigen
  cv::Mat out_rot;
  cv::Rodrigues( rvec, out_rot );


  Matrix3d eig_rot;
  cv::cv2eigen( out_rot, eig_rot );

  Vector3d eig_trans;
  cv::cv2eigen( tvec, eig_trans );


  im_T_c = Matrix4d::Identity();
  im_T_c.topLeftCorner<3,3>() = eig_rot;
  im_T_c.block<3,1>(0,3) = eig_trans;





}



// c_3dpts_4N - 4xN. 3d points in homogeneous co-ordinates
// pts2d - 2xN. Image points in undistorted-normalized-image co-ordinates.
void DataManager::estimatePnPPose_ceres( const cv::Mat& c_3dpts_4N, const cv::Mat& pts2d,
                      Matrix4d& im_T_c  )
{
  //
  // Convert input to Eigen format
  MatrixXd eigen_c_3dpts_4N, eigen_2dpts;
  cv::cv2eigen(c_3dpts_4N, eigen_c_3dpts_4N);
  cv::cv2eigen(pts2d, eigen_2dpts);


  //
  // Model the problem
  ceres::Problem problem;
  double opt_q[4] = {1.,0.,0.,0.};
  double opt_t[3] = {0.,0.,0.};
  cout << "opt_q " << opt_q[0] << " "<< opt_q[1] << " "<< opt_q[2] << " "<< opt_q[3] << endl;
  cout << "opt_t " << opt_t[0] << " "<< opt_t[1] << " "<< opt_t[2] << " "<< endl ;


  for( int i=0 ; i<eigen_2dpts.cols() ; i++ ) {
    ceres::CostFunction * cost_function = PnPReprojectionError::Create( eigen_c_3dpts_4N.col(i),
                                                                        eigen_2dpts.col(i)  );
    problem.AddResidualBlock( cost_function, new ceres::HuberLoss(0.01), opt_q, opt_t );
  }



  //
  // Set q as a Quaternion Increment
  ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
  problem.SetParameterization( opt_q, quaternion_parameterization );

  //
  // Solve
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.dogleg_type = ceres::SUBSPACE_DOGLEG;
  options.max_num_iterations = 5;
  options.max_solver_time_in_seconds = 100E-3;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);


  quaternion_to_T( opt_q, opt_t, im_T_c );
  // cout << summary.FullReport() << endl;
  // cout << "opt_t " << opt_t[0] << " "<< opt_t[1] << " "<< opt_t[2] << " "<< endl ;
  // cout << "opt_q " << opt_q[0] << " "<< opt_q[1] << " "<< opt_q[2] << " "<< opt_q[3] << endl;
  // cout << "Ceres PnP p_T_c:\n" << im_T_c << endl;

}

void DataManager::quaternion_to_T( double * opt_q, double * opt_t, Matrix4d& Tr )
{
  Quaterniond Q(opt_q[0], opt_q[1], opt_q[2], opt_q[3] );
  Vector4d t;
  t << opt_t[0], opt_t[1], opt_t[2], 1.0;

  Tr = Matrix4d::Identity();
  Tr.topLeftCorner<3,3>() = Q.toRotationMatrix();
  Tr.col(3) = t;
}


// Given an input matrix of size nXN returns (n+1)xN. Increase the dimension by 1
// Add an extra dimension with 1.0 as the entry. This is NOT an inplace operation.
void DataManager::_to_homogeneous( const cv::Mat& in, cv::Mat& out )
{
  //TODO: Use M.row() to vectorize
  cout << "Not implemented _to_homogeneous\n";
  out = cv::Mat( in.rows+1, in.cols, in.type() );
  int lastindex_of_out = out.rows-1;
  for( int iu=0 ; iu<in.cols ; iu++ ) // loop across all points
  {
    for( int iv=0 ; iv<in.rows ; iv++ ) //x,y,z, (dimensions)
    {
      out.at<float>(iv,iu) = (float)in.at<float>(iv,iu) ;
    }
    out.at<float>(lastindex_of_out, iu) = 1.0;
  }
}


// Given input of size nxN. return (n-1)xN. Decrease the dimension by 1
// Y = X[0:3,:] / X[3,:]. This is not an inplace operation
void DataManager::_from_homogeneous( const cv::Mat& in, cv::Mat& out )
{
  //TODO: Use M.row() to vectorize
  out = cv::Mat( in.rows-1, in.cols, in.type() );
  int lastindex_of_in = in.rows-1;
  for( int iu=0 ; iu<in.cols ; iu++ )
  {
    for( int iv=0 ; iv<out.rows ; iv++ ) //x,y,z,
    {
      out.at<float>(iv,iu) = (float)in.at<float>(iv,iu) / (float)in.at<float>(lastindex_of_in,iu);
    }
  }
}


// Input of nxN returns nxN. After this call the last row will be 1s.
// X = X / X[n-1,:]
void DataManager::_perspective_divide_inplace( cv::Mat& in )
{
  //TODO: Use M.row() to vectorize
  for( int iu = 0 ; iu<in.cols ; iu++ )
  {
    float to_divide = in.at<float>( in.rows-1, iu );
    for( int j=0 ; j<in.rows ; j++ )
    {
      in.at<float>( j, iu ) /= to_divide;
    }
  }
}



double DataManager::_diff_2d( const cv::Mat&A, const cv::Mat&B )
{
  int M = A.cols;
  double sum = 0.0;
  for( int i=0 ; i<M ; i++ )
  {
    double dx = double(A.at<float>(0,i) - B.at<float>(0,i)) ;
    double dy = double(A.at<float>(1,i) - B.at<float>(1,i)) ;
    sum += dx*dx + dy*dy;
  }
  sum = sqrt(sum / (double)M );
  return sum;
}


void DataManager::convert_rvec_eigen4f( const cv::Mat& rvec, const cv::Mat& tvec, Matrix4f& Tr )
{
  Tr = Matrix4f::Identity();

  // Rotation
  cv::Mat R;
  cv::Rodrigues( rvec, R );
  cout << "R:" << R << endl;

  Matrix3f e_R;
  cv::cv2eigen( R, e_R );

  Tr.topLeftCorner<3,3>() = e_R;
  cout << "Eigen Tr:" << Tr << endl;

  // Translation
  // Vector3f e_T;
  // cv::cv2eigen( tvec, e_T);
  // Tr.col(3) = e_T;
}


void DataManager::print_cvmat_info( string msg, const cv::Mat& A )
{
  cout << msg << ":" << "rows=" << A.rows << ", cols=" << A.cols << ", ch=" << A.channels() << ", type=" << type2str( A.type() ) << endl;
}

string DataManager::type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

string DataManager::matrix4f_to_string( const Matrix4f& M )
{
  Vector3f ea = M.topLeftCorner<3,3>().eulerAngles( 2,1,0 );
  Vector3f tx;
  tx << M(0,3), M(1,3), M(2,3);

  string to_return;

  char pr[200];
  sprintf( pr, "A: %4.2f, %4.2f, %4.2f;Tx: %4.2f, %4.2f, %4.2f", ea(0), ea(1), ea(2), tx(0), tx(1), tx(2) );
  to_return = string( pr );

  // to_return = string("A: ");
  // to_return += std::to_string( round( ea(0) * 1000. )/1000. ) + string( ";");
  // to_return += std::to_string( round( ea(1) * 1000. )/1000. ) + string( ";");
  // to_return += std::to_string( round( ea(2) * 1000. )/1000. ) + string( ";");
  // to_return += string( "  T: ");
  // to_return += std::to_string( round( tx(0) * 1000. )/1000. ) + string( ";");
  // to_return += std::to_string( round( tx(1) * 1000. )/1000. ) + string( ";");
  // to_return += std::to_string( round( tx(2) * 1000. )/1000. ) + string( ";");

  return to_return;
}


string DataManager::matrix4d_to_string( const Matrix4d& M )
{
  Vector3d ea = M.topLeftCorner<3,3>().eulerAngles( 2,1,0 );
  Vector3d tx;
  tx << M(0,3), M(1,3), M(2,3);

  string to_return;

  char pr[200];
  sprintf( pr, "A: %4.2f, %4.2f, %4.2f;Tx: %4.2f, %4.2f, %4.2f", ea(0), ea(1), ea(2), tx(0), tx(1), tx(2) );
  to_return = string( pr );

  return to_return;
}
