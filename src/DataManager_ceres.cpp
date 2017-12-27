#include "DataManager.h"


// FUnctions for pose graph optimization with ceres
#define print_ceres( msg ) cout << "[CERES]"; msg ;

void DataManager::ceres_main()
{
  ros::Rate loop_rate(1);
  while( ros::ok() )
  {

    print_ceres( cout << "CERES LOOP\n" );


    lock_enable_ceres.lock();
    if( enable_ceres )
    {
      doOptimization();
      enable_ceres = false;
    }
    lock_enable_ceres.unlock();


    loop_rate.sleep();
  }

  print_ceres( cout << "Terminating Ceres Thread\n" );
}



void DataManager::doOptimization()
{
  print_ceres( cout << "START OPTIMIZATION")

  //
  // Build the problem
  //
  ceres::Problem problem;
  ceres::LocalParameterization *quaternion_parameterization = new ceres::EigenQuaternionParameterization;
  // ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;

  // A - Odometry Constaints
  for( int i=0 ; i<odometryEdges.size() ; i++ )
  {
    Edge * ed = odometryEdges[i];
    Matrix4d ed___b_T_a;
    ed->getEdgeRelPose( ed___b_T_a );
    ceres::CostFunction * cost_function = Residue4DOF::Create( ed___b_T_a );

    problem.AddResidualBlock( cost_function, NULL,
      ed->a->e_q.coeffs().data(), ed->a->e_p.data(),
      ed->b->e_q.coeffs().data(), ed->b->e_p.data() );


    problem.SetParameterization( ed->a->e_q.coeffs().data(), quaternion_parameterization );
    problem.SetParameterization( ed->b->e_q.coeffs().data(), quaternion_parameterization );
  }


  // B - Loop Constaints
  for( int i=0 ; i<loopClosureEdges.size() ; i++ )
  {
    Edge * ed = loopClosureEdges[i];
    Matrix4d ed___b_T_a;
    ed->getEdgeRelPose( ed___b_T_a );
    ceres::CostFunction * cost_function = Residue4DOF::Create( ed___b_T_a );

    problem.AddResidualBlock( cost_function, NULL,
      ed->a->e_q.coeffs().data(), ed->a->e_p.data(),
      ed->b->e_q.coeffs().data(), ed->b->e_p.data() );

    problem.SetParameterization( ed->a->e_q.coeffs().data(), quaternion_parameterization );
    problem.SetParameterization( ed->b->e_q.coeffs().data(), quaternion_parameterization );
  }

  // First node as I|0
  problem.SetParameterBlockConstant( nNodes[0]->e_p.data() );
  problem.SetParameterBlockConstant( nNodes[0]->e_q.coeffs().data() );

  // Eigen Quaternion Parameterization - Optimization of Lie Manifold


  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.dogleg_type = ceres::SUBSPACE_DOGLEG;
  options.max_num_iterations = 5;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary );
  cout << summary.FullReport() << endl;



  print_ceres( cout << "END_OPT")

}
