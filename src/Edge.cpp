#include "Edge.h"

Edge::Edge( const Node *a, int a_id, const Node * b, int b_id, int type )
{
  this->a = a;
  this->b = b;
  this->type = type;
  this->sub_type = -1;

  // edge_rel_position = new double[3];
  // edge_rel_quat = new double[3];
  this->a_id = a_id;
  this->b_id = b_id;


}

void Edge::setEdgeTimeStamps( ros::Time time_a, ros::Time time_b )
{
  this->a_timestamp = time_a;
  this->b_timestamp = time_b;
}

void Edge::setLoopEdgeSubtype( int sub_type )
{
  if( this->type == EDGE_TYPE_LOOP_CLOSURE )
  {
    this->sub_type = sub_type;
  }
  else
  {
    ROS_WARN( "Setting subtype for an edge which is not a loop-closure edge in function Edge::setLoopEdgeSubtype()");
  }
}

int Edge::getEdgeType() { return this->type ;}
int Edge::getEdgeSubType() { return this->sub_type; }



void Edge::setEdgeRelPose( const Matrix4d& b_T_a )
{
  e_p << b_T_a(0,3), b_T_a(1,3), b_T_a(2,3);
  e_q = Quaterniond( b_T_a.topLeftCorner<3,3>() );
}

void Edge::getEdgeRelPose( Matrix4d& M )
{
  M = Matrix4d::Zero();
  M.col(3) << e_p, 1.0;
  Matrix3d R = e_q.toRotationMatrix();
  M.topLeftCorner<3,3>() = e_q.toRotationMatrix();
}
