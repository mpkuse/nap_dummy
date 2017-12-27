#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>

#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
using namespace std;


#include <ceres/ceres.h>
#include <Eigen/Dense>


#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>

#define ODOMETRY_EDGE 0
#define CLOSURE_EDGE 1
#define BOGUS_EDGE 2

/**
  Implementation of
  Pratik Agarwal, Gian Diego Tipaldi, Luciano Spinello, Cyrill Stachniss and Wolfram Burgard
  Robust Map Optimization using Dynamic Covariance Scaling Proceedings of the International
  Conference on Robots and Automation (ICRA), Karlsruhe, Germany, 2013
  */


// Class to represent Nodes
class Node
{
    public:
    /*Node()
    {

    }*/

    Node(int index, double x, double y, double theta)
    {
        this->index = index;
        p = new double[3];
        p[0] = x;
        p[1] = y;
        p[2] = theta;
    }

    int index;
    double *p;
};

// Class to represent Edges
class Edge
{
public:
    // Type:
    // 0 : Odometry edge
    // 1 : Loop CLosure Edge
    // 2 : Bogus Edge
    Edge(const Node* a, const Node* b, int edge_type )
    {
        this->a = a;
        this->b = b;
        this->edge_type = edge_type;
    }

    void setEdgePose( double x, double y, double theta )
    {
        this->x = x;
        this->y = y;
        this->theta = theta;
    }

    void setInformationMatrix( double I11, double  I12, double  I13, double I22, double I23, double I33 )
    {
        this->I11 = I11;
        this->I12 = I12;
        this->I13 = I13;
        this->I22 = I22;
        this->I23 = I23;
        this->I33 = I33;
    }

    const Node *a, *b;
    double x, y, theta;
    double I11, I12, I13, I22, I23, I33;
    int edge_type;
};

class ReadG2O
{
public:
    ReadG2O(const string& fName)
    {
      // Read the file in g2o format
        fstream fp;
        fp.open(fName.c_str(), ios::in);


        string line;
        int v = 0;
        int e = 0;
        while( std::getline(fp, line) )
        {
            vector<string> words;
            boost::split(words, line, boost::is_any_of(" "), boost::token_compress_on);
            if( words[0].compare( "VERTEX_SE2") == 0 )
            {
                v++;
                int node_index = boost::lexical_cast<int>( words[1] );
                double x = boost::lexical_cast<double>( words[2] );
                double y = boost::lexical_cast<double>( words[3] );
                double theta = boost::lexical_cast<double>( words[4] );

                Node * node = new Node(node_index, x, y, theta);
                nNodes.push_back( node );
            }


            if( words[0].compare( "EDGE_SE2") == 0 )
            {
              // cout << e << words[0] << endl;
                int a_indx = boost::lexical_cast<int>( words[1] );
                int b_indx = boost::lexical_cast<int>( words[2] );

                double dx = boost::lexical_cast<double>( words[3] );
                double dy = boost::lexical_cast<double>( words[4] );
                double dtheta = boost::lexical_cast<double>( words[5] );

                double I11, I12, I13, I22, I23, I33;
                I11 = boost::lexical_cast<double>( words[6] );
                I12 = boost::lexical_cast<double>( words[7] );
                I13 = boost::lexical_cast<double>( words[8] );
                I22 = boost::lexical_cast<double>( words[9] );
                I23 = boost::lexical_cast<double>( words[10] );
                I33 = boost::lexical_cast<double>( words[11] );


                if( abs(a_indx - b_indx) < 5 )
                {
                  Edge * edge = new Edge( nNodes[a_indx], nNodes[b_indx], ODOMETRY_EDGE );
                  edge->setEdgePose(dx, dy, dtheta);
                  edge->setInformationMatrix(I11, I12, I13, I22, I23, I33);
                  nEdgesOdometry.push_back(edge);
                }
                else
                {
                  Edge * edge = new Edge( nNodes[a_indx], nNodes[b_indx], CLOSURE_EDGE );
                  edge->setEdgePose(dx, dy, dtheta);
                  edge->setInformationMatrix(I11, I12, I13, I22, I23, I33);
                  nEdgesClosure.push_back(edge);
                }


                e++;
            }

        }

    }


    // write nodes to file to be visualized with python script
    void writePoseGraph_nodes( const string& fname )
    {
      cout << "writePoseGraph nodes: " << fname << endl;
      fstream fp;
      fp.open( fname.c_str(), ios::out );
      for( int i=0 ; i<this->nNodes.size() ; i++ )
      {
        fp << nNodes[i]->index << " " << nNodes[i]->p[0] << " " << nNodes[i]->p[1] << " " << nNodes[i]->p[2]  << endl;
      }
    }

    void writePoseGraph_edges( const string& fname )
    {
      cout << "writePoseGraph Edges : "<< fname << endl;
      fstream fp;
      fp.open( fname.c_str(), ios::out );
      write_edges( fp, this->nEdgesOdometry );
      write_edges( fp, this->nEdgesClosure );
      write_edges( fp, this->nEdgesBogus );
    }

    void writePoseGraph_switches( const string& fname, vector<double>& priors, vector<double*>& optimized )
    {
        cout << "#Closure Edges : "<< nEdgesClosure.size() << endl;
        cout << "#Bogus Edges : "<< nEdgesBogus.size()<< endl;
        cout << "#priors : "<< priors.size()<< endl;
        cout << "#optimized " << optimized.size()<< endl;
        fstream fp;
        fp.open( fname.c_str(), ios::out );
        for( int i=0 ; i<nEdgesOdometry.size() ; i++ )
        {
            Edge * ed = nEdgesOdometry[i];
            fp << ed->a->index << " " << ed->b->index << " " << ed->edge_type <<
                    " " << 1.0 << " " << 1.0 << endl;
        }



        for( int i=0 ; i<nEdgesClosure.size() ; i++ )
        {
            Edge * ed = nEdgesClosure[i];
            fp << ed->a->index << " " << ed->b->index << " " << ed->edge_type <<
                    " " << priors[i] << " " << *(optimized[i]) << endl;
        }

        // fp << "BOGUS EDGES AHEAD\n";
        int ofset = nEdgesClosure.size();
        for( int i=0 ; i<nEdgesBogus.size() ; i++ )
        {
            Edge * ed = nEdgesBogus[i];
            fp << ed->a->index << " " << ed->b->index << " " << ed->edge_type <<
                    " " << priors[ofset+i] << " " << *(optimized[ofset+i]) << endl;
        }

    }

    // Adding Bogus edges as described in Vertigo paper
    void add_random_C(int count )
    {
        int MIN = 0 ;
        int MAX = nNodes.size();

        for( int i = 0 ; i<count ; i++ )
        {

            int a = rand() % MAX;
            int b = rand() % MAX;
            cout << a << "<--->" << b << endl;
            Edge * edge = new Edge( nNodes[a], nNodes[b], BOGUS_EDGE );
            edge->setEdgePose( rand()/RAND_MAX, rand()/RAND_MAX, rand()/RAND_MAX );
            nEdgesBogus.push_back( edge );
        }
    }

//private:
    vector<Node*> nNodes; //storage for node
    vector<Edge*> nEdgesOdometry; //storage for edges - odometry
    vector<Edge*> nEdgesClosure; //storage for edges - odometry
    vector<Edge*> nEdgesBogus; //storage for edges - odometry

    void write_edges( fstream& fp, vector<Edge*>& vec )
    {
      for( int i=0 ; i<vec.size() ; i++ )
      {
        // fp << nEdges[i]->a->index << " " << nEdges[i]->b->index << " " << (nEdges[i]->bogus_edge?1:0)  << " " << nEdges[i]->switch_var[0]<< endl;
        fp << vec[i]->a->index << " " << vec[i]->b->index << " " << vec[i]->edge_type << endl;
      }
    }

};


// Odometry Residue
struct OdometryResidue
{
    // Observation for the edge
    OdometryResidue(double dx, double dy, double dtheta)
    {

        this->dx = dx;
        this->dy = dy;
        this->dtheta = dtheta;

        // make a_Tcap_b
        {
          double cos_t = cos( this->dtheta );
          double sin_t = sin( this->dtheta );
          a_Tcap_b(0,0) = cos_t;
          a_Tcap_b(0,1) = -sin_t;
          a_Tcap_b(1,0) = sin_t;
          a_Tcap_b(1,1) = cos_t;
          a_Tcap_b(0,2) = this->dx;
          a_Tcap_b(1,2) = this->dy;

          a_Tcap_b(2,0) = 0.0;
          a_Tcap_b(2,1) = 0.0;
          a_Tcap_b(2,2) = 1.0;
      }

    }

    // Define the residue for each edge. P1 and P2 are 3-vectors representing state of the node ie. x,y,theta
    template <typename T>
    bool operator()(const T* const P1, const T* const P2, T* e) const
    {

        // Convert P1 to T1 ^w_T_a
        Eigen::Matrix<T,3,3> w_T_a;
        {
          T cos_t = T(cos( P1[2] ));
          T sin_t = T(sin( P1[2] ));
          w_T_a(0,0) = cos_t;
          w_T_a(0,1) = -sin_t;
          w_T_a(1,0) = sin_t;
          w_T_a(1,1) = cos_t;
          w_T_a(0,2) = P1[0];
          w_T_a(1,2) = P1[1];

          w_T_a(2,0) = T(0.0);
          w_T_a(2,1) = T(0.0);
          w_T_a(2,2) = T(1.0);
      }


        // Convert P2 to T2 ^w_T_a
        Eigen::Matrix<T,3,3> w_T_b;
        {
          T cos_t = cos( P2[2] );
          T sin_t = sin( P2[2] );
          w_T_b(0,0) = cos_t;
          w_T_b(0,1) = -sin_t;
          w_T_b(1,0) = sin_t;
          w_T_b(1,1) = cos_t;
          w_T_b(0,2) = P2[0];
          w_T_b(1,2) = P2[1];

          w_T_b(2,0) = T(0.0);
          w_T_b(2,1) = T(0.0);
          w_T_b(2,2) = T(1.0);
      }

      // cast from double to T
        Eigen::Matrix<T, 3, 3> T_a_Tcap_b;
        T_a_Tcap_b <<   T(a_Tcap_b(0,0)), T(a_Tcap_b(0,1)),T(a_Tcap_b(0,2)),
                        T(a_Tcap_b(1,0)), T(a_Tcap_b(1,1)),T(a_Tcap_b(1,2)),
                        T(a_Tcap_b(2,0)), T(a_Tcap_b(2,1)),T(a_Tcap_b(2,2));

        // now we have :: w_T_a, w_T_b and a_Tcap_b
        // compute pose difference
        Eigen::Matrix<T,3,3> diff = T_a_Tcap_b.inverse() * (w_T_a.inverse() * w_T_b);

        e[0] = diff(0,2);
        e[1] = diff(1,2);
        e[2] = asin( diff(1,0) );

        return true;
    }

    double dx;
    double dy;
    double dtheta;
    Eigen::Matrix<double,3,3> a_Tcap_b;

    static ceres::CostFunction* Create(const double dx, const double dy, const double dtheta){
        return (new ceres::AutoDiffCostFunction<OdometryResidue, 3, 3, 3>(
            new OdometryResidue(dx, dy, dtheta)));
    };

};


// Switchable Loop Closure Residue
struct SwitchableClosureResidue
{
    // Observation for the edge
    SwitchableClosureResidue(double dx, double dy, double dtheta )
    {

        this->dx = dx;
        this->dy = dy;
        this->dtheta = dtheta;


        // make a_Tcap_b
        {
          double cos_t = cos( this->dtheta );
          double sin_t = sin( this->dtheta );
          a_Tcap_b(0,0) = cos_t;
          a_Tcap_b(0,1) = -sin_t;
          a_Tcap_b(1,0) = sin_t;
          a_Tcap_b(1,1) = cos_t;
          a_Tcap_b(0,2) = this->dx;
          a_Tcap_b(1,2) = this->dy;

          a_Tcap_b(2,0) = 0.0;
          a_Tcap_b(2,1) = 0.0;
          a_Tcap_b(2,2) = 1.0;
      }

    }

    // Define the residue for each edge. P1 and P2 are 3-vectors representing state of the node ie. x,y,theta
    template <typename T>
    bool operator()(const T* const P1, const T* const P2, T* e) const
    {

        // Convert P1 to T1 ^w_T_a
        Eigen::Matrix<T,3,3> w_T_a;
        {
          T cos_t = T(cos( P1[2] ));
          T sin_t = T(sin( P1[2] ));
          w_T_a(0,0) = cos_t;
          w_T_a(0,1) = -sin_t;
          w_T_a(1,0) = sin_t;
          w_T_a(1,1) = cos_t;
          w_T_a(0,2) = P1[0];
          w_T_a(1,2) = P1[1];

          w_T_a(2,0) = T(0.0);
          w_T_a(2,1) = T(0.0);
          w_T_a(2,2) = T(1.0);
      }


        // Convert P2 to T2 ^w_T_a
        Eigen::Matrix<T,3,3> w_T_b;
        {
          T cos_t = cos( P2[2] );
          T sin_t = sin( P2[2] );
          w_T_b(0,0) = cos_t;
          w_T_b(0,1) = -sin_t;
          w_T_b(1,0) = sin_t;
          w_T_b(1,1) = cos_t;
          w_T_b(0,2) = P2[0];
          w_T_b(1,2) = P2[1];

          w_T_b(2,0) = T(0.0);
          w_T_b(2,1) = T(0.0);
          w_T_b(2,2) = T(1.0);
      }

      // cast from double to T
        Eigen::Matrix<T, 3, 3> T_a_Tcap_b;
        T_a_Tcap_b <<   T(a_Tcap_b(0,0)), T(a_Tcap_b(0,1)),T(a_Tcap_b(0,2)),
                        T(a_Tcap_b(1,0)), T(a_Tcap_b(1,1)),T(a_Tcap_b(1,2)),
                        T(a_Tcap_b(2,0)), T(a_Tcap_b(2,1)),T(a_Tcap_b(2,2));

        // now we have :: w_T_a, w_T_b and a_Tcap_b
        // compute pose difference
        Eigen::Matrix<T,3,3> diff = T_a_Tcap_b.inverse() * (w_T_a.inverse() * w_T_b);

        // psi - scalar
        // T psi = T(1.0) / (T(1.0) + exp( T(-2.0)*s[0] ));
        T phi = T(.5);
        T Xl = T(10.0) * ( diff(0,2)*diff(0,2) + diff(1,2)*diff(1,2) + asin( diff(1,0) )*asin( diff(1,0) ) );
        T s = min( T(1.0),  T(2.0)*phi / (phi + Xl)   );


        e[0] = s*diff(0,2);
        e[1] = s*diff(1,2);
        e[2] = s*asin( diff(1,0) );
        e[3] = phi * (1.0 - s);

        return true;
    }

    double dx;
    double dy;
    double dtheta;
    Eigen::Matrix<double,3,3> a_Tcap_b;

    static ceres::CostFunction* Create(const double dx, const double dy, const double dtheta){
        return (new ceres::AutoDiffCostFunction<SwitchableClosureResidue, 4, 3, 3>(
            new SwitchableClosureResidue(dx, dy, dtheta)));
    };

};


class Program ;
class DummyCallback : public ceres::IterationCallback
{
public:
    explicit DummyCallback( ReadG2O * g )
    {
      this->g = g;
      pub_string = n.advertise<std_msgs::String>( "/string", 1000 );
      pub_nodes = n.advertise<visualization_msgs::Marker>( "/nodes", 1000 );
      pub_edges = n.advertise<visualization_msgs::Marker>( "/edges", 1000 );
    }

    virtual ceres::CallbackReturnType operator()( const ceres::IterationSummary& summary )
    {
        // cout << "Call back exec" << endl;
        cout <<  g->nNodes[1]->p[0] << " "<< g->nNodes[1]->p[1] << " " << g->nNodes[1]->p[2] << endl;

        // ros publish - String
        std_msgs::String msg;
        msg.data = string("Helo Nodes");
        pub_string.publish( msg );

        publish_all_nodes();

        publish_edges( g->nEdgesOdometry, ODOMETRY_EDGE );
        publish_edges( g->nEdgesClosure, CLOSURE_EDGE );
        publish_edges( g->nEdgesBogus, BOGUS_EDGE );


        ros::spinOnce();
        return ceres::SOLVER_CONTINUE;
    }

private:
  Program * program_;
  ReadG2O * g;
  ros::NodeHandle n;
  ros::Publisher pub_string;
  ros::Publisher pub_nodes;
  ros::Publisher pub_edges;
  bool getValueBetweenTwoFixedColors(double value, double &red, double &green, double &blue)
  {
    double aR = 0.0;   double aG = 0.0; double aB=1.0;  // RGB for our 1st color (blue in this case).
    double bR = 1.0;   double bG = 0.0; double bB=0.0;    // RGB for our 2nd color (red in this case).

    red   = (double)(bR - aR) * value + aR;      // Evaluated as -255*value + 255.
    green = (double)(bG - aG) * value + aG;      // Evaluates as 0.
    blue  = (double)(bB - aB) * value + aB;      // Evaluates as 255*value + 0.
  }

  void publish_edges( vector<Edge*>& edgeVec, int edge_type )
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time();
    marker.ns = "my_namespace";
    marker.id = edge_type;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0; // Don't forget to set the alpha!


    for( int i=0 ; i<edgeVec.size() ; i++ )
    {
      geometry_msgs::Point pa;
      pa.x = edgeVec[i]->a->p[0];
      pa.y = edgeVec[i]->a->p[1];
      pa.z = 0.0;
      marker.points.push_back( pa );

      geometry_msgs::Point pb;
      pb.x = edgeVec[i]->b->p[0];
      pb.y = edgeVec[i]->b->p[1];
      pb.z = 0.0;
      marker.points.push_back( pb );


      {
        std_msgs::ColorRGBA color;
        // double psi = 1.0 / ( 1.0 + exp( -2.0*edgeVec[i]->switch_opt_var ) );
        double r,g,b;
        // getValueBetweenTwoFixedColors( 2.0*max(0.0,min(0.5,psi)), r, g, b);
        if( edgeVec[i]->edge_type == ODOMETRY_EDGE )
        {
          r = 1.0; g=0.0; b=0.0;
        }
        if( edgeVec[i]->edge_type == CLOSURE_EDGE )
        {
          r = 0.0; g=1.0; b=0.0;
        }
        if( edgeVec[i]->edge_type == BOGUS_EDGE )
        {
          r = 0.0; g=0.0; b=1.0;
        }

        color.r = r;
        color.g = g;
        color.b = b;
        color.a = 1.0;
        marker.colors.push_back( color );
        marker.colors.push_back( color );
      }

    }
    pub_edges.publish( marker );
  }

  void publish_all_nodes()
  {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = ros::Time();
      marker.ns = "my_namespace";
      marker.id = 0;
      marker.type = visualization_msgs::Marker::POINTS;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = 0;
      marker.pose.position.y = 0;
      marker.pose.position.z = 0;
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.2;
      marker.scale.y = 0.2;
      marker.scale.z = 0.1;
      marker.color.a = 1.0; // Don't forget to set the alpha!
      marker.color.r = 1.0;
      marker.color.g = 1.0;
      marker.color.b = 1.0;

      for( int i=0 ; i<g->nNodes.size() ; i++ )
      {
        geometry_msgs::Point p;
        p.x = g->nNodes[i]->p[0];
        p.y = g->nNodes[i]->p[1];
        p.z = 0.0;
        marker.points.push_back( p );
      }

      pub_nodes.publish( marker );


  }

};


int main(int argc, char ** argv)
{
    // ROS INit
    ros::init( argc, argv, "ceres_vertigo_DCS" );

    /////////////////////////////////////////////
    // // // // //  Read g2o file // // // // //
    /////////////////////////////////////////////
    string fname = "/home/mpkuse/catkin_ws/src/nap/slam_data/input_M3500_g2o.g2o";
    cout << "Start Reading PoseGraph\n";
    ReadG2O g( fname );
    g.add_random_C(25);


    g.writePoseGraph_nodes("/home/mpkuse/catkin_ws/src/nap/slam_data/init_nodes.txt");
    g.writePoseGraph_edges("/home/mpkuse/catkin_ws/src/nap/slam_data/init_edges.txt");
    cout << "total nodes : "<< g.nNodes.size() << endl;
    cout << "total nEdgesOdometry : "<< g.nEdgesOdometry.size() << endl;
    cout << "total nEdgesClosure : "<< g.nEdgesClosure.size() << endl;
    cout << "total nEdgesBogus : "<< g.nEdgesBogus.size() << endl;


    ////////////////////////////////////////////////////
    // // // // // Make the cost function // // // // //
    ////////////////////////////////////////////////////
    ceres::Problem problem;

    // A - Odometry Constraints
    for( int i=0 ; i<g.nEdgesOdometry.size() ; i++ )
    {
        Edge* ed = g.nEdgesOdometry[i];
        ceres::CostFunction * cost_function = OdometryResidue::Create( ed->x, ed->y, ed->theta );

        problem.AddResidualBlock( cost_function, /*new ceres::HuberLoss(0.01)*/NULL, ed->a->p, ed->b->p );
        // cout << ed->a->index << "---> " << ed->b->index << endl;
    }


    // B - Loop Closure Constaints (switchable)
    for( int i=0 ; i<g.nEdgesClosure.size() ; i++ )
    {
        Edge* ed = g.nEdgesClosure[i];
        ceres::CostFunction * cost_function = SwitchableClosureResidue::Create( ed->x, ed->y, ed->theta );

        problem.AddResidualBlock( cost_function, /*new ceres::HuberLoss(0.01)*/NULL, ed->a->p, ed->b->p );
        // cout << ed->a->index << "---> " << ed->b->index << endl;

        //dry eval
        double * params[2];
        params[0] = ed->a->p;
        params[1] = ed->b->p;
        double res[4];
        cost_function->Evaluate( params, res, NULL );
        cout << res[0] << " " << res[1] << " " << res[2] << endl;
    }

    for( int i=0 ; i<g.nEdgesBogus.size() ; i++ )
    {
        Edge* ed = g.nEdgesBogus[i];
        ceres::CostFunction * cost_function = SwitchableClosureResidue::Create( ed->x, ed->y, ed->theta );

        problem.AddResidualBlock( cost_function, /*new ceres::HuberLoss(0.01)*/NULL, ed->a->p, ed->b->p );
        // cout << ed->a->index << "---> " << ed->b->index << endl;
    }



    ///////////////////////////////////////////////
    // // // // // Iteratively Solve // // // // //
    //////////////////////////////////////////////

    problem.SetParameterBlockConstant(g.nNodes[0]->p); //1st pose be origin


    ceres::Solver::Options options;

    //callback
    DummyCallback callback(&g);
    options.callbacks.push_back( &callback );
    options.update_state_every_iteration = true;

    options.linear_solver_type = ceres::ITERATIVE_SCHUR; //ceres::SPARSE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.minimizer_progress_to_stdout = true;
    // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.dogleg_type = ceres::SUBSPACE_DOGLEG;
    // options.use_inner_iterations = true;
    options.use_nonmonotonic_steps = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;

    // Write Pose Graph after Optimization
    g.writePoseGraph_nodes("/home/mpkuse/catkin_ws/src/nap/slam_data/after_opt_nodes.txt");
    g.writePoseGraph_edges("/home/mpkuse/catkin_ws/src/nap/slam_data/after_opt_edges.txt");
}
