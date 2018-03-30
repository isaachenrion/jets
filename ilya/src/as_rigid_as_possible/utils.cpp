#include <igl/arap.h>
#include <igl/biharmonic_coordinates.h>
#include <igl/cat.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/matrix_to_list.h>
#include <igl/parula.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/readDMAT.h>
#include <igl/readMESH.h>
#include <igl/remove_unreferenced.h>
#include <igl/slice.h>
#include <igl/writeDMAT.h>
#include <igl/viewer/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>
#include <queue>
#include <sstream>
#include <random>
#include <cstdlib>

#include "tutorial_shared_path.h"

struct Mesh
{
  Eigen::MatrixXd V,U;
  Eigen::MatrixXi T,F;
} low;

Eigen::MatrixXd W;

igl::ARAPData arap_data;

int main(int argc, char * argv[])
{
  using namespace Eigen;
  using namespace std;
  using namespace igl;


    cout << argc;
    string fileid(argv[1]);
    cout << fileid;

    if(!readOBJ("/Users/kostrikov/Blender/meshes/" + fileid + ".obj",low.V,low.F))
  {
    cout<<"failed to load mesh"<<endl;
  }

  // Resize low res (high res will also be resized by affine precision of W)
  low.V.rowwise() -= low.V.colwise().mean();
  low.V /= (low.V.maxCoeff()-low.V.minCoeff());

    std::random_device rd;

    std::mt19937 gen(rd());
    double pi = atan(1)*4;
    std::uniform_real_distribution<> dist(0, 2 * pi);

    double a1 = dist(gen);
    double a2 = dist(gen);
    double a3 = dist(gen);

    Eigen::Matrix3d Ax, Ay, Az;
    Ax << 1, 0, 0,
          0, cos(a1), -sin(a1),
          0, sin(a1), cos(a1);


    Ay << cos(a2), 0, sin(a2),
          0, 1, 0,
          -sin(a2), 0, cos(a2);

    Az << cos(a3), -sin(a3), 0,
          sin(a3),  cos(a3), 0,
          0, 0, 1;


    Eigen::Matrix3d R = Ax * Ay * Az;

    low.V = low.V * R.transpose();

    low.V.rowwise() += RowVector3d(0,-low.V.colwise().minCoeff()[1] + 0.001,0);
    low.U = low.V;

    arap_data.with_dynamics = true;
  arap_data.max_iter = 10;
  arap_data.energy = ARAP_ENERGY_TYPE_DEFAULT;
  arap_data.h = 0.01;
  arap_data.ym = 0.001;
  if(!arap_precomputation(low.V,low.F,3,VectorXi(),arap_data))
  {
    cerr<<"arap_precomputation failed."<<endl;
    return EXIT_FAILURE;
  }
  // Constant gravitational force
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(low.V,low.F,igl::MASSMATRIX_TYPE_DEFAULT,M);
  const size_t n = low.V.rows();
  arap_data.f_ext =  M * RowVector3d(0,-9.8,0).replicate(n,1);
  // Random initial velocities to wiggle things
  arap_data.vel = MatrixXd::Random(n,3);

  igl::viewer::Viewer viewer;
  viewer.data.set_mesh(low.U,low.F);

    MatrixXd C(low.F.rows(),3);
  C<<
    RowVector3d(0.8,0.5,0.2).replicate(low.F.rows(),1);
  viewer.data.set_colors(C);

  viewer.callback_key_pressed =
    [&](igl::viewer::Viewer & viewer,unsigned int key,int mods)->bool
  {
    switch(key)
    {
      default:
        return false;
      case ' ':
        viewer.core.is_animating = !viewer.core.is_animating;
        return true;
      case 'r':
        low.U = low.V;
        return true;
    }
  };

    for (int time_step = 0; time_step < 100; ++time_step) {
        arap_solve(MatrixXd(0,3),arap_data,low.U);
        for(int v = 0;v<low.U.rows();v++)
        {
            // collide with y=0 plane
            const int y = 1;
            if(low.U(v,y) < 0)
            {
                low.U(v,y) = -low.U(v,y);
                // ~ coefficient of restitution
                const double cr = 1.1;
                arap_data.vel(v,y) = - arap_data.vel(v,y) / cr;
            }
        }
        stringstream ss;
        ss << "/Users/kostrikov/arap/";
        ss << fileid << "/";
        string foldername;
        ss >> foldername;

        system(("mkdir -p " + foldername).c_str());

        ss.clear();
        ss << foldername;
        ss << std::setfill('0') << std::setw(2);
        ss << time_step;
        ss <<".obj";


        string filename;
        ss >> filename;
        writeOBJ(filename, low.U, low.F);
    }


    /*
  viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)->bool
  {
    glEnable(GL_CULL_FACE);
    if(viewer.core.is_animating)
    {
      arap_solve(MatrixXd(0,3),arap_data,low.U);
      for(int v = 0;v<low.U.rows();v++)
      {
        // collide with y=0 plane
        const int y = 1;
        if(low.U(v,y) < 0)
        {
          low.U(v,y) = -low.U(v,y);
          // ~ coefficient of restitution
          const double cr = 1.1;
          arap_data.vel(v,y) = - arap_data.vel(v,y) / cr;
        }
      }
        stringstream ss;
        ss << "/Users/kostrikov/arap/tmp";
        ss << time_step;
        ss <<".obj";
        string file_name;
        ss >> file_name;
        time_step += 1;
    writeOBJ(file_name, low.U, low.F);
      viewer.data.set_vertices(low.U);
      viewer.data.compute_normals();
    }
    return false;
  };
  viewer.core.show_lines = false;
  viewer.core.is_animating = true;
  viewer.core.animation_max_fps = 30.;
  viewer.data.set_face_based(true);
  cout<<R"(
[space] to toggle animation
'r'     to reset positions
      )";
  viewer.core.rotation_type =
    igl::viewer::ViewerCore::ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP;
  viewer.launch();
     */
}
