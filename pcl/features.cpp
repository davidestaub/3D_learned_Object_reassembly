#include <iostream>
#include <array>
#include <pcl/io/pcd_io.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include "include/npy.hpp"
#include <fstream>
using namespace std;

int main(int argc, char **argv)
{
  // necessary classes
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_inverted(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);

  // process arguments
  if (argc < 4)
  {
    throw std::runtime_error("Required arguments: str<file_in> str<file_out> str<invert_normals>");
  }
  std::string file_in = argv[1];
  std::string file_out = argv[2];
  std::string invert_normals = argv[3];

  // read file
  if (pcl::io::loadPCDFile<pcl::PointNormal>(file_in, *cloud) == -1) // load the file
  {
    PCL_ERROR("Couldn't read file\n");
    return (-1);
  }

  // write xyz coordinates
  for (auto &point : *cloud)
  {
    cloud_xyz->points.emplace_back(pcl::PointXYZ(point.x, point.y, point.z));
  }

  // check if there are no normals
  bool invalid_normal = false;
  short inv_counter = 0;
  while (!invalid_normal)
  {
    for (auto &point : *cloud)
    {
      float nx = point.normal_x;
      float ny = point.normal_y;
      float nz = point.normal_z;
      if (nx == 0 && ny == 0 && nz == 0)
      {
        inv_counter++;
      }
      if (inv_counter > 5)
      {
        invalid_normal = true;
        cloud_normals->clear();
        break;
      }
      cloud_normals->points.emplace_back(pcl::Normal(nx, ny, nz));
    }
    break;
  }

  // calculate normals else
  if (invalid_normal)
  {
    cout << "No normals found: Calculating...." << endl;
    // Compute the normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud_xyz);
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setRadiusSearch(0.05);
    normal_estimation.compute(*cloud_normals);
  }

  // invert normals if set so
  if (invert_normals == "True")
  {
    cloud_normals_inverted->points.reserve(cloud_xyz->size());
    for (auto &normal : *cloud_normals)
    {
      float nx = -1 * normal.normal_x;
      float ny = -1 * normal.normal_y;
      float nz = -1 * normal.normal_z;
      cloud_normals_inverted->points.emplace_back(pcl::Normal(nx, ny, nz));
    }
    cloud_normals = cloud_normals_inverted;
  }

  // Setup the feature computation

  pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
  // Provide the original point cloud (without normals)
  fpfh_estimation.setInputCloud(cloud_xyz);
  // Provide the point cloud with normals
  fpfh_estimation.setInputNormals(cloud_normals);

  // Use the same KdTree from the normal estimation
  fpfh_estimation.setSearchMethod(tree);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr pfh_features(new pcl::PointCloud<pcl::FPFHSignature33>);

  fpfh_estimation.setRadiusSearch(0.1);

  // Actually compute the spin images
  fpfh_estimation.compute(*pfh_features);
  size_t height = pfh_features->size();
  size_t width = 33;
  // zero pad for attention heads
  size_t additional = 4 - width % 4;

  float hist[height][width + additional];

  for (int i = 0; i < height; i++)
  {
    pcl::FPFHSignature33 descriptor = (*pfh_features)[i];
    for (int j = 0; j < width; j++)
    {
      hist[i][j] = descriptor.histogram[j];
    }
    // add additional 0 to be % 4
    for (int j = width; j < width + additional; j++)
    {
      hist[i][j] = 0;
    }
  }

  // write output
  ofstream textfile;
  textfile.open(file_out, ios::out | ios::app);
  for (auto &x : hist)
  {
    for (auto &y : x)
    {
      textfile << y << " ";
    }
    textfile << endl;
  }
  textfile.close();
  return 0;
}
