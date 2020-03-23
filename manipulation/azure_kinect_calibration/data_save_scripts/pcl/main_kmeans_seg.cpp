#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/passthrough.h>
#include <iostream>
#include <string>
#include <pcl/filters/radius_outlier_removal.h>

// Only available in 1.8+
// #include <pcl/ml/kmeans.h>

#include <boost/program_options.hpp>
#include <boost/filesystem/operations.hpp>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>


namespace po = boost::program_options;

using DataFrame = std::vector<pcl::PointXYZ>;

double square(double value) {
    return value * value;
}

double squared_l2_distance(pcl::PointXYZ first, pcl::PointXYZ second) {
    return square(first.x - second.x) + square(first.y - second.y) + square (first.z - second.z);
}

DataFrame k_means(const DataFrame& data,
                  size_t k,
                  size_t number_of_iterations,
                  std::vector<size_t>& assignments,
                  std::vector<size_t>& counts) {

    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<size_t> indices(0, data.size() - 1);


    // Pick centroids as random points from the dataset.
    DataFrame means(k);
    for (auto& cluster : means) {
        cluster = data[indices(random_number_generator)];
    }

    assignments.resize(data.size());
    counts.resize(k);

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        // Find assignments.
        for (size_t point = 0; point < data.size(); ++point) {
            double best_distance = std::numeric_limits<double>::max();
            size_t best_cluster = 0;
            for (size_t cluster = 0; cluster < k; ++cluster) {
                const double distance =
                        squared_l2_distance(data[point], means[cluster]);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = cluster;
                }
            }
            assignments[point] = best_cluster;
        }

        // Sum up and count points for each cluster.
        DataFrame new_means(k);
        // Reset counts
        for (size_t cluster = 0; cluster < k; cluster++) {
            counts[cluster] = 0;
        }

        for (size_t point = 0; point < data.size(); ++point) {
            const auto cluster = assignments[point];
            new_means[cluster].x += data[point].x;
            new_means[cluster].y += data[point].y;
            new_means[cluster].z += data[point].z;
            counts[cluster] += 1;
        }

        // Divide sums by counts to get new centroids.
        for (size_t cluster = 0; cluster < k; ++cluster) {
            // Turn 0/0 into 0/1 to avoid zero division.
            const auto count = std::max<size_t>(1, counts[cluster]);
            means[cluster].x = new_means[cluster].x / count;
            means[cluster].y = new_means[cluster].y / count;
            means[cluster].z = new_means[cluster].z / count;
        }
    }

    int total_cluster_size = 0;
    for (size_t cluster = 0; cluster < k; cluster++) {
        total_cluster_size += counts[cluster];
    }
    assert(total_cluster_size == data.size());

    return means;
}

void CreateDataFrameForPCL( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, DataFrame& data) {
    data.resize(cloud->size());
    int data_idx = 0;
    for (auto it = cloud->begin(); it != cloud->end(); it++) {
        pcl::PointXYZRGB point = *it;
        pcl::PointXYZ new_point;
        new_point.x = point.x;
        new_point.y = point.y;
        new_point.z = point.z;

        data[data_idx++] = new_point;
    }
}

void CreatePointCloudForAssignments(const DataFrame& data_frame, std::vector<size_t> cluster_assignments, int cluster_size,
                                    int cluster_idx,  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    if (cluster_idx == 2) {
        std::cout << "Wtf";
    }
    for (size_t i = 0; i < cluster_assignments.size(); i++) {
        if (cluster_assignments[i] == cluster_idx) {
            cloud->push_back(data_frame[i]);
        }
    }
    assert(cloud->size() == cluster_size);
}


std::string GetDirFromPath(const std::string& str) {
    size_t found = str.find_last_of("/\\");
    return str.substr(0, found);
}

std::string GetFileFromPath(const std::string& str) {
    size_t found = str.find_last_of("/\\");
    return str.substr(found + 1);
}

void SaveCloud(pcl::PCDWriter& writer, const std::string &filename, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr output) {
    std::cout << "Will save point cloud at: " << filename << " size: " << output->size() << std::endl;
    std::stringstream ss;
    ss << filename;
    writer.write<pcl::PointXYZRGB> (ss.str (), *output, false);
}

void SaveXYZCloud(pcl::PCDWriter& writer, const std::string &filename, const pcl::PointCloud<pcl::PointXYZ>::Ptr output) {
    std::cout << "Will save point cloud at: " << filename << " size: " << output->size() << std::endl;
    std::stringstream ss;
    ss << filename;
    writer.write<pcl::PointXYZ> (ss.str (), *output, false);
}

pcl::ModelCoefficients::Ptr GetModelCoefficientsForPlane(float a, float b, float c, float d) {
    // Create a set of planar coefficients with ax + by + cz + d = 0
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    coefficients->values.resize(4);
    coefficients->values[0] = a;
    coefficients->values[1] = b;
    coefficients->values[2] = c;
    coefficients->values[3] = d;
    return coefficients;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CreatePointCloudWithPlaneProjection (
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<float> plane_coefficients_vector, bool return_total_cloud) {

    // Project points onto the table plane to complete the object.
    pcl::ModelCoefficients::Ptr coefficients = GetModelCoefficientsForPlane(
            plane_coefficients_vector[0],
            plane_coefficients_vector[1],
            plane_coefficients_vector[2],
            plane_coefficients_vector[3]);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ProjectInliers<pcl::PointXYZRGB> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setInputCloud (cloud);
    proj.setModelCoefficients (coefficients);
    proj.filter (*cloud_projected);

    if (!return_total_cloud) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr total_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        *total_cloud += (*cloud);
        *total_cloud += (*cloud_projected);
        return total_cloud;
    } else {
        return cloud_projected;
    }
}

/**
 * The virtual table plane represents the surface on which the objects rests.
 * Let's try to find this plane by finding offsets.
 */
std::vector<float> GetVirtualTablePlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    float start_z = 0.9;
    float step_z = 0.01;
    auto total_cloud_size = cloud->points.size();

    // If > threshold percent of points are below the plane then we should stop.
    float max_percent_points_below_plane = 0.2;
    float current_z = start_z;
    float end_z = 0.7;

    while (current_z >= end_z) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

        // Create the filtering object
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.0, current_z);
        pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_filtered);

        float percent_points_below_plane = double(cloud_filtered->points.size()) / total_cloud_size;
        std::cout << "Number of points below plane at z = " << current_z << ", " << percent_points_below_plane << std::endl;
        if (percent_points_below_plane >= max_percent_points_below_plane) {
            // Ok so the number of points we let of will be less than "max_percent"
            current_z += step_z;
            break;
        }

        current_z -= step_z;
    }
    std::cout << "Found plane at 0x + 0y + 1z + " << current_z << " = 0" << std::endl;
    std::vector<float> plane{0, 0, 1, current_z};

    return plane;
}

/**
 * The virtual top plane represents the top plane for all the objects. All of the objects lie between the top plane and
 * the virtual table plane.
 */
std::vector<float> GetVirtualTopPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float table_top_z,
                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cloud) {
    // Create the point cloud which will store all the points projected onto all the planes.
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::copyPointCloud(*cloud, *projected_cloud);

    // Filtered point cloud, at each iteration we use this to store the points which are still above the plane
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_above_plane(new pcl::PointCloud<pcl::PointXYZRGB>);

    auto total_cloud_size = cloud->points.size();

    // We keep on moving our plane up as long as a certain threshold of points are above the plane. At each iteration
    // we project those points onto the plane and add the projected points onto the `total_projected_cloud`.

    float max_percent_points_above_plane = 0.1;
    float percent_points_above_plane = 1.0;
    float current_z = table_top_z;
    float step_z = 0.005;

    while (percent_points_above_plane >= max_percent_points_above_plane) {

        cloud_above_plane->clear();

        // Create the filtering object
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.0, current_z);
        pass.setFilterLimitsNegative (false);
        pass.filter (*cloud_above_plane);

        std::vector<float> plane{0.0, 0.0, 1.0, -current_z};
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_projection = CreatePointCloudWithPlaneProjection(
                cloud_above_plane, plane, false);

        // Add points to the total projected cloud.
        *projected_cloud += *plane_projection;

        float percent_points_above_plane = double(cloud_above_plane->points.size()) / total_cloud_size;
        std::cout << "Number of points above plane at z = " << current_z << ", " << percent_points_above_plane << std::endl;
        if (percent_points_above_plane < max_percent_points_above_plane) {
            // Ok so the number of points we let of will be less than "max_percent"
            break;
        }

        current_z -= step_z;
    }
    std::cout << "Found top plane at 0x + 0y + 1z + " << current_z << " = 0" << std::endl;
    std::vector<float> plane{0, 0, 1, current_z};

    return plane;
}

int main(int argc, char* argv[]) {

//    std::string pcl_path = "/home/klz/datasets/data_in_line/true/objects_3"
//                           "/try_8_Nov_30_2019_12_01_AM/temp_2/final_segmented_pcl.pcd";
    std::string pcl_path;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("pcd", po::value<std::string>(&pcl_path)->default_value(""));

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << "Usage: options_description [options]\n";
        std::cout << desc << std::endl;
        return 0;
    }

    // Use it to set abs path for debugging.
    // pcl_path = "/home/klz/datasets/data_in_line/try_2/true/objects_3/try_5_Dec_01_2019_11_12_PM/extracted_pcd_data/final_segmented_pcl.pcd";

    if (!boost::filesystem::exists(pcl_path)) {
        std::cout << "ERROR pcl_path does not exist: " << pcl_path << std::endl;
        return 0;
    }

    std::cout << "Segment objects from point cloud: " << pcl_path <<  std::endl;

    std::string pcl_dir = GetDirFromPath(pcl_path);
    pcl::PCDReader reader;
    pcl::PCDWriter writer;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    reader.read (pcl_path, *cloud);
    std::cout << "Did load PointCloud with: " << cloud->points.size () << " data points." << std::endl;

    // Get the virtual plane on which the objects lie
    std::vector<float> table_plane_vector = GetVirtualTablePlane(cloud);

    // Filter away points that lie below the virtual table.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, table_plane_vector[3]);
    // pass.setFilterLimitsNegative (false);
    pass.filter (*cloud_filtered);

    // Now fill in the points between the surfaces by moving the table plane up and projecting points onto it.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<float> top_plane = GetVirtualTopPlane(cloud_filtered, table_plane_vector[3], projected_cloud);

    std::cout << "Projected PointCloud: " << projected_cloud->points.size () << " data points." << std::endl;
    SaveCloud(writer, pcl_dir + "/projected_cloud.pcd", projected_cloud);

    // Do radius based outlier removal for points between objects
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> radius_outlier;
    radius_outlier.setInputCloud(projected_cloud);
    radius_outlier.setRadiusSearch(0.01);
    radius_outlier.setMinNeighborsInRadius(30);
    radius_outlier.filter(*projected_cloud);

    std::cout << "Projected PointCloud after radius outlier: " << projected_cloud->points.size () << " data points." << std::endl;
    SaveCloud(writer, pcl_dir + "/projected_cloud_after_radius_outlier.pcd", projected_cloud);

    DataFrame data_frame;
    CreateDataFrameForPCL( projected_cloud, data_frame);
    std::vector<size_t> cluster_assignments;
    std::vector<size_t> cluster_counts;
    int num_objects = 5;
    k_means(data_frame, num_objects, 100, cluster_assignments, cluster_counts);

    std::cout << "Extracted clusters with sizes: ";
    for (int i = 0; i < num_objects; i++) {
        std::cout << cluster_counts[i] << ", ";
    }
    std::cout << std::endl;

    for (int i = 0; i < num_objects; i++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        CreatePointCloudForAssignments(data_frame, cluster_assignments, cluster_counts[i], i, cloud_cluster);
        SaveXYZCloud(writer, pcl_dir + "/cloud_cluster_" + std::to_string(i) + ".pcd", cloud_cluster);
    }

    return 0;
}
