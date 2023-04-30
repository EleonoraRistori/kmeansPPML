#include <iostream>
#include <queue>
#include <chrono>
#include "parallel_kmeans.cu"
#include "sequential_kmeans.cpp"
#include "utils.cpp"


int main() {
    int num_points=50000000;
    const int data_point_dim = 2;
    int k = 7;
    const float sigma = 10;
    float *points;
    int *cluster_assignment;
    float *centroids;
    std::vector<std::chrono::milliseconds> sequentialDuration;
    std::vector<std::chrono::milliseconds> parallelDuration;
    points = new float[num_points*data_point_dim];
    cluster_assignment = new int[num_points];
    centroids = new float[k*data_point_dim];
    generateCluster(num_points, data_point_dim, k, sigma, points, centroids);
    kmeans(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, sigma, sigma);
    int num_it = 0;
    //Test #points
//    centroids = new float[k*data_point_dim];
//
//
//    for(int i = 125; i <=140  ; i=i+5){
//        num_it++;
//        num_points = i*1000000;
//        points = new float[num_points*data_point_dim];
//        cluster_assignment = new int[num_points];
//        std::cout << i << "\n";
//        generateCluster(num_points, data_point_dim, k, sigma, points, centroids);
//        auto begin = std::chrono::high_resolution_clock::now();
//        //kmeans_s(points, centroids, cluster_assignment, num_points, data_point_dim, k, 3, sigma, sigma);
//        auto end = std::chrono::high_resolution_clock::now();
//        sequentialDuration.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin));
//        begin = std::chrono::high_resolution_clock::now();
//        if(kmeans(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, sigma, sigma) == -1)
//            return -1;
//        end = std::chrono::high_resolution_clock::now();
//        parallelDuration.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin));
//        saveArray(num_points, data_point_dim, points, "points.csv");
//        saveArray(num_points, 1, cluster_assignment, "cluster_assignment.csv");
//        saveArray(k, data_point_dim, centroids, "centroids.csv");
//        std::cout << sequentialDuration[num_it-1].count() << "\n";
//        std::cout << parallelDuration[num_it-1].count() << "\n";
//        delete[] points;
//        delete[] cluster_assignment;
//
//    }
//    delete[] centroids;
//    //saveTime(num_it, sequentialDuration, "time_n_points.csv");
//    saveTime(num_it, parallelDuration, "p_time_n_points.csv");
//
//    sequentialDuration.clear();
//    parallelDuration.clear();



    //Test #cluster
    for(k = 10; k <= 20010; k=k+2000){
        num_it++;
        points = new float[num_points*data_point_dim];
        cluster_assignment = new int[num_points];
        centroids = new float[k*data_point_dim];
        std::cout << k << "\n";
        generateCluster(num_points, data_point_dim, k, sigma, points, centroids);
        auto begin = std::chrono::high_resolution_clock::now();
        kmeans_s(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, sigma, sigma);
        auto end = std::chrono::high_resolution_clock::now();
        sequentialDuration.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin));
        begin = std::chrono::high_resolution_clock::now();
        kmeans(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, sigma, sigma);
        end = std::chrono::high_resolution_clock::now();
        parallelDuration.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin));
        std::cout << sequentialDuration[num_it-1].count() << "\n";
        std::cout << parallelDuration[num_it-1].count() << "\n";
        delete[] points;
        delete[] cluster_assignment;
        delete[] centroids;

    }
    //saveTime(19, sequentialDuration, "time_n_centroids.csv");
    //saveTime(19, parallelDuration, "p_time_n_centroids.csv");
    return 0;
}
