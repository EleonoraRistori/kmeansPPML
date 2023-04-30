#include <iostream>
#include <string>
#include <random>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <chrono>

void generatePoints(int num_points,int data_point_dim, float*data_points){
    //Generate random points
    std::default_random_engine generator;
    for(int point_id=0; point_id<num_points; point_id++){
        for(int dim=0; dim < data_point_dim; dim++){
            std::uniform_real_distribution<float> distribution(0,100);
            data_points[point_id*data_point_dim + dim] = distribution(generator);
        }
    }
}


void generateCluster(int num_points,int data_point_dim, int k, float sigma, float*data_points, float*centroids){
    //Assigning centroids
    for(int dim=0; dim<data_point_dim; dim++){
        centroids[dim] = float(std::rand()) / RAND_MAX *sigma*sigma*k;
    }
    int num_assigned_centroids = 1;

    while(num_assigned_centroids != k){

        for(int dim=0; dim<data_point_dim; dim++){
            centroids[num_assigned_centroids*data_point_dim+dim] =  float(std::rand()) / RAND_MAX *sigma*sigma*k;
        }
        int centroid_id = 0;
        float sum_distances;
        do {
            sum_distances = 0;
            for(int dim=0; dim<data_point_dim; dim++) {
                float diff = centroids[centroid_id*data_point_dim+dim] - centroids[num_assigned_centroids*data_point_dim+dim];
                sum_distances += diff*diff;
            }
            centroid_id++;
        }while(centroid_id < num_assigned_centroids && sum_distances > sigma*sigma);
        if(centroid_id == num_assigned_centroids){
            num_assigned_centroids++;
        }
    }
    //Generate random points
    std::default_random_engine generator;
    for(int point_id=0; point_id<num_points; point_id++){
        int cluster = int(std::rand()) % k;

        for(int dim=0; dim < data_point_dim; dim++){
            std::normal_distribution<double> distribution(centroids[cluster*data_point_dim+dim],sigma);
            data_points[point_id*data_point_dim + dim] = float(distribution(generator));
        }
    }
}


template <typename T> void saveArray (int num_points, int data_point_dim, T* array, std::string filename){
    std::ofstream pointsFile;
    pointsFile.open (filename);
    for(int point_id=0; point_id<num_points; point_id++){
        for(int dim=0; dim<data_point_dim; dim++){
            if(dim < data_point_dim-1)
                pointsFile << array[point_id*data_point_dim+dim] << ",";
            else
                pointsFile << array[point_id*data_point_dim+dim];
        }
        pointsFile << "\n";
    }
}

void saveTime (int num_points, std::vector<std::chrono::milliseconds> array, std::string filename){
    std::ofstream performance;
    performance.open (filename);
    for(int point_id=0; point_id<num_points; point_id++){
        performance << array[point_id].count() ;
        performance << "\n";
    }
}
