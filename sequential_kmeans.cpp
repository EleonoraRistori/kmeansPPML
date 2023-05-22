#include <iostream>
#include <string>
#include <random>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <chrono>
#include <cooperative_groups.h>

void assign_cluster_s(int *cluster_assignment, const float* data_points, const float* centroids, int num_data_points,
                      int num_centroids, int data_point_dim, float* updated_centroids, int* assigned_points) {
    for(int data_point_id=0; data_point_id<num_data_points; data_point_id++){
        float min_dist = INFINITY;
        int min_centroid_id = -1;
        for (int centroid_id = 0; centroid_id < num_centroids; centroid_id++) {
            float dist = 0;
            for (int dim = 0; dim < data_point_dim; dim++) {
                float diff = data_points[data_point_id * data_point_dim + dim] - centroids[centroid_id * data_point_dim + dim];
                dist += diff * diff;
            }
            if(dist < min_dist){
                min_dist = dist;
                min_centroid_id = centroid_id;
            }
        }
        cluster_assignment[data_point_id] = min_centroid_id;
        for (int dim = 0; dim < data_point_dim; dim++)
            updated_centroids[min_centroid_id*data_point_dim+dim] += data_points[data_point_id*data_point_dim+dim];
        assigned_points[min_centroid_id]+= 1;
    }
}

void calculate_centroid_s(float* updated_centroids, int* assigned_points, float* centroids, float* old_centroids,
                          int num_centroids, int data_point_dim) {
    for(int centroid_id=0; centroid_id<num_centroids; centroid_id++){
        for (int dim = 0; dim < data_point_dim; dim++) {
            if (assigned_points[centroid_id] > 0) {
                old_centroids[centroid_id * data_point_dim + dim] = centroids[centroid_id * data_point_dim + dim];
                centroids[centroid_id * data_point_dim + dim] = updated_centroids[centroid_id*data_point_dim+dim] / assigned_points[centroid_id];
            }
        }
    }
}


float centroid_distance_s(float* o_centroids, float* n_centroids, int k, int data_point_dim){
    float distance = 0;
    float diff;
    for(int id = 0 ; id < k; id++){
        for(int dim=0; dim < data_point_dim; dim++){
            diff = o_centroids[id*data_point_dim+dim] - n_centroids[id*data_point_dim+dim];
            distance += diff * diff;
        }
    }
    return distance;
}

void kmeans_s(float* data_points, float* centroids, int* cluster_assignment, int num_data_points, int data_point_dim,
              int num_centroids, int max_iterations, float tolerance, float sigma) {

    float* old_centroids = new float[num_centroids * data_point_dim];
    float* updated_centroids = new float[num_centroids * data_point_dim];
    int* assigned_points = new int[num_centroids];
    // Initialize centroids randomly
    initialize_centroids(data_point_dim, centroids, data_points, num_centroids, sigma*sigma*100);


    // Main loop
    int iteration = 0;
    do {

        // Assign each data point to the nearest centroid
        assign_cluster_s(cluster_assignment, data_points, centroids, num_data_points, num_centroids, data_point_dim, updated_centroids, assigned_points);


        calculate_centroid_s(updated_centroids, assigned_points, centroids, old_centroids, num_centroids, data_point_dim);


        iteration++;

    }while (iteration < max_iterations /*&& centroid_distance_s(old_centroids, centroids, num_centroids, data_point_dim) > tolerance*/);

    delete[] old_centroids;
}

