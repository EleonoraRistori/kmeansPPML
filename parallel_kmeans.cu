#include <iostream>
#include <string>
#include <random>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <chrono>
#include <cooperative_groups.h>

__global__ void calculate_distance(float* data_points, float* centroids, float* distances, int num_data_points, int num_centroids, int data_point_dim) {
    int data_point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_point_id >= num_data_points) {
        return;
    }
    for (int centroid_id = 0; centroid_id < num_centroids; centroid_id++) {
        float distance = 0;
        for (int dim = 0; dim < data_point_dim; dim++) {
            float diff = data_points[data_point_id * data_point_dim + dim] - centroids[centroid_id * data_point_dim + dim];
            distance += diff * diff;
        }
        distances[data_point_id * num_centroids + centroid_id] = distance;
    }
}

__global__ void assign_cluster(int *cluster_assignment, const float* data_points, const float* centroids, int num_data_points,
                               int num_centroids, int data_point_dim, float* updated_centroids, int* assigned_points) {
    int data_point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_point_id >= num_data_points) {
        return;
    }
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
        atomicAdd(&updated_centroids[min_centroid_id*data_point_dim+dim], data_points[data_point_id*data_point_dim+dim]);
    atomicAdd(&assigned_points[min_centroid_id], 1);

}

__global__ void calculate_centroid(float* updated_centroids, int* assigned_points, float* centroids, float* old_centroids,
                                   int num_centroids, int data_point_dim) {
    int centroid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (centroid_id >= num_centroids) {
        return;
    }

    for (int dim = 0; dim < data_point_dim; dim++) {
        if (assigned_points[centroid_id] > 0) {
            old_centroids[centroid_id * data_point_dim + dim] = centroids[centroid_id * data_point_dim + dim];
            centroids[centroid_id * data_point_dim + dim] = updated_centroids[centroid_id*data_point_dim+dim] / assigned_points[centroid_id];
        }
    }
}

__global__ void centroid_distance(float* o_centroids, float* n_centroids, int k, int data_point_dim, float* distance){
    int centroid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (centroid_id >= k) {
        return;
    }
    distance[centroid_id] = 0;
    float diff;
    for(int dim=0; dim < data_point_dim; dim++){
        diff = o_centroids[centroid_id*data_point_dim+dim] - n_centroids[centroid_id*data_point_dim+dim];
        distance[centroid_id] += diff * diff;
    }

}


void initialize_centroids(int data_point_dim, float* centroids, const float *data_points, int k, float threshold){
    for(int dim=0; dim<data_point_dim; dim++){
        centroids[dim] = data_points[dim];
    }
    int num_assigned_centroids = 1;
    int point_id = 1;
    while(num_assigned_centroids != k){

        for(int dim=0; dim<data_point_dim; dim++){
            centroids[num_assigned_centroids*data_point_dim+dim] =  data_points[point_id*data_point_dim+dim];
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
        }while(centroid_id < num_assigned_centroids && sum_distances > threshold);
        if(centroid_id == num_assigned_centroids){
            num_assigned_centroids++;
            point_id++;
        }
        else {
            point_id++;
        }
    }
}

int kmeans(float* data_points, float* centroids, int* cluster_assignment, int num_data_points, int data_point_dim, int num_centroids, int max_iterations, float tolerance, float sigma) {
    //float* distances = new float[num_data_points * num_centroids];

    // Initialize centroids randomly
    initialize_centroids(data_point_dim, centroids, data_points, num_centroids, sigma*sigma*sigma*sigma*3);

    // Copy data points and centroids to the GPU
    float* d_data_points, *d_centroids, *old_centroids, *d_updated_centroid;
    int* d_cluster_assignment, *d_assigned_points;
    cudaError_t err0 = cudaMalloc(&d_data_points, num_data_points * data_point_dim * sizeof(float));
    cudaError_t err1 = cudaMalloc(&d_centroids, num_centroids * data_point_dim * sizeof(float));
    cudaError_t err2 = cudaMalloc(&old_centroids, num_centroids * data_point_dim * sizeof(float));
    cudaError_t err3 = cudaMalloc(&d_updated_centroid, num_centroids*data_point_dim*sizeof(float));
    cudaError_t err4 = cudaMalloc(&d_assigned_points, num_centroids* sizeof(int));
    cudaError_t err5 = cudaMalloc(&d_cluster_assignment, num_data_points * sizeof(int));
    cudaMemcpy(d_data_points, data_points, num_data_points * data_point_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, num_centroids * data_point_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(old_centroids, centroids, num_centroids * data_point_dim * sizeof(float), cudaMemcpyHostToDevice);
    // Main loop
    int iteration = 0;
    int num_threads_per_block = 256;
    int num_blocks = (num_centroids + num_threads_per_block - 1) / num_threads_per_block;
    float *d_centr_distance;
    auto *centr_distance = new float[num_centroids];
    float tot_distance;
    cudaError_t err6 = cudaMallocHost(&d_centr_distance, num_centroids * sizeof(float));
    if(err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess){
        std::cerr << "Cuda error";
        cudaFree(d_data_points);
        cudaFree(d_centroids);
        cudaFree(old_centroids);
        cudaFree(d_updated_centroid);
        cudaFree(d_cluster_assignment);
        cudaFree(d_assigned_points);
        cudaFree(d_centr_distance);
        return -1;
    }

    do {
        // Calculate centroid_distance between each data point and centroid
        cudaMemset(&d_updated_centroid, 0, num_centroids*data_point_dim*sizeof(float));
        cudaMemset(&d_assigned_points, 0, num_centroids* sizeof(int));
        num_blocks = (num_data_points + num_threads_per_block - 1) / num_threads_per_block;

        // Assign each data point to the nearest centroid
        assign_cluster<<<num_blocks, num_threads_per_block>>>(d_cluster_assignment, d_data_points, d_centroids, num_data_points,
                                                              num_centroids, data_point_dim, d_updated_centroid, d_assigned_points);

        // Calculate new centroids
        num_blocks = (num_centroids + num_threads_per_block - 1) / num_threads_per_block;

        calculate_centroid<<<num_blocks, num_threads_per_block>>>(d_updated_centroid, d_assigned_points, d_centroids, old_centroids,
                                                                  num_centroids, data_point_dim);

        // Check if centroids have moved more than tolerance
        tot_distance = 0;
        centroid_distance<<<num_blocks, num_threads_per_block>>>(old_centroids, d_centroids, num_centroids, data_point_dim, d_centr_distance);

        cudaMemcpy(centr_distance, d_centr_distance, num_centroids*sizeof(float), cudaMemcpyDeviceToHost);

        for(int i=0; i<num_centroids; i++){
            tot_distance += centr_distance[i];
        }
        iteration++;
    }while (iteration < max_iterations && tot_distance > tolerance);

    // Copy final centroids and cluster assignment back to CPU
    cudaMemcpy(centroids, d_centroids, num_centroids * data_point_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cluster_assignment, d_cluster_assignment, num_data_points * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(d_data_points);
    cudaFree(d_centroids);
    cudaFree(old_centroids);
    cudaFree(d_updated_centroid);
    cudaFree(d_cluster_assignment);
    cudaFree(d_assigned_points);
    cudaFree(d_centr_distance);

    //delete[] distances;
    delete[] centr_distance;
    return 0;
}

