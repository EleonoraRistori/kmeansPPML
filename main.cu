#include <iostream>
#include <string>
#include <random>
#include <cstdlib>
#include <fstream>


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

__global__ void assign_cluster(int *cluster_assignment, float *distances, int num_data_points, int num_centroids) {
    int data_point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_point_id >= num_data_points) {
        return;
    }
    int min_distance_centroid_id = 0;
    float min_distance = distances[data_point_id * num_centroids];
    for (int centroid_id = 1; centroid_id < num_centroids; centroid_id++) {
        float distance = distances[data_point_id * num_centroids + centroid_id];
        if (distance < min_distance) {
            min_distance = distance;
            min_distance_centroid_id = centroid_id;
        }
    }
    cluster_assignment[data_point_id] = min_distance_centroid_id;
}

__global__ void calculate_centroid(float* data_points, int* cluster_assignment, float* centroids, int num_data_points, int num_centroids, int data_point_dim) {
    int centroid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (centroid_id >= num_centroids) {
        return;
    }

    for (int dim = 0; dim < data_point_dim; dim++) {
        int num_points_assigned = 0;
        float sum = 0;
        for (int data_point_id = 0; data_point_id < num_data_points; data_point_id++) {
            if (cluster_assignment[data_point_id] == centroid_id) {
                sum += data_points[data_point_id * data_point_dim + dim];
                num_points_assigned++;
            }
        }
        if (num_points_assigned > 0) {
            centroids[centroid_id * data_point_dim + dim] = sum / num_points_assigned;
        }
    }
}

void centroid_distance(float* o_centroids, float* n_centroids){
    
}


void initialize_centroids(int data_point_dim, float* centroids, float *data_points, int k, float threshold){
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

void kmeans(float* data_points, float* centroids, int* cluster_assignment, int num_data_points, int data_point_dim, int num_centroids, int max_iterations, float tolerance) {
    float* distances = new float[num_data_points * num_centroids];

    // Initialize centroids randomly
    initialize_centroids(data_point_dim, centroids, data_points, num_centroids, 100);

    // Copy data points and centroids to the GPU
    float* d_data_points, *d_centroids, *d_distances;
    int* d_cluster_assignment;
    cudaMalloc(&d_data_points, num_data_points * data_point_dim * sizeof(float));
    cudaMalloc(&d_centroids, num_centroids * data_point_dim * sizeof(float));
    cudaMalloc(&d_distances, num_data_points * num_centroids * sizeof(float));
    cudaMalloc(&d_cluster_assignment, num_data_points * sizeof(int));
    cudaMemcpy(d_data_points, data_points, num_data_points * data_point_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, num_centroids * data_point_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Main loop
    int iteration = 0;
    while (iteration < max_iterations) {
    // Calculate distance between each data point and centroid
        int num_threads_per_block = 128;
        int num_blocks = (num_data_points + num_threads_per_block - 1) / num_threads_per_block;
        calculate_distance<<<num_blocks, num_threads_per_block>>>(d_data_points, d_centroids, d_distances, num_data_points, num_centroids, data_point_dim);

        // Assign each data point to the nearest centroid
        assign_cluster<<<num_blocks, num_threads_per_block>>>(d_cluster_assignment, d_distances, num_data_points, num_centroids);

        // Calculate new centroids
        num_blocks = (num_centroids + num_threads_per_block - 1) / num_threads_per_block;
        calculate_centroid<<<num_blocks, num_threads_per_block>>>(d_data_points, d_cluster_assignment, d_centroids, num_data_points, num_centroids, data_point_dim);

        // Check if centroids have moved more than tolerance
        // ...

        iteration++;
    }

    // Copy final centroids back to CPU
    cudaMemcpy(centroids, d_centroids, num_centroids * data_point_dim * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i<4; i++){
        std::cout << centroids[i] << "\n";
    }
    std::cout << "\n";
    // Free memory on the GPU
    cudaFree(d_data_points);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_cluster_assignment);

}


int calculate_distance_s(float* data_points, float* centroids, float* distances, int num_data_points, int num_centroids, int data_point_dim) {
    for(int data_point_id=0; data_point_id<num_data_points; data_point_id++){
        for (int centroid_id = 0; centroid_id < num_centroids; centroid_id++) {
            float distance = 0;
            for (int dim = 0; dim < data_point_dim; dim++) {
                float diff = data_points[data_point_id * data_point_dim + dim] - centroids[centroid_id * data_point_dim + dim];
                distance += diff * diff;
            }
            distances[data_point_id * num_centroids + centroid_id] = distance;
        }
    }
    return 1;
}

int assign_cluster_s(int *cluster_assignment, float *distances, int num_data_points, int num_centroids) {
    for(int data_point_id=0; data_point_id<num_data_points; data_point_id++){
        int min_distance_centroid_id = 0;
        float min_distance = distances[data_point_id * num_centroids];
        for (int centroid_id = 1; centroid_id < num_centroids; centroid_id++) {
            float distance = distances[data_point_id * num_centroids + centroid_id];
            if (distance < min_distance) {
                min_distance = distance;
                min_distance_centroid_id = centroid_id;
            }
        }
        cluster_assignment[data_point_id] = min_distance_centroid_id;
    }
    return 2;
}

int calculate_centroid_s(float* data_points, int* cluster_assignment, float* centroids, int num_data_points, int num_centroids, int data_point_dim) {
    for(int centroid_id=0; centroid_id<num_centroids; centroid_id++){
        for (int dim = 0; dim < data_point_dim; dim++) {
            int num_points_assigned = 0;
            float sum = 0;
            for (int data_point_id = 0; data_point_id < num_data_points; data_point_id++) {
                if (cluster_assignment[data_point_id] == centroid_id) {
                    sum += data_points[data_point_id * data_point_dim + dim];
                    num_points_assigned++;
                }
            }
            if (num_points_assigned > 0) {
                centroids[centroid_id * data_point_dim + dim] = sum / num_points_assigned;
            }
        }
    }
    return 3;
}




void kmeans_s(float* data_points, float* centroids, int* cluster_assignment, int num_data_points, int data_point_dim, int num_centroids, int max_iterations, float tolerance) {

    float* distances = new float[num_data_points * num_centroids];

    // Initialize centroids randomly
    initialize_centroids(data_point_dim, centroids, data_points, num_centroids, 100);


    // Main loop
    int iteration = 0;
    while (iteration < max_iterations) {

        std::cout << calculate_distance_s(data_points, centroids, distances, num_data_points, num_centroids, data_point_dim);

        // Assign each data point to the nearest centroid
        std::cout << assign_cluster_s(cluster_assignment, distances, num_data_points, num_centroids);


        std::cout << calculate_centroid_s(data_points, cluster_assignment, centroids, num_data_points, num_centroids, data_point_dim);

        // Check if centroids have moved more than tolerance
        // ...
        iteration++;

    }

    delete[] distances;
}

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
        centroids[dim] = float(std::rand()) / RAND_MAX *100*k;
    }
    int num_assigned_centroids = 1;

    while(num_assigned_centroids != k){

        for(int dim=0; dim<data_point_dim; dim++){
            centroids[num_assigned_centroids*data_point_dim+dim] =  float(std::rand()) / RAND_MAX *100*k;
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
        }while(centroid_id < num_assigned_centroids && sum_distances > sigma*sigma*100);
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


int main() {
//float data_points[] = {1.2,1.3,  5.3,5,5.2,5.1, 1.1,1};

    const int num_points=100;  // number of experiments
    const int data_point_dim = 2;
    const int k = 3;
    const float sigma = 5;

    float *points = new float[num_points*data_point_dim];
    float *centroids = new float[k*data_point_dim];
    int* cluster_assignment = new int[num_points];

    generateCluster(num_points, data_point_dim, k, sigma, points, centroids);
    //generatePoints(num_points, data_point_dim, points);
//    for(int i=0; i<num_points; i++){
//        for(int dim = 0; dim < data_point_dim; dim++){
//            std::cout << points[i*data_point_dim+dim] << "   :   ";
//        }
//        std::cout << "\n";
//    }

    saveArray(num_points, data_point_dim, points, "points.csv");
    kmeans_s(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, 2);
    kmeans(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, 2);
    saveArray(k, data_point_dim, centroids, "centroids.csv");
    saveArray(num_points, 1, cluster_assignment, "cluster_assignment.csv");

    delete[] centroids;
    delete[] points;
    delete[] cluster_assignment;
    return 0;
}
