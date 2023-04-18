#include <iostream>
#include <string>
#include <random>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <chrono>


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

__global__ void assign_cluster(int *cluster_assignment, const float* data_points, const float* centroids, int num_data_points, int num_centroids, int data_point_dim) {
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
}

__global__ void calculate_centroid(float* data_points, int* cluster_assignment, float* centroids, float* old_centroids, int num_data_points, int num_centroids, int data_point_dim) {
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
            old_centroids[centroid_id * data_point_dim + dim] = centroids[centroid_id * data_point_dim + dim];
            centroids[centroid_id * data_point_dim + dim] = sum / num_points_assigned;
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
    //float* distances = new float[num_data_points * num_centroids];

    // Initialize centroids randomly
    initialize_centroids(data_point_dim, centroids, data_points, num_centroids, 100);

    // Copy data points and centroids to the GPU
    float* d_data_points, *d_centroids, *old_centroids;
    int* d_cluster_assignment;
    cudaMalloc(&d_data_points, num_data_points * data_point_dim * sizeof(float));
    cudaMalloc(&d_centroids, num_centroids * data_point_dim * sizeof(float));
    cudaMalloc(&old_centroids, num_centroids * data_point_dim * sizeof(float));
    //cudaMalloc(&d_distances, num_data_points * num_centroids * sizeof(float));
    cudaMalloc(&d_cluster_assignment, num_data_points * sizeof(int));
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
    cudaMallocHost(&d_centr_distance, num_centroids * sizeof(float));

    do {
        // Calculate centroid_distance between each data point and centroid

        num_blocks = (num_data_points + num_threads_per_block - 1) / num_threads_per_block;
        //calculate_distance<<<num_blocks, num_threads_per_block>>>(d_data_points, d_centroids, d_distances, num_data_points, num_centroids, data_point_dim);

        // Assign each data point to the nearest centroid
        assign_cluster<<<num_blocks, num_threads_per_block>>>(d_cluster_assignment, d_data_points, d_centroids, num_data_points, num_centroids, data_point_dim);

        // Calculate new centroids
        num_blocks = (num_centroids + num_threads_per_block - 1) / num_threads_per_block;

        calculate_centroid<<<num_blocks, num_threads_per_block>>>(d_data_points, d_cluster_assignment, d_centroids, old_centroids, num_data_points, num_centroids, data_point_dim);

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
    //cudaFree(d_distances);
    cudaFree(d_cluster_assignment);
    cudaFree(d_centr_distance);

    //delete[] distances;
    delete[] centr_distance;

}



void assign_cluster_s(int* cluster_assignment, float* data_points, float* centroids, int num_data_points, int num_centroids, int data_point_dim) {
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
    }
}

void calculate_centroid_s(float* data_points, int* cluster_assignment, float* centroids, float* old_centroids, int num_data_points, int num_centroids, int data_point_dim) {
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
                old_centroids[centroid_id * data_point_dim + dim] = centroids[centroid_id * data_point_dim + dim];
                centroids[centroid_id * data_point_dim + dim] = sum / num_points_assigned;
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

void kmeans_s(float* data_points, float* centroids, int* cluster_assignment, int num_data_points, int data_point_dim, int num_centroids, int max_iterations, float tolerance) {

    //float* distances = new float[num_data_points * num_centroids];
    float* old_centroids = new float[num_centroids * data_point_dim];
    // Initialize centroids randomly
    initialize_centroids(data_point_dim, centroids, data_points, num_centroids, 100);


    // Main loop
    int iteration = 0;
     do {

        //calculate_distance_s(data_points, centroids, distances, num_data_points, num_centroids, data_point_dim);

        // Assign each data point to the nearest centroid
        assign_cluster_s(cluster_assignment, data_points, centroids, num_data_points, num_centroids, data_point_dim);


        calculate_centroid_s(data_points, cluster_assignment, centroids, old_centroids, num_data_points, num_centroids, data_point_dim);


        iteration++;

    }while (iteration < max_iterations && centroid_distance_s(old_centroids, centroids, num_centroids, data_point_dim) > tolerance);

    //delete[] distances;
    delete[] old_centroids;
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

void saveTime (int num_points, std::vector<std::chrono::milliseconds> array, std::string filename){
    std::ofstream performance;
    performance.open (filename);
    for(int point_id=0; point_id<num_points; point_id++){
        performance << array[point_id].count() ;
        performance << "\n";
    }
}

int main() {
    int num_points=500000000;
    const int data_point_dim = 2;
    int k = 7;
    const float sigma = 5;
    float *points;
    int *cluster_assignment;
    float *centroids;
    std::vector<std::chrono::milliseconds> sequentialDuration;
    std::vector<std::chrono::milliseconds> parallelDuration;


    //Test #points
    centroids = new float[k*data_point_dim];

    int num_it = 0;
    for(int i = 1; i <= 10; i++){
        num_it++;
        num_points = i*10000000;
        points = new float[num_points*data_point_dim];
        cluster_assignment = new int[num_points];
        std::cout << i << "\n";
        generateCluster(num_points, data_point_dim, k, sigma, points, centroids);
        auto begin = std::chrono::high_resolution_clock::now();
        kmeans_s(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, sigma);
        auto end = std::chrono::high_resolution_clock::now();
        sequentialDuration.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin));
        begin = std::chrono::high_resolution_clock::now();
        kmeans(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, sigma);
        end = std::chrono::high_resolution_clock::now();
        parallelDuration.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin));
        delete[] points;
        delete[] cluster_assignment;

    }
    delete[] centroids;
    saveTime(num_it, sequentialDuration, "time_n_points.csv");
    saveTime(num_it, parallelDuration, "p_time_n_points.csv");

    sequentialDuration.clear();
    parallelDuration.clear();



//    //Test #cluster
//    for(k = 2; k <= 20; k++){
//        num_it++;
//        points = new float[num_points*data_point_dim];
//        cluster_assignment = new int[num_points];
//        centroids = new float[k*data_point_dim];
//        std::cout << k;
//        generateCluster(num_points, data_point_dim, k, sigma, points, centroids);
//        auto begin = std::chrono::high_resolution_clock::now();
//        kmeans_s(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, sigma);
//        auto end = std::chrono::high_resolution_clock::now();
//        sequentialDuration.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin));
//        begin = std::chrono::high_resolution_clock::now();
//        kmeans(points, centroids, cluster_assignment, num_points, data_point_dim, k, 20, sigma);
//        end = std::chrono::high_resolution_clock::now();
//        parallelDuration.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin));
//        std::cout << k;
//        delete[] points;
//        delete[] cluster_assignment;
//        delete[] centroids;
//
//    }
//    saveTime(19, sequentialDuration, "time_n_centroids.csv");
//    saveTime(19, parallelDuration, "p_time_n_centroids.csv");
    return 0;
}
