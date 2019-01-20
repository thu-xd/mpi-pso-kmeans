#ifndef UTILS_HEADER
#define UTILS_HEADER

double euclidean_distance(double * p1,double * p2, int dim);
double mean_reconstruction_error(double **points,double *centroids,int points_num, int feature_size,
    int cluster_num);
void vector_copy(double * p1, double * p2, int dim);
void print_line(int line, char ch);
void print_time(time_t timer);

// Struct used to receive the command line arguments.
struct arguments
{
    char *data;
    int points_num, feature_size, cluster_num;
    char *mode;
    int kmeans_max_iters;
    double kmeans_critical;
    int kmeans_early_stop_iters;
    double pso_inertia_weight, pso_c1, pso_c2, pso_c3;
    int pso_steps_per_iter, pso_max_iters;
    double pso_critical;
    int pso_early_stop_iters,pso_swarm_size;
};

struct kmeans_arguments
/*
feature_size: The dimension of the data points.
cluster_num: How many cluster the data points belong
max_iters: Max iterations
early_stop_iters: Mean_reconstruction_error should decrease at least every early_stop_iters, if not,
    we think we have reach the minimal and should terminal the iteration
critical: Critical value of reletive decrease on mean_reconstruction_error between two iters, 
    below which we think there are no change on mean_reconstruction_error between two iters
*/
{
    int feature_size, cluster_num, max_iters, early_stop_iters;
    double critical;
};

struct pso_arguments
/*
    feature_size: The dimension of the data points.
    cluster_num: How many cluster the data points belong
    steps_per_iter: In each iters, swarm will forward multiple steps independent of
        other swarm in other processes.
    max_iters: Max iterations
    early_stop_iters: Global_min_err should decrease at least every early_stop_iters, if not ,
        we think we have reach the minimal and should terminal the iteration
    swarm_size: The number of particles in a swarm
    inertia_weight: Inertia weight
    c1: Acceleration parameters towards the particle best
    c2: Acceleration parameters towards the swarm best
    c3: Acceleration parameters towards the global best
    critical: Critical value of relative decrease on global_min_err between two iters, below 
        which we think there are no change on global_min_err between two iters
*/
{
    int feature_size, cluster_num, steps_per_iter, max_iters, 
        early_stop_iters,swarm_size;
    double inertia_weight, c1, c2, c3, critical;
};

#endif