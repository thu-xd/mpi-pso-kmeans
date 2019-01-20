#ifndef CLUSTER_H
#define CLUSTER_H

void mpi_pso_cluster(double **points,int points_num,double **init_pos,double **init_vel,
    double *results,struct pso_arguments args);

void mpi_kmeans_cluster(double **points,int *points_num,double *init_pos,double *results,
    struct kmeans_arguments args);
    
#endif