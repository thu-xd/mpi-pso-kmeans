/*

This is a mpi implementation of the k-means algorithm.
The basic running procedure of this program is as follows, assuming that there are N processors
(0,1,2,...,N-1) and processor 0 as the master process:
1. The master processor sends clusters' centroid to all other processors
2. Each processor calculated the ownership of its data points, each data points is assigned to its 
nearest cluster centroids. 
3. Master processor collects the ownership of all data points and calculate the new clusters' 
centroid.
4. If not converged, continue from 1

Author: Dong xie
Date: 2018-12-28
E-mail:xied15@mails.tsinghua.edu.cn

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"

void mpi_kmeans_cluster(double **points,int *points_num,double *init_pos,double *results,
    struct kmeans_arguments args)
/*
    MPI implementation of kmeans cluster algorithm.

    Args:
        points: The data points which need to be clustered
        points_num: The number of data points in each process
        init_pos: The initialization clusters' centroid
        results: optimal clusters' centroid
        args: kmeans arguments, see the defination of struct kmeans_arguments for detail.
*/
{
    int my_id, procs_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &procs_num);

    int points_num_sum=0; // How many data points in all processes.
    for(int i=0;i<procs_num;i++)
        points_num_sum+=points_num[i];

    int variables_num=args.feature_size*args.cluster_num;
    double *clusters_centroid=(double*)malloc(variables_num*sizeof(double));
    for(int i=0;i<variables_num;i++)
        clusters_centroid[i]=init_pos[i];
    
    double *clusters_centroid_accumulate=(double*)malloc(variables_num*sizeof(double));
    int *clusters_ownership_accumulate=(int*)malloc(args.cluster_num*sizeof(int));
    double reconstruction_err_accumulate;

    double *clusters_centroid_accumulate_reduce=(double*)malloc(variables_num*sizeof(double));
    int *clusters_ownership_accumulate_reduce=(int*)malloc(variables_num*sizeof(double));
    double reconstruction_err_accumulate_reduce;

    int iter=0;
    int stop_flag=0; //This flag was set by process 0 to indicate whether we should stop the 
                     //iteration, 0 means no, 1 means yes
    double mean_reconstruction_err,mean_reconstruction_err_old;
    double relative_err_decrease;
    time_t current_time;
    char time_str[100];

    int owner; // which cluster each data points owned
    double minimal_distance,distance;

    if(my_id==0)
    {
        print_line(1,'-');
    }
    
    while(iter<args.max_iters && stop_flag==0)
    {
        for(int i=0;i<variables_num;i++)
            clusters_centroid_accumulate[i]=0.0;
        for(int i=0;i<args.cluster_num;i++)
            clusters_ownership_accumulate[i]=0;
        reconstruction_err_accumulate=0.0;
        
        for(int i=0;i<points_num[my_id];i++)
        {
            minimal_distance=DBL_MAX;
            for(int j=0;j<args.cluster_num;j++)
            {
                distance=euclidean_distance(points[i],clusters_centroid+
                    args.feature_size*j,args.feature_size);
                if(distance<minimal_distance)
                {
                    owner=j;
                    minimal_distance=distance;
                }
            }
            clusters_ownership_accumulate[owner]++;
            reconstruction_err_accumulate+=pow(minimal_distance,2);
            for(int k=0;k<args.feature_size;k++)
                clusters_centroid_accumulate[owner*args.feature_size+k]+=points[i][k];
        }

        MPI_Reduce(clusters_centroid_accumulate,clusters_centroid_accumulate_reduce,variables_num,
            MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(clusters_ownership_accumulate,clusters_ownership_accumulate_reduce,
            args.cluster_num,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&reconstruction_err_accumulate,&reconstruction_err_accumulate_reduce,1,
            MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        
        if(my_id==0)
        {
            for(int i=0;i<args.cluster_num;i++)
            {
                for(int j=0;j<args.feature_size;j++)
                    clusters_centroid[i*args.feature_size+j]=
                        clusters_centroid_accumulate_reduce[i*args.feature_size+j]
                        /clusters_ownership_accumulate_reduce[i];
            }

            mean_reconstruction_err=reconstruction_err_accumulate_reduce/points_num_sum;

            if(iter%args.early_stop_iters==0)
                mean_reconstruction_err_old=mean_reconstruction_err;
            
            // Early stop critical check
            if((iter+1)%args.early_stop_iters==0)
            {
                time(&current_time);
                print_time(current_time);
                relative_err_decrease=(mean_reconstruction_err_old-mean_reconstruction_err)
                    /(mean_reconstruction_err_old+1.0e-50);
                printf("iter:%d, mean_reconstruction_error:%.6f,decrease relative to %dth iter:"
                    "%.6f%%\n",iter,mean_reconstruction_err,iter-args.early_stop_iters+1,
                        relative_err_decrease*100.0);
                if(relative_err_decrease<args.critical)
                {
                    stop_flag=1;
                    print_line(1,'-');
                    printf("Having finished iteration, the minimal mean_reconstruction_error:%.6f\n"
                        ,mean_reconstruction_err);
                }
            }
        }
        MPI_Bcast(clusters_centroid,variables_num,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(&stop_flag,1,MPI_INT,0,MPI_COMM_WORLD);
        iter++;
    }

    vector_copy(clusters_centroid,results,variables_num);

    free(clusters_centroid);
    free(clusters_centroid_accumulate);
    free(clusters_ownership_accumulate);
    free(clusters_centroid_accumulate_reduce);
    free(clusters_ownership_accumulate_reduce);

    return;
}