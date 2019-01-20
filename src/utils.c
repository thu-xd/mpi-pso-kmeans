// Some utils functions defined here

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include "utils.h"


double euclidean_distance(double * p1,double * p2, int dim)
/*
    Return the euclidean distance between two points in a space whose dimension is dim

    Args:
        p1: The coordinate of the first point
        p2: The coordinate of the second point
        dim: The dimension of the points
    
    Return:
        The euclidean distance of two points
*/

{
    double res=0.0;
    for(int i=0;i<dim;i++)
        res+=pow(p1[i]-p2[i],2);
    return sqrt(res);
}

double mean_reconstruction_error(double **points,double *centroids,int points_num, 
    int feature_size, int cluster_num)
/*
    Calculate the reconstruction error when the cluster centroids are given by centroids.
    The defination of reconstruction error please referred to 
    [https://courses.cs.washington.edu/courses/cse546/16au/slides/lecture13-annotated.pdf]

    Args:
        points: The coordinate of the points
        centroids: The coordinate of the cluster centroids
        points_num: The number of points
        feature_size: The dimension of each point
        cluster_num: How many clusters these points belongs

    Retrun:
        The reconstruction error
*/
{
    double total_err=0.0;
    double min_err=0.0;
    double err;
    for(int i=0;i<points_num;i++)
    {
        min_err=DBL_MAX;
        for(int j=0;j<cluster_num;j++)
        {
            err=euclidean_distance(points[i],centroids+j*feature_size,feature_size);
            if(err<min_err)
                min_err=err;
        }
        total_err+=pow(min_err,2);
    }
    return total_err/(double)points_num;
}

void vector_copy(double * p1, double * p2, int dim)
/*
    Copy the elements of p1 to p2.

    Args:
        p1: Pointer to the source vector
        p2: Pointer to the target vector
        dim: Dimension of the vectors
    
    Retrun:
        void
*/
{
    for(int i=0;i<dim;i++)
        p2[i]=p1[i];
}

void print_line(int line, char ch)
{
    for(int i=0;i<line;i++)
    {
        for(int j=0;j<100;j++)
            printf("%c",ch);
        printf("\n");
    }
}

void print_time(time_t timer)
{
    char buffer[26];
    struct tm *tm_info;

    tm_info = localtime(&timer);

    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    puts(buffer);

    return;
}