/*

This is a MPI implementation of Particle Swarm Optimization (pso) for cluster purpose. The basic pso 
algorithm description can be referred to wiki 
[https://en.wikipedia.org/wiki/Particle_swarm_optimization].

For cluster purpose, each particle in the swarm represents a potential solution to the 
reconstruction error minimization problem, which is a vector concatenating all cluster centroid 
coordinate and with dimension FEATURE_SIZE*CLUSTER_NUM. For details about reconstruction error 
minimization problem, please referred to 
[https://courses.cs.washington.edu/courses/cse546/16au/slides/lecture13-annotated.pdf], or you can 
see mean_reconstruction_error function in utils.c file.

The parallel implementation of the pso is referred to this paper:
Gonsalves T, Egashira A. Parallel swarms oriented particle swarm optimization[J]. 
Applied Computational Intelligence and Soft Computing, 
2013, 2013: 14.


Author: Dong xie
Date: 2019-1-13
E-mail:xied15@mails.tsinghua.edu.c

*/
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include "utils.h"

void mpi_pso_cluster(double **points,int points_num,double **init_pos,double **init_vel,
    double *results,struct pso_arguments args)
/*
    MPI implementation of pso for cluster purpose

    Args:
        points: The data points which need to be clustered
        points_num: The number of data points
        init_pos: The initialization position of the particles in swarm
        init_vel: The initialization velocity of the particles in swarm
        results: optimal results
        args: Pso arguments, see the defination of struct pso_arguments for detail.
*/
{
    int my_id, procs_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &procs_num);

    int particle_dim=args.feature_size*args.cluster_num; // The dimension of the particles

    if(my_id==0)
    {
        print_line(1,'-');
        printf("Initialize the particles' positions and particles' velocity\n");
    }
    // particle position
    double **p_pos=(double **)malloc(args.swarm_size*sizeof(double *));
    for(int i=0;i<args.swarm_size;i++)
    {
        p_pos[i]=(double *)malloc(particle_dim*sizeof(double));
        vector_copy(init_pos[i],p_pos[i],particle_dim);
    }

    // particle velocity
    double **p_vel=(double **)malloc(args.swarm_size*sizeof(double *));
    for(int i=0;i<args.swarm_size;i++)
    {
        p_vel[i]=(double *)malloc(particle_dim*sizeof(double));
        vector_copy(init_vel[i],p_vel[i],particle_dim);
    }

    // particle-best position in its history
    double **p_best=(double **)malloc(args.swarm_size*sizeof(double *));
    for(int i=0;i<args.swarm_size;i++)
    {
        p_best[i]=(double*)malloc(particle_dim*sizeof(double));
        // Initilized the particle-best to the initilized position.
        vector_copy(p_pos[i],p_best[i],particle_dim);
    }
    // Minimal reconstruction error of each particle in its history
    double *p_min_err=(double *)malloc(args.swarm_size*sizeof(double));
        
    // swarm-best position in its history
    double *s_best=(double*)malloc(particle_dim*sizeof(double));
    double s_min_err=DBL_MAX;
    int min_err_particle_index;  // The index of the particle which achieve minimal 
                                 // mean_reconstruction_error in the swarm.
    // Initilized the s-best to the  best particle in p_best
    for(int i=0;i<args.swarm_size;i++)
    {
        p_min_err[i]=mean_reconstruction_error(points,p_best[i],points_num,args.feature_size,
            args.cluster_num);
        if(p_min_err[i]<s_min_err)
        {
            s_min_err=p_min_err[i];
            min_err_particle_index=i;
        }
    }
    vector_copy(p_best[min_err_particle_index],s_best,particle_dim);

    // Swarm-best in all process are collected in process 0 to determine the global best. 
    // This process is done in asynchronous mode.
    MPI_Win s_best_collect_win, s_min_err_collect_win;
    double *s_best_collect_buffer=NULL;
    double *s_min_err_collect_buffer=NULL; 
    if(my_id==0)
    {
        s_best_collect_buffer=(double*)malloc(particle_dim*procs_num*sizeof(double));
        s_min_err_collect_buffer=(double*)malloc(procs_num*sizeof(double));
        MPI_Win_create(s_best_collect_buffer,particle_dim*procs_num*sizeof(double),sizeof(double),
            MPI_INFO_NULL,MPI_COMM_WORLD,&s_best_collect_win);
        MPI_Win_create(s_min_err_collect_buffer,procs_num*sizeof(double),sizeof(double),
            MPI_INFO_NULL,MPI_COMM_WORLD,&s_min_err_collect_win);
    }
    else
    {
        // We don't need buffer for processes except process 0.
        MPI_Win_create(NULL,0,1,MPI_INFO_NULL,MPI_COMM_WORLD,&s_best_collect_win);
        MPI_Win_create(NULL,0,1,MPI_INFO_NULL,MPI_COMM_WORLD,&s_min_err_collect_win);
    }

    // Collect all swarm best solution to process 0 and calculate global best solution
    MPI_Win_fence(0,s_best_collect_win);
    MPI_Put(s_best,particle_dim,MPI_DOUBLE,0,particle_dim*my_id,particle_dim,MPI_DOUBLE,
        s_best_collect_win);
    MPI_Win_fence(0,s_best_collect_win);

    MPI_Win_fence(0,s_min_err_collect_win);
    MPI_Put(&s_min_err,1,MPI_DOUBLE,0,my_id,1,MPI_DOUBLE,s_min_err_collect_win);
    MPI_Win_fence(0,s_min_err_collect_win);

    // global-best. The global best solution so far
    double *g_best=(double*)malloc(particle_dim*sizeof(double));
    double g_min_err=DBL_MAX;
    double g_min_err_old;
    int g_min_err_procs_id;
    // The g_best_win and g_min_err_win are used by process 0 to deliver global best information 
    // to other processes.
    MPI_Win g_best_win, g_min_err_win;
    MPI_Win_create(g_best,particle_dim*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,
        &g_best_win);
    MPI_Win_create(&g_min_err,sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,
        &g_min_err_win);

    if(my_id==0)
    {
        for(int i=0;i<procs_num;i++)
        {
            if(s_min_err_collect_buffer[i]<g_min_err)
            {
                g_min_err_procs_id=i;
                g_min_err=s_min_err_collect_buffer[i];
            }
        }
        vector_copy(s_best_collect_buffer+g_min_err_procs_id*particle_dim,g_best,particle_dim);

        // Process 0 deliver the global best to other processes
        for(int i=1;i<procs_num;i++)
        {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE,i,0,g_best_win);
            MPI_Put(g_best,particle_dim,MPI_DOUBLE,i,0,particle_dim,MPI_DOUBLE,g_best_win);
            MPI_Win_unlock(i,g_best_win);

            MPI_Win_lock(MPI_LOCK_EXCLUSIVE,i,0,g_min_err_win);
            MPI_Put(&g_min_err,1,MPI_DOUBLE,i,0,1,MPI_DOUBLE,g_min_err_win);
            MPI_Win_unlock(i,g_min_err_win);
        }
        print_line(1,'-');
    }

    // Blocked to ensure that global-best variables in all processes have been initialized.
    MPI_Barrier(MPI_COMM_WORLD);

    // Now we have finished all the initialization work, we will begin to iteration optimization 
    // process
    int iter=0;
    int stop_flag=0;  //This flag was set by process 0 to indicate whether we should stop the 
                      //iteration, 0 means no, 1 means yes
    // Process 0 uses MPI window object to set stop_flag for other processes
    MPI_Win stop_flag_win;
    MPI_Win_create(&stop_flag,sizeof(int),sizeof(int),MPI_INFO_NULL,MPI_COMM_WORLD,&stop_flag_win);

    double particle_err,r1,r2,r3,relative_err_decrease;
    time_t current_time;
    while(iter<args.max_iters && stop_flag==0)
    {
        if(my_id==0)
            printf("Iter:%d\n",iter);
        // In each iter, we forward the swarm multi steps
        for(int i=0;i<args.steps_per_iter;i++)
        {
            if(my_id==0)
                printf("\tStep: %d begin\n",i);
            min_err_particle_index=-1;
            // Calculate each particles mean_reconstruction_error and update the p_best and swarm-best
            for(int j=0;j<args.swarm_size;j++)
            {
                particle_err=mean_reconstruction_error(points,p_pos[j],points_num,
                    args.feature_size,args.cluster_num);
                if(particle_err<p_min_err[j])
                {
                    p_min_err[j]=particle_err;
                    vector_copy(p_pos[j],p_best[j],particle_dim);
                    if(p_min_err[j]<s_min_err)
                    {
                        min_err_particle_index=j;
                        s_min_err=p_min_err[j];
                    }   
                }
            }
            if(min_err_particle_index>=0)
                vector_copy(p_best[min_err_particle_index],s_best,particle_dim);

            // Update the velocity and position of each particle
            for(int j=0;j<args.swarm_size;j++)
            {
                for(int k=0;k<particle_dim;k++)
                {
                    r1=rand()/(double)RAND_MAX;
                    r2=rand()/(double)RAND_MAX;
                    r3=rand()/(double)RAND_MAX;
                    p_vel[j][k]=p_vel[j][k]*args.inertia_weight+
                        args.c1*r1*(p_best[j][k]-p_pos[j][k])+
                        args.c2*r2*(s_best[k]-p_pos[j][k])+
                        args.c3*r3*(g_best[k]-p_pos[j][k]);
                    p_pos[j][k]+=p_vel[j][k];
                }
            }
            if(my_id==0)
                printf("\tStep: %d end, process 0 swarm minimal mean_reconstruction_error: %.6f\n",
                    i,s_min_err);
        }

        // After each iter, we need to send the swarm best result to process 0 to update the 
        // global best result.
        if(s_min_err<g_min_err)
        {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE,0,0,s_best_collect_win);
            MPI_Put(s_best,particle_dim,MPI_DOUBLE,0,particle_dim*my_id,particle_dim,MPI_DOUBLE,
                s_best_collect_win);
            MPI_Win_unlock(0,s_best_collect_win);

            MPI_Win_lock(MPI_LOCK_EXCLUSIVE,0,0,s_min_err_collect_win);
            MPI_Put(&s_min_err,1,MPI_DOUBLE,0,my_id,1,MPI_DOUBLE,s_min_err_collect_win);
            MPI_Win_unlock(0,s_min_err_collect_win);
        }

        // Process 0 has the responsibility to update the global best result and deliver to other 
        // processes
        if(my_id==0)
        {
            // Process 0 determine the global best
            for(int i=0;i<procs_num;i++)
                if(s_min_err_collect_buffer[i]<g_min_err)
                {
                    g_min_err_procs_id=i;
                    g_min_err=s_min_err_collect_buffer[i];
                }
            vector_copy(s_best_collect_buffer+g_min_err_procs_id*particle_dim,g_best,particle_dim);
            printf("All processes send the swarm best results to process 0 to get the global best "
                "results, the global minimal error is: %.6f\n",
                g_min_err);
            print_line(1,'-');

            // Process 0 deliver the global best to other processes
            for(int i=1;i<procs_num;i++)
            {
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE,i,0,g_best_win);
                MPI_Put(g_best,particle_dim,MPI_DOUBLE,i,0,particle_dim,MPI_DOUBLE,g_best_win);
                MPI_Win_unlock(i,g_best_win);

                MPI_Win_lock(MPI_LOCK_EXCLUSIVE,i,0,g_min_err_win);
                MPI_Put(&g_min_err,1,MPI_DOUBLE,i,0,1,MPI_DOUBLE,g_min_err_win);
                MPI_Win_unlock(i,g_min_err_win);
            }

            if(iter%args.early_stop_iters==0)
                g_min_err_old=g_min_err;
            
            // Early stop critical check
            if((iter+1)%args.early_stop_iters==0)
            {
                time(&current_time);
                print_time(current_time);
                relative_err_decrease=(g_min_err_old-g_min_err)/(g_min_err_old+1.0e-50);
                printf("iter: %d, mean_reconstruction_error:%.6f, decrease relative to %dth iter:"
                    "%.6f%%\n",iter,g_min_err,iter-args.early_stop_iters+1,
                    relative_err_decrease*100.0);
                print_line(1,'-');
                if(relative_err_decrease<args.critical)
                {
                    stop_flag=1;
                    for(int p=1;p<procs_num;p++)
                    {
                        MPI_Win_lock(MPI_LOCK_EXCLUSIVE,p,0,stop_flag_win);
                        MPI_Put(&stop_flag,1,MPI_INT,p,0,1,MPI_INT,stop_flag_win);
                        MPI_Win_unlock(p,stop_flag_win);
                    }
                    printf("Having finished iteration, the minimal mean_reconstruction_error:%.6f\n"
                        ,g_min_err);
                }
            }
        }
        iter++;
    }

    vector_copy(g_best,results,particle_dim);

    MPI_Win_free(&s_best_collect_win);
    MPI_Win_free(&s_min_err_collect_win);
    MPI_Win_free(&g_best_win);
    MPI_Win_free(&g_min_err_win);
    MPI_Win_free(&stop_flag_win);

    for(int i=0;i<args.swarm_size;i++)
    {
        free(p_pos[i]);
        free(p_vel[i]);
        free(p_best[i]);
    }
    free(p_pos);
    free(p_vel);
    free(p_best);
    free(p_min_err);
    free(s_best);
    if(my_id==0)
    {
        free(s_best_collect_buffer);
        free(s_min_err_collect_buffer);
    }
    free(g_best);

    return;
}