/*

This is a mpi implementation of the kmeans+pso cluster algorithm. In this implementation, 
seperate k-means or pso cluster algorithm can be applied to dataset cluster. 
More important, pso cluster algorithm can to used to give better initilized clusters' center for 
kmeans cluster algorithm, which helps kmeans to escape from local minimization.

Author: Dong xie
Date: 2019-1-15
E-mail:xied15@mails.tsinghua.edu.cn

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <argp.h>
#include "mpi.h"
#include "utils.h"
#include "cluster.h"

// This part is just used for arguments parse, you can ignore it.
const char *argp_program_version="version 1.0.0";
const char *argp_program_bug_address="<xied15@mails.tsinghua.edu.cn>";
static char doc[]="This is a mpi implementation of the kmeans+pso cluster algorithm.";

static struct argp_option options[]={
    {"data",'d',"dataset name",0,"The name of the dataset"},
    {"points_num",'n',"number_points",0,"How many data points in total"},
    {"feature_size",'f',"feature_size",0,"The dimension of the data points"},
    {"cluster_num",'c',"cluster_num",0,"How many clusters these data points belong to"},
    {"mode",'m',"mode",0,"Cluster algorithm mode: kmeans, pso or hybrid"},
    {"kmeans_max_iters",257,"kmeans_max_iters",0,"Max iterations for kmeans if kmeans algorithm used"},
    {"kmeans_critical",258,"kmeans_critical",0,"Early stop critical value for kmeans if kmeans"
        " algorithm used"},
    {"kmeans_early_stop_iters",259,"kmeans_early_stop_iters",0,"Early stop iterations for kmeans"
        " if kmeans algorithm used "},
    {"pso_inertia_weight",260,"pso_inertia_weight",0,"Inertia weight for pso if pso algorithm used"},
    {"pso_c1",261,"pso_c1",0,"Acceleration parameters towards the particle best for pso if pso"
        " algorithm used"},
    {"pso_c2",262,"pso_c2",0,"Acceleration parameters towards the swarm best for pso if pso"
        " algorithm used"},
    {"pso_c3",263,"pso_c3",0,"Acceleration parameters towards the global best for pso if pso"
        " algorithm used"},
    {"pso_steps_per_iter",264,"pso_steps_per_iter",0,"Steps per iter for pso if pso algorithm used"},
    {"pso_max_iters",265,"pso_max_iters",0,"Max iterations for pso if pso algorithm used"},
    {"pso_critical",266,"pso_critical",0,"Early stop critical value for pso if pso algorithm used"},
    {"pso_early_stop_iters",267,"pso_early_stop_iters",0,"Early stop iterations for pso if pso"
        " algorithm used "},
    {"pso_swarm_size",268,"pso_swarm_size",0,"How many particles in a swarm"},
    {0}
};

// How to parse the command line arguments
static error_t parse_opt(int key, char * arg, struct argp_state *state)
{
    /* Get the input argument from argp_parse, which we
    know is a pointer to our arguments structure. */
    struct arguments *args=state->input;

    switch(key)
    {
        case 'd':
            args->data=arg;
            break;
        case 'n':
            args->points_num=atoi(arg);
            break;
        case 'f':
            args->feature_size=atoi(arg);
            break;
        case 'c':
            args->cluster_num=atoi(arg);
            break;
        case 'm':
            args->mode=arg;
            break;
        case 257:
            args->kmeans_max_iters=atoi(arg);
            break;
        case 258:
            sscanf(arg,"%lf",&(args->kmeans_critical));
            break;
        case 259:
            args->kmeans_early_stop_iters=atoi(arg);
        case 260:
            sscanf(arg,"%lf",&(args->pso_inertia_weight));
            break;
        case 261:
            sscanf(arg,"%lf",&(args->pso_c1));
            break;
        case 262:
            sscanf(arg,"%lf",&(args->pso_c2));
            break;
        case 263:
            sscanf(arg,"%lf",&(args->pso_c3));
            break;
        case 264:
            args->pso_steps_per_iter=atoi(arg);
            break;
        case 265:
            args->pso_max_iters=atoi(arg);
            break;
        case 266:
            sscanf(arg,"%lf",&(args->pso_critical));
            break;
        case 267:
            args->pso_early_stop_iters=atoi(arg);
            break;
        case 268:
            args->pso_swarm_size=atoi(arg);
            break;
        default:
            return ARGP_ERR_UNKNOWN;
     }
     return 0;
}

static struct argp parser={options,parse_opt,0,doc};

int main(int argc, char * argv[])
{
    int my_id,procs_num;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&procs_num);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_id);

    srand(time(NULL));
    clock_t begin,end;

    struct arguments args;
    // Define the default value for the args
    args.data="2d_synthetic_data";
    args.points_num=5000;
    args.feature_size=2;
    args.cluster_num=15;
    args.mode="kmeans";
    args.kmeans_max_iters=200;
    args.kmeans_critical=1e-6;
    args.kmeans_early_stop_iters=5;
    args.pso_inertia_weight=0.7298;
    args.pso_c1=1.0;
    args.pso_c2=1.0;
    args.pso_c3=1.0;
    args.pso_steps_per_iter=10;
    args.pso_max_iters=100;
    args.pso_critical=1e-3;
    args.pso_early_stop_iters=5;
    args.pso_swarm_size=20;

    /* Parse our arguments; every option seen by parse_opt will
    be reflected in arguments. */
    argp_parse(&parser,argc,argv,0,0,&args);

    // Partition the data points between processes
    int num_data_per_proc,data_begin_index,data_end_index,data_owned;
    num_data_per_proc=args.points_num/procs_num;
    data_begin_index=my_id*num_data_per_proc;   //The index of the begin data this process owns
    // The index of the end data this process owns (not includes the data_end_index), 
    // [data_begin_index,data_end_index)
    data_end_index=(my_id==procs_num-1)?(args.points_num):((my_id+1)*num_data_per_proc); 
    data_owned=data_end_index-data_begin_index;

    // Data points number in each processes.
    int *data_owned_each_process=(int*)malloc(procs_num*sizeof(int));
    for(int i=0;i<procs_num-1;i++)
        data_owned_each_process[i]=num_data_per_proc;
    data_owned_each_process[procs_num-1]=args.points_num-(procs_num-1)*num_data_per_proc;
    
    if(my_id==0)
    {
        print_line(1,'-');
        printf("Partition the data points between processes\n");
        for(int i=0;i<procs_num-1;i++)
            printf("My_id: %d, data points owned:[%d,%d)\n",i,i*num_data_per_proc,
                (i+1)*num_data_per_proc);
        printf("My_id: %d, data points owned:[%d,%d)\n",procs_num-1,(procs_num-1)*num_data_per_proc,
            args.points_num);
    }

    // Read in the data points each process owned
    if(my_id==0)
    {
        print_line(1,'-');
        printf("Read in data, please wait\n");
    }

    double **points=(double **)malloc(data_owned*sizeof(double*));
    for(int i=0;i<data_owned;i++)
        points[i]=(double*)malloc(args.feature_size*sizeof(double));
    int *labels=(int *)malloc(data_owned*sizeof(int));

    char data_points_file_path[100];
    strcpy(data_points_file_path,"data/");
    strcat(data_points_file_path,args.data);
    strcat(data_points_file_path,"/data.txt");
    FILE *data_fp=fopen(data_points_file_path,"r");
    if(data_fp==NULL)
    {
        printf("Can't open the %s file \n",data_points_file_path);
        exit(1);
    }

    char label_file_path[100];
    strcpy(label_file_path,"data/");
    strcat(label_file_path,args.data);
    strcat(label_file_path,"/label.txt");
    FILE *label_fp=fopen(label_file_path,"r");
    if(label_fp==NULL)
    {
        printf("Can't open the %s file \n",label_file_path);
        exit(1);
    }
    
    double tmp1;  // Just placeholder
    for(int i=0;i<args.points_num;i++)
    {
        if(i>=data_begin_index && i<data_end_index)
            for(int j=0;j<args.feature_size;j++)
                fscanf(data_fp,"%lf",&points[i-data_begin_index][j]);
        else
            for(int j=0;j<args.feature_size;j++)
                fscanf(data_fp,"%lf",&tmp1);
    }
    fclose(data_fp);

    int tmp2; //Just placeholder
    for(int i=0;i<args.points_num;i++)
    {
        if(i>=data_begin_index && i<data_end_index)
            fscanf(label_fp,"%d",&labels[i-data_begin_index]);
        else
            fscanf(label_fp,"%d",&tmp2);
    }
    fclose(label_fp);
    
    // Array to store the centroids of the clusters
    double *results=(double*)malloc(args.feature_size*args.cluster_num*sizeof(double));

    // Using kmeans for cluster
    if(!strcmp(args.mode,"kmeans"))
    {
        struct kmeans_arguments kmeans_args;
        kmeans_args.feature_size=args.feature_size;
        kmeans_args.cluster_num=args.cluster_num;
        kmeans_args.max_iters=args.kmeans_max_iters;
        kmeans_args.early_stop_iters=args.kmeans_early_stop_iters;
        kmeans_args.critical=args.kmeans_critical;

        begin=clock();
        double *init_pos=(double*)malloc(args.feature_size*args.cluster_num*sizeof(double));
        if(my_id==0)
        {
            print_line(1,'-');
            printf("Using kmeans for cluster\n");
            printf("The arguments are as follows:\n");
            printf("\tdata: %s\n",args.data);
            printf("\tpoints_num: %d\n",args.points_num);
            printf("\tfeature_size: %d\n",args.feature_size);
            printf("\tcluster_num: %d\n",args.cluster_num);
            printf("\tkmeans_max_iters: %d\n",args.kmeans_max_iters);
            printf("\tkmeans_critical: %6f\n",args.kmeans_critical);
            printf("\tkmeans_early_stop_iters: %d\n",args.kmeans_early_stop_iters);

            // Cluster centroid initialization
            for(int i=0;i<args.cluster_num;i++)
                vector_copy(points[rand()%data_owned],init_pos+i*args.feature_size,
                    args.feature_size);
        }
        MPI_Bcast(init_pos,args.feature_size*args.cluster_num,MPI_DOUBLE,0,MPI_COMM_WORLD);
        mpi_kmeans_cluster(points,data_owned_each_process,init_pos,results,kmeans_args);
        free(init_pos);
        end=clock();
        if(my_id==0)
        {
            print_line(1,'-');
            printf("Running time is: %.6fs\n",(double)(end-begin)/CLOCKS_PER_SEC);
        }
    }
    else if(!strcmp(args.mode,"pso"))
    {
        struct pso_arguments pso_args;
        pso_args.feature_size=args.feature_size;
        pso_args.cluster_num=args.cluster_num;
        pso_args.steps_per_iter=args.pso_steps_per_iter;
        pso_args.max_iters=args.pso_max_iters;
        pso_args.early_stop_iters=args.pso_early_stop_iters;
        pso_args.swarm_size=args.pso_swarm_size;
        pso_args.inertia_weight=args.pso_inertia_weight;
        pso_args.c1=args.pso_c1;
        pso_args.c2=args.pso_c2;
        pso_args.c3=args.pso_c3;
        pso_args.critical=args.pso_critical;

        begin=clock();
        double **init_pos=(double**)malloc(args.pso_swarm_size*sizeof(double*));
        for(int i=0;i<args.pso_swarm_size;i++)
            init_pos[i]=(double*)malloc(args.feature_size*args.cluster_num*sizeof(double));
        
        double **init_vel=(double**)malloc(args.pso_swarm_size*sizeof(double*));
        for(int i=0;i<args.pso_swarm_size;i++)
            init_vel[i]=(double*)malloc(args.feature_size*args.cluster_num*sizeof(double));
        
        for(int i=0;i<args.pso_swarm_size;i++)
            for(int j=0;j<args.cluster_num;j++)
            {
                vector_copy(points[rand()%data_owned],init_pos[i]+j*args.feature_size,
                    args.feature_size);
                for(int k=0;k<args.feature_size;k++)
                    init_vel[i][j*args.feature_size+k]=rand()/(double)RAND_MAX*0.1-0.2; 
                    // Init_vel between [-0.1,0.1]
            }
        
        if(my_id==0)
        {
            print_line(1,'-');
            printf("Using pso for cluster\n");
            printf("The arguments are as follows:\n");
            printf("\tdata: %s\n",args.data);
            printf("\tpoints_num: %d\n",args.points_num);
            printf("\tfeature_size: %d\n",args.feature_size);
            printf("\tcluster_num: %d\n",args.cluster_num);
            printf("\tpso_inertia_weight: %.6f\n",args.pso_inertia_weight);
            printf("\tpso_c1: %.6f\n",args.pso_c1);
            printf("\tpso_c2: %.6f\n",args.pso_c2);
            printf("\tpso_c3: %.6f\n",args.pso_c3);
            printf("\tpso_steps_per_iter: %d\n",args.pso_steps_per_iter);
            printf("\tpso_max_iters: %d\n",args.pso_max_iters);
            printf("\tpso_critical: %.6f\n",args.pso_critical);
            printf("\tpso_early_stop_iters: %d\n",args.pso_early_stop_iters);
            printf("\tpso_swarm_size: %d\n",args.pso_swarm_size);
        }

        mpi_pso_cluster(points,data_owned,init_pos,init_vel,results,pso_args);
        for(int i=0;i<args.pso_swarm_size;i++)
        {
            free(init_pos[i]);
            free(init_vel[i]);
        }
        free(init_pos);
        free(init_vel);

        end=clock();
        if(my_id==0)
        {
            print_line(1,'-');
            printf("Running time is: %.6fs\n",(double)(end-begin)/CLOCKS_PER_SEC);
            print_line(1,'-');
        }
    }
    else
    {
        // Stage1: pso cluster
        struct pso_arguments pso_args;
        pso_args.feature_size=args.feature_size;
        pso_args.cluster_num=args.cluster_num;
        pso_args.steps_per_iter=args.pso_steps_per_iter;
        pso_args.max_iters=args.pso_max_iters;
        pso_args.early_stop_iters=args.pso_early_stop_iters;
        pso_args.swarm_size=args.pso_swarm_size;
        pso_args.inertia_weight=args.pso_inertia_weight;
        pso_args.c1=args.pso_c1;
        pso_args.c2=args.pso_c2;
        pso_args.c3=args.pso_c3;
        pso_args.critical=args.pso_critical;

        begin=clock();
        double **init_pos=(double**)malloc(args.pso_swarm_size*sizeof(double*));
        for(int i=0;i<args.pso_swarm_size;i++)
            init_pos[i]=(double*)malloc(args.feature_size*args.cluster_num*sizeof(double));
        
        double **init_vel=(double**)malloc(args.pso_swarm_size*sizeof(double*));
        for(int i=0;i<args.pso_swarm_size;i++)
            init_vel[i]=(double*)malloc(args.feature_size*args.cluster_num*sizeof(double));
        
        for(int i=0;i<args.pso_swarm_size;i++)
            for(int j=0;j<args.cluster_num;j++)
            {
                vector_copy(points[rand()%data_owned],init_pos[i]+j*args.feature_size,args.feature_size);
                for(int k=0;k<args.feature_size;k++)
                    init_vel[i][j*args.feature_size+k]=rand()/(double)RAND_MAX*0.1-0.1; 
                    // Init_vel between [-0.1,0.1]
            }
        
        if(my_id==0)
        {
            printf("Pso+kmeans cluster\n");
            printf("The arguments are as follows:\n");
            printf("\tdata: %s\n",args.data);
            printf("\tpoints_num: %d\n",args.points_num);
            printf("\tfeature_size: %d\n",args.feature_size);
            printf("\tcluster_num: %d\n",args.cluster_num);
            print_line(1,'-');
            printf("Stage1: pso\n");
            printf("\tpso_inertia_weight: %.6f\n",args.pso_inertia_weight);
            printf("\tpso_c1: %.6f\n",args.pso_c1);
            printf("\tpso_c2: %.6f\n",args.pso_c2);
            printf("\tpso_c3: %.6f\n",args.pso_c3);
            printf("\tpso_steps_per_iter: %d\n",args.pso_steps_per_iter);
            printf("\tpso_max_iters: %d\n",args.pso_max_iters);
            printf("\tpso_critical: %.6f\n",args.pso_critical);
            printf("\tpso_early_stop_iters: %d\n",args.pso_early_stop_iters);
            printf("\tpso_swarm_size: %d\n",args.pso_swarm_size);
        }

        double *kmeans_init=(double*)malloc(args.feature_size*args.cluster_num*sizeof(double));
        mpi_pso_cluster(points,data_owned,init_pos,init_vel,kmeans_init,pso_args);
        for(int i=0;i<args.pso_swarm_size;i++)
        {
            free(init_pos[i]);
            free(init_vel[i]);
        }
        free(init_pos);
        free(init_vel);

        end=clock();
        double stage1_time=(double)(end-begin)/CLOCKS_PER_SEC;
        if(my_id==0)
        {
            print_line(1,'-');
            printf("Stage 1 (pso) running time is: %.6fs\n",stage1_time);
            print_line(1,'-');
        }

        //Stage 2: kmeans_cluster
        struct kmeans_arguments kmeans_args;
        kmeans_args.feature_size=args.feature_size;
        kmeans_args.cluster_num=args.cluster_num;
        kmeans_args.max_iters=args.kmeans_max_iters;
        kmeans_args.early_stop_iters=args.kmeans_early_stop_iters;
        kmeans_args.critical=args.kmeans_critical;

        begin=clock();
        if(my_id==0)
        {
            printf("Stage2: kmeans\n");
            printf("\tkmeans_max_iters: %d\n",args.kmeans_max_iters);
            printf("\tkmeans_critical: %6f\n",args.kmeans_critical);
            printf("\tkmeans_early_stop_iters: %d\n",args.kmeans_early_stop_iters);
        }
        mpi_kmeans_cluster(points,data_owned_each_process,kmeans_init,results,kmeans_args);
        free(kmeans_init);
        end=clock();
        double stage2_time=(double)(end-begin)/CLOCKS_PER_SEC;
        if(my_id==0)
        {
            print_line(1,'-');
            printf("Satge 2 (kmeans) running time is: %.6fs\n",stage2_time);
            print_line(1,'-');
            printf("The total running time is: %.6fs\n",stage1_time+stage2_time);
        }
    }

    if(my_id==0)
    {
        char results_file_path[100];
        strcpy(results_file_path,"data/");
        strcat(results_file_path,args.data);
        strcat(results_file_path,"/cluster_centroid_results.txt");
        FILE *results_fp=fopen(results_file_path,"w");
        if(results_fp==NULL)
        {
            printf("Can't open the %s file \n",results_file_path);
            exit(1);
        }

        for(int i=0;i<args.feature_size*args.cluster_num;i++)
        {
            fprintf(results_fp,"%.6f\t",results[i]);
            if((i+1)%args.feature_size==0)
                fprintf(results_fp,"\n");
        }

        fclose(results_fp);
    }
    free(data_owned_each_process);
    for(int i=0;i<data_owned;i++)
        free(points[i]);
    free(points);
    free(labels);
    free(results);

    MPI_Finalize();
    return 0;
}
