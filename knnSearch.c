/*Antoniadis Moschos, AEM = 8761, AUTH 2018 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <sys/time.h>
#include <float.h>

#define THREADS 2


int main(int argc, char **argv)
{
    FILE *file,*dist,*nn;
    MPI_Offset fsize;
    MPI_Status status;
    MPI_Request request;
    int  ntasks,bufsize, rank, len, rc,error,filesize;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    double **points_buf,**points_buf_send,**points_buf_recv, **distance,disbuf;
    int **nearNeighbors;
    int k,N,p,size,i,j,l,m,n,rankReceived,buf_len;
    double start, end;
    omp_set_dynamic(0);
    omp_set_num_threads(THREADS);

    // initialize MPI
    MPI_Init(&argc,&argv);

    // get number of tasks
    MPI_Comm_size(MPI_COMM_WORLD,&ntasks);

    // get my rank
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    // get processor name
    MPI_Get_processor_name(hostname, &len);
    // printf ("Number of tasks= %d My rank= %d Running on %s\n", ntasks,rank,hostname);

    /* Synchronize */
    MPI_Barrier(MPI_COMM_WORLD);
    // done with MPI

    if(argc ==1)
    {
        N=30;
        k=20;
        if((file = fopen("trainX_svd.bin", "rb+")) == NULL)
        {
            printf("\nFile not found\n");
            exit(1);
        }

    }
    else if(argc !=4)
    {
        printf("Usage: %s BIN_FILE N k\n   where N = dimensions and k = number of neighbors", argv[0]);
        exit(1);
    }
    else
    {
        if((file = fopen(argv[1], "rb+")) == NULL)
        {
            printf("\nFile not found\n");
            exit(1);
        }
        N = atoi(argv[2]);
        k = atoi(argv[3]);
    }
    if((dist = fopen("distances.bin", "wb+")) == NULL)
    {
        printf("\nFile not found\n");
        exit(1);
    }

    if((nn = fopen("nearNeighbors.bin", "wb+")) == NULL)
    {
        printf("\nFile not found\n");
        exit(1);
    }

    /* Get the size of the file */
    fseek(file, 0L, SEEK_END);
    filesize = ftell(file)/(sizeof(double)*N); //find the number of points
    rewind(file);

    if(rank==0) printf("\nTotal points=%d\n",filesize);

    /*Calculate the number of points per process
     *if there are any remainders give them to the last process*/

    if (!(filesize % ntasks) && (rank == (ntasks-1)))
    {
        p = (filesize / ntasks) + (filesize % ntasks);

    }
    else
    {
        p = (filesize / ntasks);
    }
    /*Malloc for points buffer*/
    if((points_buf = (double**) malloc(p*sizeof(double *))) == NULL)  exit(1); //malloc for array of points
    for (i=0; i<p; i++)
    {
        if((points_buf[i] = (double*)malloc(N*sizeof(double))) == NULL)  exit(1);
    }
    /*Malloc for sender buffer*/
    if((points_buf_send = (double**) malloc(((filesize / ntasks) + (filesize % ntasks))*sizeof(double *))) == NULL)  exit(1);
    for (i=0; i<((filesize / ntasks) + (filesize % ntasks)); i++)
    {
        if((points_buf_send[i] = (double*)malloc(N*sizeof(double))) == NULL)  exit(1);
    }

    /*Malloc for receiver buffer*/
    if((points_buf_recv = (double**) malloc(((filesize / ntasks) + (filesize % ntasks))*sizeof(double *))) == NULL)  exit(1);
    for (i=0; i<((filesize / ntasks) + (filesize % ntasks)); i++)
    {
        if((points_buf_recv[i] = (double*)malloc(N*sizeof(double))) == NULL)  exit(1);
    }

    /*Malloc for distance matrix of k nearest neighbors*/
    if((distance = (double**) malloc(p*sizeof(double *))) == NULL)  exit(1);
    for (i=0; i<p; i++)
    {
        if((distance[i] = (double*)malloc(k*sizeof(double))) == NULL)  exit(1);
    }

    /*Malloc for id matrix of k nearest neighbors*/
    if((nearNeighbors = (int**) malloc(p*sizeof(int *))) == NULL)  exit(1);
    for (i=0; i<p; i++)
    {
        if((nearNeighbors[i] = (int*)malloc(k*sizeof(int))) == NULL)  exit(1);
    }

    /*Read the data to buffer*/
    fseek(file,(rank*(filesize/ntasks)*N*sizeof(double)),SEEK_SET);
    for(i=0; i<p; i++)
    {
        if((fread(points_buf[i],sizeof(double),N,file))!=N)
        {
            fprintf(stderr, "Unable to read data\n");
            exit(1);
        }
    }
    /* Synchronize */
    MPI_Barrier(MPI_COMM_WORLD);

    start = MPI_Wtime(); //Start clock


    /* Initialize the dist vector */
    for(i=0; i<p; i++)
    {
        for(j=0; j<k; j++)
        {
            distance[i][j] = DBL_MAX;
        }
    }

    /*Find the k nearest neighbors for every point in self dataset*/
    for(i=0; i<p; i++)
    {
        #pragma omp parallel for schedule(static) shared(rank,ntasks,points_buf,distance,nearNeighbors) private(j,m,n,disbuf)

        for(l=0; l<p; l++)
        {
            if(i!=l)
            {
                disbuf=0;
                for(j=0; j<N; j++)
                {
                    disbuf+=((points_buf[l][j]-points_buf[i][j])*(points_buf[l][j]-points_buf[i][j]));
                }
                for(m=0; m<k; m++)
                {
                    if(distance[i][m]>disbuf)
                    {

                        for(n=k-1; n>=m+1; n--)
                        {
                            distance[i][n]=distance[i][n-1];
                            nearNeighbors[i][n]=nearNeighbors[i][n-1];

                        }
                        /*Add new point*/
                        distance[i][m]=disbuf;
                        nearNeighbors[i][m]=l+rank*(filesize/ntasks)+1;
                        break;
                    }
                }
            }
        }
    }

    /*Copy current buffer to send buffer*/
    for(i=0; i<ntasks-1; i++)
    {
        if(i != 0) memcpy(points_buf_send,points_buf,((filesize / ntasks) + (filesize % ntasks))*sizeof(MPI_DOUBLE));
        else memcpy(points_buf_send,points_buf_recv,((filesize / ntasks) + (filesize % ntasks))*sizeof(MPI_DOUBLE));

        /*Calculate the rank and the length of the received data set*/
        rankReceived=(rank + ntasks - i)%(ntasks);
        buf_len=((rankReceived+ ntasks - i -1)? (filesize/ntasks):(filesize/ntasks)+(filesize%ntasks));
        /*Send and Receive*/
        if((rank%2)==0)
        {
            MPI_Send(&(points_buf_send[0][0]),p*N,MPI_DOUBLE, (rank+1)%ntasks,66,MPI_COMM_WORLD);
            MPI_Recv(&(points_buf_recv[0][0]),p*N,MPI_DOUBLE, (rank+ntasks-1)%ntasks,66,MPI_COMM_WORLD,&status);
        }
        else
        {
            MPI_Recv(&(points_buf_recv[0][0]),buf_len*N,MPI_DOUBLE, (rank+ntasks-1)%ntasks,66,MPI_COMM_WORLD,&status);
            MPI_Send(&(points_buf_send[0][0]),buf_len*N,MPI_DOUBLE, (rank+1)%ntasks,66,MPI_COMM_WORLD);
        }
        for(i=0; i<p; i++)
        {
            #pragma omp parallel for schedule(static) shared(rankReceived,ntasks,points_buf,points_buf_recv,distance,nearNeighbors) private(j,m,n,disbuf)
            for(l=0; l<buf_len; l++)
            {
                disbuf=0;

                for(j=0; j<N; j++)
                {
                    disbuf+=((points_buf_recv[l][j]-points_buf[i][j])*(points_buf_recv[l][j]-points_buf[i][j]));

                }
                /*Update distance and nearNeighbors matrix*/
                for(m=0; m<k; m++)
                {
                    if(distance[i][m]>disbuf)
                    {
                        for(n=k-1; n>=m+1; n--)
                        {
                            distance[i][n]=distance[i][n-1];
                            nearNeighbors[i][n]=nearNeighbors[i][n-1];
                        }
                        /*New point*/
                        distance[i][m]=disbuf;
                        nearNeighbors[i][m]=l+rankReceived*(filesize/ntasks)+1;
                        break;
                    }
                }
            }
        }
    }

    /* Synchronize */
    MPI_Barrier(MPI_COMM_WORLD);
    /* End time measurement */

    end = MPI_Wtime();
    if(rank==0)printf("Time spend: %f seconds\n", (end - start));

    /* Write to files the data */
    if(rank==0)printf("Writing results to files...\n");
    fseek(file,(rank*(filesize/ntasks)*k*sizeof(double)),SEEK_SET);
    for(i=0; i<p; i++)
    {
        fwrite(nearNeighbors[i], sizeof(int), k, nn);
        fwrite(distance[i], sizeof(double), k, dist);
    }
    /* Finalize */
    fclose(file);
    fclose(dist);
    fclose(nn);
    free(points_buf[0]);
    free(points_buf);
    free(distance[0]);
    free(distance);

    MPI_Finalize();

}






