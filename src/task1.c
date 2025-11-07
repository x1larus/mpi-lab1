#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double random_double() 
{
    return (double)rand() / RAND_MAX;
}

int main(int argc, char** argv) 
{
    int rank, size;
    long long total_points = 1000000000LL;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    srand(time(NULL) + rank * 1234);
    
    long long points_per_process = total_points / size;
    
    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    long long local_inside = 0;
    for (long long i = 0; i < points_per_process; i++) 
    {
        double x = random_double() * 2.0 - 1.0;
        double y = random_double() * 2.0 - 1.0;
        
        if (x*x + y*y <= 1.0) 
        {
            local_inside++;
        }
    }
    
    long long total_inside;
    MPI_Reduce(&local_inside, &total_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) 
    {
        char filename[100];
        sprintf(filename, "out/task1_%i.json", size);
        FILE *out = fopen(filename, "w");

        fprintf(out, "{\n");
        fprintf(out, "  \"total_points\": %lld,\n", total_points);
        fprintf(out, "  \"total_inside\": %lld,\n", total_inside);
        fprintf(out, "  \"exec_time\": %.6f\n", max_time);
        fprintf(out, "}\n");

        fclose(out);
    }
    
    MPI_Finalize();
    return 0;
}