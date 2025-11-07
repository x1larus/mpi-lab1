#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

double** allocate_matrix(int rows, int cols) 
{
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL) 
    {
        return NULL;
    }
    for (int i = 0; i < rows; i++) 
    {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) 
        {
            for (int j = 0; j < i; j++) 
            {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

void free_matrix(double** matrix, int rows) 
{
    if (matrix != NULL) 
    {
        for (int i = 0; i < rows; i++) 
        {
            if (matrix[i] != NULL) 
            {
                free(matrix[i]);
            }
        }
        free(matrix);
    }
}

void initialize_matrix(double** matrix, int rows, int cols) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            matrix[i][j] = (double)rand() / RAND_MAX * 10.0;
        }
    }
}

void zero_matrix(double** matrix, int rows, int cols) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            matrix[i][j] = 0.0;
        }
    }
}

void cannon_matrix_mult(int n, int grid_size, int block_size, 
                       double** local_A, double** local_B, double** local_C,
                       MPI_Comm grid_comm) 
{
    int rank;
    MPI_Comm_rank(grid_comm, &rank);
    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];
    
    MPI_Status status;
    
    for (int shift = 0; shift < row; shift++) 
    {
        int left_neighbor, right_neighbor;
        MPI_Cart_shift(grid_comm, 1, -1, &rank, &left_neighbor);
        MPI_Cart_shift(grid_comm, 1, 1, &rank, &right_neighbor);
        
        for (int i = 0; i < block_size; i++) 
        {
            MPI_Sendrecv_replace(local_A[i], block_size, MPI_DOUBLE,
                               left_neighbor, 0, right_neighbor, 0,
                               grid_comm, &status);
        }
    }
    
    for (int shift = 0; shift < col; shift++) 
    {
        int up_neighbor, down_neighbor;
        MPI_Cart_shift(grid_comm, 0, -1, &rank, &up_neighbor);
        MPI_Cart_shift(grid_comm, 0, 1, &rank, &down_neighbor);
        
        for (int i = 0; i < block_size; i++) 
        {
            MPI_Sendrecv_replace(local_B[i], block_size, MPI_DOUBLE,
                               up_neighbor, 0, down_neighbor, 0,
                               grid_comm, &status);
        }
    }
    
    for (int step = 0; step < grid_size; step++) 
    {
        for (int i = 0; i < block_size; i++) 
        {
            for (int j = 0; j < block_size; j++) 
            {
                double sum = 0.0;
                for (int k = 0; k < block_size; k++) 
                {
                    sum += local_A[i][k] * local_B[k][j];
                }
                local_C[i][j] += sum;
            }
        }
        
        if (step < grid_size - 1) 
        {
            int left_neighbor, right_neighbor;
            MPI_Cart_shift(grid_comm, 1, -1, &rank, &left_neighbor);
            MPI_Cart_shift(grid_comm, 1, 1, &rank, &right_neighbor);
            
            for (int i = 0; i < block_size; i++) 
            {
                MPI_Sendrecv_replace(local_A[i], block_size, MPI_DOUBLE,
                                   left_neighbor, 0, right_neighbor, 0,
                                   grid_comm, &status);
            }
            
            int up_neighbor, down_neighbor;
            MPI_Cart_shift(grid_comm, 0, -1, &rank, &up_neighbor);
            MPI_Cart_shift(grid_comm, 0, 1, &rank, &down_neighbor);
            
            for (int i = 0; i < block_size; i++) 
            {
                MPI_Sendrecv_replace(local_B[i], block_size, MPI_DOUBLE,
                                   up_neighbor, 0, down_neighbor, 0,
                                   grid_comm, &status);
            }
        }
    }
}

int main(int argc, char** argv) 
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int grid_size = (int)sqrt(size);
    if (grid_size * grid_size != size) 
    {
        MPI_Finalize();
        return 1;
    }
    
    int n = 1000;
    if (argc > 1) 
    {
        n = atoi(argv[1]);
    }
    
    if (n % grid_size != 0) 
    {
        MPI_Finalize();
        return 1;
    }
    
    int block_size = n / grid_size;
    
    MPI_Comm grid_comm;
    int dimensions[2] = {grid_size, grid_size};
    int periods[2] = {1, 1};
    int reorder = 1;
    
    if (MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, reorder, &grid_comm) != MPI_SUCCESS) 
    {
        MPI_Finalize();
        return 1;
    }
    
    double** local_A = allocate_matrix(block_size, block_size);
    double** local_B = allocate_matrix(block_size, block_size);
    double** local_C = allocate_matrix(block_size, block_size);
    
    if (local_A == NULL || local_B == NULL || local_C == NULL) 
    {
        MPI_Finalize();
        return 1;
    }
    
    srand(time(NULL) + rank);
    initialize_matrix(local_A, block_size, block_size);
    initialize_matrix(local_B, block_size, block_size);
    zero_matrix(local_C, block_size, block_size);
    
    double **global_A = NULL, **global_B = NULL, **global_C = NULL, **seq_C = NULL;
    
    if (rank == 0) 
    {
        global_A = allocate_matrix(n, n);
        global_B = allocate_matrix(n, n);
        global_C = allocate_matrix(n, n);
        seq_C = allocate_matrix(n, n);
        
        if (global_A == NULL || global_B == NULL || global_C == NULL || seq_C == NULL) 
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        initialize_matrix(global_A, n, n);
        initialize_matrix(global_B, n, n);
        zero_matrix(global_C, n, n);
        zero_matrix(seq_C, n, n);
    }
    
    double* global_A_1d = NULL;
    double* global_B_1d = NULL;
    
    if (rank == 0)
    {
        global_A_1d = (double*)malloc(n * n * sizeof(double));
        global_B_1d = (double*)malloc(n * n * sizeof(double));
        
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                global_A_1d[i * n + j] = global_A[i][j];
                global_B_1d[i * n + j] = global_B[i][j];
            }
        }
    }
    
    MPI_Datatype block_type;
    MPI_Type_vector(block_size, block_size, n, MPI_DOUBLE, &block_type);
    MPI_Type_commit(&block_type);
    
    if (rank == 0) 
    {
        for (int i = 0; i < grid_size; i++) 
        {
            for (int j = 0; j < grid_size; j++) 
            {
                int dest_rank;
                int dest_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
                
                if (dest_rank == 0) 
                {
                    for (int x = 0; x < block_size; x++) 
                    {
                        for (int y = 0; y < block_size; y++) 
                        {
                            local_A[x][y] = global_A[i * block_size + x][j * block_size + y];
                            local_B[x][y] = global_B[i * block_size + x][j * block_size + y];
                        }
                    }
                } else 
                {
                    double* start_A = &global_A_1d[i * block_size * n + j * block_size];
                    double* start_B = &global_B_1d[i * block_size * n + j * block_size];
                    
                    MPI_Send(start_A, 1, block_type, dest_rank, 0, MPI_COMM_WORLD);
                    MPI_Send(start_B, 1, block_type, dest_rank, 1, MPI_COMM_WORLD);
                }
            }
        }
    } else 
    {
        MPI_Recv(&local_A[0][0], block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&local_B[0][0], block_size * block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    cannon_matrix_mult(n, grid_size, block_size, local_A, local_B, local_C, grid_comm);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double parallel_time = MPI_Wtime() - start_time;
    
    if (rank == 0) 
    {
        for (int x = 0; x < block_size; x++) 
        {
            for (int y = 0; y < block_size; y++) 
            {
                global_C[x][y] = local_C[x][y];
            }
        }
        
        for (int i = 0; i < grid_size; i++) 
        {
            for (int j = 0; j < grid_size; j++)
            {
                if (i == 0 && j == 0) continue;
                
                int source_rank;
                int source_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, source_coords, &source_rank);
                
                double* start = &global_C[i * block_size][j * block_size];
                MPI_Recv(start, 1, block_type, source_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else 
    {
        MPI_Send(&local_C[0][0], block_size * block_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
    
    if (rank == 0) 
    {
        char filename[100];
        sprintf(filename, "out/task3_%i_%i.json", size, n);
        FILE *out = fopen(filename, "w");

        fprintf(out, "{\n");
        fprintf(out, "  \"matrix_size\": %d,\n", n);
        fprintf(out, "  \"block_size\": %d,\n", block_size);
        fprintf(out, "  \"exec_time\": %.6f\n", parallel_time);
        fprintf(out, "}\n");

        fclose(out);
    }
    
    MPI_Type_free(&block_type);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}