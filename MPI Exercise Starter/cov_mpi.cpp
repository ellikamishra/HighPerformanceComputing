/*
We will compute the covariance between two large vectors of numbers in parallel.
The vectors are randomly initialized by the root process.
Because the vectors are large, the work is divided among multiple processes.
Each process will receive a chunk of the vectors, compute partial results, and then combine them with the results from the other processes.
The root process is responsible to print the final output. 
*/

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int myrank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &p); //size of communicator

    // --- Step 1: Parse command line argument ---
    // Only the root processes perform this
    int n;//total lenght of vector
    if (myrank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: mpirun -np <p> ./a.out <vector_length>" << std::endl;
            n = 0;
        } else {
            n = std::atoi(argv[1]);
        }

        if (n <= 1) {
            std::cerr << "Vector length must be > 1" << std::endl;
            MPI_Finalize();
            return 1;
        }
    }

    

    // --- Step 2: Root initializes full vectors --- still root has all the data, so far root has everything
    std::vector<double> x_global, y_global;
    if (myrank == 0) {
        x_global.resize(n);
        y_global.resize(n);
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            x_global[i] = (double)rand() / RAND_MAX; // [0,1]
            y_global[i] = (double)rand() / RAND_MAX;
        }
    }

    // --- Step 3: Root computes counts and displacements based on a data distribution policy
    // If n%p=0, each process should get n/p elements
    // If n%p!=0, some processes will get 1 more enetry
    // Any distribution is fine, but this is load-balanced
    // Each process can count it if they know the distribution policy 
    /*now root has to broadcast total length of vector to all, each process will pe n/p*/
    std::vector<int> counts, displs;
    if (myrank == 0) {
        counts.resize(p);
        displs.resize(p);

        int base = n / p;
        int rem  = n % p;
        for (int i = 0; i < p; i++) {
            if(i<rem)
                counts[i] = base+1;//how many data items to process i
            else
                counts[i] = base;
            displs[i] = (i == 0 ? 0 : displs[i-1] + counts[i-1]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  // sync before timing 
    double t_start = MPI_Wtime();

    // --- Step 4: Broadcast len to all processes. 
    
    /**** Write a broadcast Call here, all will get length now, local length for all set now */
MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);//everyone gets n
    // --- Step 5: allocate local buffers

    int n_local;
    if(myrank< (n%p) ) 
        n_local = n/p + 1;//first few get extra 
    else n_local = n/p;
    std::vector<double> x_local(n_local), y_local(n_local);

//scatterv: different portion of data to different process, needs to know how much data to send, need counter and displacement
//scatter:same amount of data to all processes
    // --- Step 6: Root create data and scatter to other processes 
   
    /**** Write a scatter Calls here 
    
    two scatterv calls one for x and one for y
    
    */
    MPI_Scatterv(x_global.data(), counts.data(), displs.data(), MPI_DOUBLE, x_local.data(), n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y_global.data(), counts.data(), displs.data(), MPI_DOUBLE, y_local.data(), n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // --- Step 7: Local sums location computation in every process---
    double local_sum_x = 0.0, local_sum_y = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:local_sum_x, local_sum_y) //call to allreduce
    for (int i = 0; i < n_local; i++) {
        local_sum_x += x_local[i];
        local_sum_y += y_local[i];
    }

    // --- Step 8: Global means via allreduce ---
    double global_sum_x, global_sum_y;

    /**** Write a reductionall Call here */
    global_sum_x = 0.0;
    global_sum_y = 0.0;
    //1 everyone contributing 1 value so it should be 1 count
    MPI_Allreduce(&local_sum_x, &global_sum_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sum_y, &global_sum_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double mean_x = global_sum_x / n;
    double mean_y = global_sum_y / n;

    // --- Step 9: Local covariance contribution ---
    double local_cov = 0.0;
    #pragma omp parallel for reduction(+:local_cov)
    for (int i = 0; i < n_local; i++) {
        local_cov += (x_local[i] - mean_x) * (y_local[i] - mean_y);
    }

    // --- Step 10: Reduce to root for printing ---
    double global_cov;
    
     /**** Write a reduction Call here */
    //MPI_reduce(&local_cov, &global_cov, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_reduce(&local_cov, &global_cov, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (myrank == 0) {
        global_cov /= (n - 1);
        std::cout << "Covariance = " << global_cov << std::endl;
        std::cout << "Elapsed time = " << (t_end - t_start) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
