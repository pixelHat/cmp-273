#include <lapacke.h>
#include <starpu.h>
#include <time.h>
#include "mutils.h"

// returns the minimum of a and b
int MIN(int a, int b)
{
    return a < b ? a : b;
}

// returns the maxinum of a and b
int MAX(int a, int b)
{
    return a > b ? a : b;
}

/* Struct that will hold the input parameters for a block QR operation */
typedef struct
{
    int nb;          // Block size
    int ldt;         // Reflector leading dimension
    int lda;         // Leading dimension of matrix A
    int m, n;        // Rows and cols of A
} block_mtx;

/* ============= DGEQRT ============= */

void cpu_dgeqrt(void *buffers[], void *cl_arg)
{
    int info = LAPACKE_dgeqrt(
        LAPACK_ROW_MAJOR,
        STARPU_MATRIX_GET_NX(buffers[0]),
        STARPU_MATRIX_GET_NY(buffers[0]),
        STARPU_MATRIX_GET_NY(buffers[0]), // block_size
        (double*) STARPU_MATRIX_GET_PTR(buffers[0]),
        STARPU_MATRIX_GET_LD(buffers[0]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[1]),
        STARPU_MATRIX_GET_LD(buffers[1])
    );
        check_err(info, "LAPACKE_dgeqrt");
}

/* DGEQRT writes on 2 blocks diagonal A and T */
struct starpu_codelet dgeqrt_cl =
    {
        .cpu_funcs = { cpu_dgeqrt },
        .cpu_funcs_name = { "cpu_dgeqrt" },
        .name = "lapack_dgeqrt",
        .modes = {STARPU_RW, STARPU_RW},
        .nbuffers = 2
    };



/* ============= DLARFB ============= */

void cpu_dlarfb(void *buffers[], void *cl_arg)
{
    /* Call Lapack routine */
    int info = LAPACKE_dlarfb(
        LAPACK_ROW_MAJOR,
        'L', 'T', 'F', 'C',
        STARPU_MATRIX_GET_NX(buffers[0]),
        STARPU_MATRIX_GET_NY(buffers[0]),
        STARPU_MATRIX_GET_NY(buffers[0]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[0]),
        STARPU_MATRIX_GET_LD(buffers[0]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[1]),
        STARPU_MATRIX_GET_LD(buffers[1]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[2]),
        STARPU_MATRIX_GET_LD(buffers[2])
    );
    check_err(info, "LAPACKE_dlarfb");
}

/* DLARFB writes on 2 blocks diagonal A and T */
struct starpu_codelet dlarfb_cl =
    {
        .cpu_funcs = { cpu_dlarfb },
        .cpu_funcs_name = { "cpu_dlarfb" },
        .name = "lapack_dlarfb",
        .modes = {STARPU_RW, STARPU_RW, STARPU_RW},
        .nbuffers = 3,
    };


/* ============= DTPQRT ============= */

void cpu_dtpqrt(void *buffers[], void *cl_arg)
{
    /* Call Lapack routine */
    int info = LAPACKE_dtpqrt(
        LAPACK_ROW_MAJOR,
        STARPU_MATRIX_GET_NX(buffers[0]),
        STARPU_MATRIX_GET_NY(buffers[0]),
        0,
        STARPU_MATRIX_GET_NY(buffers[0]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[0]),
        STARPU_MATRIX_GET_LD(buffers[0]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[1]),
        STARPU_MATRIX_GET_LD(buffers[1]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[2]),
        STARPU_MATRIX_GET_LD(buffers[2])

    );
    check_err(info, "LAPACKE_dtpqrt");
}

/* DTPQRT writes on 2 blocks diagonal ITH and T */
struct starpu_codelet dtpqrt_cl =
    {
        .cpu_funcs = { cpu_dtpqrt },
        .cpu_funcs_name = { "cpu_dtpqrt" },
        .name = "lapack_dtpqrt",
        .modes = {STARPU_RW, STARPU_RW, STARPU_RW},
        .nbuffers = 3,
    };

/* ============= DTPMQRT ============= */

void cpu_dtpmqrt(void *buffers[], void *cl_arg)
{
    /* Call Lapack routine */
    int info = LAPACKE_dtpmqrt(
        LAPACK_ROW_MAJOR, 'L', 'T',
        STARPU_MATRIX_GET_NX(buffers[0]),
        STARPU_MATRIX_GET_NY(buffers[0]),
        STARPU_MATRIX_GET_NY(buffers[0]),
        0,
        STARPU_MATRIX_GET_NY(buffers[0]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[0]),
        STARPU_MATRIX_GET_LD(buffers[0]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[1]),
        STARPU_MATRIX_GET_LD(buffers[1]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[2]),
        STARPU_MATRIX_GET_LD(buffers[2]),
        (double*) STARPU_MATRIX_GET_PTR(buffers[3]),
        STARPU_MATRIX_GET_LD(buffers[3])
    );
    check_err(info, "LAPACKE_dtpmqrt");
}

/* DTPMQRT writes on 2 blocks diagonal ITH and JTH and reads I and T2*/
struct starpu_codelet dtpmqrt_cl =
    {
        .cpu_funcs = { cpu_dtpmqrt },
        .cpu_funcs_name = { "cpu_dtpmqrt" },
        .name = "lapack_dtpmqrt",
        .modes = {STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW},
        .nbuffers = 4,
    };

int main(int argc, char const **argv)
{
    struct timeval start, end, start_compute, end_compute;
    double time, compute_time;
    gettimeofday(&start, NULL);

    check_params(argc, argv);

    unsigned long int m, n;     // base dimensions
    int min_max, seed;          // parameter for amtrix generation
    int nb = atoi(argv[2]);     // block size

    // read or generate the matrix
    double* A;
    if(argc == 3) {
        A = read_mat(argv[1], &m, &n);
    } else {
        m = n = (unsigned long int) atoi(argv[1]);
        seed = atoi(argv[3]);
        min_max = atoi(argv[4]);
        srand(seed);
        A = gen_random_mat(m, min_max);
        // print_m(m, n, A, "A");
    }


    int b = ceil(n/nb);             // number of col blocks
    int mb = ceil(m/nb);            // number of row blocks
    int lb = n - (b-1)*nb;          // size of last block
    int i, j, k, z;                 // block control parameters
    int ib, ib2, jb, kb, zb, zit;   // iterators for loop control
    int info;                       // error control
    int lda = n;                    // leading dimension for calculating submatrices

    // size for the T reflector accumulator
    int ldt = lda;
    double* T = (double*) calloc (m*n, sizeof(double));

    /* initialize StarPU */
    info = starpu_init(NULL);

    // handle_A
    starpu_data_handle_t **handle_A =
        malloc(mb*sizeof(starpu_data_handle_t *));

    int block_count = mb;
    int block_size = nb;
    for (int i = 0; i < block_count; i++) {
        handle_A[i] = malloc(block_count*sizeof(starpu_data_handle_t));

        for (int j = 0; j < block_count; j++) {
            // each block is registered as a matrix
            starpu_matrix_data_register(
                &handle_A[i][j],                      // handle
                STARPU_MAIN_RAM,                    // memory node
                (uintptr_t)(A+(j*lda+i)*block_size), // pointer
                lda,                                 // leading dimension
                MIN(block_size, n-i*block_size),    // row count
                MIN(block_size, n-j*block_size),    // column count
                sizeof(double));                    // element size
        }
    }

    // handle_T
    starpu_data_handle_t **handle_T =
        malloc(block_count*sizeof(starpu_data_handle_t *));

    for (int i = 0; i < block_count; i++) {
        handle_T[i] = malloc(block_count*sizeof(starpu_data_handle_t));

        for (int j = 0; j < block_count; j++) {
            // each block is registered as a matrix
            starpu_matrix_data_register(
                &handle_T[i][j],                      // handle
                STARPU_MAIN_RAM,                    // memory node
                (uintptr_t)(T+(j*ldt+i)*block_size), // pointer
                ldt,                                 // leading dimension
                MIN(block_size, n-i*block_size),    // row count
                MIN(block_size, n-j*block_size),    // column count
                sizeof(double));                    // element size
        }
    }

    gettimeofday(&start_compute, NULL);
    int number_of_tasks = 0;

    printf("starting...\n");
    for(int k=0; k<min(mb, b); k++) {
        starpu_iteration_push(k);
        kb = (k*b + k) * 2;   // diagonal block

        starpu_task_insert(&dgeqrt_cl,
                           STARPU_RW, handle_A[k][k],
                           STARPU_RW, handle_T[k][k],
                           0);
        number_of_tasks++;

        // update diagonal right blocks
        for(int j=k+1; j<b; j++) {
            starpu_task_insert(
                &dlarfb_cl,
                STARPU_RW, handle_A[k][k],
                STARPU_RW, handle_T[k][k],
                STARPU_RW, handle_A[j][k],
                0);
            number_of_tasks++;
        }

        // eliminate blocks below the diagonal
        for(int i=k+1; i<mb; i++) {
            starpu_task_insert(
                &dtpqrt_cl,
                STARPU_RW, handle_A[k][k],
                STARPU_RW, handle_A[k][i],
                STARPU_RW, handle_T[k][i],
                0);
            number_of_tasks++;

            // update k-th line with i-th line
            for(int zit=k+1, z=1; zit<b; zit++, z++) {
                starpu_task_insert(
                    &dtpmqrt_cl,
                    STARPU_RW, handle_A[k][i],
                    STARPU_RW, handle_T[k][i],
                    STARPU_RW, handle_A[zit][k],
                    STARPU_RW, handle_A[k+z][i],
                    0);
                number_of_tasks++;
            }
        }
        starpu_iteration_pop();
    }

    /* Wait untill all tasks are executed */
    printf("Waiting...\n");
    starpu_task_wait_for_all();
    printf("Done...\n");
    gettimeofday(&end_compute, NULL);
    compute_time =  (end_compute.tv_sec - start_compute.tv_sec) * 1000000L;
    compute_time += (end_compute.tv_usec - start_compute.tv_usec);

    // free allocated resources
    for (int i = 0; i < block_count; i++) {
        for (int j = 0; j < block_count; j++) {

            // The data handles must be unregistered. The main thread waits
            // until all related tasks have been completed and the data is
            // copied back to its original location.
            starpu_data_unregister(handle_A[i][j]);
            starpu_data_unregister(handle_T[i][j]);
        }
        free(handle_A[i]);
        free(handle_T[i]);
    }
    free(handle_A);
    free(handle_T);

    // rint_m(m, n, A, "");
    /* terminate StarPU */
    starpu_shutdown();
    gettimeofday(&end, NULL);
    time =  (end.tv_sec - start.tv_sec) * 1000000L;
    time += (end.tv_usec - start.tv_usec);
    printf("Application time in seconds %lf\n", time/1000000L);
    printf("Application compute time in seconds %lf\n", compute_time/1000000L);

    printf("Number of tasks %d", number_of_tasks);
    return 0;
}
