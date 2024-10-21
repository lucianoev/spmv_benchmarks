#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_cblas.h>      // CBLAS in GSL (the GNU Scientific Library)
#include <gsl/gsl_spmatrix.h>   // GSL sparse matrix functions
#include <gsl/gsl_vector.h>     // GSL vector functions
#include "timer.h"
#include "spmv.h"

#define DEFAULT_SIZE 16384
#define DEFAULT_DENSITY 0.10

unsigned int populate_sparse_matrix(double mat[], unsigned int n, double density, unsigned int seed)
{
    unsigned int nnz = 0;

    srand(seed);

    for (unsigned int i = 0; i < n * n; i++) {
        if ((rand() % 100) / 100.0 < density) {
            // Get a pseudorandom value between -9.99 and 9.99
            mat[i] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
            nnz++;
        } else {
            mat[i] = 0;
        }
    }

    return nnz;
}

unsigned int populate_vector(double vec[], unsigned int size, unsigned int seed)
{
    srand(seed);

    for (unsigned int i = 0; i < size; i++) {
        vec[i] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
    }

    return size;
}

int is_nearly_equal(double x, double y)
{
    const double epsilon = 1e-5; // some small number
    return fabs(x - y) <= epsilon * fabs(x);
}

unsigned int check_result(double ref[], double result[], unsigned int size)
{
    for(unsigned int i = 0; i < size; i++) {
        if (!is_nearly_equal(ref[i], result[i]))
            return 0;
    }
    return 1;
}

void gsl_spmatrix_vector_multiply(const gsl_spmatrix *spmat, const gsl_vector *vec, gsl_vector *result)
{
    for (size_t i = 0; i < spmat->size1; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < spmat->size2; j++) {
            sum += gsl_spmatrix_get(spmat, i, j) * gsl_vector_get(vec, j);
        }
        gsl_vector_set(result, i, sum);
    }
}


// Función para convertir matriz densa a CSR
MatrizCSR convert_to_csr(const double* mat, int num_filas, int num_columnas) {
    // Contar elementos no cero
    int nnz = 0;
    for (int i = 0; i < num_filas; i++) {
        for (int j = 0; j < num_columnas; j++) {
            if (mat[i * num_columnas + j] != 0) {
                nnz++;
            }
        }
    }

    // Asignar memoria para CSR
    MatrizCSR matriz_csr;
    matriz_csr.num_filas = num_filas;
    matriz_csr.num_columnas = num_columnas;
    matriz_csr.num_sin_ceros = nnz;
    matriz_csr.fila_inicio = (int*)malloc((num_filas + 1) * sizeof(int));
    matriz_csr.indices_columnas = (int*)malloc(nnz * sizeof(int));
    matriz_csr.val = (double*)malloc(nnz * sizeof(double));

    // Llenar la estructura CSR
    int idx = 0; // Índice para elementos no cero
    matriz_csr.fila_inicio[0] = 0; 
    for (int i = 0; i < num_filas; i++) {
        for (int j = 0; j < num_columnas; j++) {
            if (mat[i * num_columnas + j] != 0) {
                matriz_csr.indices_columnas[idx] = j;
                matriz_csr.val[idx] = mat[i * num_columnas + j];
                idx++;
            }
        }
        matriz_csr.fila_inicio[i + 1] = idx; // Guardar índice de inicio de la siguiente fila
    }

    return matriz_csr;
}


//Función convertir matriz a COO
MatrizCOO convert_to_coo(const double* mat, int num_filas, int num_columnas) {
    
    int nnz = 0;

    // Contar elementos no cero
    for (int i = 0; i < num_filas; i++) {
        for (int j = 0; j < num_columnas; j++) {
            if (mat[i * num_columnas + j] != 0) {
                nnz++;
            }
        }
    }

    // Asignar memoria
    MatrizCOO coo;
    coo.filas = (int*)malloc(nnz * sizeof(int));
    coo.columnas = (int*)malloc(nnz * sizeof(int));
    coo.val = (double*)malloc(nnz * sizeof(double));
    coo.num_filas = num_filas;
    coo.num_columnas = num_columnas;
    coo.num_sin_ceros = nnz;

    // Llenar COO
    int idx = 0;
    for (int i = 0; i < num_filas; i++) {
        for (int j = 0; j < num_columnas; j++) {
            if (mat[i * num_columnas + j] != 0) {
                coo.filas[idx] = i;
                coo.columnas[idx] = j;
                coo.val[idx] = mat[i * num_columnas + j];
                idx++;
            }
        }
    }

    return coo;
}


//Función convertir matriz a CSC
MatrizCSC convert_to_csc(const double* mat, int num_filas, int num_columnas) {
   
    int nnz = 0;

    // Contar elementos no cero
    for (int j = 0; j < num_columnas; j++) {
        for (int i = 0; i < num_filas; i++) {
            if (mat[i * num_columnas + j] != 0) {
                nnz++;
            }
        }
    }

    // Asignar memoria
    MatrizCSC csc;
    csc.columna_inicio = (int*)malloc((num_columnas + 1) * sizeof(int));
    csc.filas = (int*)malloc(nnz * sizeof(int));
    csc.val = (double*)malloc(nnz * sizeof(double));
    csc.num_filas = num_filas;
    csc.num_columnas = num_columnas;
    csc.num_sin_ceros = nnz;

    // Llenar CSC
    int current_nnz = 0;
    csc.columna_inicio[0] = 0;
    for (int j = 0; j < num_columnas; j++) {
        for (int i = 0; i < num_filas; i++) {
            if (mat[i * num_columnas + j] != 0) {
                csc.filas[current_nnz] = i;
                csc.val[current_nnz] = mat[i * num_columnas + j];
                current_nnz++;
            }
        }
        csc.columna_inicio[j + 1] = current_nnz;
    }

    return csc;
}



// Función principal
int main(int argc, char *argv[])
{
    int size;        // number of rows and cols (size x size matrix)
    double density;  // aprox. ratio of non-zero values

    gsl_spmatrix *gsl_spmat = gsl_spmatrix_alloc(size, size); // Declarar la matriz dispersa
    gsl_vector *gsl_vec = gsl_vector_alloc(size);             // Declarar el vector
    gsl_vector *gsl_result = gsl_vector_alloc(size);  

    if (argc < 2) {
        size = DEFAULT_SIZE;
        density = DEFAULT_DENSITY;
    } else if (argc < 3) {
        size = atoi(argv[1]);
        density = DEFAULT_DENSITY;
    } else {
        size = atoi(argv[1]);
        density = atof(argv[2]);
    }

    double *mat, *vec, *refsol, *mysol;
    MatrizCSR matriz_csr;  // Variable para la matriz CSR
    MatrizCSC csc;
    MatrizCOO coo;

    mat = (double *) malloc(size * size * sizeof(double));
    vec = (double *) malloc(size * sizeof(double));
    refsol = (double *) malloc(size * sizeof(double));
    mysol = (double *) malloc(size * sizeof(double));

    unsigned int nnz = populate_sparse_matrix(mat, size, density, 1);
    populate_vector(vec, size, 2);

    printf("Matriz size: %d x %d (%d elements)\n", size, size, size*size);
    printf("%d non-zero elements (%.2lf%%)\n\n", nnz, (double) nnz / (size*size) * 100.0);

    // Cálculo denso usando CBLAS
    printf("Dense computation\n----------------\n");
    timeinfo start, now;
    timestamp(&start);

    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, mat, size, vec, 1, 0.0, refsol, 1);

    timestamp(&now);
    printf("Time taken by CBLAS dense computation: %ld ms\n", diff_milli(&start, &now));

    // Usando tu propia implementación densa
    timestamp(&start);
    my_dense(size, mat, vec, mysol);
    timestamp(&now);
    printf("Time taken by my dense matrix-vector product: %ld ms\n", diff_milli(&start, &now));

    if (check_result(refsol, mysol, size) == 1)
        printf("Result is ok!\n");
    else
        printf("Result is wrong!\n");

    //SpMV: Producto matriz CSR
    matriz_csr = convert_to_csr(mat, size, size); // Convertir mat a formato CSR

    //implementación CSR
    timestamp(&start);
    my_sparse_csr(&matriz_csr, vec, mysol); // Multiplicación CSR
    timestamp(&now);
    printf("Time taken by my sparse CSR matrix-vector product: %ld ms\n", diff_milli(&start, &now));

    // Comprobar el resultado CSR
    if (check_result(refsol, mysol, size) == 1)
        printf("Sparse result is ok!\n");
    else
        printf("Sparse result is wrong!\n");


    //SpMV: Producto matriz CSC
    csc = convert_to_csc(mat, size, size); // Convertir mat a formato CSC

    //implementación CSC
    timestamp(&start);
    my_sparse_csc(&csc, vec, mysol); // Multiplicación CSC
    timestamp(&now);
    printf("Time taken by my sparse CSC matrix-vector product: %ld ms\n", diff_milli(&start, &now));

    // Comprobar el resultado CSC
    if (check_result(refsol, mysol, size) == 1)
        printf("Sparse result is ok!\n");
    else
        printf("Sparse result is wrong!\n");


    //SpMV: Producto matriz COO
    coo = convert_to_coo(mat, size, size); // Convertir mat a formato COO

    //implementación COO
    timestamp(&start);
    my_sparse_coo(&coo, vec, mysol); // Multiplicación COO
    timestamp(&now);
    printf("Time taken by my sparse COO matrix-vector product: %ld ms\n", diff_milli(&start, &now));

    // Comprobar el resultado COO
    if (check_result(refsol, mysol, size) == 1)
        printf("Sparse result is ok!\n");
    else
        printf("Sparse result is wrong!\n");




    // Usando la matriz dispersa de GSL y multiplicación
    gsl_spmat = gsl_spmatrix_alloc(size, size);

    // Convertir la matriz densa en dispersa en GSL
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double val = mat[i * size + j];
            if (val != 0) {
                gsl_spmatrix_set(gsl_spmat, i, j, val);
            }
        }
    }

    gsl_vec = gsl_vector_alloc(size);
    gsl_result = gsl_vector_alloc(size);

    // Llenar el vector GSL
    for (int i = 0; i < size; i++) {
        gsl_vector_set(gsl_vec, i, vec[i]);
    }

    // Multiplicación dispersa usando GSL
    timestamp(&start);
    gsl_spmatrix_vector_multiply(gsl_spmat, gsl_vec, gsl_result);
    timestamp(&now);
    printf("Time taken by GSL sparse matrix-vector product: %ld ms\n", diff_milli(&start, &now));

    // Comprobar el resultado de GSL
    int gsl_ok = 1;
    for (int i = 0; i < size; i++) {
        if (!is_nearly_equal(gsl_vector_get(gsl_result, i), refsol[i])) {
            gsl_ok = 0;
            break;
        }
    }

    if (gsl_ok)
        printf("Sparse result with GSL is ok!\n");
    else
        printf("Sparse result with GSL is wrong!\n");

    // Liberar recursos
    gsl_spmatrix_free(gsl_spmat);
    gsl_vector_free(gsl_vec);
    gsl_vector_free(gsl_result);
    free(mat);
    free(vec);
    free(refsol);
    free(mysol);
    free(matriz_csr.fila_inicio);
    free(matriz_csr.indices_columnas);
    free(matriz_csr.val);


    return 0;
}
