
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include "timer.h"
#include "spmv.h"

#define DEFAULT_SIZE 16384
#define DEFAULT_DENSITY 0.10

// Población de matriz en formato CSR
unsigned int populate_sparse_matrix_csr(MatrizCSR* csr, unsigned int n, double density, unsigned int seed) {
    unsigned int nnz = 0;
    srand(seed);
    csr->fila_inicio[0] = 0;

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if ((rand() % 100) / 100.0 < density) {
                csr->indices_columnas[nnz] = j;
                csr->val[nnz] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
                nnz++;
            }
        }
        csr->fila_inicio[i + 1] = nnz;
    }
    csr->num_sin_ceros = nnz;
    return nnz;
}

// Población de matriz en formato COO
unsigned int populate_sparse_matrix_coo(MatrizCOO* coo, unsigned int n, double density, unsigned int seed) {
    unsigned int nnz = 0;
    srand(seed);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if ((rand() % 100) / 100.0 < density) {
                coo->filas[nnz] = i;
                coo->columnas[nnz] = j;
                coo->val[nnz] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
                nnz++;
            }
        }
    }
    coo->num_sin_ceros = nnz;
    return nnz;
}

// Población de matriz en formato CSC
unsigned int populate_sparse_matrix_csc(MatrizCSC* csc, unsigned int n, double density, unsigned int seed) {
    unsigned int nnz = 0;
    srand(seed);
    csc->columna_inicio[0] = 0;

    for (unsigned int j = 0; j < n; j++) {
        for (unsigned int i = 0; i < n; i++) {
            if ((rand() % 100) / 100.0 < density) {
                csc->filas[nnz] = i;
                csc->val[nnz] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
                nnz++;
            }
        }
        csc->columna_inicio[j + 1] = nnz;
    }
    csc->num_sin_ceros = nnz;
    return nnz;
}

unsigned int populate_vector(double vec[], unsigned int size, unsigned int seed) {
    srand(seed);
    for (unsigned int i = 0; i < size; i++) {
        vec[i] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
    }
    return size;
}

// Multiplicación densa sin MKL
void dense_mv(const double* mat, const double* vec, double* result, unsigned int size) {
    for (unsigned int i = 0; i < size; i++) {
        result[i] = 0.0;
        for (unsigned int j = 0; j < size; j++) {
            result[i] += mat[i * size + j] * vec[j];
        }
    }
}

// Multiplicación dispersa CSR sin MKL
void sparse_csr_mv(const MatrizCSR* csr, const double* vec, double* result) {
    for (int i = 0; i < csr->num_filas; i++) {
        result[i] = 0.0;
        for (int idx = csr->fila_inicio[i]; idx < csr->fila_inicio[i + 1]; idx++) {
            result[i] += csr->val[idx] * vec[csr->indices_columnas[idx]];
        }
    }
}

// Multiplicación dispersa COO sin MKL
void sparse_coo_mv(const MatrizCOO* coo, const double* vec, double* result) {
    for (int i = 0; i < coo->num_filas; i++) {
        result[i] = 0.0;
    }
    for (int idx = 0; idx < coo->num_sin_ceros; idx++) {
        result[coo->filas[idx]] += coo->val[idx] * vec[coo->columnas[idx]];
    }
}

// Multiplicación dispersa CSC sin MKL
void sparse_csc_mv(const MatrizCSC* csc, const double* vec, double* result) {
    for (int i = 0; i < csc->num_filas; i++) {
        result[i] = 0.0;
    }
    for (int j = 0; j < csc->num_columnas; j++) {
        for (int idx = csc->columna_inicio[j]; idx < csc->columna_inicio[j + 1]; idx++) {
            result[csc->filas[idx]] += csc->val[idx] * vec[j];
        }
    }
}

// Funciones de MKL para CSR, COO, y CSC
void mkl_sparse_csr_mv(const MatrizCSR* csr, const double* vec, double* result) {
    sparse_matrix_t csr_matrix;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_sparse_d_create_csr(&csr_matrix, SPARSE_INDEX_BASE_ZERO, csr->num_filas, csr->num_columnas,
                            csr->fila_inicio, csr->fila_inicio + 1, csr->indices_columnas, csr->val);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_matrix, descr, vec, 0.0, result);
    mkl_sparse_destroy(csr_matrix);
}

void mkl_sparse_coo_mv(const MatrizCOO* coo, const double* vec, double* result) {
    sparse_matrix_t coo_matrix;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_sparse_d_create_coo(&coo_matrix, SPARSE_INDEX_BASE_ZERO, coo->num_filas, coo->num_columnas,
                            coo->num_sin_ceros, coo->filas, coo->columnas, coo->val);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, coo_matrix, descr, vec, 0.0, result);
    mkl_sparse_destroy(coo_matrix);
}

void mkl_sparse_csc_mv(const MatrizCSC* csc, const double* vec, double* result) {
    sparse_matrix_t csc_matrix;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Corrección en la llamada de mkl_sparse_d_create_csc
    mkl_sparse_d_create_csc(&csc_matrix, SPARSE_INDEX_BASE_ZERO, 
                            csc->num_columnas, csc->num_filas,
                            csc->columna_inicio, csc->columna_inicio + 1, 
                            csc->filas, csc->val);

    // Realiza la multiplicación matriz-vector usando MKL
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csc_matrix, descr, vec, 0.0, result);

    // Libera la matriz dispersa
    mkl_sparse_destroy(csc_matrix);
}


int main(int argc, char *argv[]) {
    int size;
    double density;
    
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

    MatrizCSR matriz_csr = {malloc((size + 1) * sizeof(int)), malloc(size * size * sizeof(int)), malloc(size * size * sizeof(double)), size, size, 0};
    MatrizCOO matriz_coo = {malloc(size * size * sizeof(int)), malloc(size * size * sizeof(int)), malloc(size * size * sizeof(double)), size, size, 0};
    MatrizCSC matriz_csc = {malloc((size + 1) * sizeof(int)), malloc(size * size * sizeof(int)), malloc(size * size * sizeof(double)), size, size, 0};
    
    double *mat = (double *)malloc(size * size * sizeof(double));
    double *vec = (double *)malloc(size * sizeof(double));
    double *result = (double *)malloc(size * sizeof(double));
    
    populate_sparse_matrix_csr(&matriz_csr, size, density, 1);
    populate_sparse_matrix_coo(&matriz_coo, size, density, 1);
    populate_sparse_matrix_csc(&matriz_csc, size, density, 1);
    populate_vector(vec, size, 2);

    timeinfo start, now;

    // Dense multiplicación
    timestamp(&start);
    dense_mv(mat, vec, result, size);
    timestamp(&now);
    printf("Time taken by dense matrix-vector product without MKL: %ld ms\n", diff_milli(&start, &now));

    timestamp(&start);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, mat, size, vec, 1, 0.0, result, 1);
    timestamp(&now);
    printf("Time taken by dense matrix-vector product with MKL: %ld ms\n", diff_milli(&start, &now));

    // CSR multiplicación
    timestamp(&start);
    sparse_csr_mv(&matriz_csr, vec, result);
    timestamp(&now);
    printf("Time taken by CSR matrix-vector product without MKL: %ld ms\n", diff_milli(&start, &now));

    timestamp(&start);
    mkl_sparse_csr_mv(&matriz_csr, vec, result);
    timestamp(&now);
    printf("Time taken by CSR matrix-vector product with MKL: %ld ms\n", diff_milli(&start, &now));

    // COO multiplicación
    timestamp(&start);
    sparse_coo_mv(&matriz_coo, vec, result);
    timestamp(&now);
    printf("Time taken by COO matrix-vector product without MKL: %ld ms\n", diff_milli(&start, &now));

    timestamp(&start);
    mkl_sparse_coo_mv(&matriz_coo, vec, result);
    timestamp(&now);
    printf("Time taken by COO matrix-vector product with MKL: %ld ms\n", diff_milli(&start, &now));

    // CSC multiplicación
    timestamp(&start);
    sparse_csc_mv(&matriz_csc, vec, result);
    timestamp(&now);
    printf("Time taken by CSC matrix-vector product without MKL: %ld ms\n", diff_milli(&start, &now));

    timestamp(&start);
    mkl_sparse_csc_mv(&matriz_csc, vec, result);
    timestamp(&now);
    printf("Time taken by CSC matrix-vector product with MKL: %ld ms\n", diff_milli(&start, &now));

    // Liberar memoria
    free(matriz_csr.fila_inicio);
    free(matriz_csr.indices_columnas);
    free(matriz_csr.val);
    free(matriz_coo.filas);
    free(matriz_coo.columnas);
    free(matriz_coo.val);
    free(matriz_csc.columna_inicio);
    free(matriz_csc.filas);
    free(matriz_csc.val);
    free(mat);
    free(vec);
    free(result);

    return 0;
}
