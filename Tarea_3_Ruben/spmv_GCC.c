#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include "spmv.h"

#define DEFAULT_SIZE 1024
#define DEFAULT_DENSITY 0.10

unsigned int populate_sparse_matrix(double mat[], unsigned int n, double density, unsigned int seed)
{
    unsigned int nnz = 0;

    srand(seed);

    for (unsigned int i = 0; i < n * n; i++)
    {
        if ((rand() % 100) / 100.0 < density)
        {
            mat[i] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
            nnz++;
        }
        else
        {
            mat[i] = 0;
        }
    }

    return nnz;
}

unsigned int populate_vector(double vec[], unsigned int size, unsigned int seed)
{
    srand(seed);

    for (unsigned int i = 0; i < size; i++)
    {
        vec[i] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
    }

    return size;
}

int is_nearly_equal(double x, double y)
{
    const double epsilon = 1e-5;
    return fabs(x - y) <= epsilon * fabs(x);
}

unsigned int check_result(double ref[], double result[], unsigned int size)
{
    for (unsigned int i = 0; i < size; i++)
    {
        if (!is_nearly_equal(ref[i], result[i]))
            return 0;
    }
    return 1;
}

MatrizCSR convert_to_csr(const double *mat, int num_filas, int num_columnas) {
    MatrizCSR csr;
    int nnz = 0;

    csr.fila_inicio = (int *)malloc((num_filas + 1) * sizeof(int));
    csr.indices_columnas = (int *)malloc(num_filas * num_columnas * sizeof(int));
    csr.val = (double *)malloc(num_filas * num_columnas * sizeof(double));

    csr.fila_inicio[0] = 0;
    for (int i = 0; i < num_filas; i++) {
        for (int j = 0; j < num_columnas; j++) {
            double value = mat[i * num_columnas + j];
            if (value != 0.0) {
                csr.indices_columnas[nnz] = j;
                csr.val[nnz] = value;
                nnz++;
            }
        }
        csr.fila_inicio[i + 1] = nnz;
    }

    csr.num_filas = num_filas;
    csr.num_columnas = num_columnas;
    csr.num_sin_ceros = nnz;

    return csr;
}
// Conversión de matriz densa a formato CSC
MatrizCSC convert_to_csc(const double *mat, int num_filas, int num_columnas) {
    MatrizCSC csc;
    int nnz = 0;

    csc.columna_inicio = (int *)malloc((num_columnas + 1) * sizeof(int));
    csc.filas = (int *)malloc(num_filas * num_columnas * sizeof(int));
    csc.val = (double *)malloc(num_filas * num_columnas * sizeof(double));

    csc.columna_inicio[0] = 0;
    for (int j = 0; j < num_columnas; j++) {
        for (int i = 0; i < num_filas; i++) {
            double value = mat[i * num_columnas + j];
            if (value != 0.0) {
                csc.filas[nnz] = i;
                csc.val[nnz] = value;
                nnz++;
            }
        }
        csc.columna_inicio[j + 1] = nnz;
    }

    csc.num_filas = num_filas;
    csc.num_columnas = num_columnas;
    csc.num_sin_ceros = nnz;

    return csc;
}

// Conversión de matriz densa a formato COO
MatrizCOO convert_to_coo(const double *mat, int num_filas, int num_columnas) {
    MatrizCOO coo;
    int nnz = 0;

    coo.filas = (int *)malloc(num_filas * num_columnas * sizeof(int));
    coo.columnas = (int *)malloc(num_filas * num_columnas * sizeof(int));
    coo.val = (double *)malloc(num_filas * num_columnas * sizeof(double));

    for (int i = 0; i < num_filas; i++) {
        for (int j = 0; j < num_columnas; j++) {
            double value = mat[i * num_columnas + j];
            if (value != 0.0) {
                coo.filas[nnz] = i;
                coo.columnas[nnz] = j;
                coo.val[nnz] = value;
                nnz++;
            }
        }
    }

    coo.num_filas = num_filas;
    coo.num_columnas = num_columnas;
    coo.num_sin_ceros = nnz;

    return coo;
}
// Función principal
int main(int argc, char *argv[])
{
    int size;       // Tamaño de la matriz (size x size)
    double density; // Densidad de valores no cero

    if (argc < 2)
    {
        size = DEFAULT_SIZE;
        density = DEFAULT_DENSITY;
    }
    else if (argc < 3)
    {
        size = atoi(argv[1]);
        density = DEFAULT_DENSITY;
    }
    else
    {
        size = atoi(argv[1]);
        density = atof(argv[2]);
    }

    if (size <= 0)
    {
        fprintf(stderr, "Error: El tama\u00f1o de la matriz debe ser un entero positivo.\n");
        return 1;
    }

    double *mat, *vec, *refsol, *mysol;
    MatrizCSR matriz_csr;
    MatrizCSC csc;
    MatrizCOO coo;

    mat = (double *)malloc(size * size * sizeof(double));
    vec = (double *)malloc(size * sizeof(double));
    refsol = (double *)malloc(size * sizeof(double));
    mysol = (double *)malloc(size * sizeof(double));

    unsigned int nnz = populate_sparse_matrix(mat, size, density, 1);
    populate_vector(vec, size, 2);

    printf("Matriz size: %d x %d (%d elements)\n", size, size, size * size);
    printf("%d non-zero elements (%.2lf%%)\n\n", nnz, (double)nnz / (size * size) * 100.0);

    // C\u00e1lculo denso usando CBLAS
    printf("Dense computation\n----------------\n");
    timeinfo start, now;
    timestamp(&start);

    my_dense(size, mat, vec, refsol);

    timestamp(&now);
    printf("Time taken by dense matrix-vector product: %ld ms\n", diff_milli(&start, &now));

    // Verificaci\u00f3n de resultado
    if (check_result(refsol, refsol, size) == 1)
        printf("Dense result is ok!\n");
    else
        printf("Dense result is wrong!\n");

    // Producto matriz CSR
    matriz_csr = convert_to_csr(mat, size, size);

    timestamp(&start);
    my_sparse_csr(&matriz_csr, vec, mysol);
    timestamp(&now);
    printf("Time taken by sparse CSR matrix-vector product: %ld ms\n", diff_milli(&start, &now));

    if (check_result(refsol, mysol, size) == 1)
        printf("CSR result is ok!\n");
    else
        printf("CSR result is wrong!\n");

    // Producto matriz CSC
    csc = convert_to_csc(mat, size, size);

    timestamp(&start);
    my_sparse_csc(&csc, vec, mysol);
    timestamp(&now);
    printf("Time taken by sparse CSC matrix-vector product: %ld ms\n", diff_milli(&start, &now));

    if (check_result(refsol, mysol, size) == 1)
        printf("CSC result is ok!\n");
    else
        printf("CSC result is wrong!\n");

    // Producto matriz COO
    coo = convert_to_coo(mat, size, size);

    timestamp(&start);
    my_sparse_coo(&coo, vec, mysol);
    timestamp(&now);
    printf("Time taken by sparse COO matrix-vector product: %ld ms\n", diff_milli(&start, &now));

    if (check_result(refsol, mysol, size) == 1)
        printf("COO result is ok!\n");
    else
        printf("COO result is wrong!\n");

    // Liberar memoria
    free(mat);
    free(vec);
    free(refsol);
    free(mysol);
    free(matriz_csr.fila_inicio);
    free(matriz_csr.indices_columnas);
    free(matriz_csr.val);
    free(csc.columna_inicio);
    free(csc.filas);
    free(csc.val);
    free(coo.filas);
    free(coo.columnas);
    free(coo.val);

    return 0;
}
