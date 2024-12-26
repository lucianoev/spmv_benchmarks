#include "spmv.h"
#include "stdio.h"
#include <stdlib.h>


int my_sparse_csr(const MatrizCSR* matriz_csr, double vec[], double result[]) {
    #pragma omp parallel for
    for (int fila = 0; fila < matriz_csr->num_filas; fila++) {
        double temp = 0.0;
        #pragma omp simd
        for (int idx = matriz_csr->fila_inicio[fila]; idx < matriz_csr->fila_inicio[fila + 1]; idx++) {
            temp += matriz_csr->val[idx] * vec[matriz_csr->indices_columnas[idx]];
        }
        result[fila] = temp;
    }
    return 0;
}
