#include "spmv.h"
#include <omp.h>
#include <stdlib.h>

int my_sparse_coo(const MatrizCOO* matriz_coo, double vec[], double result[]) {
    // Inicia el vector resultado en 0
    #pragma omp parallel for
    for (int i = 0; i < matriz_coo->num_filas; i++) {
        result[i] = 0.0;
    }

    // Realiza el producto matriz-vector
    #pragma omp parallel for
    for (int i = 0; i < matriz_coo->num_sin_ceros; i++) {
        int fila = matriz_coo->filas[i];
        double valor = matriz_coo->val[i] * vec[matriz_coo->columnas[i]];
        #pragma omp atomic
        result[fila] += valor;
    }

    return 0;
}
