#include "spmv.h"
#include <omp.h>
#include <stdlib.h>

int my_sparse_csc(const MatrizCSC* matriz_csc, double vec[], double result[]) {
    // Inicia el vector resultado en 0
    #pragma omp parallel for
    for (int i = 0; i < matriz_csc->num_filas; i++) {
        result[i] = 0.0;
    }

    // Realiza el producto matriz-vector
    #pragma omp parallel for
    for (int columna = 0; columna < matriz_csc->num_columnas; columna++) {
        for (int idx = matriz_csc->columna_inicio[columna]; idx < matriz_csc->columna_inicio[columna + 1]; idx++) {
            int fila = matriz_csc->filas[idx];
            double valor = matriz_csc->val[idx] * vec[columna];
            #pragma omp atomic
            result[fila] += valor;
        }
    }

    return 0;
}
