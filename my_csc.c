#include "spmv.h"
#include "stdio.h"
#include <stdlib.h>


//Función para el producto natriz CSC
int my_sparse_csc(const MatrizCSC* matriz_csc, double vec[], double result[]) {
    for (int i = 0; i < matriz_csc->num_filas; i++) {
        result[i] = 0.0;
    }

    for (int j = 0; j < matriz_csc->num_columnas; j++) {
        for (int idx = matriz_csc->columna_inicio[j]; idx < matriz_csc->columna_inicio[j + 1]; idx++) {
            result[matriz_csc->filas[idx]] += matriz_csc->val[idx] * vec[j];
        }
    }

    return 0;
}
