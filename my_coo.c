#include "spmv.h"
#include "stdio.h"
#include <stdlib.h>


//Función para el producto matriz COO 
int my_sparse_coo(const MatrizCOO* matriz_coo, double vec[], double result[]) {
    for (int i = 0; i < matriz_coo->num_filas; i++) {
        result[i] = 0.0;
    }

    for (int idx = 0; idx < matriz_coo->num_sin_ceros; idx++) {
        result[matriz_coo->filas[idx]] += matriz_coo->val[idx] * vec[matriz_coo->columnas[idx]];
    }

    return 0;
}