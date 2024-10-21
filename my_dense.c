#include "spmv.h"

int my_dense(const unsigned int n, const double mat[], double vec[], double result[]) {
    for (unsigned int i = 0; i < n; i++) {
        result[i] = 0.0;  // Inicializar el resultado
        for (unsigned int j = 0; j < n; j++) {
            result[i] += mat[i * n + j] * vec[j];  // Producto
        }
    }
    return 0;  // Indicar éxito
}

