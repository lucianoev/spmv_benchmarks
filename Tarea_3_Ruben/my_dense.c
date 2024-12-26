#include "spmv.h"

int my_dense(const unsigned int n, const double mat[], double vec[], double result[]) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double temp = 0.0;
        #pragma omp simd
        for (int j = 0; j < n; j++) {
            temp += mat[i * n + j] * vec[j];
        }
        result[i] = temp;
    }
    return 0;
}
