#ifndef SPMV_H
#define SPMV_H

// Estructura para matriz 
typedef struct {
    int* fila_inicio;       // Índices de inicio de cada fila
    int* indices_columnas;  // Índices de las columnas de los elementos no cero
    double* val;            // Valores de los elementos no cero
    int num_filas;          // Número de filas
    int num_columnas;       // Número de columnas
    int num_sin_ceros;      // Número de elementos no cero
} MatrizCSR;

typedef struct {
   int* columna_inicio;   // Índices de inicio de cada columna
    int* filas;            // Índices de las filas de los elementos no cero
    double* val;           // Valores de los elementos no cero
    int num_filas;         // Número de filas
    int num_columnas;      // Número de columnas
    int num_sin_ceros;     // Número de elementos no cero
} MatrizCSC;

typedef struct {
    int* filas;        // Índices de las filas de los elementos no cero
    int* columnas;     // Índices de las columnas de los elementos no cero
    double* val;       // Valores de los elementos no cero
    int num_filas;     // Número de filas
    int num_columnas;  // Número de columnas
    int num_sin_ceros; // Número de elementos no cero
} MatrizCOO;

// Declaracion de funciones
int my_dense(const unsigned int n, const double mat[], double vec[], double result[]);
MatrizCSR convert_to_csr(const double* mat, int num_filas, int num_columnas); 
int my_sparse_csr(const MatrizCSR* matriz_csr, double vec[], double result[]); 
MatrizCOO convert_to_coo(const double* mat, int num_filas, int num_columnas);
int my_sparse_coo(const MatrizCOO* matriz_coo, double vec[], double result[]);
MatrizCSC convert_to_csc(const double* mat, int num_filas, int num_columnas);
int my_sparse_csc(const MatrizCSC* matriz_csc, double vec[], double result[]);
#endif // SPMV_H

