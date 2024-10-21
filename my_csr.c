#include "spmv.h"
#include "stdio.h"
#include <stdlib.h>


//Función para el producto entre matriz y vector 
int my_sparse_csr(const MatrizCSR* matriz_csr, double vector[], double resultado[]){

	//Inicializo vector
	for (int fila = 0; fila < matriz_csr->num_filas; fila++) {
	        resultado[fila] = 0.0;
	}

	//Producto de matriz CSR con vector
	for (int fila = 0; fila < matriz_csr->num_filas; fila++) {
		        for (int idx = matriz_csr->fila_inicio[fila]; idx < matriz_csr->fila_inicio[fila + 1]; idx++) {
				            resultado[fila] += matriz_csr->val[idx] * vector[matriz_csr->indices_columnas[idx]];    
			}
	}
	return 0; // Retorna 0 si todo salió bien
}
