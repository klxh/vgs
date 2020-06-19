#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>


#define vegas_cycles 1 // numero di iterazioni vegas
#define dim 2 // numero di variabili della funzione integranda
#define Nc 10 // numero di suddivisioni degli intervalli di integrazione


#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void generate_kernel(curandStateMtgp32 *state, 
                                int n,
                                int *result)
{
    int id = threadIdx.x + blockIdx.x * gridDim.x;
    int count = 0;
    unsigned int x;
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < n; i++) {
        x = curand(&state[blockIdx.x]);
        /* Check if low bit set */
        if(x & 1) {
            count++;
        }
    }
    /* Store results */
    result[id] += count;
}



// definizione della funzione integranda
float f(float r[dim])
{
	return r[0] + r[1];	
}


int main ()
{
	// definizione della griglia di integrazione e inizializzazione con equispaziature
	float grid[dim * (Nc + 1)];
	for(int i = 0; i < dim; i++)
	{
		grid[i * (Nc + 1)] = 0;
		printf("grid[%d][%d] = %f\n", i, 0, grid[i * (Nc + 1)]);
		for(int j = 1; j < Nc + 1; j++)
		{
			grid[i * (Nc + 1) + j] = grid[i * (Nc + 1) + j - 1] + 1./Nc;
		  printf("grid[%d][%d] = %f\n", i, j, grid[i * (Nc + 1) + j]);
		}
	}

	// definizione della griglia delle spaziature
	float spacings[dim * Nc];
	for(int i = 0; i < dim * Nc; i++) { spacings[i] = 1./Nc; }

  curandStateMtgp32 *devMTGPStates;
  mtgp32_kernel_params *devKernelParams;

	// cicli vegas
	for (int it = 0; it < vegas_cycles; it++)
	{
		printf("\n###### ITERAZIONE VEGAS %d ######\n\n", it);
	}
	
	return 0;
}
