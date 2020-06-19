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
#define NTHREADS 16
#define NBLOCKS 4
#define RNG_MAX 4294967295 
#define DEFAULT_SEED 5234

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

		
__global__ void generate_kernel(curandStateMtgp32 *state, double *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Generate pseudo-random unsigned ints */
    result[id] = (double) curand(&state[blockIdx.x]) / RNG_MAX;
}

__global__ void integration(curandStateMtgp32 *state, float *grid, float *spacings, float *I, float *E)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x; 
}

// definizione della funzione integranda
__device__ float f(float r[dim])
{
	return r[0] + r[1];	
}


int main ()
{
	int seed = DEFAULT_SEED;

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

	// set up per generazione di numeri random
  curandStateMtgp32 *devMTGPStates;
  mtgp32_kernel_params *devKernelParams;
	
	double *dev_results;

  /* Allocate space for prng states on device */
  CUDA_CALL(cudaMalloc((void **)&devMTGPStates, NBLOCKS * sizeof(curandStateMtgp32)));
 

  /* Setup MTGP prng states */
  
  /* Allocate space for MTGP kernel parameters */
  CUDA_CALL(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));
  
  /* Reformat from predefined parameter sets to kernel format, */
  /* and copy kernel parameters to device memory               */
  CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams));
  
  /* Initialize one state per thread block */
  CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, 
              mtgp32dc_params_fast_11213, devKernelParams, NBLOCKS, seed));
  
	CURAND_CALL(cudaMalloc((void**)&dev_results, NBLOCKS * NTHREADS * sizeof(double)));

  /* State setup is complete */
  
	generate_kernel<<<NBLOCKS,NTHREADS>>>(devMTGPStates, dev_results);

	double host_results[NBLOCKS * NTHREADS];

  CUDA_CALL(cudaMemcpy(host_results, dev_results, NBLOCKS * NTHREADS * sizeof(double), cudaMemcpyDeviceToHost));

	for(int i = 0; i < NBLOCKS * NTHREADS; i++) { printf("host_result[%d] = %f\n", i, host_results[i]); }

	// riprendo il discorso principale

	float *dev_grid;
	CUDA_CALL(cudaMalloc((void**)&dev_grid, dim * (Nc + 1) * sizeof(float)));
	CUDA_CALL(cudaMemcpy(dev_grid, grid,  dim * (Nc + 1) * sizeof(float), cudaMemcpyHostToDevice));

	float *dev_spacings;
	CUDA_CALL(cudaMalloc((void**)&dev_spacings, dim * Nc * sizeof(float)));
	CUDA_CALL(cudaMemcpy(dev_spacings, spacings,  dim * Nc * sizeof(float), cudaMemcpyHostToDevice));

	float I[vegas_cycles]; 
	float E[vegas_cycles];

	float *dev_I, *dev_E;

	CUDA_CALL(cudaMalloc((void**)&dev_I, vegas_cycles * sizeof(float)));
	CUDA_CALL(cudaMemset(dev_I, 0, vegas_cycles * sizeof(float)));

	CUDA_CALL(cudaMalloc((void**)&dev_E, vegas_cycles * sizeof(float)));
	CUDA_CALL(cudaMemset(dev_E, 0, vegas_cycles * sizeof(float)));

	// cicli vegas
	for (int it = 0; it < vegas_cycles; it++)
	{
		printf("\n###### ITERAZIONE VEGAS %d ######\n\n", it);

		integration<<<NBLOCKS,NTHREADS>>>(devMTGPStates, dev_grid, dev_spacings, dev_I, dev_E);
	}
	
	CUDA_CALL(cudaMemcpy(I, dev_I, vegas_cycles * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < vegas_cycles; i++) { printf("I[%d] = %f\n", i, I[i]); }

  CUDA_CALL(cudaFree(devMTGPStates));
  CUDA_CALL(cudaFree(dev_results));
  CUDA_CALL(cudaFree(dev_grid));
  CUDA_CALL(cudaFree(dev_spacings));
  CUDA_CALL(cudaFree(dev_I));
  CUDA_CALL(cudaFree(dev_E));
 
 	return EXIT_SUCCESS;
}


