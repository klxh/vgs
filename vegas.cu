#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>


#define vegas_cycles 3 // numero di iterazioni vegas
#define dim 4 // numero di variabili della funzione integranda
#define Nc 10 // numero di suddivisioni degli intervalli di integrazione
#define NTHREADS 256
#define NBLOCKS 64
#define RNG_MAX 4294967295 
#define DEFAULT_SEED 5234
#define N 100
const unsigned int Nh = pow(Nc, dim);

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__device__ double myatomicAdd(double* address, double val)
{
 unsigned long long int* address_as_ull =
 (unsigned long long int*) address;
 unsigned long long int old = *address_as_ull;
 unsigned long long int assumed;
 do {
 assumed = old;
 old = atomicCAS(address_as_ull, assumed,
 __double_as_longlong(val + __longlong_as_double(assumed)));
 // Note: uses integer comparison to avoid hang in case
 // of NaN (since NaN != NaN)
 } while (assumed != old);
 return __longlong_as_double(old);
}


// definizione della funzione integranda
__device__ double f(double r[dim])
{
//	return r[0] + r[1];	
  return exp(-50 * pow((r[0] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[1] - 0.5), 2)) / sqrt(2 * M_PI * 0.01)
  * exp(-50 * pow((r[2] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[3] - 0.5), 2)) / sqrt(2 * M_PI * 0.01);
}

		
__global__ void generate_kernel(curandStateMtgp32 *state, double *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Generate pseudo-random unsigned ints */
    result[id] = (double) curand(&state[blockIdx.x]) / RNG_MAX;
}

__global__ void integration(curandStateMtgp32 *state, double *grid, double *spacings, double *I, double *E, int Nh)
{
  __shared__ double cache_S[NTHREADS];
  __shared__ double cache_err[NTHREADS];

  int cacheIndex = threadIdx.x;

	int tid = threadIdx.x + blockIdx.x * blockDim.x; 

  double temp_S = 0;
  double temp_err = 0;
  while(tid < Nh)
  {
    double hypervolume = 0;
    int index[dim];
    int buffer[dim];
    index[0] = tid % Nc;
    buffer[0] = index[0];
    hypervolume = grid[index[0] + 1] - grid[index[0]];
    for(int dir = 1; dir < dim; dir++)
    {
      buffer[dir] = buffer[dir - 1] + (tid - buffer[dir - 1]) % (int) pow((double) Nc, (double) dir + 1);
      index[dir] = ( (tid - buffer[dir - 1]) % (int) pow((double) Nc, (double) dir + 1) ) / pow((double) Nc, (double) dir);
      hypervolume *= (grid[dir * (Nc + 1) + index[dir] + 1] - grid[dir * (Nc + 1) + index[dir]]);
    }
    double quads = 0;
    double partial_S = 0;
    for(int count = 0; count < N; count++)
    {
      double r[dim];
      r[0] = grid[index[0]] + (double) curand(&state[blockIdx.x]) / RNG_MAX * (grid[index[0] + 1] - grid[index[0]]);
      for(int dir = 1; dir < dim; dir++)
      {
        r[dir] = grid[dir * (Nc + 1) + index[dir]] + (double) curand(&state[blockIdx.x]) / RNG_MAX
                 * (grid[dir * (Nc + 1) + index[dir] + 1] - grid[dir * (Nc + 1) + index[dir]]);
      }
      quads += f(r) * f(r);
      temp_S += f(r) * hypervolume / N;
      partial_S += f(r) * hypervolume / N;
		}
    temp_err += pow(hypervolume, 2) / (N * (N - 1)) * quads - 1./N * pow(partial_S, 2) ;
 
		tid += blockDim.x * gridDim.x;
  }
  cache_S[cacheIndex] = temp_S;
  cache_err[cacheIndex] = temp_err;

   __syncthreads();

  int k = blockDim.x / 2;
  while(k != 0)
  {
    if (cacheIndex < k)
    {
      cache_S[cacheIndex] += cache_S[cacheIndex + k];
      cache_err[cacheIndex] += cache_err[cacheIndex + k];
    }

    __syncthreads();
    k/=2;
  }
  if (cacheIndex == 0) { myatomicAdd(I, cache_S[0]); myatomicAdd(E, cache_err[0]); }
}


__global__ void adaptation()
{

}


int main ()
{
  CUDA_CALL(cudaSetDevice(0));

	int seed = DEFAULT_SEED;

	// definizione della griglia di integrazione e inizializzazione con equispaziature
	double grid[dim * (Nc + 1)];
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
	double spacings[dim * Nc];
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
  
/*	generate_kernel<<<NBLOCKS,NTHREADS>>>(devMTGPStates, dev_results);

	double host_results[NBLOCKS * NTHREADS];

  CUDA_CALL(cudaMemcpy(host_results, dev_results, NBLOCKS * NTHREADS * sizeof(double), cudaMemcpyDeviceToHost));

	for(int i = 0; i < NBLOCKS * NTHREADS; i++) { printf("host_result[%d] = %f\n", i, host_results[i]); }
*/

	// riprendo il discorso principale

	double *dev_grid;
	CUDA_CALL(cudaMalloc((void**)&dev_grid, dim * (Nc + 1) * sizeof(double)));
	CUDA_CALL(cudaMemcpy(dev_grid, grid,  dim * (Nc + 1) * sizeof(double), cudaMemcpyHostToDevice));

	double *dev_spacings;
	CUDA_CALL(cudaMalloc((void**)&dev_spacings, dim * Nc * sizeof(double)));
	CUDA_CALL(cudaMemcpy(dev_spacings, spacings,  dim * Nc * sizeof(double), cudaMemcpyHostToDevice));

	double I[vegas_cycles]; 
	double E[vegas_cycles];

	double *dev_I, *dev_E;

	// cicli vegas
	for (int it = 0; it < vegas_cycles; it++)
	{
		double S = 0;
		double err = 0;
		CUDA_CALL(cudaMalloc((void**)&dev_I, sizeof(double)));
 	  CUDA_CALL(cudaMemcpy(dev_I, &S, sizeof(double), cudaMemcpyHostToDevice));

		CUDA_CALL(cudaMalloc((void**)&dev_E, sizeof(double)));
	  CUDA_CALL(cudaMemcpy(dev_E, &err, sizeof(double), cudaMemcpyHostToDevice));

		printf("\n###### ITERAZIONE VEGAS %d ######\n\n", it);

		integration<<<NBLOCKS,NTHREADS>>>(devMTGPStates, dev_grid, dev_spacings, dev_I, dev_E, Nh);

		CUDA_CALL(cudaMemcpy(&S, dev_I, sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(&err, dev_E, sizeof(double), cudaMemcpyDeviceToHost));

		adaptation<<<1,1>>>();

		I[it] = S;
		E[it] = err;
		printf("I[%d] = %f\nE[%d] = %f\n", it, I[it], it, E[it]); 
	}
	
  CUDA_CALL(cudaFree(devMTGPStates));
  CUDA_CALL(cudaFree(dev_results));
  CUDA_CALL(cudaFree(dev_grid));
  CUDA_CALL(cudaFree(dev_spacings));
  CUDA_CALL(cudaFree(dev_I));
  CUDA_CALL(cudaFree(dev_E));
 
 	return EXIT_SUCCESS;
}


