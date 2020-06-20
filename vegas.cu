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
#define dim 8 // numero di variabili della funzione integranda
#define Nc 5 // numero di suddivisioni degli intervalli di integrazione
#define NTHREADS 256
#define NBLOCKS 128
#define RNG_MAX 4294967295 
#define DEFAULT_SEED 5234
#define N 10
#define alpha 1.5
#define K 1000
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
  * exp(-50 * pow((r[2] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[3] - 0.5), 2)) / sqrt(2 * M_PI * 0.01)
  * exp(-50 * pow((r[4] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[5] - 0.5), 2)) / sqrt(2 * M_PI * 0.01)
  * exp(-50 * pow((r[6] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[7] - 0.5), 2)) / sqrt(2 * M_PI * 0.01);
}

		
__global__ void generate_kernel(curandStateMtgp32 *state, double *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Generate pseudo-random unsigned ints */
    result[id] = (double) curand(&state[blockIdx.x]) / RNG_MAX;
}

__global__ void 
integration(curandStateMtgp32 *state, double *grid, double *spacings, double *I, double *A, double *abs_I, double *E, int Nh)
{
  __shared__ double cache_S[NTHREADS];
  __shared__ double cache_err[NTHREADS];
	__shared__ double cache_abs[NTHREADS];

  int cacheIndex = threadIdx.x;

	int tid = threadIdx.x + blockIdx.x * blockDim.x; 

  double temp_S = 0;
  double temp_err = 0;
	double temp_abs = 0;
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
      A[tid] += fabs(f(r)) * hypervolume / N;
      temp_abs += fabs(f(r)) * hypervolume / N;
		}
    temp_err += pow(hypervolume, 2) / (N * (N - 1)) * quads - 1./N * pow(partial_S, 2) ;
 
		tid += blockDim.x * gridDim.x;
  }
  cache_S[cacheIndex] = temp_S;
  cache_err[cacheIndex] = temp_err;
  cache_abs[cacheIndex] = temp_abs;

   __syncthreads();

  int k = blockDim.x / 2;
  while(k != 0)
  {
    if (cacheIndex < k)
    {
      cache_S[cacheIndex] += cache_S[cacheIndex + k];
      cache_err[cacheIndex] += cache_err[cacheIndex + k];
      cache_abs[cacheIndex] += cache_abs[cacheIndex + k];
    }

    __syncthreads();
    k /= 2;
  }
  if (cacheIndex == 0) { myatomicAdd(I, cache_S[0]); myatomicAdd(E, cache_err[0]); myatomicAdd(abs_I, cache_abs[0]); }
}


__global__ void marginals(double *pdfs, double *A, double *abs_I, int Nh)
{
    for(int i = 0; i < Nh; i++)
    {
      int index[dim];
      int buffer[dim];
      index[0] = i % Nc;
      buffer[0] = index[0];
      pdfs[index[0]] += A[i] / *abs_I;
      for(int dir = 1; dir < dim; dir++)
      {
        buffer[dir] = buffer[dir - 1] + (i - buffer[dir - 1]) % (int) pow((double )Nc, (double) dir + 1);
        index[dir] = ( (i - buffer[dir - 1]) % (int) pow((double) Nc, (double) dir + 1) ) / pow((double) Nc, (double) dir);
        pdfs[dir * Nc + index[dir]] += A[i] / *abs_I;
      }
    }
}

void adaptation(double *grid, double *spacings, double * pdfs)
{
	// grid update
  for(int direction = 0; direction < dim; direction ++)
  {
    double M = 0;
    double m[Nc];
    for(int i = 0; i < Nc; i++)
    {
      double r = pdfs[direction * Nc + i];
      m[i] = K * pow( (r - 1) / log(r), alpha );
      if(m[i] == 0) { m[i] = 1; }
      M += m[i];
    }

    int L = floor (M / Nc) * Nc;
    int l[Nc];
    int sum = 0;
    for(int elem = 0; elem < Nc; elem++) { l[elem] = floor (m[elem]); sum += l[elem]; }

    if(sum < L)
    {
      int D = L - sum;
      while(D != 0)
      {
        int q = random() % Nc;
        l[q]++;
        D--;
      }
    }

    else if(sum > L)
    {
      int D = sum - L;
      while(D != 0)
      {
        int q = random() % Nc;
        l[q]--;
        D--;
      }
    }

    // ora devo aggiornare la griglia di integrazione lungo l'asse d
    double diffs[Nc];
    for(int j = 0; j < Nc; j++)
    {
      diffs[j] = spacings[direction * Nc + j] / l[j];
      spacings[direction * Nc + j] = 0;
    }

    int p = 0;
    double count = 0;
    for(int k = 0; k < Nc; k++)
    {
      for(int a = 0; a < L/Nc; a++)
      {
        l[p]--;
        spacings[direction * Nc + k] += diffs[p];
        if (l[p] == 0) { p++; }
      }
      count += spacings[direction * Nc + k]; // risommo i sottointervalli (check)
			printf("spacings[%d][%d] = %f\n", direction, k, spacings[direction * Nc + k]);
    }
		printf("\ncount = %f\n##############################\n", count);
  }

  for(int i = 0; i < dim; i++)
  {
    grid[i * (Nc + 1)] = 0;
		printf("\ngrid[%d][0] = %f\n", i, grid[i * (Nc + 1)]);
    for(int j = 1; j < Nc + 1; j++)
    {
      grid[i * (Nc + 1) + j] = 0;
      grid[i * (Nc + 1) + j] += grid[i * (Nc + 1) + j - 1] + spacings[i * Nc + j - 1];
			printf("grid[%d][%d] = %f\n", i, j, grid[i * (Nc + 1) + j]);
    }
		printf("\n##############################\n");
  }
}


int main ()
{
  CUDA_CALL(cudaSetDevice(0));

	int seed = DEFAULT_SEED;

	float time = 0;

  cudaEvent_t start, stop;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));

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
  
	// riprendo il discorso principale

	double *dev_grid;
	CUDA_CALL(cudaMalloc((void**)&dev_grid, dim * (Nc + 1) * sizeof(double)));

	double *dev_spacings;
	CUDA_CALL(cudaMalloc((void**)&dev_spacings, dim * Nc * sizeof(double)));

	double I[vegas_cycles]; 
	double E[vegas_cycles];

	double *dev_I, *dev_A, *dev_abs_I, *dev_E, *dev_pdfs;

	CUDA_CALL(cudaMalloc((void**)&dev_I, sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_E, sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_A, Nh * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_abs_I, sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&dev_pdfs, dim * Nc * sizeof(double)));

	// cicli vegas
	for (int it = 0; it < vegas_cycles; it++)
	{

		CUDA_CALL(cudaMemcpy(dev_grid, grid,  dim * (Nc + 1) * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_spacings, spacings,  dim * Nc * sizeof(double), cudaMemcpyHostToDevice));

		double S = 0;
		double err = 0;
		double abs_I = 0;
 	  CUDA_CALL(cudaMemcpy(dev_I, &S, sizeof(double), cudaMemcpyHostToDevice));
	  CUDA_CALL(cudaMemcpy(dev_E, &err, sizeof(double), cudaMemcpyHostToDevice));
	  CUDA_CALL(cudaMemcpy(dev_abs_I, &abs_I, sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemset(dev_A, 0, Nh * sizeof(double)));
		CUDA_CALL(cudaMemset(dev_pdfs, 0, dim * Nc * sizeof(double)));
		
		printf("\n###### ITERAZIONE VEGAS %d ######\n\n", it);

    CUDA_CALL(cudaEventRecord(start, 0));

		integration<<<NBLOCKS,NTHREADS>>>(devMTGPStates, dev_grid, dev_spacings, dev_I, dev_A, dev_abs_I, dev_E, Nh);

		CUDA_CALL(cudaMemcpy(&S, dev_I, sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(&err, dev_E, sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(&abs_I, dev_abs_I, sizeof(double), cudaMemcpyDeviceToHost));

		marginals<<<1,1>>>(dev_pdfs, dev_A, dev_abs_I, Nh);

		double pdfs[dim * Nc] = {0};
		CUDA_CALL(cudaMemcpy(pdfs, dev_pdfs, dim * Nc * sizeof(double), cudaMemcpyDeviceToHost));

		adaptation(grid, spacings, pdfs);

    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    float elapsed = 0;
    CUDA_CALL(cudaEventElapsedTime(&elapsed, start, stop));
    time += elapsed;    

		I[it] = S;
		E[it] = err;
		printf("I[%d] = %f\nE[%d] = %f\n", it, I[it], it, E[it]); 
		printf("A[%d] = %f\n", it, abs_I);
	}

	// stima globale dell'integrale e dell'errore; calcolo del chi2
  double I_avg = 0;
  double inv = 0;
  double chi2 = 0;
  for(int i = 0; i < vegas_cycles; i++) { inv += 1. / E[i]; }
  double E_avg = 1. / inv;
  for (int i = 0; i < vegas_cycles; i++) { I_avg += E_avg * I[i] / E[i]; }
  for(int i = 0; i < vegas_cycles; i++) { chi2 += pow( (I[i] - I_avg) , 2) / E[i]; }

	printf("\n#####################################################################\n\n\n");

	printf("Stima globale dell'integrale = %f\n", I_avg);
	printf("Stima globale dell'errore = %f\n", sqrt(E_avg));
	printf("chi quadro = %f\n", chi2);
	printf("chi quadro per iterazione (meno uno) = %f\n", chi2 / (vegas_cycles - 1));

	printf("\nElapsed_time = %f\n\n", time / 1000.);

  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(stop));

  CUDA_CALL(cudaFree(devMTGPStates));
  CUDA_CALL(cudaFree(dev_results));
  CUDA_CALL(cudaFree(dev_grid));
  CUDA_CALL(cudaFree(dev_spacings));
  CUDA_CALL(cudaFree(dev_I));
  CUDA_CALL(cudaFree(dev_E));
	CUDA_CALL(cudaFree(dev_A));
	CUDA_CALL(cudaFree(dev_abs_I));
	CUDA_CALL(cudaFree(dev_pdfs));
 
 	return EXIT_SUCCESS;
}


