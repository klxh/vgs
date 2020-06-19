#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define NTHREADS 256
#define NBLOCKS 64
#define RNG_MAX 4294967295 
		
__global__ void generate_kernel(curandStateMtgp32 *state, double *result)
{
    int id = threadIdx.x + blockIdx.x * NTHREADS;
    /* Generate pseudo-random unsigned ints */
    result[id] = (double) curand(&state[blockIdx.x]) / RNG_MAX;
}



int main()
{
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
              mtgp32dc_params_fast_11213, devKernelParams, NBLOCKS, 1234));
  
	CURAND_CALL(cudaMalloc((void**)&dev_results, NBLOCKS * NTHREADS * sizeof(double)));

  /* State setup is complete */
  
	generate_kernel<<<NBLOCKS,NTHREADS>>>(devMTGPStates, dev_results);

	double host_results[NBLOCKS * NTHREADS];

  CUDA_CALL(cudaMemcpy(host_results, dev_results, NBLOCKS * NTHREADS * sizeof(double), cudaMemcpyDeviceToHost));

	for(int i = 0; i < NBLOCKS * NTHREADS; i++) { printf("host_result[%d] = %f\n", i, host_results[i]); }

  CUDA_CALL(cudaFree(devMTGPStates));
  CUDA_CALL(cudaFree(dev_results));

 	return EXIT_SUCCESS;
}
