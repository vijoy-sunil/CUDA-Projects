// 
// Vector Reduction
//

// Includes
#include <stdio.h>
#include <cutil_inline.h>

// Input Array Variables
float* h_In = NULL;
float* d_In = NULL;

// Output Array
float* h_Out = NULL;
float* d_Out = NULL;

// Variables to change
int GlobalSize = 50000;
int BlockSize = 32;

// Functions
void Cleanup(void);
void RandomInit(float*, int);
void PrintArray(float*, int);
float CPUReduce(float*, int);
void ParseArguments(int, char**);

//timers
unsigned int timer_CPU = 0; //time to calculate partial sums in cpu
unsigned int timer_GPU = 0;
unsigned int timer_mem = 0;
unsigned int timer_total = 0;

// Device code
__global__ void VecReduce(float* g_idata, float* g_odata, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdata[]; 

  unsigned int tid = threadIdx.x; 
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x; 

  // For thread ids greater than data space
  sdata[tid] = g_idata[globalid]; 


  // each thread loads one element from global to shared mem
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) { 
         sdata[tid] = sdata[tid] + sdata[tid+ s];
     }
     __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)  {
     g_odata[blockIdx.x] = sdata[0];
  }
}


// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);

    int N = GlobalSize;
    printf("Vector reduction: size %d\n", N);
    size_t in_size = N * sizeof(float);
    float CPU_result = 0.0, GPU_result = 0.0;

    // Allocate input vectors h_In and h_B in host memory
    h_In = (float*)malloc(in_size);
    if (h_In == 0) 
      Cleanup();

    // Initialize input vectors
    RandomInit(h_In, N);

    //create timer
    cutilCheckError(cutCreateTimer(&timer_mem));
    cutilCheckError(cutCreateTimer(&timer_total));
    cutilCheckError(cutCreateTimer(&timer_GPU));
    cutilCheckError(cutCreateTimer(&timer_CPU));

    // Set the kernel arguments
    int threadsPerBlock = BlockSize;
    int sharedMemSize = threadsPerBlock * sizeof(float);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t out_size = blocksPerGrid * sizeof(float);

    // Allocate host output
    h_Out = (float*)malloc(out_size);
    if (h_Out == 0) 
      Cleanup();

    // STUDENT: CPU computation - time this routine for base comparison
    
    cutilCheckError(cutStartTimer(timer_CPU));

    CPU_result = CPUReduce(h_In, N);

    cutilCheckError(cutStopTimer(timer_CPU));
    printf("CPU Reduction time: %f (ms) \n",cutGetTimerValue(timer_CPU));

    // Allocate vectors in device memory
    cutilSafeCall( cudaMalloc((void**)&d_In, in_size) );
    cutilSafeCall( cudaMalloc((void**)&d_Out, out_size) );

    // compute memory transfer time - CPU to GPU
    cutilCheckError(cutStartTimer(timer_mem));
    cutilCheckError(cutStartTimer(timer_total));

    // STUDENT: Copy h_In from host memory to device memory
    cutilSafeCall( cudaMemcpy(d_In,h_In,in_size,cudaMemcpyHostToDevice));   

    cutilCheckError(cutStopTimer(timer_mem));
    printf("CPU to GPU Transfer Time: %f (ms) \n",cutGetTimerValue(timer_mem));
    // compute gpu execution time   
    cutilCheckError(cutStartTimer(timer_GPU));

    // Invoke kernel
    VecReduce<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_In, d_Out, N);
    cutilCheckMsg("kernel launch failure");
    cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel

    // gpu exec timer stop
    cutilCheckError(cutStopTimer(timer_GPU));

    //compute memmory transger time - GPU to CPU
    cutilCheckError(cutCreateTimer(&timer_mem));
    cutilCheckError(cutStartTimer(timer_mem));

    // STUDENT: copy results back from GPU to the h_Out
    cutilSafeCall( cudaMemcpy(h_Out,d_Out,out_size,cudaMemcpyDeviceToHost));

    cutilCheckError(cutStopTimer(timer_mem));
    cutilCheckError(cutStopTimer(timer_total));

    printf("GPU Execution time: %f (ms) \n",cutGetTimerValue(timer_GPU));
    printf("GPU to CPU Transfer Time: %f (ms) \n",cutGetTimerValue(timer_mem));
    printf("Overall Execution Time (Memory + GPU): %f (ms) \n", cutGetTimerValue(timer_total));

    // STUDENT: Perform the CPU addition of partial results
    // update variable GPU_result

    int i;
    cutilCheckError(cutStartTimer(timer_CPU));
   
    for( i= 0;i < blocksPerGrid; i++ ) 
    {
        GPU_result = GPU_result + h_Out[i];
    }  

    cutilCheckError(cutStopTimer(timer_CPU));
    printf("CPU Partial Sums Execution time: %f (ms) \n",cutGetTimerValue(timer_CPU));

    // STUDENT Check results to make sure they are the same
    printf("CPU results : %f\n", CPU_result);
    printf("GPU results : %f\n", GPU_result);
 
    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_In)
        cudaFree(d_In);
    if (d_Out)
        cudaFree(d_Out);

    // Free host memory
    if (h_In)
        free(h_In);
    if (h_Out)
        free(h_Out);
    
    cutilCheckError(cutDeleteTimer(timer_GPU));        
    cutilCheckError(cutDeleteTimer(timer_CPU));        
    cutilCheckError(cutDeleteTimer(timer_mem));        
    cutilCheckError(cutDeleteTimer(timer_total));
        
    cutilSafeCall( cudaThreadExit() );
    
    exit(0);
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = rand() / (float)RAND_MAX;
}

void PrintArray(float* data, int n)
{
    for (int i = 0; i < n; i++)
        printf("[%d] => %f\n",i,data[i]);
}

float CPUReduce(float* data, int n)
{
  float sum = 0;
    for (int i = 0; i < n; i++)
        sum = sum + data[i];

  return sum;
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
                  GlobalSize = atoi(argv[i+1]);
		  i = i + 1;
        }
        if (strcmp(argv[i], "--blocksize") == 0 || strcmp(argv[i], "-blocksize") == 0) {
                  BlockSize = atoi(argv[i+1]);
		  i = i + 1;
	}
    }
}
