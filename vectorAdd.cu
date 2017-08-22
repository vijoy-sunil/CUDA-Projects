/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

// Includes
#include <stdio.h>
#include <cutil_inline.h>

// Variables
float* h_A;
float* h_B;
float* h_C;
float* d_A;
float* d_B;
float* d_C;

// create and start timer as unsigned integer
unsigned int timer_mem = 0;
unsigned int timer_total = 0;
unsigned int timer_GPU = 0;

// Functions
void Cleanup(void);
void RandomInit(float*, int);
void ParseArguments(int, char**);

// Device code
__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int GlobalSize = 50000;

// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);

    int N = GlobalSize;
    printf("Vector addition : size %d\n", N);
    size_t size = N * sizeof(float);

    // STUDENT: Allocate input vectors h_A h_B h_C in host memory
    
    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // STUDENT: Allocate vectors (d_A,d_B,d_C) in device memory

    // Initialize the timer to zero cycles.
    cutilCheckError(cutCreateTimer(&timer_mem));
    cutilCheckError(cutCreateTimer(&timer_total));
    cutilCheckError(cutCreateTimer(&timer_GPU));

    // Start the timer
    cutilCheckError(cutStartTimer(timer_mem));
    cutilCheckError(cutStartTimer(timer_total));

    // STUDENT: Copy A,B vectors from host memory to device memory

    // stop timer
    cutilCheckError(cutStopTimer(timer_mem));

    // Set the kernel arguments
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
   
    // Print the timer
    printf("CPU to GPU Transfer Time: %f (ms) \n", cutGetTimerValue(timer_mem));
    
    // Start GPU timer
    cutilCheckError(cutStartTimer(timer_GPU));
  
    // STUDENT: Invoke kernel

    // Host wait for the kernel to finish
    cudaThreadSynchronize(); 

    // stop GPU timer
    cutilCheckError(cutStopTimer(timer_GPU));

    // Reset the timer for memory
    cutilCheckError(cutCreateTimer(&timer_mem));

    // Start the timer
    cutilCheckError(cutStartTimer(timer_mem));

    // STUDENT: Copy result from device memory to host memory
    // h_C contains the result in host memory
    
    // stop and destroy timer
    cutilCheckError(cutStopTimer(timer_mem));
    cutilCheckError(cutStopTimer(timer_total));

    // Print the timer
    printf("GPU to CPU Transfer Time: %f (ms) \n", cutGetTimerValue(timer_mem));
    printf("Overall Execution Time (Memory + GPU): %f (ms) \n", cutGetTimerValue(timer_total));
    printf("GPU Execution Time: %f (ms) \n", cutGetTimerValue(timer_GPU));

    // Verify result with the CPU
    int i;
    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];
        if (fabs(h_C[i] - sum) > 1e-5)
            break;
    }
    printf("%s \n", (i == N) ? "PASSED" : "FAILED");
    
    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
  
    // Destroy (Free) timer   
    cutilCheckError(cutDeleteTimer(timer_mem));
    cutilCheckError(cutDeleteTimer(timer_total));
    cutilCheckError(cutDeleteTimer(timer_GPU));
      
    cutilSafeCall( cudaThreadExit() );
    
    exit(0);
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
                  GlobalSize = atoi(argv[i+1]);
		  i = i + 1;
        }

    }
}
