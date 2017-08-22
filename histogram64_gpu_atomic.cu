
// Includes
#include <stdio.h>
#include <cutil_inline.h>

// Variables
int* inp_arr = NULL;	//input array in host
int* hist = NULL;   	//partial histogram in host
int* h_hist = NULL; 	//cpu execution for comparison
int* h_global = NULL;	//global histogram in host - using atomic add

int* d_arr = NULL;  		//input array in device
int* d_hist = NULL; 		//partial histogram created by each block
int* global64_hist = NULL; 	//global histogram in device - using atomic add

//atomic add execution time
int* d_timer = NULL;
int* h_timer = NULL;

//cuda Event timer
cudaEvent_t cstart,cstop;  	//measure gpu time
float cdiff;

cudaEvent_t hstart,hstop;  	//measure cpu time
float hdiff;


unsigned int timer_dh = 0;	//memory transfer time device - host
unsigned int timer_hd = 0;	//memory transfer time host - device

// kernel parameters
int BlockSize =32;
int GlobalSize = 50000;

// Functions
void Cleanup(void);
void RandomInit(int*, int);
void ParseArguments(int, char**);
void Print_Hist(void);

#define HISTOGRAM64_THREADBLOCK_SIZE 32
#define HISTOGRAM64_BIN_COUNT 64
#define MERGE_THREADBLOCK_SIZE 32

// Device code
__global__ void Histogram(int* d_arr, int* d_hist, int* d_timer, int* global64_hist, int N)
{

    clock_t atomic_start;
    clock_t atomic_stop;

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;
   
    const int sharedMem = HISTOGRAM64_THREADBLOCK_SIZE * HISTOGRAM64_BIN_COUNT;
    __shared__ int s_Hist[sharedMem];
   
    for(int i = 0; i < HISTOGRAM64_BIN_COUNT; i ++)
        s_Hist[threadIdx.x * HISTOGRAM64_THREADBLOCK_SIZE + i] = 0;

    int THREAD_N = blockDim.x * gridDim.x;

        for (int pos = tid; pos < N; pos = pos + THREAD_N)
        {  
            int data = d_arr[pos];
            s_Hist[data + threadIdx.x * HISTOGRAM64_BIN_COUNT]+= 1;
        }


    __syncthreads();


    for (int i = 0; i < sharedMem ; i+= HISTOGRAM64_BIN_COUNT) 
    {
        d_hist[threadIdx.x + blockIdx.x * HISTOGRAM64_BIN_COUNT] += s_Hist[threadIdx.x + i];  
        d_hist[threadIdx.x + HISTOGRAM64_THREADBLOCK_SIZE + blockIdx.x * HISTOGRAM64_BIN_COUNT] += s_Hist[threadIdx.x + HISTOGRAM64_THREADBLOCK_SIZE+ i];
    }   

    __syncthreads();

    atomic_start = clock();
   
    atomicAdd(&global64_hist[threadIdx.x],d_hist[threadIdx.x + blockIdx.x * HISTOGRAM64_BIN_COUNT]);
    atomicAdd(&global64_hist[threadIdx.x + HISTOGRAM64_THREADBLOCK_SIZE],d_hist[threadIdx.x + HISTOGRAM64_THREADBLOCK_SIZE + blockIdx.x * HISTOGRAM64_BIN_COUNT]);
    atomic_stop = clock();

    if(threadIdx.x == 0)
        d_timer[blockIdx.x] = atomic_stop - atomic_start;
}

// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);

    int Grid_Size = 0;

    int N = GlobalSize;
    printf("Histogram : %d BIN \n", HISTOGRAM64_BIN_COUNT);
    printf("Input Array size: %d \n",N);
    size_t size = N * sizeof(int);
    
    Grid_Size = N/HISTOGRAM64_THREADBLOCK_SIZE;
    if (Grid_Size > 90)
        Grid_Size = 90;
    if (Grid_Size < 1)
        Grid_Size = 1;

    printf("Grid Size: %d \n",Grid_Size);
    // Allocate array to store block exec times in host- start and end time
    int h_timer[Grid_Size];

   // STUDENT: Allocate input vectors in host memory

    inp_arr = (int *)malloc(size);    
   
    // Initialize input vectors
    RandomInit(inp_arr, N);

    // Set the kernel argumentsi
    int partial_index = Grid_Size * HISTOGRAM64_BIN_COUNT;
    size_t hist_size =  partial_index *  sizeof(int);
    hist = (int *)malloc(hist_size);

    for (int i = 0; i< partial_index; i ++)
        hist[i] = 0;

    h_global = (int*)malloc(HISTOGRAM64_BIN_COUNT * sizeof(int));
    for (int i =0; i< HISTOGRAM64_BIN_COUNT; i++)
        h_global[i] = 0;
    // STUDENT: Allocate vectors  in device memory

    cutilSafeCall(cudaMalloc((void**)&d_arr, size));
    cutilSafeCall(cudaMalloc((void**)&d_hist, hist_size));
    cutilSafeCall(cudaMalloc((void**)&global64_hist, HISTOGRAM64_BIN_COUNT * sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&d_timer, sizeof(int) * Grid_Size ));

    cutilCheckError(cutCreateTimer(&timer_dh));
    cutilCheckError(cutCreateTimer(&timer_hd));
    
    cudaEventCreate(&cstart);
    cudaEventCreate(&cstop);

    cudaEventCreate(&hstart);
    cudaEventCreate(&hstop);


    cutilCheckError(cutStartTimer(timer_hd));

    // STUDENT: Copy input vectors from host memory to device memory

    cutilSafeCall(cudaMemcpy( d_arr, inp_arr, size, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy( d_hist, hist, hist_size, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy( global64_hist, h_global, HISTOGRAM64_BIN_COUNT * sizeof(int), cudaMemcpyHostToDevice));

    cutilCheckError(cutStopTimer(timer_hd));

    cudaEventRecord(cstart,0);

    // STUDENT: Invoke kernel
    Histogram<<<Grid_Size,HISTOGRAM64_THREADBLOCK_SIZE>>>(d_arr, d_hist, d_timer, global64_hist, N);

    // Host wait for the kernel to finish
    cudaThreadSynchronize(); 

    cudaEventRecord(cstop,0);
    cudaEventSynchronize(cstop);

    cutilCheckError(cutStartTimer(timer_dh));
    
    // STUDENT: Copy result from device memory to host memory
    cutilSafeCall(cudaMemcpy(h_global, global64_hist, HISTOGRAM64_BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost)); 
    cutilSafeCall(cudaMemcpy(h_timer, d_timer,sizeof(int) * Grid_Size, cudaMemcpyDeviceToHost));

    cutilCheckError(cutStopTimer(timer_dh));
    
    printf("CPU to GPU Transfer Time: %f (ms) \n",cutGetTimerValue(timer_hd));
    printf("GPU to CPU Transfer Time: %f (ms) \n",cutGetTimerValue(timer_dh));
    cudaEventElapsedTime(&cdiff,cstart,cstop);
    printf("GPU Execution Time: %.3f (ms) \n", cdiff);

    int i,fail = 0;
   
    // Verify result with the CPU
    size_t h_hist_size = HISTOGRAM64_BIN_COUNT * sizeof(int);
    h_hist = (int *)malloc(h_hist_size);
  
    for (i = 0; i< HISTOGRAM64_BIN_COUNT; i++)
        h_hist[i] = 0;

    cudaEventRecord(hstart,0);
    for (i = 0; i < N; i++) 
        h_hist[inp_arr[i]]++; 
    cudaEventRecord(hstop,0);
    cudaEventSynchronize(hstop);
    cudaEventElapsedTime(&hdiff,hstart,hstop);

    printf("CPU Execution time : %.3f (ms) \n",hdiff);

    for (i =0; i < HISTOGRAM64_BIN_COUNT; i++)
    {  
        if(h_hist[i] !=h_global[i]) 
        {
            fail = 1;
            break;
        }
    }
    if(fail == 0)
        printf("CPU and GPU histogram result SAME \n");
    else
        printf("CPU and GPU histogram result DIFFERENT \n");
    
    Print_Hist();

    int B = 0;
    
    printf("------------------------------------------\n");
    printf("Atomic Add times of each block\n");
    printf("------------------------------------------\n");
    printf("BLOCK\tTIME\n");

    for (i = 0; i < Grid_Size; i++)
    {
        printf("B[%d]\t%d\n",B,h_timer[i]);
        B = B + 1;
    }
    Cleanup();
}

void Print_Hist(void)
{
    printf("CPU_HISTOGRAM\tGPU_HISTOGRAM\n");
 
    for (int i =0; i< HISTOGRAM64_BIN_COUNT; i++)
    {
        printf("H_CPU[%d] : %d \t H_GPU[%d] : %d \n",i,h_hist[i],i,h_global[i]);       
    }
}


void Cleanup(void)
{
    // Free device memory
    if (d_arr)
        cudaFree(d_arr);
    if (d_hist)
        cudaFree(d_hist);
    if (d_timer)
        cudaFree(d_timer);
    if (global64_hist)
        cudaFree(global64_hist);

    // Free host memory
    if (inp_arr)
        free(inp_arr);
    if (hist)
        free(hist);
    if (h_hist)
        free(h_hist);
    if (h_global)
        free(h_global);
  
    cudaEventDestroy(cstart);
    cudaEventDestroy(cstop);

    cudaEventDestroy(hstart);
    cudaEventDestroy(hstop);

    cutilSafeCall( cudaThreadExit() );
    
    exit(0);
}

// Allocates an array with random float entries.
void RandomInit(int* data, int n)
{
    srand(1);
    for (int i = 0; i < n; i++)
        data[i] = rand() % 64 ;
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
                  GlobalSize = atoi(argv[i+1]);
		  i = i + 1;
        }

        if (strcmp(argv[i], "--blocksize") == 0 || strcmp(argv[i], "-blocksize") == 0)
        {
                  BlockSize = atoi(argv[i + 1]);
                  i = i + 1;
        }
    }
}
