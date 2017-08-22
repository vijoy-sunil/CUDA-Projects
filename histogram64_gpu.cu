
// Includes
#include <stdio.h>
#include <cutil_inline.h>

// Variables
int* inp_arr = NULL;//input array in host
int* hist = NULL;   //partial histogram in host
int* h_hist = NULL; //cpu execution for comparison

int* d_arr = NULL;  //input array in device
int* d_hist = NULL; //partial histogram in device

//block execution times
clock_t* d_timer_start = NULL;
clock_t* d_timer_stop = NULL;

//cuda Event timer
cudaEvent_t cstart,cstop;  	//measure gpu time
float cdiff;

cudaEvent_t hstart,hstop;  	//measure cpu time
float hdiff;

cudaEvent_t psum_start,psum_stop; //measure partial sum compute time
float pdiff;

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
__global__ void Histogram(int* d_arr, int* d_hist, clock_t* d_timer_start, clock_t* d_timer_stop, int N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x ;

   
    //block time log by thread0
    if(tid % HISTOGRAM64_THREADBLOCK_SIZE == 0)
        d_timer_start[blockIdx.x] = clock(); 

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

    if(tid % HISTOGRAM64_THREADBLOCK_SIZE == 0)
        d_timer_stop[blockIdx.x] = clock();
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
    clock_t h_timer_start[Grid_Size];
    clock_t h_timer_stop[Grid_Size];

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

    // STUDENT: Allocate vectors  in device memory

    cutilSafeCall(cudaMalloc((void**)&d_arr, size));
    cutilSafeCall(cudaMalloc((void**)&d_hist, hist_size));
    cutilSafeCall(cudaMalloc((void**)&d_timer_start, sizeof(clock_t) * Grid_Size ));
    cutilSafeCall(cudaMalloc((void**)&d_timer_stop, sizeof(clock_t) * Grid_Size ));

    cutilCheckError(cutCreateTimer(&timer_dh));
    cutilCheckError(cutCreateTimer(&timer_hd));
    
    cudaEventCreate(&cstart);
    cudaEventCreate(&cstop);

    cudaEventCreate(&hstart);
    cudaEventCreate(&hstop);

    cudaEventCreate(&psum_start);
    cudaEventCreate(&psum_stop);

    cutilCheckError(cutStartTimer(timer_hd));
    // STUDENT: Copy input vectors from host memory to device memory
    cutilSafeCall(cudaMemcpy( d_arr, inp_arr, size, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy( d_hist, hist, hist_size, cudaMemcpyHostToDevice));
    

    cutilCheckError(cutStopTimer(timer_hd));

    cudaEventRecord(cstart,0);
    // STUDENT: Invoke kernel
    Histogram<<<Grid_Size,HISTOGRAM64_THREADBLOCK_SIZE>>>(d_arr,d_hist,d_timer_start,d_timer_stop,N);

    // Host wait for the kernel to finish
    cudaThreadSynchronize(); 

    cudaEventRecord(cstop,0);
    cudaEventSynchronize(cstop);

    cutilCheckError(cutStartTimer(timer_dh));
    
    // STUDENT: Copy result from device memory to host memory
    cutilSafeCall(cudaMemcpy(hist, d_hist,hist_size, cudaMemcpyDeviceToHost)); 
    cutilSafeCall(cudaMemcpy(h_timer_start, d_timer_start,sizeof(clock_t) * Grid_Size, cudaMemcpyDeviceToHost));

    cutilSafeCall(cudaMemcpy(h_timer_stop, d_timer_stop,sizeof(clock_t) * Grid_Size, cudaMemcpyDeviceToHost));
    cutilCheckError(cutStopTimer(timer_dh));
    
    printf("CPU to GPU Transfer Time: %f (ms) \n",cutGetTimerValue(timer_hd));
    printf("GPU to CPU Transfer Time: %f (ms) \n",cutGetTimerValue(timer_dh));
    cudaEventElapsedTime(&cdiff,cstart,cstop);
    printf("GPU Execution Time: %.3f (ms) \n", cdiff);

    int i,j,fail = 0;
    
    //Partial sums compute
    cudaEventRecord(psum_start,0); 
   
    for(i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
    {
        for(j = 64; j < Grid_Size * HISTOGRAM64_BIN_COUNT; j += 64)
        {
            hist[i] = hist[i] + hist[i + j];
        } 
    }
    cudaEventRecord(psum_stop,0);

    cudaEventSynchronize(psum_stop);
    cudaEventElapsedTime(&pdiff,psum_start,psum_stop);

    printf("CPU Partial Sums Execution time : %.3f (ms) \n",pdiff);
    
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
        if(h_hist[i] !=hist[i]) 
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
    unsigned int B_TIME[Grid_Size];

    printf("\n\nBLOCK EXECUTION TIME\n");
    printf("Block\tStart_time\tEnd_time\tExec_time\n");
    for (int i = 0; i < Grid_Size ; i ++)
    {
        printf("B%d\t%d\t%d\t",B,h_timer_start[i],h_timer_stop[i]);
        B_TIME[i] = h_timer_stop[i] - h_timer_start[i];
        printf("%d\n",B_TIME[i]);
        B = B + 1;
    }

    // execution time report - max,min print
    clock_t max_start = h_timer_start[0];
//    clock_t max_end = h_timer_stop[0];
    int short_block = B_TIME[0];
    int long_block = B_TIME[Grid_Size];

    for (int i = 0; i < Grid_Size; i++)
    {
        max_start = h_timer_start[i] > max_start ? h_timer_start[i] : max_start;
//        max_end = h_timer_stop[i] > max_end ? h_timer_stop[i] : max_end;
        short_block = B_TIME[i] < short_block ? B_TIME[i] : short_block;
        long_block = B_TIME[i] > long_block ? B_TIME[i] : long_block;
    } 

    int index = 0;
    for (int i = 0; i < Grid_Size; i++)
    {
        if(h_timer_start[i] == max_start){
            index = i;
            break;
        }
    }

    printf("------------------------------------------------------------------\n");
    printf("Exec time for shortest running block: %d\n",short_block);
    printf("Exec time for longest running block: %d\n",long_block);
    printf("Exec time for the last Block - B[%d](start time :%d) :%d\n",index,max_start,B_TIME[index]);    
    printf("------------------------------------------------------------------\n");
    Cleanup();
}

void Print_Hist(void)
{
    printf("CPU_HISTOGRAM\tGPU_HISTOGRAM\n");
 
    for (int i =0; i< HISTOGRAM64_BIN_COUNT; i++)
    {
        printf("H_CPU[%d] : %d \t H_GPU[%d] : %d \n",i,h_hist[i],i,hist[i]);       
    }
}


void Cleanup(void)
{
    // Free device memory
    if (d_arr)
        cudaFree(d_arr);
    if (d_hist)
        cudaFree(d_hist);
    if (d_timer_start)
        cudaFree(d_timer_start);
    if (d_timer_stop)
        cudaFree(d_timer_stop);

    // Free host memory
    if (inp_arr)
        free(inp_arr);
    if (hist)
        free(hist);
    if (h_hist)
        free(h_hist);
    
  
    cudaEventDestroy(cstart);
    cudaEventDestroy(cstop);

    cudaEventDestroy(hstart);
    cudaEventDestroy(hstop);

    cudaEventDestroy(psum_start);
    cudaEventDestroy(psum_stop);
      
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
