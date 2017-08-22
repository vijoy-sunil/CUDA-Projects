
#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_

__global__ void SobelFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
   __shared__ unsigned char sharedMem[BLOCK_HEIGHT * BLOCK_WIDTH];
   float s_SobelMatrix[9];

    s_SobelMatrix[0] = -1;
    s_SobelMatrix[1] = 0;
    s_SobelMatrix[2] = 1;

    s_SobelMatrix[3] = -2;
    s_SobelMatrix[4] = 0;
    s_SobelMatrix[5] = 2;

    s_SobelMatrix[6] = -1;
    s_SobelMatrix[7] = 0;
    s_SobelMatrix[8] = 1;

   // Computer the X and Y global coordinates
   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

   // Get the Global index into the original image
   int index = y * (width) + x;

   // Perform the first load of values into shared memory
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
   sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();

   // STUDENT:  Check 1
   // Handle the extra thread case where the image width or height 
   // 

   if (x >= width || y >= height)
      return;

   // STUDENT: Check 2
   // Handle the border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if(x == blockIdx.x * TILE_WIDTH) //left
       return;

    if(x > blockIdx.x * TILE_WIDTH + TILE_WIDTH) //right
       return;

    if(y == blockIdx.y * TILE_HEIGHT) //top
       return;

    if(y > blockIdx.y * TILE_HEIGHT + TILE_HEIGHT)
       return; //bottom
   
   // STUDENT: Make sure only the thread ids should write the sum of the neighbors.
                // g_DataOut[index] = abs(sumX) + abs(sumY) > EDGE_VALUE_THRESHOLD ? 255 : 0;
    float sumX = 0,sumY = 0;
    for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++)
    {
        for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++)
        {
            int my_index = sharedIndex + (dy * blockDim.x + dx);
            float Pixel =  (float)(sharedMem[my_index]);
            //int my_index = y * width + x + (dy * width + dx);

            sumX += Pixel * s_SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx + FILTER_RADIUS)];

            sumY += Pixel * s_SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy + FILTER_RADIUS)];
        }
    }
    g_DataOut[index] = abs(sumX) + abs(sumY) > EDGE_VALUE_THRESHOLD ? 255 : 0;

}


__global__ void AverageFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
    __shared__ unsigned char sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

   // Get the Global index into the original image
   int index = y * (width) + x;

  // STUDENT: write code for Average Filter : use Sobel as base code
  // Perform the first load of values into shared memory
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
   sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();

   // STUDENT:  Check 1
   // Handle the extra thread case where the image width or height 
   // 

   if (x >= width || y >= height)
      return;

   // STUDENT: Check 2
   // Handle the border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if(x == blockIdx.x * TILE_WIDTH) //left
       return;

    if(x > blockIdx.x * TILE_WIDTH + TILE_WIDTH) //right
       return;

    if(y == blockIdx.y * TILE_HEIGHT) //top
       return;

    if(y > blockIdx.y * TILE_HEIGHT + TILE_HEIGHT)
       return; //bottom
   
   // STUDENT: Make sure only the thread ids should write the sum of the neighbors.

    float sum = 0;
    for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++)
    {
        for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++)
        {
            int my_index = sharedIndex + (dy * blockDim.x + dx);
            float Pixel =  (float)(sharedMem[my_index]);            

            sum += Pixel; 

        }
    }
    g_DataOut[index] = (unsigned char)(sum/FILTER_AREA);
}



__global__ void HighBoostFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
   __shared__ unsigned char sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

   // Get the Global index into the original image
   int index = y * (width) + x;


   // Perform the first load of values into shared memory
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
   sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();

   // STUDENT:  Check 1
   // Handle the extra thread case where the image width or height 
   // 

   if (x >= width || y >= height)
      return;

   // STUDENT: Check 2
   // Handle the border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if(x == blockIdx.x * TILE_WIDTH) //left
       return;

    if(x > blockIdx.x * TILE_WIDTH + TILE_WIDTH) //right
       return;

    if(y == blockIdx.y * TILE_HEIGHT) //top
       return;

    if(y > blockIdx.y * TILE_HEIGHT + TILE_HEIGHT)
       return; //bottom
   
   // STUDENT: Make sure only the thread ids should write the sum of the neighbors.
  // STUDENT: write code for High Boost Filter : use Sobel as base code

    unsigned char centerPixel = sharedMem[sharedIndex];

    float sum = 0;
    for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++)
    {
        for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++)
        {
            int my_index = sharedIndex + (dy * blockDim.x + dx);
            float Pixel =  (float)(sharedMem[my_index]);            

            sum += Pixel; 

        }
    }
    g_DataOut[index] = CLAMP_8bit(centerPixel + HIGH_BOOST_FACTOR * (unsigned char)(centerPixel - sum/FILTER_AREA));
}


#endif // _FILTER_KERNEL_H_


