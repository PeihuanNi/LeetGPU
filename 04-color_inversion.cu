#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < width * height) {
        image[idx*4] = 255 - image[idx*4];
        image[idx*4+1] = 255 - image[idx*4+1];
        image[idx*4+2] = 255 - image[idx*4+2];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}