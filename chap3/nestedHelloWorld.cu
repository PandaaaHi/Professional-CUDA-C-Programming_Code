#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nestedHelloWorld(int const iSize, int iDepth) {
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d"
        " block %d\n", iDepth, tid, blockIdx.x);

    if (iSize == 1) return;

    int nthreads = iSize >> 1;

    if (tid == 0 && nthreads > 0) {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-----> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("device %d: %s", dev, deviceProp.name);
    cudaSetDevice(dev);

    int grid = 1;
    int block = 8;
    if (argc > 1) {
        grid = atoi(argv[1]);
    }
    if (argc > 2) {
        block = atoi(argv[2]);
    }

    nestedHelloWorld<<<grid, block>>>(block, 0);
    cudaDeviceSynchronize();

    cudaDeviceReset();

    return EXIT_SUCCESS;
}