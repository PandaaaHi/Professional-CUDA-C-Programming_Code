#include <stdio.h>

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU\n");
    // if (threadIdx.x == 5) {
    //     printf("Hello World from GPU thread 5\n");
    // }
    // printf("Hello World from GPU thread %d\n", threadIdx.x);
}

int main(void) {
    printf("Hello World from CPU\n");

    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    // cudaDeviceSynchronize();

    return 0;
}