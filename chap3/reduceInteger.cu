#include<stdio.h>
#include<sys/time.h>
#include<cuda_runtime.h>

double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int recursiveReduce(int *data, int const size) {
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++) {
        data[i] += data[i+stride];
    }

    return recursiveReduce(data, stride);
}

__global__ void warmup(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride) == 0)) {
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride) == 0)) {
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index+stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx+blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_idata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    if (idx + blockDim.x * 3 < n) {
        for (int i = 1; i <= 3; i++) {
            g_idata[idx] += g_idata[idx+blockDim.x*i];
        }
        // int a0 = g_idata[idx];
        // int a1 = g_idata[idx+blockDim.x];
        // int a2 = g_idata[idx+2*blockDim.x];
        // int a3 = g_idata[idx+3*blockDim.x];
        // g_idata[idx] = a0 + a1 + a2 + a3;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_idata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < n) {
        for (int i = 1; i <= 7; i++) {
            g_idata[idx] += g_idata[idx+blockDim.x*i];
        }
        // int a0 = g_idata[idx];
        // int a1 = g_idata[idx+blockDim.x];
        // int a2 = g_idata[idx+2*blockDim.x];
        // int a3 = g_idata[idx+3*blockDim.x];
        // int a4 = g_idata[idx+4*blockDim.x];
        // int a5 = g_idata[idx+5*blockDim.x];
        // int a6 = g_idata[idx+6*blockDim.x];
        // int a7 = g_idata[idx+7*blockDim.x];
        // g_idata[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_idata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < n) {
        // for (int i = 1; i <= 7; i++) {
        //     g_idata[idx] += g_idata[idx+blockDim.x*i];
        // }
        int a0 = g_idata[idx];
        int a1 = g_idata[idx+blockDim.x];
        int a2 = g_idata[idx+2*blockDim.x];
        int a3 = g_idata[idx+3*blockDim.x];
        int a4 = g_idata[idx+4*blockDim.x];
        int a5 = g_idata[idx+5*blockDim.x];
        int a6 = g_idata[idx+6*blockDim.x];
        int a7 = g_idata[idx+7*blockDim.x];
        g_idata[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }

    if (tid == 0) g_idata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < n) {
        int a0 = g_idata[idx];
        int a1 = g_idata[idx+blockDim.x];
        int a2 = g_idata[idx+2*blockDim.x];
        int a3 = g_idata[idx+3*blockDim.x];
        int a4 = g_idata[idx+4*blockDim.x];
        int a5 = g_idata[idx+5*blockDim.x];
        int a6 = g_idata[idx+6*blockDim.x];
        int a7 = g_idata[idx+7*blockDim.x];
        g_idata[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid+512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid+256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid+128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid+64];
    __syncthreads();

    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }

    if (tid == 0) g_idata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < n) {
        int a0 = g_idata[idx];
        int a1 = g_idata[idx+blockDim.x];
        int a2 = g_idata[idx+2*blockDim.x];
        int a3 = g_idata[idx+3*blockDim.x];
        int a4 = g_idata[idx+4*blockDim.x];
        int a5 = g_idata[idx+5*blockDim.x];
        int a6 = g_idata[idx+6*blockDim.x];
        int a7 = g_idata[idx+7*blockDim.x];
        g_idata[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid+512];
    __syncthreads();
    if (iBlockSize >= 512 && tid < 256) idata[tid] += idata[tid+256];
    __syncthreads();
    if (iBlockSize >= 256 && tid < 128) idata[tid] += idata[tid+128];
    __syncthreads();
    if (iBlockSize >= 128 && tid < 64) idata[tid] += idata[tid+64];
    __syncthreads();

    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }

    if (tid == 0) g_idata[blockIdx.x] = idata[0];
}

__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize) {
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int istride = isize >> 1;
    if (istride > 1 && tid < istride) {
        idata[tid] += idata[tid + istride];
    }

    __syncthreads();

    if (tid == 0) {
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
        // cudaDeviceSynchronize(); // removed for compute capability 9.0 and higher
    }

    __syncthreads();
}

__global__ void gpuRecursiveReduceNosync(int *g_idata, int *g_odata, unsigned int isize) {
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int istride = isize >> 1;
    if (istride > 1 && tid < istride) {
        idata[tid] += idata[tid+istride];
        if (tid == 0) {
            gpuRecursiveReduceNosync<<<1, istride>>>(idata, odata, istride);
        }
    }
}

__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, int iStride, int const iDim) {
    int *idata = g_idata + blockIdx.x * iDim;

    if (iStride == 1 && threadIdx.x == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    idata[threadIdx.x] += idata[threadIdx.x+iStride];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gpuRecursiveReduce2<<<gridDim.x, iStride/2>>>(g_idata, g_odata, iStride/2, iDim);
    }
}

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false;

    // initialization
    int size = 1<<24;
    printf(" with array size %d ", size);

    // execution configuration
    int blocksize = 512;
    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }
    dim3 block(blocksize, 1);
    dim3 grid((size+block.x-1)/blocksize, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocatte host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int*)malloc(bytes);
    int *h_odata = (int*)malloc(grid.x*sizeof(int));
    int *tmp = (int*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) {
        h_idata[i] = (int)(rand() % 0xff);
    }
    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void**)&d_idata, bytes);
    cudaMalloc((void**)&d_odata, grid.x*sizeof(int));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce elapsed %f s cpu_sum: %d\n", iElaps, cpu_sum);

    // warmup
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu Warmup elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);
    
    // kernel 1: reduceNeighbored
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceNeighbored elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighboredLess
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceNeighboredLess elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceInterleaved elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling2<<<grid.x/2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceUnrolling2 elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/2, block.x);

    // kernel 5: reduceUnrolling4
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling4<<<grid.x/4, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceUnrolling4 elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/4, block.x);

    // kernel 6: reduceUnrolling8
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceUnrolling8 elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

    // kernel 7: reduceUnrollWarps8
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceUnrollWarps8 elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

    // kernel 8: reduceCompleteUnrollWarps8
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceCompleteUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceCompleteUnrollWarps8 elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

    // kernel 9: reduceCompleteUnroll
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    switch (blocksize)
    {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 512:
        reduceCompleteUnroll<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 256:
        reduceCompleteUnroll<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 128:
        reduceCompleteUnroll<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 64:
        reduceCompleteUnroll<64><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    default:
        break;
    }
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceCompleteUnroll elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

    // kernel 9: gpuRecursiveReduce
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    gpuRecursiveReduce<<<grid.x, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu gpuRecursiveReduce elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 10: gpuRecursiveReduceNosync
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    gpuRecursiveReduceNosync<<<grid.x, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu gpuRecursiveReduceNosync elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 11: gpuRecursiveReduce2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    gpuRecursiveReduce2<<<grid.x, block>>>(d_idata, d_odata, size/2, block.x);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu gpuRecursiveReduce2 elapsed %f s gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);

    // reset device
    cudaDeviceReset();

    // check the results
    bResult = (gpu_sum == cpu_sum);
    if (!bResult) printf("Test failed!\n");
    else printf("Test succeed!\n");

    return EXIT_SUCCESS;
}