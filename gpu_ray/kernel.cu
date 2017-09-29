#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" cudaError_t InitCuda(int w,int h,unsigned char** dev_bitmap);
extern "C" cudaError_t CalculateCuda(int w, int h, unsigned char* dev_bitmap,unsigned char* host_bitmap);
extern "C" void DeinitCuda(unsigned char* dev_bitmap);

__global__ void RayKernel(int w, int h,unsigned char* dev_bitmap)
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	dev_bitmap[i * 4] = int(255.99*double(i)/double(w));
	dev_bitmap[i * 4 + 1] = int(255.99*double(j) / double(w));
	dev_bitmap[i * 4 + 2] = int(255.99*0.2);
	dev_bitmap[i * 4 + 3] = 1;
}

cudaError_t CalculateCuda(int w, int h, unsigned char* dev_bitmap, unsigned char* host_bitmap) {
	cudaError_t cudaStatus;
	int image_size = w * h * 4;

	// Launch a kernel on the GPU with one thread for each element.
	dim3 grid(w, h);
	RayKernel << <grid, 1 >> >(w,h,dev_bitmap);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_bitmap, dev_bitmap, image_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	return cudaStatus;
}

cudaError_t InitCuda(int w, int h, unsigned char** dev_bitmap) {
	cudaError_t cudaStatus;
	int image_size = w * h * 4;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	// Allocate GPU buffers for three vectors (two input, one output)   
	cudaStatus = cudaMalloc((void**)dev_bitmap, image_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	return cudaStatus;
}

void DeinitCuda(unsigned char* dev_bitmap) {
	cudaFree(dev_bitmap);
}
