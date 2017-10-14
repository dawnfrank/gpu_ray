#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mymath.cuh"
#include "sphere.cuh"
#include "ray.cuh"

extern "C" cudaError_t InitCuda(int w, int h, unsigned char** dev_bitmap);
extern "C" cudaError_t CalculateCuda(int w, int h, unsigned char* dev_bitmap, unsigned char* host_bitmap);
extern "C" void DeinitCuda(unsigned char* dev_bitmap);

__device__ Vec3 color(Ray &r, Sphere *dev_sp) {
//	hit_record rec;
//	if (dev_sp->hit(r, 0.001, DBL_MAX, rec)) {
//		return 0.5*Vec3(rec.normal.x + 1, rec.normal.y + 1, rec.normal.z + 1);
//	}
	double t = 0.5*(r.direction.y + 1.0);
	return Vec3(1.0, 1.0, 1.0)*(1.0 - t) + Vec3(0.5, 0.7, 1.0)*t;
}

__global__ void RayKernel(int w, int h, unsigned char* dev_bitmap,Sphere *dev_sp)
{
	Vec3 v1(1, 1, 1);
	Vec3 v2(0, 0, 0);

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	
	
	if (x<w && y<h) {
		int offset = x + y*w;

		double u, v;
		u = double(x) / double(w);
		v = double(y) / double(h);

		Vec3 lower_left_corner(-2.0, -1.5, -1.0);
		Vec3 horizontal(4.0, 0.0, 0.0);
		Vec3 vertical(0.0, 3.0, 0.0);
		Vec3 origin(0.0, 0.0, 0.0);

		Ray r(origin, lower_left_corner + horizontal*u + vertical*v);
		Vec3 pixel = color(r, dev_sp);
		
		dev_bitmap[offset * 4] = int(255.99*pixel.r);
		dev_bitmap[offset * 4 + 1] = int(255.99*pixel.g);
		dev_bitmap[offset * 4 + 2] = int(255.99*pixel.b);
		dev_bitmap[offset * 4 + 3] = 1;
	}
}

cudaError_t CalculateCuda(int w, int h, unsigned char* dev_bitmap, unsigned char* host_bitmap) {
	cudaError_t cudaStatus;
	int image_size = w * h * 4;

	Sphere *sp1 = (Sphere *)malloc(sizeof(Sphere));
	sp1->radius = 0.5;
	sp1->center.x = 0;
	sp1->center.y = 0;
	sp1->center.z = -1;
	
	Sphere *dev_sp=nullptr;
	cudaMalloc((void**)&dev_sp, sizeof(Sphere));
	cudaMemcpy(dev_sp, sp1, sizeof(Sphere), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU with one thread for each element.
	dim3 grids((w+31)/32, (h+31)/32); 
	dim3 threads(32, 32);
	RayKernel << <grids, threads >> >(w, h,dev_bitmap, dev_sp);

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

	cudaFree(dev_sp);
	free(sp1);

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
