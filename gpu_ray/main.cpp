#include <iostream>
#include <cfloat>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "renderer.h"
#include "Window.h"
#include "mymath.h"

extern "C" cudaError_t InitCuda(int w, int h, unsigned char** dev_bitmap);
extern "C" cudaError_t CalculateCuda(int w, int h, unsigned char* dev_bitmap, unsigned char* host_bitmap);
extern "C" void DeinitCuda(unsigned char* dev_bitmap);


int main() {
	const int x = 400;
	const int y = 300;
	HDC hdc;
	Renderer render;
	auto w = render.OpenWindow(x, y, TEXT("test"));

	unsigned char host_bitmap[x*y * 4];
	unsigned char *dev_bitmap=nullptr;
	cudaError_t cudaStatus;

	cudaStatus=InitCuda(x, y,&dev_bitmap);

	while (render.Run()) {
		hdc = BeginPaint(w->GetHandler(), &w->GetPainter());

		CalculateCuda(x, y, dev_bitmap, host_bitmap);

		/*
		for (int j = 0; j != y; ++j) {
			for (int i = 0; i != x; ++i) {
				Vec3 pixel(double(i) / double(x), double(j) / double(y), 0.2);
				SetPixel(hdc, i, j, RGB(int(pixel[0] * 255.99), int(pixel[1] * 255.99), int(pixel[2] * 255.99)));
			}
		}
		*/
		EndPaint(w->GetHandler(), &w->GetPainter());
	}

	DeinitCuda(dev_bitmap);

	return 0;
}