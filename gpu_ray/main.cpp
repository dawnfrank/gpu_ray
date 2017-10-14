#include <iostream>
#include <cfloat>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "renderer.h"
#include "Window.h"

extern "C" cudaError_t InitCuda(int w, int h, unsigned char** dev_bitmap);
extern "C" cudaError_t CalculateCuda(int w, int h, unsigned char* dev_bitmap, unsigned char* host_bitmap);
extern "C" void DeinitCuda(unsigned char* dev_bitmap);

int main() {
	const int x = 400;
	const int y = 300;
	const int image_size = x * y * 4;

	Renderer render;
	auto w = render.OpenWindow(x, y, "test");

	unsigned char* dev_bitmap=nullptr;
	unsigned char host_bitmap[image_size];
	HDC hdc;
	int r, g, b;
	int pixel;

	InitCuda(x, y, &dev_bitmap);

	while (render.Run()) {
		hdc = BeginPaint(w->GetHandler(), &w->GetPainter());

		CalculateCuda(x, y, dev_bitmap, host_bitmap);

		for (int j = 0; j != y; ++j) {
			for (int i = 0; i != x; ++i) {
				pixel = i + x*(y-j-1);

				r = host_bitmap[pixel*4];
				g = host_bitmap[pixel*4+1];
				b = host_bitmap[pixel*4+2];

				SetPixel(hdc, i, j, RGB(r, g, b));
			}
		}
		
		EndPaint(w->GetHandler(), &w->GetPainter());
	}

	DeinitCuda(dev_bitmap);

	return 0;
}