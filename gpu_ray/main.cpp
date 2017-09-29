#include <iostream>
#include <cfloat>

#include "renderer.h"
#include "Window.h"
#include "mymath.h"

int main() {
	int x = 400;
	int y = 300;
	HDC hdc;
	Renderer render;
	auto w = render.OpenWindow(x, y, TEXT("test"));

	while (render.Run()) {
		hdc = BeginPaint(w->GetHandler(), &w->GetPainter());
		for (int j = 0; j != y; ++j) {
			for (int i = 0; i != x; ++i) {
				Vec3 pixel(double(i) / double(x), double(j) / double(y), 0.2);
				SetPixel(hdc, i, j, RGB(int(pixel[0] * 255.99), int(pixel[1] * 255.99), int(pixel[2] * 255.99)));
			}
		}

		EndPaint(w->GetHandler(), &w->GetPainter());
	}
	return 0;
}