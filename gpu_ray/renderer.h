#pragma once

#include <vector>

#include "fps.h"
#include "windows.h"

class Window;

class Renderer
{
public:
	Renderer();
	~Renderer();

	Window* OpenWindow(uint32_t size_x, uint32_t size_y, std::wstring name);

	bool Run();

private:

	Window *_window = nullptr;
	FPS _fps;
};