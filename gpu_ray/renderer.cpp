#include <iostream>

#include "Renderer.h"
#include "Window.h"

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
	delete _window;
}

Window * Renderer::OpenWindow(uint32_t size_x, uint32_t size_y, std::wstring name)
{
	_window = new Window(this, size_x, size_y, name);

	return _window;
}

bool Renderer::Run()
{
	if (nullptr != _window) {
		_fps.Update();
		return _window->Update();
	}
	return true;
}
