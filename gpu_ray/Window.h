#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "windows.h"
#include "renderer.h"

class Window
{
public:
	Window(Renderer *render, uint32_t size_x, uint32_t size_y, std::wstring name);
	~Window();

	void Close();
	bool Update();

	Renderer* GetRenderer() { return _renderer; }
	HWND GetHandler() { return _win32_window; }
	PAINTSTRUCT GetPainter() { return _ps; }

	//	HDC GetHDC() { return _hdc; }
	//	void SetHDC(HDC hdc) { _hdc = hdc; }

private:
	void _InitOSWindow();
	void _DeInitOsWindow();
	void _UpdateOSWindow();


	Renderer *_renderer = nullptr;

	uint32_t _surface_size_x = 512;
	uint32_t _surface_size_y = 512;
	std::wstring _window_name;

	bool _window_should_run = true;

	HINSTANCE	_win32_instance = NULL;
	HWND _win32_window = NULL;
	std::wstring _win32_class_name;
	static uint64_t _win32_class_id_counter;

	PAINTSTRUCT   _ps;
	//	HDC           _hdc;
};