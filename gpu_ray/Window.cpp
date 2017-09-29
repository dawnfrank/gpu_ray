#include <iostream>
#include <assert.h>

#include "Window.h"
#include "renderer.h"

Window::Window(Renderer * renderer, uint32_t size_x, uint32_t size_y, std::wstring name)
{
	_renderer = renderer;
	_surface_size_x = size_x;
	_surface_size_y = size_y;
	_window_name = name;

	_InitOSWindow();
}

Window::~Window()
{
	_DeInitOsWindow();
}

void Window::Close()
{
	_window_should_run = false;
}

bool Window::Update()
{
	_UpdateOSWindow();
	return _window_should_run;
}


LRESULT CALLBACK WindowsEventHandler(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	Window *window = reinterpret_cast<Window*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
	switch (uMsg) {
	case WM_CLOSE:
		window->Close();
		return 0;
	case WM_SIZE:
		break;
	case WM_PAINT:
		break;
	default:
		break;
	}
	return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}

uint64_t Window::_win32_class_id_counter = 0;

void Window::_InitOSWindow() {
	WNDCLASSEX win_class{};
	assert(_surface_size_x > 0);
	assert(_surface_size_y > 0);

	_win32_instance = GetModuleHandle(nullptr);
	_win32_class_name = _window_name + TEXT("_") + std::to_wstring(_win32_class_id_counter);
	_win32_class_id_counter++;

	//Initlize the window class structure
	win_class.cbSize = sizeof(WNDCLASSEX);
	win_class.style = CS_HREDRAW | CS_VREDRAW;
	win_class.lpfnWndProc = WindowsEventHandler;
	win_class.cbClsExtra = 0;
	win_class.cbWndExtra = 0;
	win_class.hInstance = _win32_instance;
	win_class.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	win_class.hCursor = LoadCursor(NULL, IDC_ARROW);
	win_class.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	win_class.lpszMenuName = NULL;
	win_class.lpszClassName = _win32_class_name.c_str();
	win_class.hIconSm = LoadIcon(NULL, IDI_WINLOGO);
	//Register WIndow class
	if (!RegisterClassEx(&win_class)) {
		DWORD errCode = GetLastError();
		std::cout << errCode << std::endl;;
		assert(0 && "Cannot create a window in which to draw!\n");
		fflush(stdout);
		std::exit(-1);
	}

	DWORD ex_style = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
	DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;

	//create window with the registered class;
	RECT wr = { 0, 0, LONG(_surface_size_x), LONG(_surface_size_y) };
	AdjustWindowRectEx(&wr, style, FALSE, ex_style);
	_win32_window = CreateWindowEx(0,
		_win32_class_name.c_str(),
		_window_name.c_str(),
		style,
		CW_USEDEFAULT, CW_USEDEFAULT,
		wr.right - wr.left,
		wr.bottom - wr.top,
		NULL,
		NULL,
		_win32_instance,
		NULL);
	if (!_win32_window) {
		assert(1 && "Cannot create a window in which to draw!\n");
		std::exit(-1);
	}
	SetWindowLongPtr(_win32_window, GWLP_USERDATA, (LONG_PTR)this);

	ShowWindow(_win32_window, SW_SHOW);
	SetForegroundWindow(_win32_window);
	SetFocus(_win32_window);
}

void Window::_DeInitOsWindow()
{
	DestroyWindow(_win32_window);
	UnregisterClass(_win32_class_name.c_str(), _win32_instance);
}

void Window::_UpdateOSWindow()
{
	MSG msg;
	if (PeekMessage(&msg, _win32_window, 0, 0, PM_REMOVE)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}