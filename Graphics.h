#pragma once
#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <helper_gl.h>
#include <GL/freeglut.h>
#include <shader.h>

class Graphics 
{
public:
	bool initialize(int *argc, char **argv);
	void drawCircle(float2 center, float r, int num_segments);
private: 
	const unsigned int window_width = 512;
	const unsigned int window_height = 512;
	const char *windowTitle = "CUDA_Flocking";
	Shader* shPointer; 
};

#endif