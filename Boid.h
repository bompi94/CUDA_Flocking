#pragma once
#include <vector_functions.h>
float boidRadius = .05;
const unsigned int numberOfBoids = 1024;
float quadVertices[] = {
	// positions     // colors
	0.0f,  0.01f,  1.0f, 0.0f, 0.0f,
	0.01f, -0.01f,  0.0f, 1.0f, 0.0f,
	-0.01f, -0.01f,  0.0f, 0.0f, 1.0f
};