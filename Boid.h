#ifndef BOID_H
#define BOID_H
#include <vector_functions.h>

float boidRadius = .05;
const unsigned int numberOfBoids = 2000;
float boidVertices[] = {
	// positions     // colors
	0.0f,  0.01f,  1.0f, 0.0f, 0.0f,
	0.01f, -0.01f,  0.0f, 1.0f, 0.0f,
	-0.01f, -0.01f,  0.0f, 0.0f, 1.0f
};

#endif