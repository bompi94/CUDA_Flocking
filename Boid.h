#pragma once

#ifndef BOID_H
#define BOID_H

const float boidRadius = .05;
const unsigned int numberOfBoids = 20000;
const int neighbourLimit = 7;
const float boidVertices[] = {
	// positions     // colors
	0.0f,  0.01f,  1.0f, 0.0f, 0.0f,
	0.01f, -0.01f,  0.0f, 1.0f, 0.0f,
	-0.01f, -0.01f,  0.0f, 0.0f, 1.0f
};

#endif