#pragma once

#include <vector_functions.h>
float boidRadius = 0.05;

const unsigned int numberOfBoids = 100;

class Boid {

public:
	Boid()
	{

	}

	Boid(float2 position)
	{
		this->position = position; 
	}
	float2 position; 
	float2 velocity; 

};