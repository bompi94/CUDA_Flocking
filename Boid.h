#pragma once

#include <vector_functions.h>
float boidRadius = 1;
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