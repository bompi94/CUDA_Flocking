#pragma once
#ifndef OBSTACLE_H
#define OBSTACLE_H
#include <vector_functions.h>
class Obstacle
{
public:
	float2 center;
	float radius;
	bool initialized = false;
};

const unsigned int numberOfObstacles = 3;
float obstacleRadius = 0.05;
float2 obstacleCenters[numberOfObstacles];
float obstacleRadii[numberOfObstacles];

#endif
