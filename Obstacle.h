#pragma once
#ifndef OBSTACLE_H
#define OBSTACLE_H
class Obstacle
{
public:
	float2 center;
	float radius;
	bool initialized = false;
};

const unsigned int numberOfObstacles = 3;
const float obstacleRadius = 0.05;


#endif
