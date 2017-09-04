#pragma once
#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <helper_gl.h>
#include <GL/freeglut.h>
#include <shader.h>
#include <helper_timer.h>
#include <windows.h>

#define MAX(a,b) ((a > b) ? a : b)

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10//ms

const unsigned int window_width = 512;
const unsigned int window_height = 512;

class Graphics
{

public:
	bool initialize(int *argc, char **argv);
	void drawCircle(float2 center, float r, int num_segments);
	void createGLStructures();
	void saveBoidsRenderingData( float* boidVertices, int numberOfBoids);
	void loadBoidsVertices();
	void loadBoidsColor();
	void loadBoidsPosition(float2 * positions, int numberOfBoids);
	void allowInstancing();
	void drawObstacles(int numberOfObstacles, float2 * obstacleCenters, float * obstacleRadii);
	void drawBoids(int numberOfBoids, float2 * positions);
	void computeFPS();
	void startOfFrame();
	void endOfFrame();

	const char *windowTitle = "CUDA_Flocking";

private:
	Shader* shPointer;
	int fpsCount;        // FPS count for averaging
	int fpsLimit;        // FPS limit for sampling
	unsigned int frameCount;
	float avgFPS;
	StopWatchInterface *timer;
	unsigned int VAO;
	GLuint vbo;
	GLuint translationsVBO;
};

#endif