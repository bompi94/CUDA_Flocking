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
	void createGLStructures(GLuint *vbo, GLuint *VAO); 
	void saveBoidsRenderingData(GLuint * vbo, float* boidVertices, int numberOfBoids); 
	void loadBoidsVertices(GLuint * vbo); 
	void loadBoidsColor(GLuint * vbo); 
	void loadBoidsPosition(GLuint * vbo, GLuint * translationsVBO, float2 * positions, int numberOfBoids); 
	void allowInstancing(); 
	void drawObstacles(int numberOfObstacles, float2 * obstacleCenters, float * obstacleRadii);
	void drawBoids(int numberOfBoids, GLuint * translationsVBO, float2 * positions); 

	const char *windowTitle = "CUDA_Flocking";


private: 
	const unsigned int window_width = 512;
	const unsigned int window_height = 512;
	Shader* shPointer; 
};

#endif