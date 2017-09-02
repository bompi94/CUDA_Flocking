#include "Graphics.h"

bool Graphics::initialize(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("CUDA flocking");

	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	Shader shader("shaders/flock.vert", "shaders/flock.frag");
	shader.use();
	shPointer = (Shader*)malloc(sizeof(Shader));
	shPointer = &shader;

	return true;
}

void Graphics::drawCircle(float2 center, float r, int num_segments)
{
	r *= 2;
	glBegin(GL_LINE_LOOP);
	for (int ii = 0; ii < num_segments; ii++)
	{
		float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle
		float x = r * cosf(theta);//calculate the x component
		float y = r * sinf(theta);//calculate the y component
		glVertex2f(x + center.x * 2, y + center.y * 2);//output vertex
	}
	glEnd();
}

void Graphics::createGLStructures(GLuint *vbo, GLuint *VAO)
{
	assert(vbo);
	glGenVertexArrays(1, VAO);
	glGenBuffers(1, vbo);
	glBindVertexArray(*VAO);
}

void Graphics::saveBoidsRenderingData(GLuint * vbo, float* boidVertices, int numberOfBoids)
{
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, numberOfBoids * sizeof(float), &boidVertices[0], GL_DYNAMIC_DRAW);
	loadBoidsVertices(vbo);
	loadBoidsColor(vbo);
}

void Graphics::loadBoidsVertices(GLuint * vbo)
{
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
}

void Graphics::loadBoidsColor(GLuint * vbo)
{
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
}

void Graphics::loadBoidsPosition(GLuint * vbo, GLuint* translationsVBO, float2 * positions, int numberOfBoids)
{
	glGenBuffers(1, translationsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, *translationsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * numberOfBoids, &positions[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);
}

void Graphics::allowInstancing()
{
	glVertexAttribDivisorARB(2, 1);
}

void Graphics::drawObstacles(int numberOfObstacles, float2 * obstacleCenters, float * obstacleRadii)
{
	for (int i = 0; i < numberOfObstacles; i++)
	{
		float2 center = obstacleCenters[i];
		float radius = obstacleRadii[i];
		drawCircle(center, radius, 100);
	}
}

void Graphics::drawBoids(int numberOfBoids, GLuint * translationsVBO, float2 * positions)
{
	glBindBuffer(GL_ARRAY_BUFFER, *translationsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * numberOfBoids, &positions[0], GL_DYNAMIC_DRAW);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 3, numberOfBoids);
}


