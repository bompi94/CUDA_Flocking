#include "CudaFlocking.h" 
#include "Obstacle.h"
#include "Graphics.h"

Graphics graphics;
CUDAFlocking simulation; 
int mouse_buttons = 0;
int *pArgc = NULL;
char **pArgv = NULL;


void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void prepareGraphicsToRenderBoids();
void startApplication(int argc, char ** argv);
void endApplication();
void registerGlutCallbacks();
void sendFlockToMouseClick(int x, int y);
float2 mouseToWorldCoordinates(int x, int y); 

int main(int argc, char **argv)
{
	printf("---CUDA FLOCKING--- \nboids %d  grid %dx%d neighbour limit %d \n", numberOfBoids, numberOfCells, numberOfCells, neighbourLimit);
	startApplication(argc, argv);
	printf("running...\n"); 
	glutMainLoop();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}


void cleanup()
{
	printf("clean up\n"); 
	simulation.freeCUDADataStructures(); 
	endApplication(); 
}

void registerGlutCallbacks()
{
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutCloseFunc(cleanup);
	printf("registered glut callbacks\n");
}


void display()
{
	graphics.startOfFrame();
	graphics.drawObstacles(numberOfObstacles, simulation.getObstacleCenters(), simulation.getObstacleRadii());
	simulation.calculateBoidsPositions();
	graphics.drawBoids(numberOfBoids, simulation.getPositions());
	graphics.endOfFrame();
}

void startApplication(int argc, char **argv)
{
	pArgc = &argc;
	pArgv = argv;
	printf("%s starting...\n", graphics.windowTitle);
	graphics.initialize(&argc, argv);
	registerGlutCallbacks();
	simulation.init(); 
	prepareGraphicsToRenderBoids(); 
}

void endApplication()
{
	printf("%s completed\n average time per frame -> %f \n\n", graphics.windowTitle, graphics.getAvgTimePerFrame());
	exit(0); 
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27):
		glutDestroyWindow(glutGetWindow());
		return;
	}
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
		sendFlockToMouseClick(x, y);
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

}

void prepareGraphicsToRenderBoids()
{
	graphics.createGLStructures();
	printf(" graphics->createdGLstructures\n");

	graphics.saveBoidsRenderingData((float*)boidVertices, 15);
	printf(" graphics->savedBoidsRenderingData\n");

	graphics.loadBoidsPosition(simulation.getPositions(), numberOfBoids);
	printf(" graphics->loadedBoidsPosition\n");

	graphics.allowInstancing();
	printf(" graphics->allowedInstancing\n");

	printf("prepared graphics to rendering\n");
}

void sendFlockToMouseClick(int x, int y)
{
	float2 destination = mouseToWorldCoordinates(x, y);
	simulation.setFlockDestination(destination);
}

float2 mouseToWorldCoordinates(int x, int y)
{
	float fX = (float)x / window_width;
	float fY = (float)y / window_width;
	fX = fX * 2 - 1;
	fY = -fY * 2 + 1;
	return make_float2(fX, fY);
}

