#include "CudaFlocking.h" 
#include "Obstacle.h"
#include "Graphics.h"

Graphics graphics;
CUDAFlocking simulation; 
int g_Index = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;
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
	printf("quinto (stream) approccio boids -> %d\n", numberOfBoids);
	startApplication(argc, argv);
	glutMainLoop();
	endApplication();
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
	//freeCUDADataStructures();
	printf("%s completed, returned %s\n", graphics.windowTitle, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
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

	mouse_old_x = x;
	mouse_old_y = y;
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

