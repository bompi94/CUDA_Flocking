#version 330 core

uniform float time;
uniform vec2 resolution;
uniform vec4 mouse;

//position of the player controlled racket
uniform vec3 racketPos;
//size of the racket
uniform vec3 racketSize; 
//position of the racket not controlled by the player
uniform vec3 otherRacketPos; 
//position of the ball
uniform vec3 ballPos; 
//radius of the ball
uniform float ballRadius; 
//tells if the ball should be perturbated or not 
uniform bool perturbation; 


out vec4 fragColor;

//------ costanti 

#define PI 3.1415926535898 


//----- variabili

const float eps = 0.005;

const int maxIterations = 128;
const float stepScale = 0.5;
const float stopThreshold = 0.005;

//----- SDF e scena

float sphere(in vec3 p, in vec3 centerPos, float radius){

  return length(p-centerPos) - radius;
}

float box(vec3 p, in vec3 centerPos, vec3 b){

  vec3 d = abs(p - centerPos) - b;

  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sinusoidalPlasma(in vec3 p){

    return sin(p.x+time*2.)*cos(p.y+time*2.1)*sin(p.z+time*2.3) + 0.25*sin(p.x*2.)*cos(p.y*2.)*sin(p.z*2.);
}

float scene(in vec3 p) {

  float d0 = sphere(p, ballPos, ballRadius); 
  if(perturbation)
  	d0 += 0.03*sinusoidalPlasma(p*8.0); 
  else
  	d0 += 0.05*sinusoidalPlasma(p*11.0); 
  float d1 = box(p, racketPos, racketSize);
  float d2 = box(p, otherRacketPos, racketSize);

  return min(d2,min(d0,d1)); 
}


//----- funzioni di utilità 

vec3 getNormal(in vec3 p) {

  return normalize(vec3(
    scene(vec3(p.x+eps,p.y,p.z))-scene(vec3(p.x-eps,p.y,p.z)),
    scene(vec3(p.x,p.y+eps,p.z))-scene(vec3(p.x,p.y-eps,p.z)),
    scene(vec3(p.x,p.y,p.z+eps))-scene(vec3(p.x,p.y,p.z-eps))
  ));
}

//----- algoritmo di raymarching 

float rayMarching( vec3 origin, vec3 dir, float start, float end ) {

  float sceneDist = 1e4;
  float rayDepth = start; 
  for ( int i = 0; i < maxIterations; i++ ) {

    sceneDist = scene( origin + dir * rayDepth ); // Distance from the point along the ray to the nearest surface point in the scene.

    if (( sceneDist < stopThreshold ) || (rayDepth >= end)) {
      break;
    }
    // Non abbiamo colpito nulla, perciò aumentiamo la profondità del raggio proporzionalmente alla distanza della scena
    rayDepth += sceneDist * stepScale;

  }

  if ( sceneDist >= stopThreshold ) 
    rayDepth = end;
  else 
    rayDepth += sceneDist;

  return rayDepth;
}

//----- funzione principale

void main(void) {

  vec2 aspect = vec2(resolution.x/resolution.y, 1.0);
  vec2 screenCoords = (2.0*gl_FragCoord.xy/resolution.xy - 1.0)*aspect;

  vec3 lookAt = vec3(0.,0.,0.);  
  vec3 camPos = vec3(0., 4., -5.); 

  // Camera setup.
  vec3 forward = normalize(lookAt-camPos);
  vec3 right = normalize(vec3(forward.z, 0., -forward.x )); 
  vec3 up = normalize(cross(forward,right)); 

  // FOV - Field of view.
  float FOV = 0.5;

  // ro - Ray origin.
  vec3 ro = camPos;
  // rd - Ray direction. 
  vec3 rd = normalize(forward + FOV*screenCoords.x*right + FOV*screenCoords.y*up);

  const float clipNear = 0.0;
  const float clipFar = 8.0;
  float dist = rayMarching(ro, rd, clipNear, clipFar ); 
  if ( dist >= clipFar ) {
  	  //Se siamo a questo punto, non abbiamo colpito nulla e mostriamo il colore di sfondo
      fragColor = vec4(0.5,0.5,0.5, 1.0);
      return;
  }

  // sp - Surface position. Se siamo a questo punto, abbiamo colpito qualcosa.
  vec3 sp = ro + rd*dist;

  vec3 surfNormal = getNormal(sp);

  //lp = light position
  vec3 lp = vec3(0,2,0);
  // ld - Light direction.
  vec3 ld = vec3(0,1,0);
  // lcolor - Light color. 
  vec3 lcolor = vec3(1.,0.97,0.92);


  // Light falloff (attenuation)
  float len = length( ld ); // Distanza dalla luce alla superficie colpita
  float lightAtten = min( 1.0 / ( 0.25*len*len ), 1.0 ); // Mantiene il valore fra 0 e 1, proporzionalmente al quadrato della distanza

  ld /= len; // Normalizzazione del vettore ld

  vec3 ref = reflect(-ld, surfNormal);

  vec3 sceneColor = vec3(0.0); //colore che verrà modificato e restituito

  //colore per tutti gli oggetti della scena
  vec3 objColor = vec3(0,1,0);

  float ambient = .05; 
  float diffuse = max( 0.0, dot(surfNormal, ld) );
  float specular = max( 0.0, dot( ref, normalize(camPos-sp)) );

  sceneColor += (objColor*(diffuse*0.8+ambient)+specular*0.2)*lcolor;

  fragColor = vec4(clamp(sceneColor, 0.0, 1.0), 1.0);

}