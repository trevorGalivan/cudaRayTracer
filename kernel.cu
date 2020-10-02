#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#define CUDA_VERSION 101000
#include <glm/glm.hpp>

#include <ctime>

__device__
const float pi = 3.1415962;



using vec3 = glm::vec3;
using vec4 = glm::vec4;
typedef struct Ray{
    vec3 pos;
    vec3 dir;
    __device__
    Ray(vec3 pos_, vec3 dir_) : pos(pos_), dir(dir_) {}
}Ray;

typedef struct Material { // REPLACE with model that accounts for glossiness, and diffuse reflection
    float roughness;
    float metalness;
    float emmisiveness;
    float IOR;
    vec3 colour;
    __device__
    Material() {};
    __device__
    Material(float roughness_, float metalness_, float emmisiveness_, float IOR_, vec3 colour_) : roughness(roughness_), metalness(metalness_), emmisiveness(emmisiveness_), IOR(IOR_), colour(colour_) {}
}Material;

typedef struct Intersection {
    vec3 pos;
    vec3 normal;
    Material mat;
    __device__
    Intersection(vec3 pos_, vec3 normal_, Material mat_) : pos(pos_), normal(normal_), mat(mat_) {}
}Intersection;

typedef struct Sphere {
    vec3 pos;
    Material mat;
    float radius;
    __device__
    Sphere() {};
    __device__
    Sphere(vec3 pos_, Material mat_, float radius_) : pos(pos_), mat(mat_), radius(radius_) {}
}Sphere;




__device__
float intersect(Ray ray, vec3 ballPos, float ballRad);

__device__
vec3 trace(const Ray ray, const Sphere scene[], const int numSpheres, const int iterCap, curandState* cuRandPtr, int index);

__device__
vec3 getDiffuseRay(curandState* globalRandState, vec3 normal, int ind);

__device__
vec3 tonemap(vec3 input, float exposure);

__global__
void setupRNG(curandState* state, unsigned long int seed, unsigned int xRes, unsigned int yRes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < xRes * yRes) {
        curand_init(seed, index, 0, &state[index]);
    }
}

__host__
curandState* setupRaytracer(unsigned int xRes, unsigned int yRes)
{
    curandState* cuRandPtr;
    cudaMalloc(&cuRandPtr, xRes * yRes * sizeof(curandState));

    srand(time(NULL));

    setupRNG <<<(xRes * yRes) / 128 + 1, 128 >>> (cuRandPtr, rand(), xRes, yRes);

    return cuRandPtr;
}

__global__
void traceImage(cudaSurfaceObject_t output, unsigned int xRes, unsigned int yRes, int iterCap, vec3 camPos, vec3 camDir, vec3 camUp, vec3 camRight, curandState* cuRandPtr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;

    if(index < xRes * yRes)
    {
        int x = index % xRes;
        int y = index / xRes;

        
        glm::vec2 normalized = glm::vec2(x, y) / glm::vec2(xRes, yRes) * 2.f - 1.f; // [0, 1)

        vec3 rayDirection = camDir * 2.f + camUp * normalized.y + camRight * normalized.x;
        rayDirection = normalize(rayDirection);

        Ray baseRay = { camPos, rayDirection };


        const Sphere scene[9] = { Sphere(vec3(0.f), Material(1.f, 0.f, 0.f, 2.5f,  vec3(0.2f, 0.9f, 0.2f)), 1.f),
                                  Sphere(vec3(3.f), Material(0.5f, 1.f, 0.0f, 1.5f, vec3(0.9f, 0.9f, 0.2f)), 0.5f),
                                  Sphere(vec3(1.5f), Material(1.f, 0.f, 0.f, 1.1f, vec3(0.2f, 0.9f, 0.9f)), 1.f),
                                  Sphere(vec3(6.f), Material(1.f, 0.f, 0.f, 1.1f, vec3(0.9f, 0.9f, 0.1f)), 1.f),
                                  Sphere(vec3(1.f, -5.f, 1.f), Material(0.1f, 0.f, 1.5f, 1.5f, vec3(0.9f, 0.2f, 0.2f)), 3.9f),
                                  Sphere(vec3(1.f, 5.f, 10.f), Material(0.1f, 0.f, 2.5f, 1.5f, vec3(0.3f, 0.5f, 0.8f)), 1.5f),
                                  Sphere(vec3(-1.f, 0.f, 5.f), Material(1.f, 1.f, 0.f, 2.5f, vec3(0.2f, 0.2f, 0.9f)), 1.f),
                                  Sphere(vec3(0.f, -50010.f, 0.f), Material(0.1f, 0.0f, 0.f, 1.1f, vec3(0.25f, 0.2f, 0.25f)), 50000.f),
                                  Sphere(vec3(0.f, 0.f, -50010.f), Material(0.1f, 1.0f, 0.f, 7.1f, vec3(0.5f, 0.5f, 0.5f)), 50000.f)};

        float4 data = *(reinterpret_cast<float4*> (&vec4( tonemap( trace(baseRay, scene, 9, iterCap, cuRandPtr, index), 1.f ), 1) ));

        surf2Dwrite(data, output, x * sizeof(float4), y);
    }


}

__device__
vec3 tonemap(vec3 input, float exposure)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return exposure * (input * (a * input + b)) / (input * (c * input + d) + e);
}

__device__
float intersect(const Ray ray, const vec3 ballPos, const float ballRad)
{
    vec3 oc = ray.pos - ballPos;
    //float a = dot(rayDirection, rayDirection);
    float b = 2.0 * dot(oc, ray.dir);
    float c = dot(oc, oc) - ballRad * ballRad;
    float discriminant = b * b - 4.f * c;
    return (discriminant > 0.f && -1.f * sqrt(discriminant) - b > 0.f) ? (-1.f * sqrt(discriminant) - b) / (2.f) : 1000000000.f;//dot((camPosition + ( ((-1*sqrt(discriminant) - b) / (2.*a)) * rayDirection) - ballPos), vec3(0., 1., 0) ): 0.1;
}

__device__
vec3 trace(const Ray ray, const Sphere scene[], const int numSpheres, const int iterCap, curandState* cuRandPtr, int index)
{
    if (iterCap == 0) {
        return vec3(0.f);
    }

    const Material skyMaterial = { 0.f, 0.f, 0.f, 1.f, vec3{0.0f, 0.0f, 0.0f} };

    Sphere nearestHit;
    float hitDist = 10000000.f;
    float recentHit;

    for (int i = 0; i < numSpheres; i++) {
        if ((recentHit = intersect(ray, scene[i].pos, scene[i].radius)) < hitDist) {
            hitDist = recentHit;
            nearestHit = scene[i];
        }
    }

    if (hitDist == 10000000.f) {
        return skyMaterial.colour;
    }

    vec3 position = ray.dir * hitDist + ray.pos;

    vec3 normal = glm::normalize(position - nearestHit.pos);

    float R = pow((1.f - nearestHit.mat.IOR) / (1.f + nearestHit.mat.IOR), 2.f);
    float fresnel = R + (1.f - R) * pow( (1.f + glm::dot(ray.dir, normal) ) , 5.f);
    
    vec3 finalColour = fresnel * (vec3(1.f) * (1.f - nearestHit.mat.metalness) + nearestHit.mat.colour * nearestHit.mat.metalness) * trace(Ray(position, glm::reflect(ray.dir, normal)), scene, numSpheres, iterCap - 1, cuRandPtr, index);
    finalColour += (1.f - fresnel) * (1.f - nearestHit.mat.metalness) * nearestHit.mat.colour * trace(Ray(position, getDiffuseRay(cuRandPtr, normal, index)), scene, numSpheres, iterCap-1, cuRandPtr, index); // This line is buggy, but the bug looks really cool
    finalColour += nearestHit.mat.emmisiveness * nearestHit.mat.colour;

    return vec3(finalColour);
}

__device__
vec3 getDiffuseRay(curandState* globalRandState, vec3 normal, int ind)
{
    // Generates two random angles
    curandState localState = globalRandState[ind];
    float horiz = curand_uniform(&localState) * 2.f * pi; // (0, 2*pi]
    float vert  = curand_uniform(&localState)  * pi; // (0, pi]
    globalRandState[ind] = localState;
    // Creates spherical distribution of unit vectors
    vec3 diffuse(sin(vert) * sin(horiz), cos(vert), sin(vert) * cos(horiz));

    // Converts to hemispherical distribution in direction of normal vector
    return glm::dot(diffuse, normal) > 0.f ? diffuse : -1.f * diffuse;
}