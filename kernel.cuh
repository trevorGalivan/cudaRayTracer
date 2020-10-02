#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#define CUDA_VERSION 101000
#include <glm/glm.hpp>

__global__
void traceImage(cudaSurfaceObject_t output, unsigned int xRes, unsigned int yRes, int iterCap, glm::vec3 camPos, glm::vec3 camDir, glm::vec3 camUp, glm::vec3 camRight, curandState* cuRandtr);

__host__
curandState* setupRaytracer(unsigned int xRes, unsigned int yRes);