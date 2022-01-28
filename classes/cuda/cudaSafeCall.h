/*
 * Author: Tanczos Ladislav
 * Source: ml/EulerVijayasundaram.h
 */
#include "stdio.h"

#ifndef CUDA_SAFE_CALL
    #if 0
        #define CUDA_SAFE_CALL
    #else
        #define CUDA_SAFE_CALL( call) do {\
                    cudaError_t err = call;                                                    \
                    if( cudaSuccess != err) {                                                \
                        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                        __FILE__, __LINE__, cudaGetErrorString( err) );              \
                    } } while (0)
    #endif
#endif
