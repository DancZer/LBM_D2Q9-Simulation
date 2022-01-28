/*
 * Author: Tanczos Ladislav
 */
#ifndef LBMPARTICLE_H
#define LBMPARTICLE_H

#define REAL float

//#define CUDA

//const values for calculatin ffeq
#define LBM_G1 (3.0f)
#define LBM_G2 (4.5f)
#define LBM_G3 (-1.5f)

//const values for calculating tau
#define LBM_M1 (0.5f)
#define LBM_M2 (3.0f)

//lattice weights
#define LBM_T0 (4./9.)
#define LBM_TS (1./9.)
#define LBM_TL (1./36.)

#include "cuda/cudaAlign.h"

#define LBM_VECT_COUNT 9

struct LBM2DQ9_Values{
    REAL* v0;             //center
    CudaS8A16<REAL>* v18; //eight direction velocites(4 side and 4 corner)
};

struct LBM2DQ9_Pointers{
    REAL** p0;              //center
    CudaS8A16<REAL*>* p18;  //eight direction velocites(4 side and 4 corner)
};

struct LBM2DQ9_Indexes{
    int cellIdx[LBM_VECT_COUNT];      //contains idx of cell where to move the vector value at stream step
    short vectIdx[LBM_VECT_COUNT];    //contains idx of vector where to move the vector value at stream step
};

#endif // LBMPARTICLE_H
