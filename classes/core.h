#ifndef LBMCORE_H
#define LBMCORE_H

#include "lbmparticle.h"

#define LBM_COLLISION

#ifdef CUDA_KERNEL
__global__
void lbmStreamAndCollisionOnCuda(REAL* v0Values, CudaS8A16<REAL>* v18Values, REAL** v0Pointers, CudaS8A16<REAL*>* v18Pointers, int realCellCount, int bcCellCount, REAL* bcRho, REAL* bcVx, REAL* bcVy, REAL tau)
#else
void lbmStreamAndCollision(REAL* v0Values, CudaS8A16<REAL>* v18Values, REAL** v0Pointers, CudaS8A16<REAL*>* v18Pointers, int realCellCount, int bcCellCount, REAL* bcRho, REAL* bcVx, REAL* bcVy, REAL tau)
#endif
{
    #ifdef CUDA_KERNEL
        int mIdx = blockDim.x * blockIdx.x + threadIdx.x;

        if(mIdx<realCellCount+bcCellCount) {
    #else
        for(int mIdx=0;mIdx<realCellCount+bcCellCount;mIdx++){
    #endif
            REAL f0;
            CudaS8A16<REAL> f18;

            #ifdef LBM_COLLISION
            REAL t1,tn;
            REAL vxc,vyc,rhoc,wrhoc;
            REAL ffeq;

            //
            //macroscopic
            if(mIdx <realCellCount){

                REAL* p0 = v0Pointers[mIdx];
                CudaS8A16<REAL*> p18 = v18Pointers[mIdx];

                //stream read only for visible cells
                f0 = *p0;
                f18.v1 = *(p18.v1);
                f18.v2 = *(p18.v2);
                f18.v3 = *(p18.v3);
                f18.v4 = *(p18.v4);
                f18.v5 = *(p18.v5);
                f18.v6 = *(p18.v6);
                f18.v7 = *(p18.v7);
                f18.v8 = *(p18.v8);

                rhoc = f0 + f18.v1 + f18.v2 + f18.v3 + f18.v4 + f18.v5 + f18.v6 + f18.v7 + f18.v8;
                vxc = (f18.v1 - f18.v3 + f18.v5 - f18.v6 - f18.v7 + f18.v8)/rhoc;
                vyc = (f18.v2 - f18.v4 + f18.v5 + f18.v6 - f18.v7 - f18.v8)/rhoc;


                tn = (vxc * vxc + vyc * vyc) * LBM_G3 + 1.;

                //
                //collision
                ffeq=LBM_T0*rhoc*tn;
                f0 -= (f0-ffeq)/tau;

                //vertical and horizontal forces
                wrhoc = LBM_TS*rhoc;

                t1=vxc;
                ffeq= wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
                f18.v1 -= (f18.v1-ffeq)/tau;

                t1 = vyc;
                ffeq= wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
                f18.v2 -= (f18.v2-ffeq)/tau;

                t1 = -vxc;
                ffeq= wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
                f18.v3 -= (f18.v3-ffeq)/tau;

                t1 = -vyc;
                ffeq= wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
                f18.v4 -= (f18.v4-ffeq)/tau;

                //diagonal forces
                wrhoc = LBM_TL*rhoc;

                t1 = vxc + vyc;
                ffeq= wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
                f18.v5 -= (f18.v5-ffeq)/tau;

                t1 = -vxc + vyc;
                ffeq= wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
                f18.v6 -= (f18.v6-ffeq)/tau;

                t1 = -vxc - vyc;
                ffeq= wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
                f18.v7 -= (f18.v7-ffeq)/tau;

                t1 = vxc - vyc;
                ffeq= wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
                f18.v8 -= (f18.v8-ffeq)/tau;

            }else{
                //boundary condition(in or out flow)

                rhoc = bcRho[mIdx-realCellCount];
                vxc = bcVx[mIdx-realCellCount];
                vyc = bcVy[mIdx-realCellCount];

                tn = (vxc * vxc + vyc * vyc) * LBM_G3 + 1.f;

                //
                //collision
                f0=LBM_T0*rhoc*tn;

                //vertical and horizontal forces
                wrhoc = LBM_TS*rhoc;

                t1 = vxc;
                f18.v1 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

                t1 = vyc;
                f18.v2 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

                t1 = -vxc;
                f18.v3 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

                t1 = -vyc;
                f18.v4 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

                //diagonal forces
                wrhoc = LBM_TL*rhoc;

                t1 = vxc + vyc;
                f18.v5 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

                t1 = -vxc + vyc;
                f18.v6 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

                t1 = -vxc - vyc;
                f18.v7 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

                t1 = vxc - vyc;
                f18.v8 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
            }
            #endif

            v0Values[mIdx] = f0;
            v18Values[mIdx] = f18;
        }
}

#ifdef CUDA_KERNEL
__global__
void lbmSetupStreamPointersOnCuda(REAL* v0Values, CudaS8A16<REAL>* v18Values, REAL** v0Pointers, CudaS8A16<REAL*>* v18Pointers, LBM2DQ9_Indexes* indexes, int count)
#else
void lbmSetupStreamPointers(REAL* v0Values, CudaS8A16<REAL>* v18Values, REAL** v0Pointers, CudaS8A16<REAL*>* v18Pointers, LBM2DQ9_Indexes* indexes, int count)
#endif
{
    #ifdef CUDA_KERNEL
        int mIdx = blockDim.x * blockIdx.x + threadIdx.x;

        if(mIdx<count) {
    #else
        for(int mIdx=0;mIdx<count;mIdx++){
    #endif
            REAL* pointer = 0;

            /*v0Values[mIdx] = mIdx;
            v18Values[mIdx].v1 = mIdx;
            v18Values[mIdx].v2 = mIdx;
            v18Values[mIdx].v3 = mIdx;
            v18Values[mIdx].v4 = mIdx;
            v18Values[mIdx].v5 = mIdx;
            v18Values[mIdx].v6 = mIdx;
            v18Values[mIdx].v7 = mIdx;
            v18Values[mIdx].v8 = mIdx;*/

            for(int i=0;i<LBM_VECT_COUNT;i++){

                switch(indexes[mIdx].vectIdx[i]){
                    case 0 :
                        pointer = &(v0Values[indexes[mIdx].cellIdx[i]]);
                    break;

                    case 1 :
                        pointer = &(v18Values[indexes[mIdx].cellIdx[i]].v1);
                    break;

                    case 2 :
                        pointer = &(v18Values[indexes[mIdx].cellIdx[i]].v2);
                    break;

                    case 3 :
                        pointer = &(v18Values[indexes[mIdx].cellIdx[i]].v3);
                    break;

                    case 4 :
                        pointer = &(v18Values[indexes[mIdx].cellIdx[i]].v4);
                    break;

                    case 5 :
                        pointer = &(v18Values[indexes[mIdx].cellIdx[i]].v5);
                    break;

                    case 6 :
                        pointer = &(v18Values[indexes[mIdx].cellIdx[i]].v6);
                    break;

                    case 7 :
                        pointer = &(v18Values[indexes[mIdx].cellIdx[i]].v7);
                    break;

                    case 8 :
                        pointer = &(v18Values[indexes[mIdx].cellIdx[i]].v8);
                    break;
                }

                switch(i){
                    case 0 :
                        v0Pointers[mIdx] = pointer;
                    break;

                    case 1 :
                        v18Pointers[mIdx].v1 = pointer;
                    break;

                    case 2 :
                        v18Pointers[mIdx].v2 = pointer;
                    break;

                    case 3 :
                        v18Pointers[mIdx].v3 = pointer;
                    break;

                    case 4 :
                        v18Pointers[mIdx].v4 = pointer;
                    break;

                    case 5 :
                        v18Pointers[mIdx].v5 = pointer;
                    break;

                    case 6 :
                        v18Pointers[mIdx].v6 = pointer;
                    break;

                    case 7 :
                        v18Pointers[mIdx].v7 = pointer;
                    break;

                    case 8 :
                        v18Pointers[mIdx].v8 = pointer;
                    break;
                }
            }
        }
}

#ifdef CUDA_KERNEL

#include "cuda/cudaSafeCall.h"

extern "C" {

    void lbmCudaStreamAndCollision(LBM2DQ9_Values* values, LBM2DQ9_Pointers* pointers, int realCellCount, int boundaryCellCount, REAL* bcRho, REAL* bcVx, REAL* bcVy, REAL tau, int gridSize, int blockSize){
        lbmStreamAndCollisionOnCuda<<<gridSize,blockSize>>>( values->v0, values->v18, pointers->p0,  pointers->p18, realCellCount, boundaryCellCount, bcRho, bcVx, bcVy, tau);
        CUDA_SAFE_CALL(cudaGetLastError());
    }

    void lbmCudaSetupStreamPointers(LBM2DQ9_Values* values, LBM2DQ9_Pointers* pointers, LBM2DQ9_Indexes* indexes, int count, int gridSize, int blockSize){
        lbmSetupStreamPointersOnCuda<<<gridSize,blockSize>>>(values->v0, values->v18, pointers->p0, pointers->p18, indexes, count);
        CUDA_SAFE_CALL(cudaGetLastError());
    }
}
#endif

#endif // LBM2DQ9CORE_H
