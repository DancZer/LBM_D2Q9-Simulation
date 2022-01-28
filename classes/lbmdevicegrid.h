#ifndef LBMDEVICEGRID_H
#define LBMDEVICEGRID_H

#include "cuda/cudaSafeCall.h"
#include "classes/lbmparticle.h"
#include <cuda_runtime.h>

class LBMDeviceGrid {
    LBM2DQ9_Values mValues;
    LBM2DQ9_Pointers mPointers;
    LBM2DQ9_Indexes* mIndexes;

    REAL *mBcRho;
    REAL *mBcVx;
    REAL *mBcVy;

    int mLength;
    int mBcCount;

public:
    LBMDeviceGrid (int visibleCells, int bcCells, bool allocPointers, bool allocIndexes=false):
        mLength(visibleCells+bcCells),
        mBcCount(bcCells)
    {
        CUDA_SAFE_CALL( cudaMalloc((void**)&(mValues.v0), sizeof(REAL)*mLength));
        CUDA_SAFE_CALL( cudaMalloc((void**)&(mValues.v18), sizeof(CudaS8A16<REAL>)*mLength));

        if(bcCells>0){
            CUDA_SAFE_CALL( cudaMalloc((void**)&(mBcRho), sizeof(REAL)*bcCells));
            CUDA_SAFE_CALL( cudaMalloc((void**)&(mBcVx), sizeof(REAL)*bcCells));
            CUDA_SAFE_CALL( cudaMalloc((void**)&(mBcVy), sizeof(REAL)*bcCells));
        }else{
            mBcRho = 0;
            mBcVx = 0;
            mBcVy = 0;
        }

        if(allocPointers){
            CUDA_SAFE_CALL( cudaMalloc((void**)&(mPointers.p0), sizeof(REAL*)*mLength));
            CUDA_SAFE_CALL( cudaMalloc((void**)&(mPointers.p18), sizeof(CudaS8A16<REAL*>)*mLength));
        }
        else {
            mPointers.p0 = 0;
            mPointers.p18 = 0;
        }

        if(allocIndexes){
            CUDA_SAFE_CALL( cudaMalloc((void**)&(mIndexes), sizeof(LBM2DQ9_Indexes)*mLength));
        }
        else {
            mIndexes = 0;
        }
    }

    ~LBMDeviceGrid()
    {
        CUDA_SAFE_CALL( cudaFree(mValues.v0));
        CUDA_SAFE_CALL( cudaFree(mValues.v18));

        if(mPointers.p0 != 0){
            CUDA_SAFE_CALL( cudaFree(mPointers.p0));
        }

        if(mPointers.p18 != 0){
            CUDA_SAFE_CALL( cudaFree(mPointers.p18));
        }

        if(mIndexes != 0){
            CUDA_SAFE_CALL( cudaFree(mIndexes));
        }

        CUDA_SAFE_CALL( cudaFree(mBcRho));
        CUDA_SAFE_CALL( cudaFree(mBcVx));
        CUDA_SAFE_CALL( cudaFree(mBcVy));
    }

    void copyValuesHostToDevice(LBM2DQ9_Values* hostValues){
        CUDA_SAFE_CALL( cudaMemcpy(mValues.v0, hostValues->v0, sizeof(REAL)*mLength, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL( cudaMemcpy(mValues.v18, hostValues->v18, sizeof(CudaS8A16<REAL>)*mLength, cudaMemcpyHostToDevice));
    }

    void copyIndexesHostToDevice(LBM2DQ9_Indexes* hostIndexes){
        CUDA_SAFE_CALL( cudaMemcpy(mIndexes, hostIndexes, sizeof(LBM2DQ9_Indexes)*mLength, cudaMemcpyHostToDevice));
    }

    void copyBcHostToDevice(REAL *bcRho,REAL *bcVx, REAL *bcVy){
        CUDA_SAFE_CALL( cudaMemcpy(mBcRho, bcRho, sizeof(REAL)*mBcCount, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL( cudaMemcpy(mBcVx, bcVx, sizeof(REAL)*mBcCount, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL( cudaMemcpy(mBcVy, bcVy, sizeof(REAL)*mBcCount, cudaMemcpyHostToDevice));
    }

    //back
    void copyValuesDeviceToHost(LBM2DQ9_Values* hostValues){
        CUDA_SAFE_CALL( cudaMemcpy( hostValues->v0, mValues.v0, sizeof(REAL)*mLength, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL( cudaMemcpy( hostValues->v18, mValues.v18, sizeof(CudaS8A16<REAL>)*mLength, cudaMemcpyDeviceToHost));
    }


    LBM2DQ9_Values* values()
    {
        return &mValues;
    }

    LBM2DQ9_Pointers* pointers()
    {
        return &mPointers;
    }

    LBM2DQ9_Indexes* indexes()
    {
        return mIndexes;
    }

    REAL* bcRho()
    {
        return mBcRho;
    }

    REAL* bcVx()
    {
        return mBcVx;
    }

    REAL* bcVy()
    {
        return mBcVy;
    }

    int length() const
    {
        return mLength;
    }
};

#endif // LBMDEVICEGRID_H
