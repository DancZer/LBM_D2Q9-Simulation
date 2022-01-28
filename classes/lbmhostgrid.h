#ifndef LBMHOSTGRID_H
#define LBMHOSTGRID_H

#include "classes/lbmparticle.h"

class LBMHostGrid {
    LBM2DQ9_Values mValues;
    LBM2DQ9_Pointers mPointers;
    LBM2DQ9_Indexes* mIndexes;

    REAL *mBcRho;
    REAL *mBcVx;
    REAL *mBcVy;

    int mLength;

public:
    LBMHostGrid (int visibleCells, int bcCells, bool allocPointers, bool allocIndexes=false):
        mLength(visibleCells+bcCells)
    {
        mValues.v0 = new REAL[mLength];
        mValues.v18 = new CudaS8A16<REAL>[mLength];

        if(bcCells>0){
            mBcRho = new REAL[bcCells];
            mBcVx = new REAL[bcCells];
            mBcVy = new REAL[bcCells];
        }else{
            mBcRho = 0;
            mBcVx = 0;
            mBcVy = 0;
        }

        if(allocPointers){
            mPointers.p0 = new REAL*[mLength];
            mPointers.p18 = new CudaS8A16<REAL*>[mLength];
        }
        else {
            mPointers.p0 = 0;
            mPointers.p18 = 0;
        }

        if(allocIndexes){
            mIndexes = new LBM2DQ9_Indexes[mLength];
        }
        else {
            mIndexes = 0;
        }
    }

    ~LBMHostGrid()
    {
        delete[] mValues.v0;
        delete[] mValues.v18;

        if(mPointers.p0 != 0){
            delete[] mPointers.p0;
        }

        if(mPointers.p18 != 0){
            delete[] mPointers.p18;
        }

        if(mIndexes != 0){
            delete[] mIndexes;
        }


        delete[] mBcRho;
        delete[] mBcVx;
        delete[] mBcVy;
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

#endif // LBMHOSTGRID_H
