#ifndef _LBMSIMULATION_H
#define _LBMSIMULATION_H

#include <QThread>
#include <QMutex>
#include <QMutexLocker>

#include "classes/lbmhostgrid.h"
#include "classes/lbmmesh.h"
#include "classes/meshrenderer.h"
#include "classes/settings.h"

#ifdef CUDA_BUILD
#include "classes/lbmdevicegrid.h"
extern "C"
{
    void lbmCudaStreamAndCollision(LBM2DQ9_Values* values, LBM2DQ9_Pointers* pointers, int realCellCount, int boundaryCellCount, REAL* bcRho, REAL* bcVx, REAL* bcVy, REAL tau, int gridSize, int blockSize);
    void lbmCudaSetupStreamPointers(LBM2DQ9_Values* values, LBM2DQ9_Pointers* pointers, LBM2DQ9_Indexes* indexes, int count, int gridSize, int blockSize);
}
#endif

class LBMSimulation : public QThread
{
    Q_OBJECT
public:
    LBMSimulation(LBMMesh* mesh, BoundaryConditionList* bcList, Settings *settings, MeshRenderer* render);
    ~LBMSimulation();

    bool getQuit(){
        QMutexLocker locker(&mMutex);
        return mQuit;
    }

    bool getPause(){
        QMutexLocker locker(&mMutex);
        return mPause;
    }

    void preProcess();
    void iterate();
    void postProcess();

public slots:
    void pause();
    void resume();
    void quit();

protected:
    void run();

signals:
    void meshDataUpdated(int iterCount);

private:
    LBMMesh *mMesh;
    BoundaryConditionList *mBCList;
    Settings *mSettings;
    MeshRenderer* mRender;

    QMutex mMutex;
    bool mQuit,mPause;

    long mIterCount;
    long mMaxIterCount;
    long mMaxRunTime;
    long mRenderAfterIterCount;
    long mRenderFromIterCount;
    REAL mTau;

    bool mOnGPU;
    int mCudaBlockSize;
    int mCudaGridSize;

    //for performacen calc
    time_t mStartTime;
    #ifdef CUDA_BUILD
    cudaEvent_t mKernelStart, mKernelStop;
    #endif

    //LBM Grids
    LBMHostGrid* mHostGridA; //primary
    LBMHostGrid* mHostGridB; //secondary

    #ifdef CUDA_BUILD
    LBMDeviceGrid* mDeviceGridA; //primary
    LBMDeviceGrid* mDeviceGridB; //secondary
    #endif

    void initIndexes();
    void setupCell(int idx, int* neighbours);

    int getGhostCellSourceCellIdx(int ghostCellIdx);

    void sideStreamIndex(int cellIdx, int neighborIdx, short streamVectIdx, short reflectVectIdx);
    void cornerStreamIndex(int cellIdx, int cornerNeighborIdx, int horNeighborIdx, int verNeighborIdx, short streamVectIdx, short reflectVectIdx, short horReflectVectIdx, short verReflectVectIdx );

    void setupDataFromGridToMesh();

    void copyDatafromMeshToGrid(LBMHostGrid *hostGrid);
    void copyDatafromGridToMesh(LBMHostGrid* hostGrid);

    void setupSimulation();
};

#endif //_LBMSIMULATION_H
