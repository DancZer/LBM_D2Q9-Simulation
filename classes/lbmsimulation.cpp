#include "lbmsimulation.h"
#include "core.h"
#include "time.h"
#include "math.h"
#include <QDebug>
#include <QDateTime>
#include "classes/logger.h"

//#define DEBUG_STREAM

LBMSimulation::LBMSimulation(LBMMesh* mesh, BoundaryConditionList* bcList, Settings *settings, MeshRenderer* render ):
    QThread(){

    mMesh = mesh;
    mBCList = bcList;
    mSettings = settings;
    mRender = render;

    setupSimulation();
}

LBMSimulation::~LBMSimulation(){

    delete mHostGridA;

    if(mOnGPU){
        #ifdef CUDA_BUILD
        delete mDeviceGridA;
        delete mDeviceGridB;
        #endif
    }else{
        delete mHostGridB;
    }
}

void LBMSimulation::preProcess()
{
    Logger::instance()->appendLog(QString("Szimuláció előkészítése."));

    mHostGridA = new LBMHostGrid(mMesh->getVisibleCellCount(),mMesh->getBoundaryCellCount(),true,true); //this contains the indexes array

    initIndexes();

    copyDatafromMeshToGrid(mHostGridA);

    if(mOnGPU){
        #ifdef CUDA_BUILD
        mDeviceGridA = new LBMDeviceGrid(mMesh->getVisibleCellCount(),mMesh->getBoundaryCellCount(),true,true); //this contains the indexes array
        mDeviceGridB = new LBMDeviceGrid(mMesh->getVisibleCellCount(),mMesh->getBoundaryCellCount(),true,false); //this contains the indexes array

        mCudaBlockSize = qMin(mCudaBlockSize,mHostGridA->length());
        mCudaGridSize = ceil(mHostGridA->length()/(float)(mCudaBlockSize));


        Logger::instance()->appendLog(QString("Cuda Blokk Méret: %1").arg(mCudaBlockSize));
        Logger::instance()->appendLog(QString("Cuda Rács Méret: %1").arg(mCudaGridSize));

        Logger::instance()->appendLog(QString("Adatok másolása a GPU memóriájába."));
        mDeviceGridA->copyValuesHostToDevice(mHostGridA->values());
        mDeviceGridA->copyIndexesHostToDevice(mHostGridA->indexes());
        mDeviceGridA->copyBcHostToDevice(mHostGridA->bcRho(),mHostGridA->bcVx(),mHostGridA->bcVy());

        Logger::instance()->appendLog(QString("Mutatók elkészítése a GPU-n."));
        lbmCudaSetupStreamPointers(mDeviceGridA->values(),mDeviceGridA->pointers(),mDeviceGridA->indexes(),mDeviceGridA->length(),mCudaGridSize,mCudaBlockSize);
        lbmCudaSetupStreamPointers(mDeviceGridB->values(),mDeviceGridB->pointers(),mDeviceGridA->indexes(),mDeviceGridB->length(),mCudaGridSize,mCudaBlockSize);
        #endif
    }else{
        mHostGridB = new LBMHostGrid(mMesh->getVisibleCellCount(),mMesh->getBoundaryCellCount(),true,false);

        Logger::instance()->appendLog(QString("Mutatók elkészítése."));
        lbmSetupStreamPointers(mHostGridA->values()->v0,mHostGridA->values()->v18,mHostGridA->pointers()->p0, mHostGridA->pointers()->p18,mHostGridA->indexes(),mHostGridA->length());
        lbmSetupStreamPointers(mHostGridB->values()->v0,mHostGridB->values()->v18,mHostGridB->pointers()->p0, mHostGridB->pointers()->p18,mHostGridA->indexes(),mHostGridB->length());
    }

    mIterCount = 0;

    setupDataFromGridToMesh();

    if(mOnGPU){
        #ifdef CUDA_BUILD
        cudaEventCreate(&mKernelStart);
        cudaEventCreate(&mKernelStop);

        cudaEventRecord(mKernelStart,0);
        #endif
    }else{
        mStartTime = clock();
    }

    Logger::instance()->appendLog(QString("Szimuláció előkészítése befejeződött."));
}

void LBMSimulation::iterate()
{
    if(mIterCount%2){
        if(mOnGPU)
        {
            #ifdef CUDA_BUILD
            lbmCudaStreamAndCollision(mDeviceGridA->values(),mDeviceGridB->pointers(),mMesh->getVisibleCellCount(),mMesh->getBoundaryCellCount(),mDeviceGridA->bcRho(),mDeviceGridA->bcVx(),mDeviceGridA->bcVy(),mTau,mCudaGridSize,mCudaBlockSize);
            #endif
        }
        else
        {
            lbmStreamAndCollision(mHostGridA->values()->v0,mHostGridA->values()->v18,mHostGridB->pointers()->p0, mHostGridB->pointers()->p18,mMesh->getVisibleCellCount(),mMesh->getBoundaryCellCount(),mHostGridA->bcRho(),mHostGridA->bcVx(),mHostGridA->bcVy(),mTau);
        }

    }else{
        if(mOnGPU)
        {
            #ifdef CUDA_BUILD
            lbmCudaStreamAndCollision(mDeviceGridB->values(),mDeviceGridA->pointers(),mMesh->getVisibleCellCount(),mMesh->getBoundaryCellCount(),mDeviceGridA->bcRho(),mDeviceGridA->bcVx(),mDeviceGridA->bcVy(),mTau,mCudaGridSize,mCudaBlockSize);
            #endif
        }
        else
        {
            lbmStreamAndCollision(mHostGridB->values()->v0,mHostGridB->values()->v18,mHostGridA->pointers()->p0, mHostGridA->pointers()->p18,mMesh->getVisibleCellCount(),mMesh->getBoundaryCellCount(),mHostGridA->bcRho(),mHostGridA->bcVx(),mHostGridA->bcVy(),mTau);
        }
    }

    //increment iteration count
#ifdef CUDA_BUILD
    cudaThreadSynchronize();
#endif
    mIterCount++;

    if(mIterCount%mRenderAfterIterCount == 0 && mIterCount>mRenderFromIterCount)
    {
#ifndef DEBUG_STREAM
        if(mRender->isRendered()){
#endif
            setupDataFromGridToMesh();
#ifndef DEBUG_STREAM
        }
#endif
    }

}

void LBMSimulation::postProcess()
{
    float runTime;

    if(mOnGPU){
        #ifdef CUDA_BUILD
        cudaEventRecord(mKernelStop,0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&runTime, mKernelStart, mKernelStop); //ms
        cudaEventDestroy(mKernelStart);
        cudaEventDestroy(mKernelStop);
        runTime/=1000.0f;
        #endif
    }else{
        runTime = (float)(clock()-mStartTime)/(float)CLOCKS_PER_SEC;
    }

    Logger::instance()->appendLog(QString("Szimuláció futási ideje: %1").arg(runTime));
    Logger::instance()->appendLog(QString("Szimuláció futási teljesítménye: %1 MegaCell/sec").arg(mHostGridA->length()*((mIterCount/1.0e6)/runTime)));

    setupDataFromGridToMesh();

}

void LBMSimulation::initIndexes(){

    int neighbours[LBM_VECT_COUNT];

    for(int i=0; i<mMesh->getCellCount();i++){
        mMesh->getNeighbours(i,neighbours);

        setupCell(i,neighbours);
    }
}

void LBMSimulation::setupCell(int idx, int* neighbours){

    //vector 0
    sideStreamIndex(idx,neighbours[0],0,0);

    //stream from left
    sideStreamIndex(idx,neighbours[3],1,3);

    //stream from bottom
    sideStreamIndex(idx,neighbours[4],2,4);

    //stream from right
    sideStreamIndex(idx,neighbours[1],3,1);

    //stream from top
    sideStreamIndex(idx,neighbours[2],4,2);

    //stream from bottom left
    cornerStreamIndex(idx,neighbours[7],neighbours[3],neighbours[4],5,7,8,6);

    //stream from bottom right
    cornerStreamIndex(idx,neighbours[8],neighbours[1],neighbours[4],6,8,7,5);

    //stream from top right
    cornerStreamIndex(idx,neighbours[5],neighbours[1],neighbours[2],7,5,6,8);

    //stream from top left
    cornerStreamIndex(idx,neighbours[6],neighbours[3],neighbours[2],8,6,5,7);
}

/**
    * This function setup side stream indexes for cellIdx considering the wall(-1)
    */
void LBMSimulation::sideStreamIndex(int cellIdx, int neighborIdx, short streamVectIdx, short reflectVectIdx){

    int* cellIndex = &(mHostGridA->indexes()[cellIdx].cellIdx[streamVectIdx]);
    short* vectorIndex = &(mHostGridA->indexes()[cellIdx].vectIdx[streamVectIdx]);

    if(neighborIdx!=cellIdx){
        *(vectorIndex) = streamVectIdx;
        *(cellIndex) = neighborIdx;
    }else{
        *(vectorIndex) = reflectVectIdx;
        *(cellIndex) = cellIdx;
    }
}

/**
    * This function setup corner stream indexes for cellIdx considering the wall(-1)
    */
void LBMSimulation::cornerStreamIndex(int cellIdx, int cornerNeighborIdx, int horNeighborIdx, int verNeighborIdx, short streamVectIdx, short reflectVectIdx, short horReflectVectIdx, short verReflectVectIdx ){

    int* cellIndex = &(mHostGridA->indexes()[cellIdx].cellIdx[streamVectIdx]);
    short* vectorIndex = &(mHostGridA->indexes()[cellIdx].vectIdx[streamVectIdx]);

    if(cornerNeighborIdx!=cellIdx){
        if(verNeighborIdx == cellIdx && horNeighborIdx == cellIdx){
            *(vectorIndex) = reflectVectIdx;
            *(cellIndex) = cellIdx;
        }else{
            *(vectorIndex) = streamVectIdx;
            *(cellIndex) = cornerNeighborIdx;
        }
    }else{
        if((verNeighborIdx == cellIdx && horNeighborIdx == cellIdx) || (verNeighborIdx != cellIdx && horNeighborIdx != cellIdx)){
            *(vectorIndex) = reflectVectIdx;
            *(cellIndex) = cellIdx;
        }else{
            if(verNeighborIdx == cellIdx){
                *(vectorIndex) = horReflectVectIdx;
                *(cellIndex) = horNeighborIdx;
            }else{
                *(vectorIndex) = verReflectVectIdx;
                *(cellIndex) = verNeighborIdx;
            }
        }
    }
}

inline void LBMSimulation::setupDataFromGridToMesh()
{
    if(mIterCount%2){
        if(mOnGPU)
        {
            #ifdef CUDA_BUILD
            mDeviceGridB->copyValuesDeviceToHost(mHostGridA->values());
            #endif
        }
        else
        {
            copyDatafromGridToMesh(mHostGridB);
        }
    }else{
        if(mOnGPU)
        {
            #ifdef CUDA_BUILD
            mDeviceGridA->copyValuesDeviceToHost(mHostGridA->values());
            #endif
        }
        else
        {
            copyDatafromGridToMesh(mHostGridA);
        }
    }

    if(mOnGPU)
    {
        copyDatafromGridToMesh(mHostGridA);
    }
}

/**
    * This function setup vectors values, which are connected to the mesh.
    * Check the global rho and if is NaN then exit.
    */
void LBMSimulation::copyDatafromGridToMesh(LBMHostGrid* hostGrid){

    LBM2DQ9_Values* data = hostGrid->values();

    REAL globalRho=0.0,rhoc,vxc,vyc;

#ifdef DEBUG_STREAM
    int i=0;
    qDebug()<<i<<". "<<data->v0[i]<<","<<data->v18[i].v1<<","<<data->v18[i].v2<<","<<data->v18[i].v3<<","<<data->v18[i].v4<<","<<data->v18[i].v5<<","<<data->v18[i].v6<<","<<data->v18[i].v7<<","<<data->v18[i].v8;
#else
    //copy only visible cell values, ghost cells excluded
    for(int i=0;i<mMesh->getVisibleCellCount();i++){

        rhoc = data->v0[i] + data->v18[i].v1 + data->v18[i].v2 + data->v18[i].v3 + data->v18[i].v4 + data->v18[i].v5 + data->v18[i].v6 + data->v18[i].v7 + data->v18[i].v8;
        vxc = data->v18[i].v1 - data->v18[i].v3 + data->v18[i].v5 - data->v18[i].v6 - data->v18[i].v7 + data->v18[i].v8;
        vyc = data->v18[i].v2 - data->v18[i].v4 + data->v18[i].v5 + data->v18[i].v6 - data->v18[i].v7 - data->v18[i].v8;

        vxc /= rhoc;
        vyc /= rhoc;

        mRender->setRhoc(i,rhoc);
        mRender->setVxc(i,vxc);
        mRender->setVyc(i,vyc);

        globalRho += rhoc;
    }

    if(globalRho != globalRho){
        Logger::instance()->appendLog(QString("!!!!!!!Kritikus áramlás jött létre, ezért a szimuláció leált!!!!!!!"));
        quit();
    }else{
        //notify GUI thread to render mesh
        mRender->setRendered(false);
        emit meshDataUpdated(mIterCount);
    }
#endif
}


void LBMSimulation::copyDatafromMeshToGrid(LBMHostGrid *hostGrid)
{
    Logger::instance()->appendLog(QString("-Adatok inicializálása a memóriában."));

    REAL rhoc,vxc,vyc,tn,t1,wrhoc;
    LBM2DQ9_Values* data = hostGrid->values();

#ifdef DEBUG_STREAM
    for(int i=0;i<mMesh->getCellCount();i++){
        data->v0[i] = i;
        data->v18[i].v1 = i;
        data->v18[i].v2 = i;
        data->v18[i].v3 = i;
        data->v18[i].v4 = i;
        data->v18[i].v5 = i;
        data->v18[i].v6 = i;
        data->v18[i].v7 = i;
        data->v18[i].v8 = i;
    }
#else
    //visible Cells
    int offset = 0;

    for(int i=offset;i<mMesh->getVisibleCellCount();i++){

        rhoc = mRender->getRhoc(i);
        vxc = mRender->getVxc(i);
        vyc = mRender->getVyc(i);

        tn = (vxc * vxc + vyc * vyc)*LBM_G3 + 1.;

        t1=0.0;
        data->v0[i] = LBM_T0*rhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        wrhoc = LBM_TS*rhoc;
        t1=vxc;
        data->v18[i].v1 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=vyc;
        data->v18[i].v2 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=-vxc;
        data->v18[i].v3 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=-vyc;
        data->v18[i].v4 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        wrhoc = LBM_TL*rhoc;
        t1=vxc + vyc;
        data->v18[i].v5 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=-vxc + vyc;
        data->v18[i].v6 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=-vxc - vyc;
        data->v18[i].v7 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=vxc - vyc;
        data->v18[i].v8 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
    }

    //input ghost cells
    offset +=mMesh->getVisibleCellCount();
    for(int i=offset;i<offset+mMesh->getBoundaryCellCount();i++){
        //input ghost cells boundary condition
        hostGrid->bcRho()[i-offset] = mBCList->getByID(mMesh->getBoundaryCellTagId(i))->getRho(mSettings->getPressure());
        hostGrid->bcVx()[i-offset] = mBCList->getByID(mMesh->getBoundaryCellTagId(i))->getVx(mSettings->getSpeedOfSound());
        hostGrid->bcVy()[i-offset] = mBCList->getByID(mMesh->getBoundaryCellTagId(i))->getVy(mSettings->getSpeedOfSound());

        //mivel homogen a suruseg
        rhoc = hostGrid->bcRho()[i-offset];
        vxc = hostGrid->bcVx()[i-offset];
        vyc = hostGrid->bcVy()[i-offset];

        tn = (vxc * vxc + vyc * vyc)*LBM_G3 + 1.;

        t1=0.0;
        data->v0[i] = LBM_T0*rhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        wrhoc = LBM_TS*rhoc;
        t1=vxc;
        data->v18[i].v1 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=vyc;
        data->v18[i].v2 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=-vxc;
        data->v18[i].v3 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=-vyc;
        data->v18[i].v4 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        wrhoc = LBM_TL*rhoc;
        t1=vxc + vyc;
        data->v18[i].v5 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=-vxc + vyc;
        data->v18[i].v6 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=-vxc - vyc;
        data->v18[i].v7 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);

        t1=vxc - vyc;
        data->v18[i].v8 = wrhoc*(LBM_G1*t1+LBM_G2*t1*t1+tn);
    }
#endif
}

void LBMSimulation::run()
{
    mMutex.lock();
    mQuit = false;
    mPause = false;    
    mMutex.unlock();
    time_t start;
    float runSec=0.0;
    preProcess();

    Logger::instance()->appendLog(QString("Szimuláció elkezdődött."));
    while(!getQuit()){
        while((mIterCount < mMaxIterCount || runSec< mMaxRunTime || (mMaxIterCount<0 && mMaxRunTime<0)) && !getQuit() && !getPause()){
            start = clock();
            iterate();
            runSec += (float)(clock()-start)/(float)CLOCKS_PER_SEC;
        }
        if((mMaxIterCount>0 && mIterCount >= mMaxIterCount) || (runSec>= mMaxRunTime && mMaxRunTime>0)){
            quit();
        }else{
            msleep(100);
        }
    }
    Logger::instance()->appendLog(QString("Szimuláció befejezdődött."));
    postProcess();
}

void LBMSimulation::quit()
{
    QMutexLocker locker(&mMutex);
    mQuit = true;
}

void LBMSimulation::pause()
{
    QMutexLocker locker(&mMutex);
    mPause = true;
}

void LBMSimulation::resume()
{
    QMutexLocker locker(&mMutex);
    mPause = false;
}

void LBMSimulation::setupSimulation()
{
    mTau = LBM_M1+LBM_M2*mSettings->getViscosity();
    mMaxIterCount = mSettings->getMaxIteration();
    mMaxRunTime = mSettings->getMaxRunTime();
    mRenderAfterIterCount = mSettings->getRenderAfter();
    mRenderFromIterCount = mSettings->getRenderFrom();
    mCudaBlockSize = mSettings->getBlockSize();
    mOnGPU = (mSettings->getArch() == Settings::GPU);
}

