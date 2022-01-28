#ifndef MESHSETTINGS_H
#define MESHSETTINGS_H

#include <QObject>
#include <QLabel>
#include <QRadioButton>
#include <QSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLineEdit>
#include <QGroupBox>
#include <math.h>
#include "widgets/colorssample.h"
#include <QDebug>

#define CUDA_BUILD

#ifdef CUDA_BUILD
#include "classes/cuda/cudaSafeCall.h"
#include <cuda_runtime.h>
#endif

#define LBM_CS sqrt(1./3.)

//#define ENABLE_SAVE_SETTINGS

class Settings : public QObject
{
    Q_OBJECT
public:
    enum ARCHITECTURE {UNKNOWN_ARCH=0,CPU=1, GPU=2};
    enum DISPLAY_VALUE {UNKNOWN_VALUE=0, RHOC = 1,VXYC = 2, PRESSURE=3};
    enum VALUE_SCALE {UNKNOWN_SCALE=0, AUTO = 1,USER = 2};

    explicit Settings(QObject *parent = 0):
        QObject(parent){
        mBackground = 0;
    }

    //get values
    ARCHITECTURE getArch();
    int getBlockSize();

    double getViscosity();
    double getGridStepSize();
    double getSpeedOfSound();
    double getSpeedOfSound2();
    double getDensity();
    double getPressure();
    double getTimeStep();

    int getRenderAfter();
    int getRenderFrom();
    bool getSyncRender(){return mW_SyncRender->isChecked();}

    bool getBackground(){return mW_Background->isChecked();}
    QImage* getBackgroundImg();

    bool getSaveImage(){return mW_SaveImage->isChecked();}
    QString getSaveImagePath(int iteration);
    int getRenderColorCount();

    DISPLAY_VALUE getRenderDisplay();
    VALUE_SCALE getRenderDisplayScale();
    double getRenderDisplayScaleMin();
    double getRenderDisplayScaleMax();

    bool getRenderInfoGrid(){return mW_RenderInfoGrid->isChecked();}
    bool getRenderInfoIterCount(){return mW_RenderInfoIterCount->isChecked();}
    bool getRenderInfoSimTime(){return mW_RenderInfoSimTime->isChecked();}
    bool getRenderInfoColorLegend(){return mW_RenderInfoColorLegend->isChecked();}
    bool getRenderInfoVectors(){return mW_RenderInfoVectors->isChecked();}
    int getRenderInfoVectorsLength(){return mW_RenderInfoVectorsLength->value();}

    int getMaxIteration();
    int getMaxRunTime();

    //groupboxes
    void setGroupBoxes(QGroupBox* arch, QGroupBox* grid, QGroupBox* render, QWidget* runCond ){mArch = arch; mGrid = grid; mRender = render; mRunCond = runCond;}

    //widgets setup
    void setArchWidgets(QRadioButton* cpu, QRadioButton* gpu){mW_CPU = cpu; mW_GPU=gpu;}
    void setBlockSizeWidget(QSpinBox* cudaBlck){mW_CudaBlck = cudaBlck;}

    void setGridStepSizeWidget(QDoubleSpinBox* w){mW_GridStepSize = w;}
    void setViscosityWidget(QDoubleSpinBox* w){mW_Viscosity = w;}
    void setSpeedOfSoundWidget(QDoubleSpinBox* w){mW_SpeedOfSound = w;}
    void setDensityWidget(QDoubleSpinBox* w){mW_Density = w;}
    void setTimeStepWidget(QLabel* w){mW_TimeStep = w;}

    void setRenderLabels(QLabel* l1,QLabel* l2,QLabel* l3){mW_RenderTextL1 = l1;mW_RenderTextL2 = l2;mW_RenderTextL3 = l3;}
    void setRenderAfterWidget(QSpinBox* w){mW_RenderAfter = w;}
    void setRenderFromWidget(QSpinBox* w){mW_RenderFrom = w;}
    void setSyncRenderWidget(QCheckBox* checkBox){mW_SyncRender = checkBox;}
    void setBackgroundWidget(QCheckBox* checkBox, QLineEdit *pathLine, QPushButton *btn){mW_Background = checkBox; mW_BackgroundPath = pathLine; mW_BackgroundPathBt=btn;}
    void setSaveImageWidget(QCheckBox* checkBox, QLineEdit *pathLine, QPushButton *btn){mW_SaveImage = checkBox; mW_SaveImagePath = pathLine; mW_SaveImagePathBt=btn;}
    void setRenderValueVWidget(QRadioButton* radioV, QRadioButton* userScale, QDoubleSpinBox* min,  QDoubleSpinBox* max, QRadioButton* autoScale){
        mW_RenderValueV=radioV;
        mW_RenderValueV_UserScale = userScale;
        mW_RenderValueV_UserScale_Min = min;
        mW_RenderValueV_UserScale_Max = max;
        mW_RenderValueV_AutoScale = autoScale;
    }
    void setRenderValueRhoWidget(QRadioButton* radioRho, QRadioButton* userScale, QDoubleSpinBox* min,  QDoubleSpinBox* max, QRadioButton* autoScale){
        mW_RenderValueRho=radioRho;
        mW_RenderValueRho_UserScale = userScale;
        mW_RenderValueRho_UserScale_Min = min;
        mW_RenderValueRho_UserScale_Max = max;
        mW_RenderValueRho_AutoScale = autoScale;
    }

    void setRenderValuePWidget(QRadioButton* radioP, QRadioButton* userScale, QDoubleSpinBox* min,  QDoubleSpinBox* max, QRadioButton* autoScale){
        mW_RenderValueP=radioP;
        mW_RenderValueP_UserScale = userScale;
        mW_RenderValueP_UserScale_Min = min;
        mW_RenderValueP_UserScale_Max = max;
        mW_RenderValueP_AutoScale = autoScale;
    }
    void setRenderColorCount(QSpinBox* value, ColorsSample* colors){mW_RenderColorCount = value; mW_RenderColors = colors;}

    void setRenderInfoGrid(QCheckBox* cb){mW_RenderInfoGrid = cb;}
    void setRenderInfoIterCount(QCheckBox* cb){mW_RenderInfoIterCount = cb;}
    void setRenderInfoSimTime(QCheckBox* cb){mW_RenderInfoSimTime = cb;}
    void setRenderInfoColorLegend(QCheckBox* cb){mW_RenderInfoColorLegend = cb;}
    void setRenderInfoVectors(QCheckBox* cb, QLabel* label, QSpinBox* length){mW_RenderInfoVectors = cb; mW_RenderInfoVectorsL = label; mW_RenderInfoVectorsLength=length;}

    QList<QColor> getRenderColors(){ return mRenderColors;}

    void setMaxIterationWidget(QCheckBox* cb, QSpinBox* line){mW_MaxIterationCB = cb; mW_MaxIteration = line;}
    void setMaxRunTimeWidget(QCheckBox* cb, QSpinBox* line){mW_MaxRunTimeCB = cb; mW_MaxRunTime = line;}

    //beállítja a widgetek közötti kapcsolatokat és az egyes widgetek tulajdonságait
    void connectWidgets();

    void setupDefaultValues();

    void setEnable(bool enable){
        mArch->setEnabled(enable);
        mGrid->setEnabled(enable);

        //mRender->setEnabled(enable);
        mW_RenderTextL1->setEnabled(enable);
        mW_RenderTextL2->setEnabled(enable);
        mW_RenderTextL3->setEnabled(enable);
        mW_RenderAfter->setEnabled(enable);
        mW_RenderFrom->setEnabled(enable);
        mW_SyncRender->setEnabled(enable);

//a felhasznalo a futtatast kovetoen is meg tudja adni a mentés helyét illetve a hátteret
//        mW_SaveImage->setEnabled(enable);
//        mW_Background->setEnabled(enable);

//        if(mW_SaveImage->isChecked()){
//            mW_SaveImagePath->setEnabled(enable);
//            mW_SaveImagePathBt->setEnabled(enable);
//        }

//        if(mW_Background->isChecked()){
//            mW_BackgroundPath->setEnabled(enable);
//            mW_BackgroundPathBt->setEnabled(enable);
//        }

        mRunCond->setEnabled(enable);
    }

    void setArch(Settings::ARCHITECTURE arch);

    void saveSettings();
    void loadSettings();

private slots:
    void updateCudaBlck();
    void updateTimeStep();
    void changeColorCount(int count);

    void openBackgroundTextChanged();
    void openBackgroundPath();

    void openSaveImageTextChanged();
    void openSaveImagePath();

private:
    #ifdef CUDA_BUILD
        cudaDeviceProp mDeviceProp;
    #endif

    int mCudaDeviceCount;

    QGroupBox* mArch;
    QGroupBox* mGrid;
    QGroupBox* mRender;
    QWidget* mRunCond;

    QRadioButton* mW_CPU;
    QRadioButton* mW_GPU;
    QSpinBox* mW_CudaBlck;


    QDoubleSpinBox* mW_GridStepSize;
    QDoubleSpinBox* mW_Viscosity;
    QDoubleSpinBox* mW_SpeedOfSound;
    double mSpeedOfSound2;
    QDoubleSpinBox* mW_Density;
    QLabel* mW_TimeStep;
    double mTimeStep;


    QLabel* mW_RenderTextL1;
    QLabel* mW_RenderTextL2;
    QLabel* mW_RenderTextL3;
    QSpinBox* mW_RenderAfter;
    QSpinBox* mW_RenderFrom;
    QCheckBox* mW_SyncRender;

    QCheckBox* mW_Background;
    QLineEdit* mW_BackgroundPath;
    QPushButton* mW_BackgroundPathBt;
    QImage* mBackground;

    QCheckBox* mW_SaveImage;
    QLineEdit* mW_SaveImagePath;
    QPushButton* mW_SaveImagePathBt;
    QString mSaveImagePath;

    QRadioButton* mW_RenderValueV;
    QRadioButton* mW_RenderValueV_UserScale;
    QDoubleSpinBox* mW_RenderValueV_UserScale_Min;
    QDoubleSpinBox* mW_RenderValueV_UserScale_Max;
    QRadioButton* mW_RenderValueV_AutoScale;

    QRadioButton* mW_RenderValueRho;
    QRadioButton* mW_RenderValueRho_UserScale;
    QDoubleSpinBox* mW_RenderValueRho_UserScale_Min;
    QDoubleSpinBox* mW_RenderValueRho_UserScale_Max;
    QRadioButton* mW_RenderValueRho_AutoScale;

    QRadioButton* mW_RenderValueP;
    QRadioButton* mW_RenderValueP_UserScale;
    QDoubleSpinBox* mW_RenderValueP_UserScale_Min;
    QDoubleSpinBox* mW_RenderValueP_UserScale_Max;
    QRadioButton* mW_RenderValueP_AutoScale;

    QSpinBox* mW_RenderColorCount;
    ColorsSample* mW_RenderColors;
    QList<QColor> mRenderColors;

    QCheckBox* mW_RenderInfoGrid;
    QCheckBox* mW_RenderInfoIterCount;
    QCheckBox* mW_RenderInfoSimTime;
    QCheckBox* mW_RenderInfoColorLegend;
    QCheckBox* mW_RenderInfoVectors;
    QLabel* mW_RenderInfoVectorsL;
    QSpinBox* mW_RenderInfoVectorsLength;


    QCheckBox* mW_MaxIterationCB;
    QSpinBox* mW_MaxIteration;
    QCheckBox* mW_MaxRunTimeCB;
    QSpinBox* mW_MaxRunTime;

    void refreshConnections();
    void openBackground(QString path);
};

#endif // MESHSETTINGS_H
