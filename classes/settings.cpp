#include "classes/settings.h"
#include <QFileDialog>
#include <QApplication>
#include <QMessageBox>
#include <QSettings>

void Settings::connectWidgets(){

    //architecture group
    connect(mW_CPU,SIGNAL(toggled(bool)),this,SLOT(updateCudaBlck()));
    connect(mW_GPU,SIGNAL(toggled(bool)),this,SLOT(updateCudaBlck()));
    mW_CudaBlck->setMinimum(1);

    //grid settings group
    connect(mW_GridStepSize,SIGNAL(valueChanged(double)),this,SLOT(updateTimeStep()));
    connect(mW_SpeedOfSound,SIGNAL(valueChanged(double)),this,SLOT(updateTimeStep()));

    connect(mW_Background,SIGNAL(toggled(bool)),mW_BackgroundPath,SLOT(setEnabled(bool)));
    connect(mW_Background,SIGNAL(toggled(bool)),mW_BackgroundPathBt,SLOT(setEnabled(bool)));
    connect(mW_SaveImage,SIGNAL(toggled(bool)),mW_SaveImagePath,SLOT(setEnabled(bool)));
    connect(mW_SaveImage,SIGNAL(toggled(bool)),mW_SaveImagePathBt,SLOT(setEnabled(bool)));

    connect(mW_BackgroundPath,SIGNAL(textChanged(QString)),this,SLOT(openBackgroundTextChanged()));
    connect(mW_BackgroundPathBt,SIGNAL(clicked()),this,SLOT(openBackgroundPath()));

    connect(mW_SaveImagePath,SIGNAL(textChanged(QString)),this,SLOT(openSaveImageTextChanged()));
    connect(mW_SaveImagePathBt,SIGNAL(clicked()),this,SLOT(openSaveImagePath()));

    connect(mW_RenderInfoVectors,SIGNAL(toggled(bool)),mW_RenderInfoVectorsL,SLOT(setEnabled(bool)));
    connect(mW_RenderInfoVectors,SIGNAL(toggled(bool)),mW_RenderInfoVectorsLength,SLOT(setEnabled(bool)));

    connect(mW_RenderColorCount,SIGNAL(valueChanged(int)),this,SLOT(changeColorCount(int)));

    connect(mW_MaxIterationCB,SIGNAL(toggled(bool)),mW_MaxIteration,SLOT(setEnabled(bool)));
    connect(mW_MaxRunTimeCB,SIGNAL(toggled(bool)),mW_MaxRunTime,SLOT(setEnabled(bool)));

    refreshConnections();
}

void Settings::refreshConnections()
{
    mW_CPU->setChecked(true);
    mW_GPU->setChecked(true);

    mW_Background->setChecked(true);
    mW_Background->setChecked(false);

    mW_SaveImage->setChecked(true);
    mW_SaveImage->setChecked(false);

    mW_MaxIterationCB->setChecked(true);
    mW_MaxIterationCB->setChecked(false);

    mW_MaxRunTimeCB->setChecked(true);
    mW_MaxRunTimeCB->setChecked(false);

    mW_RenderInfoVectors->setChecked(true);
    mW_RenderInfoVectors->setChecked(false);

    changeColorCount(mW_RenderColorCount->value());

    #ifdef CUDA_BUILD
        CUDA_SAFE_CALL(cudaGetDeviceCount(&mCudaDeviceCount));

        if(mCudaDeviceCount>0){
            int devices;

            CUDA_SAFE_CALL(cudaGetDevice(&devices));
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&mDeviceProp,devices));

            mW_CudaBlck->setValue(mDeviceProp.maxThreadsPerBlock);
            mW_CudaBlck->setMaximum(mDeviceProp.maxThreadsPerBlock);
        }
    #else
        mCudaDeviceCount = 0;
    #endif

    if(mCudaDeviceCount<=0){
        mW_CudaBlck->setDisabled(true);
        mW_GPU->setDisabled(true);
        mW_CPU->setChecked(true);
        mW_CudaBlck->setValue(0);
#ifdef CUDA_BUILD
        QMessageBox::warning((QWidget*)this->parent(),QString::fromUtf8("Kártya nem található!"),QString::fromUtf8("CUDA kártya nem található! Csak CPU-n tud szimulációt futtatni!"));
#endif
    }

    //a LBMSimulation osztály iterate metodusaban hasznalt szinkronizaci gyorsabb futast eredmenyez, mint a timeres megoldas ezert elrejtettem az opciot
    mW_SyncRender->setVisible(false);
}

Settings::ARCHITECTURE Settings::getArch()
{
    return mW_CPU->isChecked()? Settings::CPU : Settings::GPU;
}

void Settings::setArch(Settings::ARCHITECTURE arch)
{
    if(arch == Settings::GPU && mCudaDeviceCount>0){
        mW_GPU->setChecked(true);
    }else{
        mW_CPU->setChecked(true);
    }
}

int Settings::getBlockSize()
{
    return mW_CudaBlck->value();
}


double Settings::getViscosity()
{
    return mW_Viscosity->value();
}

double Settings::getGridStepSize()
{
    return mW_GridStepSize->value();
}

double Settings::getSpeedOfSound()
{
    return mW_SpeedOfSound->value();
}

double Settings::getDensity()
{
    return mW_Density->value();
}

double Settings::getPressure()
{
    return mW_Density->value()*mSpeedOfSound2;
}

double Settings::getTimeStep()
{
    return mTimeStep;
}


int Settings::getRenderAfter()
{
    return mW_RenderAfter->value();
}


int Settings::getMaxIteration()
{
    if(mW_MaxIterationCB->isChecked())
        return mW_MaxIteration->value();
    else
        return -1; //végtelen
}

int Settings::getMaxRunTime()
{
    if(mW_MaxRunTimeCB->isChecked())
        return mW_MaxRunTime->value();
    else
        return -1; //végtelen
}


void Settings::updateCudaBlck()
{
    mW_CudaBlck->setEnabled(mW_GPU->isChecked());
}

void Settings::updateTimeStep(){

    if(mW_SpeedOfSound->value()>0.){
        mTimeStep = mW_GridStepSize->value()*(LBM_CS / mW_SpeedOfSound->value());
    }else{
        mTimeStep = 0.;
    }

    mSpeedOfSound2 = mW_SpeedOfSound->value()*mW_SpeedOfSound->value();

    mW_TimeStep->setText(QString("%1").arg(mTimeStep));
}

Settings::DISPLAY_VALUE Settings::getRenderDisplay()
{
    if(mW_RenderValueV->isChecked()){
        return VXYC;
    }else if(mW_RenderValueRho->isChecked()){
        return RHOC;
    }else{
        return PRESSURE;
    }
}

Settings::VALUE_SCALE Settings::getRenderDisplayScale()
{
    if(mW_RenderValueV->isChecked()){
        if(mW_RenderValueV_UserScale->isChecked()){
            return USER;
        }else{
            return AUTO;
        }
    }else if(mW_RenderValueRho->isChecked()){
        if(mW_RenderValueRho_UserScale->isChecked()){
            return USER;
        }else{
            return AUTO;
        }
    }else{
        if(mW_RenderValueP_UserScale->isChecked()){
            return USER;
        }else{
            return AUTO;
        }
    }
}

double Settings::getRenderDisplayScaleMin()
{
    if(mW_RenderValueV->isChecked()){
        if(mW_RenderValueV_UserScale->isChecked()){
            return mW_RenderValueV_UserScale_Min->value();
        }else{
            return 0.0;
        }
    }else if(mW_RenderValueRho->isChecked()){
        if(mW_RenderValueRho_UserScale->isChecked()){
            return mW_RenderValueRho_UserScale_Min->value();
        }else{
            return 0.0;
        }
    }else{
        if(mW_RenderValueP_UserScale->isChecked()){
            return mW_RenderValueP_UserScale_Min->value();
        }else{
            return 0.0;
        }
    }
}

double Settings::getRenderDisplayScaleMax()
{
    if(mW_RenderValueV->isChecked()){
        if(mW_RenderValueV_UserScale->isChecked()){
            return mW_RenderValueV_UserScale_Max->value();
        }else{
            return 0.0;
        }
    }else if(mW_RenderValueRho->isChecked()){
        if(mW_RenderValueRho_UserScale->isChecked()){
            return mW_RenderValueRho_UserScale_Max->value();
        }else{
            return 0.0;
        }
    }else{
        if(mW_RenderValueP_UserScale->isChecked()){
            return mW_RenderValueP_UserScale_Max->value();
        }else{
            return 0.0;
        }
    }
}

double Settings::getSpeedOfSound2()
{
    return mSpeedOfSound2;
}

int Settings::getRenderColorCount()
{
    return mW_RenderColorCount->value();
}

void Settings::changeColorCount(int count)
{
    mRenderColors.clear();

    switch(count){
        case 2:
            mRenderColors.append(QColor(Qt::blue));
            mRenderColors.append(QColor(Qt::red));
        break;

        case 3:
            mRenderColors.append(QColor(Qt::blue));   //Min
            mRenderColors.append(QColor(Qt::green));
            mRenderColors.append(QColor(Qt::red));
        break;

        case 4:
            mRenderColors.append(QColor(Qt::blue));   //Min
            mRenderColors.append(QColor(Qt::green));
            mRenderColors.append(QColor(Qt::red));
            mRenderColors.append(QColor(Qt::black)); //Max
        break;

        case 5:
            mRenderColors.append(QColor(Qt::blue));   //Min
            mRenderColors.append(QColor(Qt::cyan));
            mRenderColors.append(QColor(Qt::green));
            mRenderColors.append(QColor(Qt::red));
            mRenderColors.append(QColor(Qt::black)); //Max
        break;
    }

    mW_RenderColors->setColors(mRenderColors);
}

QImage* Settings::getBackgroundImg()
{
    return mBackground;
}

QString Settings::getSaveImagePath(int iteration)
{
    if(mSaveImagePath.length()>0){
        return mSaveImagePath.arg(iteration,9,10,QChar('0'));
    }else{
        return mSaveImagePath;
    }
}

void Settings::openBackgroundPath()
{
    mW_BackgroundPath->setText(QFileDialog::getOpenFileName((QWidget*)this->parent(),"Háttér megnyitása",qApp->applicationDirPath(),"Images (*.png)"));
}

void Settings::openSaveImagePath()
{
    mW_SaveImagePath->setText(QFileDialog::getSaveFileName((QWidget*)this->parent(),"Renderelt kép mentése fájlba",qApp->applicationDirPath()+"/ImageSequence_%1","Image (*.png)"));
}

void Settings::openBackground(QString path)
{
    if(mBackground){
        delete mBackground;
    }

    if(path.length()>0 && QFile::exists(path)){
        mBackground = new QImage(path);
    }else{
        mBackground = 0;
    }
}

void Settings::openSaveImageTextChanged()
{
    mSaveImagePath = mW_SaveImagePath->text();
}

void Settings::openBackgroundTextChanged()
{
    if(QFile::exists(mW_BackgroundPath->text())){
        openBackground(mW_BackgroundPath->text());
    }
}

int Settings::getRenderFrom()
{
    return mW_RenderFrom->value();
}

void Settings::saveSettings()
{
#ifdef ENABLE_SAVE_SETTINGS
    QSettings settings(qApp->applicationDirPath()+"/settings.ini",QSettings::IniFormat);

    settings.beginGroup("Settings");
    settings.setValue("Architecture",getArch());
    settings.setValue("CudaBlockSize", getBlockSize());
    settings.setValue("GridStepSize", getGridStepSize());
    settings.setValue("Viscosity", getViscosity());
    settings.setValue("SpeedOfSound", getSpeedOfSound());
    settings.setValue("Density", getDensity());
    settings.setValue("RenderAfter", getRenderAfter());
    settings.setValue("RenderFrom", getRenderFrom());
    settings.setValue("DisplayValue", getRenderDisplay());

    settings.setValue("DisplayColorsCount", getRenderColorCount());
    settings.setValue("DisplayGridStep", getRenderInfoGrid());
    settings.setValue("DisplayIterationCount", getRenderInfoIterCount());
    settings.setValue("DisplaySimulationTime", getRenderInfoSimTime());
    settings.setValue("DisplayColorLegend", getRenderInfoColorLegend());
    settings.setValue("DisplayVectors", getRenderInfoVectors());
    settings.setValue("DisplayVectorsLength", getRenderInfoVectorsLength());

    settings.endGroup();
#endif
}

void Settings::loadSettings()
{
    QSettings settings(qApp->applicationDirPath()+"/settings.ini",QSettings::IniFormat);

    setArch((Settings::ARCHITECTURE)settings.value("Architecture",Settings::GPU).toInt());
    mW_CudaBlck->setValue(settings.value("CudaBlockSize", 1024).toInt());
    mW_GridStepSize->setValue(settings.value("GridStepSize", 0.001).toDouble());
    mW_Viscosity->setValue(settings.value("Viscosity", 0.0011041).toDouble());
    mW_SpeedOfSound->setValue(settings.value("SpeedOfSound", 1497).toDouble());
    mW_Density->setValue(settings.value("Density", 997.0479).toDouble());

    if(getArch() == GPU){
        mW_RenderAfter->setValue(settings.value("RenderAfter", 100).toInt());
    }else{
        mW_RenderAfter->setValue(settings.value("RenderAfter", 10).toInt());
    }
    mW_RenderFrom->setValue(settings.value("RenderFrom", 1).toInt());


    mW_RenderColorCount->setValue(settings.value("DisplayColorsCount", 3).toInt());
    mW_RenderInfoGrid->setChecked(settings.value("DisplayGridStep", true).toBool());
    mW_RenderInfoIterCount->setChecked(settings.value("DisplayIterationCount", true).toBool());
    mW_RenderInfoSimTime->setChecked(settings.value("DisplaySimulationTime", true).toBool());
    mW_RenderInfoColorLegend->setChecked(settings.value("DisplayColorLegend", true).toBool());
    mW_RenderInfoVectors->setChecked(settings.value("DisplayVectors", false).toBool());
    mW_RenderInfoVectorsLength->setValue(settings.value("DisplayVectorsLength", 10).toInt());
}

void Settings::setupDefaultValues()
{
    //1mm
    mW_GridStepSize->setValue(0.001);

    //http://en.wikipedia.org/wiki/Speed_of_sound
    mW_SpeedOfSound->setValue(1497);

    //http://en.wikipedia.org/wiki/Viscosity
    mW_Viscosity->setValue(0.0011041);

    //http://en.wikipedia.org/wiki/Density
    mW_Density->setValue(997.0479);

    if(mW_GPU->isEnabled()){
        mW_GPU->setChecked(true);
    }

    mW_MaxIterationCB->setChecked(true);
    mW_MaxIteration->setValue(1000000);

    mW_MaxRunTimeCB->setChecked(false);

    mW_RenderAfter->setValue(1000);
    mW_RenderValueV->setChecked(true);
    mW_RenderValueV_AutoScale->setChecked(true);

    mW_RenderInfoGrid->setChecked(true);
    mW_RenderInfoIterCount->setChecked(true);
    mW_RenderInfoSimTime->setChecked(true);
    mW_RenderInfoColorLegend->setChecked(true);
    mW_RenderInfoVectors->setChecked(true);

    mW_RenderValueRho_AutoScale->setChecked(true);
}
