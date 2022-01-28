#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <qmath.h>
#include <QDebug>
#include <QFileDialog>
#include "newmaskdialog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    mUi(new Ui::MainWindow),
    mPaused(false),
    mSettings(new Settings(this)),
    mTimer(new QTimer(this))
{
    mUi->setupUi(this);
    mSimulation = 0;
    mLBMMesh = 0;
    mMeshRenderer = 0;

    //színek küldése az editornak
    connect(mUi->widgetBoundaryConditions,SIGNAL(newColorAvailable(QColor)),mUi->widgetEdit,SLOT(addColor(QColor)));
    connect(mUi->widgetBoundaryConditions,SIGNAL(colorDeleted(QColor)),mUi->widgetEdit,SLOT(removeColor(QColor)));

    //settings osztály beállítása
    mSettings->setGroupBoxes(mUi->groupBoxArch,mUi->groupBoxGrid,mUi->groupBoxRender,mUi->widgetRunConditions);

    mSettings->setArchWidgets(mUi->radioButtonCPU,mUi->radioButtonGPU);
    mSettings->setBlockSizeWidget(mUi->spinBoxCudaBckSize);

    mSettings->setViscosityWidget(mUi->doubleSpinBoxViscosity);
    mSettings->setSpeedOfSoundWidget(mUi->doubleSpinBoxSpeedOfSound);
    mSettings->setGridStepSizeWidget(mUi->doubleSpinBoxGridSize);
    mSettings->setDensityWidget(mUi->doubleSpinBoxDensity);
    mSettings->setTimeStepWidget(mUi->labelTimeStep);

    mSettings->setRenderLabels(mUi->label_3,mUi->label_2,mUi->label_11);
    mSettings->setRenderAfterWidget(mUi->spinBoxRenderAfter);
    mSettings->setRenderFromWidget(mUi->spinBoxRenderFrom);
    mSettings->setSyncRenderWidget(mUi->checkBoxSyncRender);
    mSettings->setBackgroundWidget(mUi->checkBoxBackground,mUi->lineEditBackground,mUi->pushButtonBackground);
    mSettings->setSaveImageWidget(mUi->checkBoxSaveImage,mUi->lineEditSaveImage,mUi->pushButtonSaveImage);
    mSettings->setRenderValueVWidget(  mUi->radioButtonRenderV,  mUi->radioButtonRenderVUserScale,  mUi->doubleSpinBoxRenderVUserScaleMin,  mUi->doubleSpinBoxRenderVUserScaleMax,  mUi->radioButtonRenderVAutoScale);
    mSettings->setRenderValueRhoWidget(mUi->radioButtonRenderRho,mUi->radioButtonRenderRhoUserScale,mUi->doubleSpinBoxRenderRhoUserScaleMin,mUi->doubleSpinBoxRenderRhoUserScaleMax,mUi->radioButtonRenderRhoAutoScale);
    mSettings->setRenderValuePWidget(  mUi->radioButtonRenderP,  mUi->radioButtonRenderPUserScale,  mUi->doubleSpinBoxRenderPUserScaleMin,  mUi->doubleSpinBoxRenderPUserScaleMax,  mUi->radioButtonRenderPAutoScale);
    mSettings->setRenderColorCount(mUi->spinBoxRenderColorCount,mUi->widgetRenderColors);

    mSettings->setRenderInfoGrid(mUi->checkBoxRenderInfoGrid);
    mSettings->setRenderInfoIterCount(mUi->checkBoxRenderInfoIterCount);
    mSettings->setRenderInfoSimTime(mUi->checkBoxRenderInfoSimTime);
    mSettings->setRenderInfoColorLegend(mUi->checkBoxRenderInfoColorLegend);
    mSettings->setRenderInfoVectors(mUi->checkBoxRenderInfoVectors,mUi->labelRenderInfoLength,mUi->spinBoxRenderVectorLength);

    mSettings->setMaxIterationWidget(mUi->checkBoxMaxIteration,mUi->spinBoxMaxIteration);
    mSettings->setMaxRunTimeWidget(mUi->checkBoxRunTime,mUi->spinBoxMaxRunTime);

    mSettings->connectWidgets();
    mSettings->loadSettings();

    mUi->widgetBoundaryConditions->setSettings(mSettings);

    mTimer->setInterval(10);
    connect(mTimer,SIGNAL(timeout()),this,SLOT(timerTick()));

    mUi->pBStop->setEnabled(false);

    connect(Logger::instance(),SIGNAL(newLogAdded(QString)),mUi->plainTextEditLogger,SLOT(appendPlainText(QString)),Qt::QueuedConnection);

    Logger::instance()->appendLog(QString("Elindult az LBM D2Q9 szimulációs szoftver."));
}

MainWindow::~MainWindow()
{
    mSettings->saveSettings();
    deleteObjects();

    delete mUi;
}

void MainWindow::on_pBStart_clicked()
{
    if(mSettings->getSyncRender()){
        if(mSimulation){
            if(mTimer->isActive()){
                mTimer->stop();
                changeStartBtnIcon(false);
            }else{
                mTimer->start();
                changeStartBtnIcon(true);
                mUi->tabWidget->setCurrentIndex(0);
            }
        }else{
            createObjects();
            mUi->tabWidget->setCurrentIndex(0);
            mSimulation->preProcess();
            mTimer->start();
            changeStartBtnIcon(true);
            mUi->pBStop->setEnabled(true);
        }
    }else{
        if(mSimulation){
            //ha fut azaz ha szimulál vagy pause
            if(mSimulation->isRunning()){
                if(mSimulation->getPause())
                {
                    mSimulation->resume();

                    changeStartBtnIcon(true);
                    mUi->tabWidget->setCurrentIndex(0);

                }else{
                    mSimulation->pause();

                    changeStartBtnIcon(false);
                }
            }
        }else{
            createObjects();
            mUi->tabWidget->setCurrentIndex(0);
            mSimulation->start();

            mUi->pBStop->setEnabled(true);

            //change icon to pause
            changeStartBtnIcon(true);
        }
    }
}

void MainWindow::on_pBStop_clicked()
{
    mTimer->stop();
    deleteObjects();
    changeStartBtnIcon(false);
    mUi->pBStop->setEnabled(false);
}

void MainWindow::createObjects()
{
    Logger::instance()->appendLog(QString("Objektumok inicializálása."));
    lockControls(false);

    QImage* mask = mUi->widgetEdit->getMask();
    BoundaryConditionList* bcList = mUi->widgetBoundaryConditions->getBCList();

    mLBMMesh = new LBMMesh(mask, bcList);
    mMeshRenderer = new MeshRenderer(mUi->widgetView,mLBMMesh,mask->width(),mask->height(),mSettings);
    mSimulation = new LBMSimulation(mLBMMesh,bcList,mSettings,mMeshRenderer);

    connect(mSimulation,SIGNAL(meshDataUpdated(int)),this->mMeshRenderer,SLOT(render(int)),Qt::QueuedConnection);
    connect(mSimulation,SIGNAL(finished()),this,SLOT(on_pBStop_clicked()));
}

void MainWindow::on_actionOpen_mask_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Rcs megnyits",qApp->applicationDirPath(),"Images (*.png)");

    if(QFile::exists(fileName)){
        mUi->widgetEdit->openMaskFromFile(fileName);
        mUi->tabWidget->setCurrentIndex(1);
    }
}

void MainWindow::timerTick()
{
    mSimulation->iterate();

    if(mSimulation->getQuit()){
        on_pBStop_clicked();
    }
}

void MainWindow::deleteObjects()
{
    if(mSimulation)
    {
        mSimulation->quit();
        mSimulation->wait();
    }

    if(mLBMMesh){
        delete mLBMMesh;
    }

    if(mMeshRenderer){
        if(mSimulation){
            disconnect(mSimulation,SIGNAL(meshDataUpdated(int)),mMeshRenderer,SLOT(render(int)));
            disconnect(mSimulation,SIGNAL(finished()),this,SLOT(on_pBStop_clicked()));
        }
        delete mMeshRenderer;
    }

    if(mSimulation){
        delete mSimulation;
    }

    mLBMMesh = 0;
    mMeshRenderer = 0;
    mSimulation = 0;
    lockControls(true);
}

void MainWindow::changeStartBtnIcon(bool pauseIcon)
{
    if(pauseIcon){
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icon_pause"), QSize(), QIcon::Normal, QIcon::Off);
        mUi->pBStart->setIcon(icon);
    }else{
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/icon_play"), QSize(), QIcon::Normal, QIcon::Off);
        mUi->pBStart->setIcon(icon);
    }
}

void MainWindow::on_actionNew_mask_triggered()
{
    NewMaskDialog dialog(this);

    dialog.exec();

    if(dialog.result() == QDialog::Accepted){
        mUi->widgetEdit->newDrawing(dialog.getMaskWidth(),dialog.getMaskHeight());
    }
}

void MainWindow::lockControls(bool lock)
{
    mSettings->setEnable(lock);
    mUi->widgetBoundaryConditions->setEnabled(lock);
}

void MainWindow::on_actionQuit_triggered()
{
    this->close();
}

void MainWindow::on_actionSave_Image_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(this,QString::fromUtf8("Kép mentése"),qApp->applicationDirPath(),"Images (*.png)");

    if(fileName.length()>0){
        mUi->widgetView->saveImage(fileName);
    }
}
