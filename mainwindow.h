#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStringList>
#include "classes/settings.h"
#include "classes/lbmmesh.h"
#include "classes/lbmsimulation.h"
#include "classes/meshrenderer.h"
#include "classes/logger.h"
#include "QTimer"

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pBStart_clicked();
    void on_pBStop_clicked();

    void on_actionOpen_mask_triggered();

    void timerTick();

    void on_actionNew_mask_triggered();

    void on_actionQuit_triggered();

    void on_actionSave_Image_triggered();

private:
    Ui::MainWindow *mUi;
    bool mPaused;

    Settings* mSettings;
    LBMMesh* mLBMMesh;
    MeshRenderer *mMeshRenderer;

    LBMSimulation* mSimulation;
    QTimer* mTimer;

    void createObjects();
    void deleteObjects();

    void changeStartBtnIcon(bool pauseIcon);
    void lockControls(bool lock);
};

#endif // MAINWINDOW_H
