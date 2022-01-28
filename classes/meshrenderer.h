#ifndef MESHRENDERER_H
#define MESHRENDERER_H

#include <QObject>
#include <QMutex>
#include <QMutexLocker>

#include "widgets/viewerwidget.h"
#include "classes/lbmparticle.h"
#include "classes/lbmmesh.h"
#include "classes/settings.h"

class MeshRenderer : public QObject
{
    Q_OBJECT
public:

    explicit MeshRenderer(ViewerWidget *view, LBMMesh *mesh, int imgWidth, int imgHeight, Settings* settings, QObject *parent=0);

    ~MeshRenderer(){
        delete mImgFlow;
    }

    void setDrawColorList(QList<QColor> colors){
        mRenderColors = colors;
        mRenderColorStep = 1.0/(REAL)(mRenderColors.count()-1);
    }

    REAL getRhoc(int index){return mRhoc[index];}
    REAL getVxc(int index){return mVxc[index];}
    REAL getVyc(int index){return mVyc[index];}

    void setRhoc(int index, REAL value){ mRhoc[index] = value;}
    void setVxc(int index, REAL value){ mVxc[index] = value;}
    void setVyc(int index, REAL value){ mVyc[index] = value;}

    void fillVectors(REAL rho, REAL vx, REAL vy );

    void setRendered(bool value){QMutexLocker locker(&mMutex); mRendered = value;}
    bool isRendered(){QMutexLocker locker(&mMutex); return mRendered;}

public slots:
    void render(int iterCount);

private:
    mutable QMutex mMutex;
    bool mRendered;

    QList<QColor> mRenderColors;
    REAL mRenderColorStep;

    //pointers for objects, don't delete it!
    Settings* mSettings;
    LBMMesh *mMesh;
    ViewerWidget *mView;

    QImage* mImgFlow;
    int mWidth,mHeight;
    int mImgOffsetY;
    int mImgOffsetX; 

    QList<REAL> mRhoc;
    QList<REAL> mVxc;
    QList<REAL> mVyc;

    QColor getDrawColor(float percent);
    QColor getFadeColor(QColor from, QColor to, REAL percent);
    int getFadeValue(int from, int to, REAL percent);

    //rendering
    REAL mMinRho;
    REAL mMaxRho;
    REAL mMinV;
    REAL mMaxV;
    Settings::VALUE_SCALE mCalculatedScale;
    Settings::DISPLAY_VALUE mCalculatedValue;
    REAL mCalculatedValueMin;
    REAL mCalculatedValueMax;


    void calcMinMaxValues();
    void renderValue(QPainter* p, Settings::DISPLAY_VALUE value, Settings::VALUE_SCALE scale, int offsetX, int offsetY);
    REAL getVectLength(int index);

    void drawTimer(QPainter* p, int x, int y, int itercount);
    void drawIterCount(QPainter* p, int x, int y, int itercount);
    void drawColorLegend(QPainter* p);
    void drawGridSnap(QPainter* p, int offsetX, int offsetY);
    void drawVectors(QPainter* p, int vectLength, int offsetX, int offsetY);

    bool drawIsSettingsChanged();
};

#endif // MESHRENDERER_H
