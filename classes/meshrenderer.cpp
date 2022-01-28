#include "meshrenderer.h"
#include <QPainter>
#include <QApplication>
#include "classes/logger.h"
#include "time.h"

#define TEXT_OFFSET_X 3
#define TEXT_HEIGHT 15
#define GRID_HEIGHT 16
#define COLORLEGEND_OFFSET 50
#define COLORLEGEND_WIDTH 250

MeshRenderer::MeshRenderer(ViewerWidget *view, LBMMesh *mesh, int imgWidth, int imgHeight, Settings *settings, QObject *parent):
    QObject(parent)
{
    mMesh=mesh;
    mView=view;
    mSettings = settings;
    mImgFlow = 0;

    mCalculatedScale = Settings::UNKNOWN_SCALE;
    mRendered = false;

    fillVectors(1.0f,0.f,0.f);

    mWidth = imgWidth;
    mHeight = imgHeight;

    mRenderColors.clear();
}

void MeshRenderer::render(int iterCount)
{
    if(mRenderColors.count() != mSettings->getRenderColors().count()){
        mRenderColors = mSettings->getRenderColors();
        mRenderColorStep = 1.0 / (float)(mRenderColors.count()-1);
    }

    if(drawIsSettingsChanged()){
        if(mSettings->getRenderDisplayScale() == Settings::AUTO){
            calcMinMaxValues();
        }else{
            if(mSettings->getRenderDisplay() == Settings::VXYC){
                mMinV = mSettings->getRenderDisplayScaleMin()/mSettings->getSpeedOfSound();
                mMaxV = mSettings->getRenderDisplayScaleMax()/mSettings->getSpeedOfSound();

                mCalculatedValueMin = mMinV;
                mCalculatedValueMax = mMaxV;
            } else if(mSettings->getRenderDisplay() == Settings::RHOC){
                mMinRho = mSettings->getRenderDisplayScaleMin()/mSettings->getDensity();
                mMaxRho = mSettings->getRenderDisplayScaleMax()/mSettings->getDensity();

                mCalculatedValueMin = mMinRho;
                mCalculatedValueMax = mMaxRho;
            }else {
                mMinRho = (mSettings->getRenderDisplayScaleMin()*1000000.)/(mSettings->getSpeedOfSound2()*mSettings->getDensity());
                mMaxRho = (mSettings->getRenderDisplayScaleMax()*1000000.)/(mSettings->getSpeedOfSound2()*mSettings->getDensity());

                mCalculatedValueMin = mMinRho;
                mCalculatedValueMax = mMaxRho;
            }
        }

        mCalculatedScale = mSettings->getRenderDisplayScale();
        mCalculatedValue = mSettings->getRenderDisplay();
    }

    mImgOffsetY=0;
    mImgOffsetX=0;

    if(mSettings->getRenderInfoIterCount()){
        mImgOffsetY += TEXT_HEIGHT;
    }

    if(mSettings->getRenderInfoSimTime()){
        mImgOffsetY += TEXT_HEIGHT;
    }

    if(mSettings->getRenderInfoColorLegend()){
        if(mImgOffsetY<TEXT_HEIGHT*2){
            mImgOffsetY = TEXT_HEIGHT*2;
        }
    }

    if(mSettings->getRenderInfoGrid()){
        mImgOffsetX += GRID_HEIGHT;
        mImgOffsetY += GRID_HEIGHT;
    }

    if(!mImgFlow){
        mImgFlow = new QImage(mWidth+mImgOffsetX+1,mHeight+mImgOffsetY+1,QImage::Format_ARGB32);
    }

    mImgFlow->fill(qRgba(255,255,255,255));

    QPainter p(mImgFlow);

    if(mSettings->getBackground() && mSettings->getBackgroundImg()){
        p.drawImage(mImgOffsetX,mImgOffsetY,*mSettings->getBackgroundImg());
    }

    renderValue(&p,mSettings->getRenderDisplay(),mSettings->getRenderDisplayScale(),mImgOffsetX,mImgOffsetY);

    if(mSettings->getRenderInfoGrid()){
        drawGridSnap(&p, mImgOffsetX-1, mImgOffsetY-1);
    }

    if(mSettings->getRenderInfoVectors()){
        drawVectors(&p,mSettings->getRenderInfoVectorsLength(),mImgOffsetX,mImgOffsetY);
    }

    mImgOffsetY=0;

    if(mSettings->getRenderInfoColorLegend()){
        drawColorLegend(&p);
    }

    if(mSettings->getRenderInfoIterCount()){
        drawIterCount(&p,TEXT_OFFSET_X,mImgOffsetY,iterCount);
        mImgOffsetY += TEXT_HEIGHT;
    }

    if(mSettings->getRenderInfoSimTime()){
        drawTimer(&p,TEXT_OFFSET_X,mImgOffsetY,iterCount);
        mImgOffsetY += TEXT_HEIGHT;
    }

    if(mSettings->getSaveImage() && mSettings->getSaveImagePath(iterCount).length()>0 ){
        mImgFlow->save(mSettings->getSaveImagePath(iterCount),"PNG",100);
    }

    mView->drawImage(mImgFlow);
    setRendered(true);
}

inline void MeshRenderer::renderValue(QPainter *p, Settings::DISPLAY_VALUE value,  Settings::VALUE_SCALE scale, int offsetX, int offsetY)
{
    bool repaint=true;
    REAL percent;
    Point point;

    if(value == Settings::VXYC){
        while(repaint){
            repaint=false;
            REAL sumV,stepV = (mMaxV-mMinV);

            for(int i=0;i<mVxc.count();i++){
                sumV = getVectLength(i);

                if(scale == Settings::AUTO){
                    if(sumV>mMaxV){
                        mMaxV = sumV;
                        repaint=true;
                    }else if(sumV<mMinV){
                        mMinV = sumV;
                        repaint=true;
                    }
                }else{
                    if(sumV>mMaxV){
                        percent = 1.0;
                    }else if(sumV<mMinV){
                        percent = 0.0;
                    }else if(stepV != 0.0){
                        percent = (sumV-mMinV) / stepV;
                    }else{
                        percent = 0.0;
                    }
                }

                //feleslegesen nem rajzol
                if(!repaint){
                    if(scale == Settings::AUTO){
                        if(stepV != 0.0){
                            percent = (sumV-mMinV) / stepV;
                        }else{
                            percent = 0.0;
                        }
                    }

                    point = mMesh->getVisibleCellPos(i);
                    p->fillRect(offsetX+point.x(),offsetY+point.y(),1,1,getDrawColor(percent));

                }
            }
        }
    }else{
        while(repaint){
            repaint=false;
            REAL stepRho = (mMaxRho-mMinRho);

            for(int i=0;i<mRhoc.count();i++){

                if(scale == Settings::AUTO){
                    if(mRhoc[i]>mMaxRho){
                        mMaxRho = mRhoc[i];
                        repaint=true;
                    }else if(mRhoc[i]<mMinRho){
                        mMinRho = mRhoc[i];
                        repaint=true;
                    }
                }else{
                    if(mRhoc[i]>mMaxRho){
                        percent = 1.0;
                    }else if(mRhoc[i]<mMinRho){
                        percent = 0.0;
                    }else if(stepRho != 0.0){
                        percent = (mRhoc[i]-mMinRho) / stepRho;
                    }else{
                        percent = 0.0;
                    }
                }

                //feleslegesen nem rajzol
                if(!repaint){
                    if(scale == Settings::AUTO){
                        if(stepRho != 0.0){
                            percent = (mRhoc[i]-mMinRho) / stepRho;
                        }else{
                            percent = 0.0;
                        }
                    }

                    point = mMesh->getVisibleCellPos(i);
                    p->fillRect(offsetX+point.x(),offsetY+point.y(),1,1,getDrawColor(percent));
                }
            }
        }
    }
}

void MeshRenderer::calcMinMaxValues()
{
    int sumV;

    for(int i=0;i<mRhoc.count();i++){
        if(i==0){
            mMinRho = mMaxRho = mRhoc[i];
            mMinV = mMaxV = getVectLength(i);
        }else{
            if(mMinRho>mRhoc[i]){
                mMinRho = mRhoc[i];
            }else if(mMaxRho<mRhoc[i]){
                mMaxRho = mRhoc[i];
            }

            sumV = getVectLength(i);
            if(mMinV>sumV){
                mMinV = sumV;
            }else if(mMaxV<sumV){
                mMaxV = sumV;
            }
        }
    }
}


//0.0-1.0
inline QColor MeshRenderer::getDrawColor(float percent)
{
    if(percent<=0.0){
        return mRenderColors.first();
    }

    if(percent>=1.0){
        return mRenderColors.last();
    }

    int colorIdx = 0;
    while (percent> mRenderColorStep) {
        colorIdx ++;
        percent -= mRenderColorStep;
    }

    if(colorIdx+1 < mRenderColors.count())
        return getFadeColor(mRenderColors[colorIdx],mRenderColors[(colorIdx+1)],percent/mRenderColorStep);
    else
        return mRenderColors.last();
}

QColor MeshRenderer::getFadeColor(QColor from, QColor to, REAL percent)
{
    return QColor(getFadeValue(from.red(),to.red(),percent), getFadeValue(from.green(),to.green(),percent), getFadeValue(from.blue(),to.blue(),percent));
}

int MeshRenderer::getFadeValue(int from, int to, REAL percent)
{
    if(from>to){
        return from-((REAL)(abs(from-to))*percent);
    }else{
        return from+((REAL)(abs(from-to))*percent);
    }
}

void MeshRenderer::fillVectors(float rho, float vx, float vy)
{
    mRhoc.clear();
    mVxc.clear();
    mVyc.clear();

    for (int idx = 0; idx < mMesh->getVisibleCellCount(); ++idx) {
        mRhoc.append(rho);
        mVxc.append(vx);
        mVyc.append(vy);
    }
}

inline float MeshRenderer::getVectLength(int index)
{
    return sqrt(mVxc[index]*mVxc[index]+ mVyc[index]*mVyc[index]);
}

inline void MeshRenderer::drawTimer(QPainter *p, int x, int y, int itercount)
{
    p->setPen(QColor(Qt::black));
    p->drawText(x,y,mImgFlow->width(),30,Qt::AlignLeft & Qt::AlignVCenter,QString::fromUtf8("Szimulált idő: %1").arg(itercount*mSettings->getTimeStep()));
}

inline void MeshRenderer::drawIterCount(QPainter *p, int x, int y, int itercount)
{
    p->setPen(QColor(Qt::black));
    p->drawText(x,y,mImgFlow->width(),30,Qt::AlignLeft & Qt::AlignVCenter,QString::fromUtf8("Iterációk száma: %1").arg(itercount));
}

inline void MeshRenderer::drawGridSnap(QPainter *p, int offsetX, int offsetY)
{
    int sY = offsetY-(GRID_HEIGHT/2);
    int lY = offsetY-GRID_HEIGHT;

    p->setPen(QColor(Qt::black));

    for (int x = offsetX; x <= mWidth+offsetX; x+=10) {
        if((x-offsetX) % 100 == 0){
            p->drawLine(x,lY,x,offsetY);
        }else{
            p->drawLine(x,sY,x,offsetY);
        }
    }

    sY = offsetX-(GRID_HEIGHT/2);
    lY = offsetX-GRID_HEIGHT;

    for (int y = offsetY; y <= mHeight+offsetY; y+=10) {
        if((y-offsetY) % 100 == 0){
            p->drawLine(lY,y,offsetX,y);
        }else{
            p->drawLine(sY,y,offsetX,y);
        }
    }
}

inline void MeshRenderer::drawColorLegend(QPainter *p)
{
    //0%-100%
    int off = (mImgFlow->width()-COLORLEGEND_WIDTH-COLORLEGEND_OFFSET);
    p->setPen(getDrawColor(0.));
    p->drawLine(off,0,off,TEXT_HEIGHT);
    for (int i = 1; i <= COLORLEGEND_WIDTH; i++) {
        p->setPen(getDrawColor((float)i/(float)COLORLEGEND_WIDTH));
        p->drawLine(off+i,0,off+i,TEXT_HEIGHT);
    }

    p->setPen(Qt::black);

    if(mSettings->getRenderDisplay() == Settings::VXYC){
        p->drawText(off-COLORLEGEND_WIDTH/4,TEXT_HEIGHT,COLORLEGEND_WIDTH/2,TEXT_HEIGHT,Qt::AlignCenter,QString("%1 m/s").arg(mMinV*mSettings->getSpeedOfSound()));
        p->drawText(off+COLORLEGEND_WIDTH/4,TEXT_HEIGHT,COLORLEGEND_WIDTH/2,TEXT_HEIGHT,Qt::AlignCenter,QString("%1 m/s").arg((mMinV+(mMaxV-mMinV)/2.)*mSettings->getSpeedOfSound()));
        p->drawText(off+COLORLEGEND_WIDTH-COLORLEGEND_WIDTH/4,TEXT_HEIGHT,COLORLEGEND_WIDTH/2,TEXT_HEIGHT,Qt::AlignCenter,QString("%1 m/s").arg(mMaxV*mSettings->getSpeedOfSound()));
    }else if(mSettings->getRenderDisplay() == Settings::RHOC){
        p->drawText(off-COLORLEGEND_WIDTH/4,TEXT_HEIGHT,COLORLEGEND_WIDTH/2,TEXT_HEIGHT,Qt::AlignCenter,QString("%1 kg/m2").arg(mMinRho*mSettings->getDensity()));
        p->drawText(off+COLORLEGEND_WIDTH/4,TEXT_HEIGHT,COLORLEGEND_WIDTH/2,TEXT_HEIGHT,Qt::AlignCenter,QString("%1 kg/m2").arg((mMinRho+(mMaxRho-mMinRho)/2.)*mSettings->getDensity()));
        p->drawText(off+COLORLEGEND_WIDTH-COLORLEGEND_WIDTH/4,TEXT_HEIGHT,COLORLEGEND_WIDTH/2,TEXT_HEIGHT,Qt::AlignCenter,QString("%1 kg/m2").arg(mMaxRho*mSettings->getDensity()));
    }else{
        p->drawText(off-COLORLEGEND_WIDTH/4,TEXT_HEIGHT,COLORLEGEND_WIDTH/2,TEXT_HEIGHT,Qt::AlignCenter,QString("%1 MPa").arg(mMinRho*mSettings->getDensity()*mSettings->getSpeedOfSound2()/1000000.));
        p->drawText(off+COLORLEGEND_WIDTH/4,TEXT_HEIGHT,COLORLEGEND_WIDTH/2,TEXT_HEIGHT,Qt::AlignCenter,QString("%1 MPa").arg((mMinRho+(mMaxRho-mMinRho)/2.)*mSettings->getDensity()*mSettings->getSpeedOfSound2()/1000000.));
        p->drawText(off+COLORLEGEND_WIDTH-COLORLEGEND_WIDTH/4,TEXT_HEIGHT,COLORLEGEND_WIDTH/2,TEXT_HEIGHT,Qt::AlignCenter,QString("%1 MPa").arg(mMaxRho*mSettings->getDensity()*mSettings->getSpeedOfSound2()/1000000.));
    }
}

inline void MeshRenderer::drawVectors(QPainter *p, int vectLength, int offsetX, int offsetY)
{
    int idx;
    float step = mMaxV/vectLength;

    p->setPen(Qt::white);

    if(step != 0.0){
        for (int x = offsetX+vectLength/2; x < mWidth+offsetX; x+=vectLength) {
            for (int y = offsetY+vectLength/2; y <= mHeight+offsetY; y+=vectLength) {
                idx = mMesh->getVisibleCellIdx(Point(x-offsetX,y-offsetY));

                if(idx>=0){
                    p->drawLine( x, y, x+mVxc[idx]/step, y+mVyc[idx]/step);
                }
            }
        }
    }
}

bool MeshRenderer::drawIsSettingsChanged()
{
    return (mSettings->getRenderDisplay() != mCalculatedValue ||
            mSettings->getRenderDisplayScale() != mCalculatedScale ||
            mSettings->getRenderDisplayScaleMin() != mCalculatedValueMin ||
            mSettings->getRenderDisplayScaleMax() != mCalculatedValueMax);
}
