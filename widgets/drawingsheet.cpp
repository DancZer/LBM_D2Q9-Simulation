#include "drawingsheet.h"
#include <QPainter>
#include <QMouseEvent>
#include <QDebug>

#include "classes/point.h"

DrawingSheet::DrawingSheet(QWidget *parent) :
    QWidget(parent)
{
    mScale = 1;
    mEnableFillTool = false;

    //halo szine
    mGridPen.setColor(QColor(20,20,20));
    mGridPen.setWidth(1);

    mColorDraw = QColor(Qt::black);
    mColorEmpty = QColor(Qt::white);

    mCurrPosX = 0;
    mCurrPosY = 0;
    mColorLine = mColorEmpty;

    mEnableFillTool = false;
    mEnableLineTool = false;
    mLineDraging = false;
}

void DrawingSheet::newDrawing(int width, int height)
{
    mImgMask = QImage(width,height,QImage::Format_ARGB32);
    clearImage();
    backupMask();

    this->setMinimumWidth(mImgMask.width());
    this->setMinimumHeight(mImgMask.height());

    drawTestRectangle();
}


void DrawingSheet::setMask(QImage img)
{
    mImgMask = img;
    backupMask();

    this->setMinimumWidth(mImgMask.width());
    this->setMinimumHeight(mImgMask.height());

    this->repaint();
}

void DrawingSheet::drawTestRectangle()
{
    for (int x = 0; x < mImgMask.width(); x++) {
        for (int y = 0; y < mImgMask.height(); y++) {
            if(x==0)
            {
                drawPixel(x,y,QColor(Qt::red));
            }
            else if(x==mImgMask.width()-1)
            {
                drawPixel(x,y,QColor(Qt::green));
            }
            else
            {
                drawPixel(x,y,mColorDraw);
            }
        }
    }
}

void DrawingSheet::paintEvent(QPaintEvent *)
{
    QPainter p(this);

    p.fillRect(0,0,this->width(),this->height(),Qt::gray);
    p.drawImage(QRectF(0.0, 0.0, mImgMask.width()*mScale, mImgMask.height()*mScale),mImgMask);

    if(mEnableLineTool && mLineDraging){
        p.setPen(QPen(QBrush(mColorLine),mScale));
        p.drawLine((mScale/2)+mPrevPosX*mScale,(mScale/2)+mPrevPosY*mScale,(mScale/2)+mCurrPosX*mScale,(mScale/2)+mCurrPosY*mScale);
    }

    if(mScale>1){
        drawGrid(&p);
    }
}

void DrawingSheet::mousePressEvent(QMouseEvent *e)
{
    backupMask();

    int x = e->x()/mScale;
    int y = e->y()/mScale;

    if(mEnableFillTool){

        if(e->buttons() & Qt::LeftButton){
            fillArea(x,y,QColor(mImgMask.pixel(x,y)),mColorDraw);
        }else if(e->buttons() & Qt::RightButton){
            fillArea(x,y,QColor(mImgMask.pixel(x,y)),mColorEmpty);
        }
    }else if(mEnableLineTool){

        if(e->buttons() & Qt::LeftButton){
            mColorLine = mColorDraw;
            mPrevPosX = x;
            mPrevPosY = y;
            mCurrPosX = x;
            mCurrPosY = y;
        }else if(e->buttons() & Qt::RightButton){
            mColorLine = mColorEmpty;
            mPrevPosX = x;
            mPrevPosY = y;
            mCurrPosX = x;
            mCurrPosY = y;
        }else if(e->buttons() & Qt::MiddleButton){
            mPrevPosX = mCurrPosX;
            mPrevPosY = mCurrPosY;
            mCurrPosX = x;
            mCurrPosY = y;
        }

        mLineDraging = true;
        this->repaint();

    }else{
        mPrevPosX = x;
        mPrevPosY = y;

        if(e->buttons() & Qt::LeftButton){
            drawLine(mPrevPosX,mPrevPosY,mPrevPosX,mPrevPosY,mColorDraw);
        }else if(e->buttons() & Qt::RightButton){
            drawLine(mPrevPosX,mPrevPosY,mPrevPosX,mPrevPosY,mColorEmpty);
        }
    }
}

void DrawingSheet::mouseMoveEvent(QMouseEvent *e)
{
    int x = e->x()/mScale;
    int y = e->y()/mScale;

    if(!mEnableFillTool && !mEnableLineTool){
        if(e->buttons() & Qt::LeftButton){
            drawLine(mPrevPosX,mPrevPosY,x,y,mColorDraw);
        }else if(e->buttons() & Qt::RightButton){
            drawLine(mPrevPosX,mPrevPosY,x,y,mColorEmpty);
        }

        mPrevPosX = x;
        mPrevPosY = y;
    }

    if(mEnableLineTool){
        mCurrPosX = x;
        mCurrPosY = y;
        mLineDraging = true;
        this->repaint();
    }

}

void DrawingSheet::mouseReleaseEvent(QMouseEvent *)
{
    if(mEnableLineTool){
        drawLine(mPrevPosX,mPrevPosY,mCurrPosX,mCurrPosY,mColorLine);
        mLineDraging = false;
        this->repaint();
    }
}

void DrawingSheet::clearImage()
{
    QPainter p(&mImgMask);

    p.fillRect(0,0,mImgMask.width(),mImgMask.height(),mColorEmpty);

    this->repaint();
}

void DrawingSheet::setScale(int scale)
{
    mScale = scale;

    this->setMinimumWidth(mImgMask.width()*mScale);
    this->setMinimumHeight(mImgMask.height()*mScale);

    this->repaint();
}

void DrawingSheet::drawLine(int x1, int y1, int x2, int y2, QColor color)
{
    QPainter p(&mImgMask);
    p.setPen(QPen(QBrush(color),1.0));
    p.drawLine(x1,y1,x2,y2);

    this->repaint();
}

void DrawingSheet::drawPixel(int posX,int posY,QColor color)
{
    QPainter p(&mImgMask);

    p.fillRect(posX,posY,1,1,color);
}


void DrawingSheet::drawGrid(QPainter *p)
{
    QPen old = p->pen();

    p->setPen(mGridPen);

    for(int pos=0;pos<=qMax(mImgMask.width()*mScale,mImgMask.height()*mScale);pos+=mScale)
    {
        //x irnyba
        if(pos<=mImgMask.height()*mScale)
        {
            p->drawLine(0,pos,mImgMask.width()*mScale,pos);
        }

        //y irnyba
        if(pos<=mImgMask.width()*mScale)
        {
            p->drawLine(pos,0,pos,mImgMask.height()*mScale);
        }
    }

    p->setPen(old);
}


void DrawingSheet::fillArea(int x, int y, QColor targetColor, QColor newColor)
{
    QList<Point> list;

    if(x>=0 && y>=0 && x<mImgMask.width() && y<mImgMask.height())
    {
        QColor c = QColor(mImgMask.pixel(x,y));

        if(c != targetColor || c == newColor)
            return;

        int pX,pY;
        int wX,wY;
        int eX,eY;

        list.append(Point(x,y));

        for (int i=0;i<list.count();i++) { 

            pX = wX = eX = list[i].x();
            pY = wY = eY = list[i].y();

            //átlók
          /*  if(pX>0){
                if(pY>0 && QColor(mImgMask.pixel(pX-1,pY-1)) == targetColor){
                    list.append(Point(pX-1,pY-1));
                }

                if(pY<mImgMask.height()-1 && QColor(mImgMask.pixel(pX-1,pY+1)) == targetColor){
                    list.append(Point(pX-1,pY+1));
                }
            }else if(pX<mImgMask.width()-1){
                if(pY>0 && QColor(mImgMask.pixel(pX+1,pY-1)) == targetColor){
                    list.append(Point(pX+1,pY-1));
                }

                if(pY<mImgMask.height()-1 && QColor(mImgMask.pixel(pX+1,pY+1)) == targetColor){
                    list.append(Point(pX+1,pY+1));
                }
            }*/
            //

            //WEST
            wX--; //skip current
            while(wX>=0 && QColor(mImgMask.pixel(wX,wY)) == targetColor){
                drawPixel(wX,wY,newColor);
                wX--;
            }
            wX++;

            //EAST
            while(eX<mImgMask.width() && QColor(mImgMask.pixel(eX,eY)) == targetColor){
                drawPixel(eX,eY,newColor);
                eX++;
            }
            eX--;

            //checkt pixels the above the painted row
            pX = wX;
            pY = wY-1;

            if(pY>=0){
                while(pX<=eX){
                    if(QColor(mImgMask.pixel(pX,pY)) == targetColor){
                        list.append(Point(pX,pY));
                    }
                    pX++;
                }
            }

            //checkt pixels the below the painted row
            pX = wX;
            pY = wY+1;

            if(pY<mImgMask.height()){
                while(pX<=eX){
                    if(QColor(mImgMask.pixel(pX,pY)) == targetColor){
                        list.append(Point(pX,pY));
                    }
                    pX++;
                }
            }
        }
    }

    this->repaint();
}

void DrawingSheet::enableLine(bool enable)
{
    mEnableLineTool = enable;
    if(enable){
        mEnableFillTool = !enable;
    }
}

void DrawingSheet::enableFill(bool enable)
{
    mEnableFillTool = enable;
    if(enable){
        mEnableLineTool = !enable;
    }
}

void DrawingSheet::backupMask()
{
    mImgMaskBackup = QImage(mImgMask);

    emit undoAvailable(true);
}

void DrawingSheet::undo()
{
    mImgMask = QImage(mImgMaskBackup);
    this->repaint();

    emit undoAvailable(false);
}
