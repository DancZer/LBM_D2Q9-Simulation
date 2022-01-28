#include "colorwidget.h"
#include <QPaintEvent>
#include "classes/lbmmesh.h"

#include <QPainter>
#include <QDebug>

ColorWidget::ColorWidget(QWidget *parent) :
    QWidget(parent)
{
    mSelectedColorIndex = 0;

    //fluid
    mAvailableColors.append(QColor(Qt::black));

    this->setMinimumHeight(ICONSIZE*2);

    mOffsetY = ICONSIZE/2;

    emit colorChanged(mAvailableColors[mSelectedColorIndex]);
}

void ColorWidget::paintEvent(QPaintEvent *)
{
    QPainter p(this);

    p.setPen(QColor(96,96,96));
    p.drawRect(0,0,this->width()-1,this->height()-1);

    mOffsetY = (this->height()/2)-ICONSIZE/2;

    for(int i=0;i<mAvailableColors.count();i++)
    {
        QColor c = mAvailableColors[i];

        if(i==mSelectedColorIndex)
        {
            p.fillRect(ICONSIZE/2+(SELECTED_ICONSIZE-ICONSIZE)/2+i*(ICONSIZE+SELECTED_ICONSIZE/2)-(SELECTED_ICONSIZE-ICONSIZE)/2 + SELECTED_ICONSIZE/5, mOffsetY-(SELECTED_ICONSIZE-ICONSIZE)/3 + SELECTED_ICONSIZE/5, SELECTED_ICONSIZE,SELECTED_ICONSIZE,QColor(255,215,0));
            p.fillRect(ICONSIZE/2+(SELECTED_ICONSIZE-ICONSIZE)/2+i*(ICONSIZE+SELECTED_ICONSIZE/2)-(SELECTED_ICONSIZE-ICONSIZE)/2 + SELECTED_ICONSIZE/5+1, mOffsetY-(SELECTED_ICONSIZE-ICONSIZE)/3 + SELECTED_ICONSIZE/5+1, SELECTED_ICONSIZE-2,SELECTED_ICONSIZE-2,QColor(255,255,255));
            p.fillRect(ICONSIZE/2+(SELECTED_ICONSIZE-ICONSIZE)/2+i*(ICONSIZE+SELECTED_ICONSIZE/2)-(SELECTED_ICONSIZE-ICONSIZE)/2, mOffsetY-(SELECTED_ICONSIZE-ICONSIZE)/2, SELECTED_ICONSIZE,SELECTED_ICONSIZE,QColor(255,215,0));
        }

        p.fillRect(ICONSIZE/2+(SELECTED_ICONSIZE-ICONSIZE)/2+ i*(ICONSIZE+SELECTED_ICONSIZE/2),mOffsetY,ICONSIZE,ICONSIZE,c);
    }

}

void ColorWidget::addColor(QColor color)
{
    mAvailableColors.append(color);

    this->setMinimumWidth(ICONSIZE/2+(SELECTED_ICONSIZE-ICONSIZE)/2+mAvailableColors.count()*(ICONSIZE+SELECTED_ICONSIZE/2)-(SELECTED_ICONSIZE-ICONSIZE)/2+1);
    this->repaint();
}

void ColorWidget::removeColor(QColor color)
{
    mAvailableColors.removeAt(mAvailableColors.indexOf(color));
    mSelectedColorIndex = mAvailableColors.count()-1;
    emit colorChanged(mAvailableColors[mSelectedColorIndex]);

    this->repaint();
}

void ColorWidget::mousePressEvent(QMouseEvent *e)
{
    int mouseX = e->x();
    int mouseY = e->y();

    int cX, cY, selectedColor=-1;

    cY = mOffsetY;


    for(int i=0;i<mAvailableColors.count();i++)
    {
        cX=ICONSIZE/2+(SELECTED_ICONSIZE-ICONSIZE)/2+ i*(ICONSIZE+SELECTED_ICONSIZE/2);

        if(mouseX>cX && mouseY>cY && mouseX<cX+ICONSIZE && mouseY<cY+ICONSIZE)
        {
            selectedColor = i;
            break;
        }
    }

    if(selectedColor>=0 && selectedColor != mSelectedColorIndex)
    {
        mSelectedColorIndex = selectedColor;
        repaint();
        emit colorChanged(mAvailableColors[mSelectedColorIndex]);
    }
}

QColor ColorWidget::getSelectedColor()
{
    return mAvailableColors[mSelectedColorIndex];
}

QList<QColor> ColorWidget::getColors()
{
    return mAvailableColors;
}
