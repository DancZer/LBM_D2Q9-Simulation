#include "colorssample.h"
#include <QPaintEvent>
#include <QPainter>

ColorsSample::ColorsSample(QWidget *parent) :
    QWidget(parent)
{
}

void ColorsSample::paintEvent(QPaintEvent *)
{
    int x=0,y=0;
    QPainter p(this);


    x = SAMPLE_SIZE/2;
    y = this->height()/2-SAMPLE_SIZE/2;

    p.drawText(x,this->height()/2-TEXT_HEIGHT/2,TEXT_HEIGHT,TEXT_HEIGHT+TEXT_HEIGHT/2,Qt::AlignLeft & Qt::AlignVCenter,"Min");
    x += TEXT_HEIGHT+TEXT_HEIGHT/2;
    foreach (QColor c, mColors) {
        p.fillRect(x,y,SAMPLE_SIZE,SAMPLE_SIZE,c);
        x += SAMPLE_SIZE + SAMPLE_SIZE/2;
    }

    p.drawText(x,this->height()/2-TEXT_HEIGHT/2,TEXT_HEIGHT*2,TEXT_HEIGHT,Qt::AlignLeft & Qt::AlignVCenter,"Max");

}
