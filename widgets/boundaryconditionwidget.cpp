#include "boundaryconditionwidget.h"
#include "classes/lbmmesh.h"
#include <QDebug>

BoundaryConditionWidget::BoundaryConditionWidget( BoundaryCondition *bc, QWidget *parent) :
    QWidget(parent),
    mBc(bc)
{
    QHBoxLayout* lay = new QHBoxLayout();

    QLabel *colorLabel = new QLabel();
    mSpinBoxP = new QDoubleSpinBox();
    mSpinBoxVx = new QDoubleSpinBox();
    mSpinBoxVy = new QDoubleSpinBox();

    mSpinBoxP->setDecimals(8);
    mSpinBoxP->setMaximum(1000000);
    mSpinBoxP->setMinimum(0);
    mSpinBoxP->setPrefix("P: ");
    mSpinBoxP->setSuffix(" MPa");
    mSpinBoxP->setAlignment(Qt::AlignRight);

    mSpinBoxVx->setDecimals(8);
    mSpinBoxVx->setMaximum(1000000);
    mSpinBoxVx->setMinimum(-1000000);
    mSpinBoxVx->setPrefix("vX: ");
    mSpinBoxVx->setSuffix(" m/s");
    mSpinBoxVx->setAlignment(Qt::AlignRight);

    mSpinBoxVy->setDecimals(8);
    mSpinBoxVy->setMaximum(1000000);
    mSpinBoxVy->setMinimum(-1000000);
    mSpinBoxVy->setPrefix("vY: ");
    mSpinBoxVy->setSuffix(" m/s");
    mSpinBoxVy->setAlignment(Qt::AlignRight);

    mImg = QImage(15,15,QImage::Format_RGB32);
    mImg.fill(bc->getColor().rgb());
    colorLabel->setToolTip(QString("RGB(%1,%2,%3)").arg(bc->getColor().red()).arg(bc->getColor().green()).arg(bc->getColor().blue()));

    colorLabel->setPixmap(QPixmap::fromImage(mImg));

    lay->addWidget(colorLabel);
    lay->addWidget(mSpinBoxP,1);
    lay->addWidget(mSpinBoxVx,1);
    lay->addWidget(mSpinBoxVy,1);

    lay->setMargin(3);
    this->setLayout(lay);

    connect(mSpinBoxP, SIGNAL(valueChanged(double)),this,SLOT(updateP(double)));
    connect(mSpinBoxVx,SIGNAL(valueChanged(double)),this,SLOT(updateVx(double)));
    connect(mSpinBoxVy,SIGNAL(valueChanged(double)),this,SLOT(updateVy(double)));

}

void BoundaryConditionWidget::setP(double p)
{
    mSpinBoxP->setValue(p);
}

void BoundaryConditionWidget::setVx(double vx)
{
    mSpinBoxVx->setValue(vx);
}

void BoundaryConditionWidget::setVy(double vy)
{
    mSpinBoxVy->setValue(vy);
}

void BoundaryConditionWidget::updateP(double p)
{
    mBc->setP(p);
}

void BoundaryConditionWidget::updateVx(double vx)
{
    mBc->setVx(vx);
}

void BoundaryConditionWidget::updateVy(double vy)
{
    mBc->setVy(vy);
}
