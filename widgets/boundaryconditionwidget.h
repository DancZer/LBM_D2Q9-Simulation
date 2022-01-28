#ifndef BOUNDARYCONDITIONWIDGET_H
#define BOUNDARYCONDITIONWIDGET_H

#include <QWidget>
#include <QColor>
#include <QHBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <QDoubleSpinBox>

#include "classes/boundarycondition.h"

class BoundaryConditionWidget : public QWidget
{
    Q_OBJECT
public:
    explicit BoundaryConditionWidget(BoundaryCondition *bc, QWidget *parent = 0);

    void setP(double p);
    void setVx(double vx);
    void setVy(double vy);

    int getID(){
        return mBc->getID();
    }

    QImage getIcon(){
        return mImg;
    }

private slots:
    void updateP(double p);
    void updateVx(double vx);
    void updateVy(double vy);

private:
    BoundaryCondition *mBc;

    QImage mImg;
    QDoubleSpinBox *mSpinBoxP;
    QDoubleSpinBox *mSpinBoxVx;
    QDoubleSpinBox *mSpinBoxVy;

};

#endif // BOUNDARYCONDITIONWIDGET_H
