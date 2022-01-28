#ifndef BOUNDARYCONDITION_H
#define BOUNDARYCONDITION_H

#include <QColor>

class BoundaryCondition
{
public:
    BoundaryCondition(int id):
        mID(id){}

    int getID(){return mID;}

    float getP(){return mP;}

    double getRho(double pressure){return (mP*1000000.)/pressure;}
    double getVx(double spdOfSound=1.0f){return mVx/spdOfSound;}
    double getVy(double spdOfSound=1.0f){return mVy/spdOfSound;}

    QColor getColor(){return mColor;}
    void setColor(QColor color){mColor = color;}

    void setP(double p){mP = p;}
    void setVx(double vx){mVx = vx;}
    void setVy(double vy){mVy = vy;}

private:
    int mID;

    QColor mColor;

    double mP; //MPa
    double mVx; //m/s
    double mVy; //m/s
};

#endif // BOUNDARYCONDITION_H
