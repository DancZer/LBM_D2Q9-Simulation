#ifndef POINT_H
#define POINT_H

#include <QHash>

class Point
{
public:
    Point(){
        mX = 0;
        mY = 0;
    }

    Point(int x, int y){
        mX = x;
        mY = y;
    }

    int x()const{return mX;}
    int x(int offset)const{return mX+offset;}

    int y()const{return mY;}
    int y(int offset)const{return mY+offset;}

    void setX(int offset){mX+=offset;}
    void setY(int offset){mY+=offset;}

    Point offset(int x, int y)
    {
        return Point(mX+x,mY+y);
    }

    bool operator < (const Point &point) const
    {
        return mX < point.x() && mY < point.y();
    }

    bool operator==(const Point &point) const
    {
        if(this == &point)
            return true;
        return mX == point.x() && mY == point.y();
    }

private:
    int mX;
    int mY;

};


inline uint qHash(const Point& p)
{
    return qHash(p.x()+p.y());
}


#endif // POINT_H
