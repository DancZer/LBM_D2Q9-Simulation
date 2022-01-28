#ifndef LBMMESH_H
#define LBMMESH_H

#include <QImage>
#include <QHash>
#include <QDebug>

#include "classes/point.h"
#include "classes/boundaryconditionlist.h"
#include "classes/lbmparticle.h"

class LBMMesh : public QObject
{
    Q_OBJECT
public:
    LBMMesh(QImage *mask, BoundaryConditionList *bcList, QObject *parent=0);

    inline Point getVisibleCellPos(int index){
        return mVisibleCellsPos.value(index);
    }
    inline int getVisibleCellIdx(Point p){
        return mVisibleCells.value(p,-1);
    }

    Point getCellPos(int index);

    void getNeighbours(int index, int *neighbours);

    int getVisibleCellCount(){return mVisibleCells.count();}
    int getBoundaryCellCount(){return mBoundaryCells.count();}

    int getCellCount(){
        return mVisibleCells.count()+mBoundaryCells.count();
    }

    int getBoundaryCellTagId(int index){  return mBoundaryCellTagIDs[index-mVisibleCells.count()];}

private:
    QHash<Point,int> mVisibleCells;
    QHash<int,Point> mVisibleCellsPos;

    QHash<Point,int> mBoundaryCells;
    QHash<int,Point> mBoundaryCellsPos;
    QList<int> mBoundaryCellTagIDs;

    QColor mFluidColor;
    QColor mEmptyColor;

    int getNeighbourIdx(int index,int oX, int oY);
    void generateMeshGrid(QImage *maskImg, BoundaryConditionList *bcList);
};

#endif // LBMMESH_H
