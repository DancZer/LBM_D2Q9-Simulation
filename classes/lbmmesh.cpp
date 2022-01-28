#include "lbmmesh.h"
#include <QColor>
#include <QPainter>

#include <QDebug>
#include "classes/logger.h"

LBMMesh::LBMMesh(QImage *mask, BoundaryConditionList *bcList, QObject *parent):
    QObject(parent)
{
    mFluidColor = QColor(Qt::black);
    mEmptyColor = QColor(Qt::white);

    generateMeshGrid(mask,bcList);
}

void LBMMesh::generateMeshGrid(QImage *maskImg, BoundaryConditionList *bcList)
{
    QColor pixelColor;
    BoundaryCondition* ghostCell;

    int visibleCount=0;
    int bcCount=0;

    for (int x = 0; x < maskImg->width(); ++x)
    {
        for (int y = 0; y < maskImg->height(); ++y)
        {
            pixelColor = QColor(maskImg->pixel(x,y));

            if(pixelColor != mEmptyColor)
            {
                if(pixelColor == mFluidColor)
                {
                    mVisibleCells[Point(x,y)]=visibleCount;
                    mVisibleCellsPos[visibleCount]=Point(x,y);
                    visibleCount++;
                }
                else
                {
                    ghostCell = bcList->getByColor(pixelColor);

                    if(ghostCell != 0){
                        mBoundaryCells[Point(x,y)]=bcCount;
                        mBoundaryCellsPos[bcCount]=Point(x,y);
                        mBoundaryCellTagIDs.append(ghostCell->getID());
                        bcCount++;
                    }
                }
            }
        }
    }

    Logger::instance()->appendLog(QString("Cella információk: látható cellák száma: %1, külső hatású cellák száma: %2.").arg(mVisibleCells.count()).arg(mBoundaryCells.count()));
}

void LBMMesh::getNeighbours(int index,int *neighbours)
{
    neighbours[0] = index;

    //a QImage bal felso sarkaban van az origo
    neighbours[1] = getNeighbourIdx(index,1,0);
    neighbours[2] = getNeighbourIdx(index,0,-1);
    neighbours[3] = getNeighbourIdx(index,-1,0);
    neighbours[4] = getNeighbourIdx(index,0,1);

    neighbours[5] = getNeighbourIdx(index,1,-1);
    neighbours[6] = getNeighbourIdx(index,-1,-1);
    neighbours[7] = getNeighbourIdx(index,-1,1);
    neighbours[8] = getNeighbourIdx(index,1,1);
 }

int LBMMesh::getNeighbourIdx(int index,int oX, int oY)
{
    int neighIdx, relIdx;
    Point pos;

    relIdx = index;

    if(relIdx<mVisibleCells.count()){
        pos = mVisibleCellsPos.value(relIdx);
    }else{
        relIdx -= mVisibleCells.count();

        if(relIdx<mBoundaryCells.count()){
            pos = mBoundaryCellsPos.value(relIdx);
        }else{
            return index;
        }
    }

    pos = pos.offset(oX,oY);

    //megkeresi a pozcit
    neighIdx = mVisibleCells.value(pos,-1);

    if(neighIdx<0){
        neighIdx = mBoundaryCells.value(pos,-1);

        if(neighIdx>=0){
            neighIdx += mVisibleCells.count();
        }
    }

    return neighIdx>=0 ? neighIdx : index;
}

Point LBMMesh::getCellPos(int index)
{
    int relIdx;
    Point pos;
    pos.setX(-1);
    pos.setY(-1);

    relIdx = index;

    if(relIdx<mVisibleCells.count()){
        pos = mVisibleCellsPos.value(relIdx);
    }else{
        relIdx -= mVisibleCells.count();

        if(relIdx<mBoundaryCells.count()){
            pos = mBoundaryCellsPos.value(relIdx);
        }
    }

    return pos;
}
