#ifndef BPUNDARYCONDITIONLIST_H
#define BPUNDARYCONDITIONLIST_H

#include <QList>
#include "lbmparticle.h"
#include "boundarycondition.h"

class BoundaryConditionList : public QList<BoundaryCondition*>
{
public:
    const static int GLOBALCOLOR_MIN = 7;
    const static int GLOBALCOLOR_MAX = 18;

    explicit BoundaryConditionList():
        QList<BoundaryCondition*>(),
        mID(1){}

    BoundaryCondition* createNew(){
        QColor newC;
        bool found=false;

        for (int c = GLOBALCOLOR_MIN; c <= GLOBALCOLOR_MAX; c++) {
            found = false;

            QColor col((Qt::GlobalColor)c);

            foreach (BoundaryCondition *bc, *this) {
                if(bc->getColor() == col)
                {
                    found = true;
                    break;
                }
            }

            if(!found){
                newC = col;
                break;
            }
        }


        if(!found){
            BoundaryCondition *bc = new BoundaryCondition(mID++);
            this->append(bc);
            bc->setColor(newC);
            bc->setP(1.0);
            bc->setVx(0.0);
            bc->setVy(0.0);          

            return bc;
        }else{
            return NULL;
        }
    }

    void deleteAt(int index){
        delete (*this).at(index);
        this->removeAt(index);
    }

    void deleteByID(int id){
        int i=0;
        foreach (BoundaryCondition* bc, *this) {
            if(bc->getID() == id)
            {
                deleteAt(i);
                break;
            }else{
                i++;
            }
        }
    }

    BoundaryCondition* getByID(int id){
        foreach (BoundaryCondition* bc, *this) {
            if(bc->getID() == id)
            {
                return bc;
            }
        }

        return 0;
    }

    BoundaryCondition* getByColor(QColor color){
        foreach (BoundaryCondition* bc, *this) {
            if(bc->getColor() == color)
            {
                return bc;
            }
        }

        return 0;
    }

    static int maxCount(){
        return GLOBALCOLOR_MAX-GLOBALCOLOR_MIN;
    }

private:
    int mID;

};

#endif // BPUNDARYCONDITIONLIST_H
