#include "boundaryconditionlistwidget.h"
#include <QMessageBox>

BoundaryConditionListWidget::BoundaryConditionListWidget(QWidget *parent) :
    QWidget(parent)
{
    mBcAddBtn = new QPushButton(QString::fromUtf8("Új hozzáadása"));
    mBcDelBtn = new QPushButton(QString::fromUtf8("Eltávolít"));
    mBcDeleteIndex = new QComboBox();
    mBcDeleteIndex->setMinimumWidth(130);
    mScrollAreaWidget = new QWidget();

    QScrollArea *scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(1);
    scrollArea->setBackgroundRole(QPalette::Light);
    scrollArea->setWidget(mScrollAreaWidget);

    QHBoxLayout* hbox = new QHBoxLayout();
    hbox->addWidget(mBcDeleteIndex,0);
    hbox->addWidget(mBcDelBtn,0);
    hbox->addSpacerItem(new QSpacerItem(20,20,QSizePolicy::Expanding,QSizePolicy::Minimum));
    hbox->addWidget(mBcAddBtn,0);

    QVBoxLayout* vbox = new QVBoxLayout();
    vbox->addWidget(scrollArea);
    vbox->addLayout(hbox);

    this->setLayout(vbox);
    this->setMinimumWidth(500);

    connect(mBcAddBtn,SIGNAL(clicked()),this,SLOT(addBoundaryCondition()));
    connect(mBcDelBtn,SIGNAL(clicked()),this,SLOT(deleteBoundaryCondition()));
    createScrollAreaLayout();
}

void BoundaryConditionListWidget::createScrollAreaLayout()
{
    delete mScrollAreaWidget->layout();

    QVBoxLayout* hBox = new QVBoxLayout();

    int idx = mBcDeleteIndex->currentIndex();
    if(idx < 0){
        idx = 0;
    }
    mBcDeleteIndex->clear();

    mBcAddBtn->setEnabled(mBCWidgetList.count()<BoundaryConditionList::maxCount());

    mBcDeleteIndex->setEnabled(mBCWidgetList.count()>0);
    mBcDelBtn->setEnabled(mBCWidgetList.count()>0);


    foreach (BoundaryConditionWidget* bcw, mBCWidgetList) {
        hBox->addWidget(bcw);

        mBcDeleteIndex->addItem(QIcon(QPixmap::fromImage(bcw->getIcon())),QString::fromUtf8("peremfeltétel"),bcw->getID());
    }

    if(idx<mBcDeleteIndex->count()){
        mBcDeleteIndex->setCurrentIndex(idx);
    }else{
        mBcDeleteIndex->setCurrentIndex(mBcDeleteIndex->count()-1);
    }

    hBox->addSpacerItem(new QSpacerItem(0,10,QSizePolicy::Expanding,QSizePolicy::Expanding));

    hBox->setSpacing(0);
    hBox->setMargin(0);
    mScrollAreaWidget->setLayout(hBox);
}

void BoundaryConditionListWidget::addBoundaryCondition()
{
    if(mBCList.count() < BoundaryConditionList::maxCount())
    {
        BoundaryCondition* bc = mBCList.createNew();
        emit newColorAvailable(bc->getColor());

        BoundaryConditionWidget* bcW = new BoundaryConditionWidget(bc,mScrollAreaWidget);
        mBCWidgetList.append(bcW);
        createScrollAreaLayout();

        bcW->setP(mSettings->getPressure()/1000000.);
    }else{
        QMessageBox::warning(this,QString::fromUtf8("Peremfeltétel limit"),QString::fromUtf8("Nem lehet új peremfeltetélt hozzáadni!"));
    }
}

void BoundaryConditionListWidget::deleteBoundaryCondition()
{
    if(mBCList.count()>0){
        int id =mBcDeleteIndex->itemData(mBcDeleteIndex->currentIndex()).toInt();

        bool found=false;
        int i=0;

        while(i<mBCList.count() && !found){
            if(mBCList.at(i)->getID() == id){

                emit colorDeleted(mBCList.at(i)->getColor());

                delete mBCWidgetList[i];//delete object
                mBCWidgetList.removeAt(i);//remove from the list

                mBCList.deleteAt(i);
                found = true;
            }else{
                i++;
            }
        }

        if(found){
            createScrollAreaLayout();
        }
    }
}
