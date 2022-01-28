#ifndef BOUNDARYCONDITIONLISTWIDGET_H
#define BOUNDARYCONDITIONLISTWIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QScrollArea>
#include <QSpacerItem>
#include <QComboBox>
#include "classes/boundaryconditionlist.h"
#include "widgets/boundaryconditionwidget.h"
#include "classes/settings.h"

class BoundaryConditionListWidget : public QWidget
{
    Q_OBJECT
public:
    explicit BoundaryConditionListWidget(QWidget *parent = 0);

    BoundaryConditionList* getBCList(){return &mBCList;}

    void setSettings(Settings* settings){
        mSettings = settings;
    }

signals:
    void newColorAvailable(QColor c);
    void colorDeleted(QColor c);

public slots:
    void addBoundaryCondition();
    void deleteBoundaryCondition();

private:
    QList<BoundaryConditionWidget*> mBCWidgetList;
    BoundaryConditionList mBCList;
    Settings* mSettings;

    QWidget* mScrollAreaWidget;

    QPushButton* mBcAddBtn;
    QPushButton* mBcDelBtn;
    QComboBox* mBcDeleteIndex;

    void createScrollAreaLayout();
};

#endif // BOUNDARYCONDITIONLISTWIDGET_H
