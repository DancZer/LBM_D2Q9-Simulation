#include "editorwidget.h"

#include <QPushButton>
#include <QApplication>

EditorWidget::EditorWidget(QWidget *parent) :
    QWidget(parent)
{
    createLayout();
}

void EditorWidget::createLayout()
{
    QVBoxLayout *layMain = new QVBoxLayout();

    QHBoxLayout *layTools = new QHBoxLayout();
    mWidgetColors = new ColorWidget(this);
    ZoomWidget *widgetZoom = new ZoomWidget(this);
    mComboBox = new QComboBox(this);
    mComboBox->addItem(QIcon(":/tools/pencil"),QString::fromUtf8("Ceruza"));
    mComboBox->addItem(QIcon(":/tools/line"),QString::fromUtf8("Vonal"));
    mComboBox->addItem(QIcon(":/tools/fill"),QString::fromUtf8("Kiöntés"));

    mPBUndo = new QPushButton(QString::fromUtf8("Vissza"));

    mPBUndo->setIcon(QIcon(":/icons/icon_undo"));

    layTools->addWidget(mWidgetColors,1);
    layTools->addWidget(widgetZoom);
    layTools->addWidget(mComboBox);
    layTools->addSpacerItem(new QSpacerItem(30,10,QSizePolicy::Expanding));
    layTools->addWidget(mPBUndo);

    QScrollArea *scrollArea = new QScrollArea(this);
    mDrawingSheet = new DrawingSheet(this);

    scrollArea->setWidgetResizable(1);
    scrollArea->setBackgroundRole(QPalette::Light);
    scrollArea->setWidget(mDrawingSheet);

    layMain->setMargin(0);
    layMain->addLayout(layTools);
    layMain->addWidget(scrollArea,1);
    setLayout(layMain);    

    //update main color
    QObject::connect(mWidgetColors,SIGNAL(colorChanged(QColor)),mDrawingSheet,SLOT(setupColorDraw(QColor)));
    QObject::connect(widgetZoom,SIGNAL(zoomValueChanged(int)),mDrawingSheet,SLOT(setScale(int)));
    QObject::connect(mComboBox,SIGNAL(currentIndexChanged(int)),this,SLOT(drawingToolChanged(int)));

    QObject::connect(mPBUndo,SIGNAL(clicked()),mDrawingSheet,SLOT(undo()));
    QObject::connect(mDrawingSheet,SIGNAL(undoAvailable(bool)),mPBUndo,SLOT(setEnabled(bool)));

    mDrawingSheet->newDrawing(1000,200);
}

void EditorWidget::addColor(QColor c)
{
    mWidgetColors->addColor(c);
}

void EditorWidget::removeColor(QColor c)
{
    mWidgetColors->removeColor(c);
}


void EditorWidget::openMaskFromFile(QString filePath)
{
    mDrawingSheet->setMask(QImage(filePath,"PNG"));
    mFilePath = filePath;
}

void EditorWidget::drawingToolChanged(int index)
{
    switch(index){
    case 0:
        mDrawingSheet->enableFill(false);
        mDrawingSheet->enableLine(false);
        break;
    case 1:
        mDrawingSheet->enableLine(true);
        break;

    case 2:
        mDrawingSheet->enableFill(true);
        break;

    default:
        break;
    }
}
