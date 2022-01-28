#ifndef PAINTERWIDGET_H
#define PAINTERWIDGET_H

#include <QWidget>
#include <QLabel>
#include <QScrollArea>
#include <QHBoxLayout>
#include <QPushButton>
#include <QComboBox>

#include "colorwidget.h"
#include "drawingsheet.h"
#include "zoomwidget.h"

class EditorWidget : public QWidget
{
    Q_OBJECT
public:
    explicit EditorWidget(QWidget *parent = 0);

    //forward methods
    void newDrawing(int width, int height){mDrawingSheet->newDrawing(width,height);}
    QImage* getMask(){return mDrawingSheet->getMask();}

    void openMaskFromFile(QString filePath);
    QString getFilePath(){return mFilePath;}

signals:
    void drawingChanged();

public slots:
    void addColor(QColor);
    void removeColor(QColor);

private slots:
    void drawingToolChanged(int);

private:
    DrawingSheet *mDrawingSheet;
    ColorWidget *mWidgetColors;
    QString mFilePath;

    QPushButton *mPBUndo;

    QComboBox* mComboBox;


    void createLayout();
};

#endif // PAINTERWIDGET_H
