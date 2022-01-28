#ifndef DRAWINGSHEET_H
#define DRAWINGSHEET_H

#include <QWidget>
#include <QPen>
#include "QDebug"


class DrawingSheet : public QWidget
{
    Q_OBJECT
public:
    explicit DrawingSheet(QWidget *parent = 0);

    void newDrawing(int width, int height);

    void setMask(QImage img);

    QImage* getMask(){return &mImgMask;}
signals:
    void undoAvailable(bool available);

public slots:
    void clearImage();

    void setupColorDraw(QColor newColor){mColorDraw = newColor;}
    void setScale(int scale);
    void enableFill(bool enable);
    void enableLine(bool enable);

    void undo();

protected:
    void paintEvent(QPaintEvent *);

    void mousePressEvent(QMouseEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);

private:
    QImage mImgMaskBackup;
    QImage mImgMask;
    int mScale;

    QPen mGridPen;
    QColor mColorDraw;
    QColor mColorEmpty;

    int mPrevPosX,mCurrPosX;
    int mPrevPosY,mCurrPosY;

    bool mEnableFillTool;
    bool mEnableLineTool;
    bool mLineDraging;

    QColor mColorLine;

    void drawLine(int x1, int y1, int x2, int y2,QColor color);
    void drawPixel(int posX, int posY,QColor color);

    void drawGrid(QPainter *p);
    void fillArea(int x,int y, QColor targetColor, QColor newColor);

    void drawTestRectangle();
    void backupMask();
};

#endif // DRAWINGSHEET_H
