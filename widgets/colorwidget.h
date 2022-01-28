#ifndef COLORWIDGET_H
#define COLORWIDGET_H

#include <QWidget>
#include <QList>
#include <QColor>

class ColorWidget : public QWidget
{
    Q_OBJECT
public:

    static const int ICONSIZE = 16;
    static const int SELECTED_ICONSIZE = 20;

    explicit ColorWidget(QWidget *parent = 0);

    QList<QColor> getColors();
    QColor getSelectedColor();

public slots:
    void addColor(QColor color);
    void removeColor(QColor color);

signals:
    void colorChanged(QColor newColor);

protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);

private:
    QList<QColor> mAvailableColors;
    int mSelectedColorIndex;
    int mOffsetY;

    void createLayout();

};

#endif // COLORWIDGET_H
