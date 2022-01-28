#ifndef COLORSSAMPLE_H
#define COLORSSAMPLE_H

#include <QWidget>

#define SAMPLE_SIZE 15
#define TEXT_HEIGHT 15

class ColorsSample : public QWidget
{
    Q_OBJECT
public:
    explicit ColorsSample(QWidget *parent = 0);

    void setColors(QList<QColor> colors){
        mColors.clear();
        mColors = colors;
        repaint();
    }

protected:
    void paintEvent(QPaintEvent *);

private:
    QList<QColor> mColors;
};

#endif // COLORSSAMPLE_H
