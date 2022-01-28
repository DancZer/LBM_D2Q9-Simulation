#ifndef ZOOMWIDGET_H
#define ZOOMWIDGET_H

#include <QWidget>
#include <QRadioButton>

class ZoomWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ZoomWidget(QWidget *parent = 0);

signals:
    void zoomValueChanged(int);

private slots:
    void selectionChanged();

private:
    QList<QRadioButton*> mRadioButtons;
    QList<int> mZoomFactor;
};

#endif // ZOOMWIDGET_H
