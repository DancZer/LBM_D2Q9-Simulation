#ifndef VIEWERWIDGET_H
#define VIEWERWIDGET_H

#include <QWidget>
#include <QScrollArea>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>

class ViewerWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ViewerWidget(QWidget *parent = 0);

    void drawImage(QImage* img);

    void saveImage(QString path);

private:
    QLabel *mViewer;
    QImage mImage;
};

#endif // VIEWERWIDGET_H
