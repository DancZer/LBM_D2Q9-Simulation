#include "viewerwidget.h"
#include <QPainter>

ViewerWidget::ViewerWidget(QWidget *parent) :
    QWidget(parent)
{
    QVBoxLayout *layMain = new QVBoxLayout();

    QScrollArea *scrollArea = new QScrollArea(this);
    mViewer = new QLabel(this);

    scrollArea->setWidgetResizable(1);
    scrollArea->setBackgroundRole(QPalette::Light);
    scrollArea->setWidget(mViewer);

    layMain->addWidget(scrollArea);
    setLayout(layMain);

    layMain->setMargin(0);
}

void ViewerWidget::drawImage(QImage* img)
{
    mImage = QImage(*img);
    mViewer->setPixmap(QPixmap::fromImage(mImage));
}

void ViewerWidget::saveImage(QString path)
{
    mImage.save(path,"PNG",100);
}
