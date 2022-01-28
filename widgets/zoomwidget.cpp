#include "zoomwidget.h"
#include <QHBoxLayout>

ZoomWidget::ZoomWidget(QWidget *parent) :
    QWidget(parent)
{
    mRadioButtons.append(new QRadioButton("1x",this));
    mRadioButtons.append(new QRadioButton("2x",this));
    mRadioButtons.append(new QRadioButton("4x",this));
    mRadioButtons.append(new QRadioButton("8x",this));
    mRadioButtons.append(new QRadioButton("16x",this));

    mZoomFactor.append(1);
    mZoomFactor.append(2);
    mZoomFactor.append(4);
    mZoomFactor.append(8);
    mZoomFactor.append(16);

    QHBoxLayout *lay = new QHBoxLayout();

    mRadioButtons[0]->setChecked(true);

    foreach (QRadioButton* rb, mRadioButtons) {
        lay->addWidget(rb);
        QObject::connect(rb,SIGNAL(toggled(bool)),this,SLOT(selectionChanged()));
    }

    setLayout(lay);
}

void ZoomWidget::selectionChanged()
{
    for (int index = 0; index < mRadioButtons.count(); ++index) {
        if(mRadioButtons[index]->isChecked()){
            emit zoomValueChanged(mZoomFactor[index]);
            break;
        }
    }
}
