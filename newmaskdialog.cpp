#include "newmaskdialog.h"
#include "ui_newmaskdialog.h"

NewMaskDialog::NewMaskDialog(QWidget *parent) :
    QDialog(parent),
    mUi(new Ui::NewMaskDialog)
{
    mUi->setupUi(this);
    updateWidget();
}

NewMaskDialog::~NewMaskDialog()
{
    delete mUi;
}

int NewMaskDialog::getMaskWidth()
{
    return mUi->spinBoxWidth->value();
}

int NewMaskDialog::getMaskHeight()
{
    return mUi->spinBoxHeight->value();
}

void NewMaskDialog::on_spinBoxWidth_valueChanged(int )
{
    updateWidget();
}

void NewMaskDialog::on_spinBoxHeight_valueChanged(int )
{
    updateWidget();
}

void NewMaskDialog::updateWidget()
{
    float ratio = (float)mUi->spinBoxWidth->value()/(float)mUi->spinBoxHeight->value();
    int maxW=300,maxH;

    if(ratio>1.0){
        maxH = (float)maxW/ratio;
    }else{
        maxH = maxW;
        maxW = ratio*(float)maxH;
    }

    maxH = qMax(maxH,1);
    maxW = qMax(maxW,1);

    mUi->widget->setMinimumWidth(maxW);
    mUi->widget->setMaximumWidth(maxW);

    mUi->widget->setMinimumHeight(maxH);
    mUi->widget->setMaximumHeight(maxH);
}
