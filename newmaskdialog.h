#ifndef NEWMASKDIALOG_H
#define NEWMASKDIALOG_H

#include <QDialog>

namespace Ui {
    class NewMaskDialog;
}

class NewMaskDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NewMaskDialog(QWidget *parent = 0);
    ~NewMaskDialog();

    int getMaskWidth();
    int getMaskHeight();

private slots:
    void on_spinBoxWidth_valueChanged(int );

    void updateWidget();

    void on_spinBoxHeight_valueChanged(int );

private:
    Ui::NewMaskDialog *mUi;
};

#endif // NEWMASKDIALOG_H
