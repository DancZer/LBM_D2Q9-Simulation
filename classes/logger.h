#ifndef LOGGER_H
#define LOGGER_H

#include <QObject>
#include <QPlainTextEdit>
#include <QMutex>
#include <QMutexLocker>
#include <QStringBuilder>

class Logger : public QObject
{
    Q_OBJECT
public:
    static Logger* instance();

    void appendLog(QString text);

signals:
    void newLogAdded(QString text);

protected:
    explicit Logger(QObject *parent = 0);

private:
    static Logger* mInstance;
    QMutex mMutex;
    QPlainTextEdit* mTextBox;

    QString getTimeStamp();
};

#endif // LOGGER_H
