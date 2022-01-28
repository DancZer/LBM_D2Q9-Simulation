#include "logger.h"
#include "QDateTime"

Logger* Logger::mInstance = 0;

Logger * Logger::instance()
{
    if(!mInstance){
        mInstance = new Logger();
    }

    return mInstance;
}

Logger::Logger(QObject *parent) :
    QObject(parent)
{}

void Logger::appendLog(QString text)
{
    newLogAdded(getTimeStamp()+" "+QString::fromUtf8(text.toUtf8()));
}

QString Logger::getTimeStamp()
{
    return QTime::currentTime().toString();
}
