#-------------------------------------------------
#
# Project created by QtCreator 2011-11-03T20:06:52
#
#-------------------------------------------------

QT       += core gui widgets

TARGET = LBM_D2Q9-Simulation
TEMPLATE = app

RESOURCES += \
    resources.qrc

SOURCES += main.cpp\
        mainwindow.cpp \
    widgets/zoomwidget.cpp \
    widgets/viewerwidget.cpp \
    widgets/editorwidget.cpp \
    widgets/drawingsheet.cpp \
    widgets/colorwidget.cpp \
    widgets/boundaryconditionwidget.cpp \
    widgets/boundaryconditionlistwidget.cpp \
    classes/lbmsimulation.cpp \
    classes/lbmmesh.cpp \
    classes/meshrenderer.cpp \
    classes/settings.cpp \
    widgets/colorssample.cpp \
    newmaskdialog.cpp \
    classes/logger.cpp \
    classes/cudakernel.cu

HEADERS  += mainwindow.h \
    widgets/zoomwidget.h \
    widgets/viewerwidget.h \
    widgets/editorwidget.h \
    widgets/drawingsheet.h \
    widgets/colorwidget.h \
    classes/point.h \
    widgets/boundaryconditionwidget.h \
    widgets/boundaryconditionlistwidget.h \
    classes/boundaryconditionlist.h \
    classes/boundarycondition.h \
    classes/lbmsimulation.h \
    classes/lbmparticle.h \
    classes/lbmmesh.h \
    classes/lbmhostgrid.h \
    classes/lbmdevicegrid.h \
    classes/core.h \
    classes/cuda/cudaSafeCall.h \
    classes/cuda/cudaAlign.h \
    classes/meshrenderer.h \
    classes/settings.h \
    widgets/colorssample.h \
    newmaskdialog.h \
    classes/logger.h

FORMS    += mainwindow.ui \
    newmaskdialog.ui

SOURCES -= classes/cudakernel.cu
unix{
    QMAKE_CFLAGS_RELEASE     = -O3
    QMAKE_CXXFLAGS_RELEASE   = -O3
}else{
    QMAKE_CFLAGS_RELEASE     = -O2 /MD
    QMAKE_CXXFLAGS_RELEASE   = -O2 /MD
}

#Ahoz hogy CUDA módot beleepitsuk a kodba az alabbi feltetelt allitsuk att igazra
#valamint a settings.h-ban uncommenteljuk a: #define CUDA_BUILD
if (true){
    unix {
        #cuda toolkit directory
        CUDA_DIR = /usr/local/cuda
        QMAKE_LIBDIR += $$CUDA_DIR/lib64
    } else {
        #cuda toolkit directory
        CUDA_DIR = "c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
        QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
    }

    cuda.output = ${QMAKE_FILE_BASE}.obj
    cuda.commands = $$CUDA_DIR/bin/nvcc -c -Xcompiler $$join(QMAKE_CXXFLAGS_RELEASE,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} -arch=sm_60

    cuda.input = classes/cudakernel.cu
    QMAKE_EXTRA_COMPILERS += cuda

    INCLUDEPATH += $$CUDA_DIR/include
    LIBS += -lcudart
}

QMAKE_LFLAGS += "/nodefaultlib:msvcrtd"
