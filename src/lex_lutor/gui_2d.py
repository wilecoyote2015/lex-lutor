import colour
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import (Property, QObject, QPropertyAnimation, Signal)
from PySide6.QtWidgets import QWidget, QPushButton, QGraphicsWidget, QHBoxLayout, QVBoxLayout, QApplication
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.QtGui import (QGuiApplication, QMatrix4x4, QQuaternion, QVector3D)
from PySide6.Qt3DInput import Qt3DInput
from PySide6.Qt3DRender import Qt3DRender
# from PySide6 import Qt3DCore, Qt3DExtras, Qt3DInput, Qt3DRender
import cv2
import sys

# TODO: Async LUT trafo!
#   See https://realpython.com/python-pyqt-qthread/

class WorkerLut(QObject):
    finished = Signal(QtGui.QImage)
    progress = Signal(int)

    def __init__(self, image: np.ndarray, lut: colour.LUT3D, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image = image
        self.lut = lut



    def run(self, ):
        """Long-running task."""
        image_transformed = self.lut.apply(self.image)
        img_uint = (image_transformed * 255).astype(np.uint8, order='c')
        qimage = QtGui.QImage(
            img_uint,
            self.image.shape[1],
            self.image.shape[0],
            self.image.shape[1] * 3,
            QtGui.QImage.Format_RGB888
        )

        self.finished.emit(qimage)

        return qimage

class MenuWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MenuWidget, self).__init__(parent)
        # self.worker_image.progress.connect(self.reportProgress)

        button_test = QPushButton('test')
        button_test2 = QPushButton('test2')

        self.thread_image = None

        self.label_image = QtWidgets.QLabel()
        self.img_base = cv2.resize(colour.io.read_image(
            '/home/bjoern/PycharmProjects/darktabe_hald_generator/samples/provia/DSCF0326.JPG'
        ), (600, 800))
        # TODO:Color management. https://doc.qt.io/qt-6/qcolorspace.html
        img_uint = (self.img_base*255).astype(np.uint8, order='c')
        qimage =             QtGui.QImage(
                img_uint,
                self.img_base.shape[1],
                self.img_base.shape[0],
                self.img_base.shape[1]*3,
                QtGui.QImage.Format_RGB888
            )
        self.label_image.setPixmap(
            QtGui.QPixmap(qimage)

        )

        layout = QVBoxLayout()
        layout.addWidget(self.label_image)
        layout.addWidget(button_test2)

        self.setLayout(layout)

    @QtCore.Slot(QtGui.QImage)
    def update_image_async(self, image_updated):
        self.label_image.setPixmap(
            QtGui.QPixmap(image_updated)
        )
        self.thread_image = None


    @QtCore.Slot(colour.LUT3D)
    def update_image(self, lut):
        # print('emit')
        image_transformed = lut.apply(self.img_base)
        img_uint = (image_transformed * 255).astype(np.uint8, order='c')
        qimage = QtGui.QImage(
            img_uint,
            self.img_base.shape[1],
            self.img_base.shape[0],
            self.img_base.shape[1] * 3,
            QtGui.QImage.Format_RGB888
        )
        self.label_image.setPixmap(
            QtGui.QPixmap(qimage)

        )

    @QtCore.Slot(colour.LUT3D)
    def start_update_image(self, lut):
        # print('start')
        # TODO / FIXME: this way, a new thread is only started when
        #   a previous is finished. But this means that the last movement
        #   input before stopping the cursor will not be
        #   computed, which is bad for fast cursor movements.
        if self.thread_image is not None:
            return

        self.thread_image = QtCore.QThread()
        self.worker_image = WorkerLut(self.img_base, lut)
        self.worker_image.moveToThread(self.thread_image)
        self.worker_image.finished.connect(self.thread_image.quit)
        self.worker_image.finished.connect(self.worker_image.deleteLater)
        self.worker_image.finished.connect(self.update_image_async)
        self.thread_image.finished.connect(self.thread_image.deleteLater)
        self.thread_image.started.connect(self.worker_image.run)


        self.thread_image.start()