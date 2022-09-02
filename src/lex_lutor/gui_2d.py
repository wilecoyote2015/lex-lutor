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
        self.thread_image_waiting = None
        self.worker_image = None
        self.worker_image_waiting = None

        self.threads_image: [QtCore.QThread, QtCore.QObject] = []

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

    # @QtCore.Slot(tuple, colour.LUT3D)
    # def slot_hover_node_stop(self, indices, lut):
    #     print('stopped hover')
    #     # FIXME: This destroys update on dragging
    #     # self.start_update_image(lut)

    # @QtCore.Slot(tuple, QVector3D, colour.LUT3D)
    # def slot_hover_node_start(self, indices, coordinates, lut):
    #     # Make empty LUT of size
    #     # TODO: handle domain!
    #     lut_use = colour.LUT3D(
    #         colour.LUT3D.linear_table(lut.size)
    #     )
    #
    #     lut_use.table = np.mean(lut_use.table, axis=3)[..., np.newaxis]
    #     lut_use.table[lut.indices] = [1., 0., 0.]
    #
    #     self.start_update_image()
    #
    #     # Make selected red
    #
    #     # render
    #     print('started hover')

    @QtCore.Slot(QtGui.QImage)
    def update_image_async(self, image_updated):
        self.label_image.setPixmap(
            QtGui.QPixmap(image_updated)
        )

        # if thread is None, the worker was worker_waiting
        if self.thread_image is None:
            self.worker_image_waiting = None
            self.thread_image_waiting = None

        self.thread_image = None
        self.worker_image = None

    @QtCore.Slot()
    def start_update_image_waiting(self):
        if self.thread_image_waiting is not None and self.worker_image_waiting is not None:
            self.thread_image.start()

    def clean_threads(self):
        # TODO: does this intruduce a memory leak?

        threads_new = []
        for thread, worker in self.threads_image:
            try:
                finished = thread.isFinished()
            except RuntimeError:
                continue

            if not finished:
                threads_new.append((thread, worker))

        self.threads_image = threads_new

        # self.threads_image = [t for t in self.threads_image if t[0] and not t[0].isFinished()]

    @QtCore.Slot(colour.LUT3D)
    def start_update_image(self, lut):
        # print('start')
        # TODO / FIXME: this way, a new thread is only started when
        #   a previous is finished. But this means that the last movement
        #   input before stopping the cursor will not be
        #   computed, which is bad for fast cursor movements.
        #   tried with second thread. But logic is complicated!
        self.clean_threads()

        if self.threads_image:
            for thread, worker in self.threads_image:
                worker.finished.disconnect(self.start_update_image_waiting)
                #
                # try:
                # except RuntimeError:
                #     pass

                thread.quit()



        # if self.thread_image is not None:
        #     self.thread_image.terminate()
            # self.thread_image = None
            # self.thread_image.wait()
            # self.thread_image_waiting = QtCore.QThread()
            # self.worker_image_waiting = WorkerLut(self.img_base, lut)
            # self.worker_image_waiting.moveToThread(self.thread_image_waiting)
            # self.worker_image_waiting.finished.connect(self.thread_image_waiting.quit)
            # self.worker_image_waiting.finished.connect(self.worker_image_waiting.deleteLater)
            # self.worker_image_waiting.finished.connect(self.update_image_async)
            # self.thread_image_waiting.finished.connect(self.thread_image_waiting.deleteLater)
            # self.thread_image_waiting.started.connect(self.worker_image_waiting.run)
            # TODO: works better if just returning!
            #   But hover does not work well. Try to get termination working
            # return

        self.threads_image.append((QtCore.QThread(), WorkerLut(self.img_base, lut)))
        self.threads_image[-1][1].moveToThread(self.threads_image[-1][0])
        self.threads_image[-1][1].finished.connect(self.threads_image[-1][0].quit)
        self.threads_image[-1][1].finished.connect(self.threads_image[-1][1].deleteLater)
        self.threads_image[-1][1].finished.connect(self.update_image_async)
        self.threads_image[-1][0].finished.connect(self.threads_image[-1][0].deleteLater)
        self.threads_image[-1][0].started.connect(self.threads_image[-1][1].run)


        self.threads_image[-1][0].start()