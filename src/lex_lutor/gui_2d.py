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

class LabelClickable(QtWidgets.QLabel):
    double_clicked = Signal(QtGui.QMouseEvent)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.double_clicked.emit(event)

class LabelClickable(QtWidgets.QLabel):
    double_clicked = Signal(QtGui.QMouseEvent)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.double_clicked.emit(event)

class WorkerLut(QObject):
    finished = Signal(QtGui.QImage, str)
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

        self.finished.emit(qimage, str(id(self)))

        return qimage

class MenuWidget(QtWidgets.QWidget):
    select_nodes_affecting_pixel = Signal(QVector3D, bool)
    select_node_closest_pixel = Signal(QVector3D, bool)

    def __init__(self, parent=None):
        super(MenuWidget, self).__init__(parent)
        button_test2 = QPushButton('test2')

        self.threads_image = []

        self.threads_image: [QtCore.QThread, QtCore.QObject] = []

        self.label_image = LabelClickable()
        image = colour.io.read_image(
            '/home/bjoern/PycharmProjects/darktabe_hald_generator/samples/provia/DSCF0326.JPG'
        )
        aspect = image.shape[0] / image.shape[1]
        width = 400
        height = int(width * aspect)
        self.img_base = cv2.resize(image, (width, height))
        # TODO:Color management. https://doc.qt.io/qt-6/qcolorspace.html
        img_uint = (self.img_base*255).astype(np.uint8, order='c')
        qimage = QtGui.QImage(
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

        self.label_image.double_clicked.connect(self.mouseDoubleClickEvent)


        # self.mouseDoubleClickEvent.connect(self.image_double_clicked)

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

    # @QtCore.Slot(QtGui.QMouseEvent)
    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        # TODO / FIXME: When proper image display scaling is implemented,
        #   the coords do not correspond to pixel coords directly anymore (maybe the don't even nof with display scaling?)
        #   hence, transorfmation must be performed. Is this done using qpainter?
        #

        # Fixme: Not possible to get relative position in widget?
        pos_image = self.label_image.pos()
        pos_image = self.label_image.mapFromParent(self.label_image.pos())
        print(pos_image)

        pos_pixel = [
            event.y() - pos_image.y(),
            event.x() - pos_image.x(),
        ]



        print(pos_pixel)

        pixel_image = self.img_base[
            pos_pixel[0],
            pos_pixel[1]

        ]

        # TODO: STRG makes only one
        expand_selection = event.modifiers() in (
            QtCore.Qt.Modifier.SHIFT,
            QtCore.Qt.Modifier.SHIFT + QtCore.Qt.Modifier.CTRL,
        )
        select_closest = event.modifiers() in (
            QtCore.Qt.Modifier.CTRL,
            QtCore.Qt.Modifier.SHIFT + QtCore.Qt.Modifier.CTRL,
        )


        # image is float, so pixel is coords.
        if select_closest:
            self.select_node_closest_pixel.emit(QVector3D(*pixel_image), expand_selection)
        else:
            self.select_nodes_affecting_pixel.emit(QVector3D(*pixel_image), expand_selection)

    @QtCore.Slot(QtGui.QImage, int)
    def update_image_async(self, image_updated, id_worker):
        if id_worker == self.threads_image[-1][-1]:
            self.label_image.setPixmap(
                QtGui.QPixmap(image_updated)
            )

    @QtCore.Slot()
    def start_update_image_waiting(self):
        if self.thread_image_waiting is not None and self.worker_image_waiting is not None:
            self.thread_image.start()

    def clean_threads(self):
        # TODO: does this intruduce a memory leak?

        threads_new = []
        for thread, worker, id_worker in self.threads_image:
            try:
                finished = thread.isFinished()
            except RuntimeError:
                continue

            if not finished:
                threads_new.append((thread, worker, id_worker))

        self.threads_image = threads_new

        # self.threads_image = [t for t in self.threads_image if t[0] and not t[0].isFinished()]

    @QtCore.Slot(colour.LUT3D)
    def start_update_image(self, lut):
        print('start')
        self.clean_threads()

        if self.threads_image:
            for thread, worker, id_worker in self.threads_image:
                try:
                    worker.finished.disconnect(self.start_update_image_waiting)
                except RuntimeError:
                    pass

                thread.terminate()

        thread, worker = QtCore.QThread(), WorkerLut(self.img_base, lut)
        self.threads_image.append((thread, worker, str(id(worker))))
        
        worker.moveToThread(thread)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self.update_image_async)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)


        self.threads_image[-1][0].start()