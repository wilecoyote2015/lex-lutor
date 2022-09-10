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
from tqdm import tqdm
from scipy.sparse import csc_matrix

# TODO: Async LUT trafo!
#   See https://realpython.com/python-pyqt-qthread/

INTER_TRILINEAR = 'Trilinear'
INTER_TETRAHEDRAL = 'Tetrahedral'

INTERPOLATORS = {
    INTER_TRILINEAR: colour.algebra.table_interpolation_trilinear,
    INTER_TETRAHEDRAL: colour.algebra.table_interpolation_tetrahedral,
}


class LabelClickable(QtWidgets.QLabel):
    double_clicked = Signal(QtGui.QMouseEvent)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.double_clicked.emit(event)


class LabelClickable(QtWidgets.QLabel):
    double_clicked = Signal(QtGui.QMouseEvent)

    def __init__(self, parent=None):
        super().__init__(parent)

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed,
        )
        sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy)

    def hasHeightForWidth(self) -> bool:
        return True

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.double_clicked.emit(event)

    def heightForWidth(self, width: int) -> int:
        # FIXME: does not work somehow.
        aspect_pixmap = self.pixmap().height() / self.pixmap().width()

        # return int(width * aspect_pixmap)
        return aspect_pixmap * width



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
        self.interpolation_matrix = None
        button_test2 = QPushButton('test2')

        self.queue_updates_image: [QtCore.QThread, QtCore.QObject, str] = []

        self.label_image = LabelClickable()
        self.label_image.setScaledContents(True)
        self.img_base = None

        layout = QVBoxLayout()
        layout.addWidget(self.label_image)
        # layout.addStretch(2)
        layout.addWidget(button_test2)

        self.setLayout(layout)

        self.label_image.double_clicked.connect(self.mouseDoubleClickEvent)

    @QtCore.Slot(str)
    def load_image(self, path_image, lut):
        image = colour.io.read_image(
            path_image
        )
        aspect = image.shape[0] / image.shape[1]
        width = 400
        height = int(width * aspect)
        self.img_base = cv2.resize(image, (width, height))
        # TODO:Color management. https://doc.qt.io/qt-6/qcolorspace.html
        img_uint = (self.img_base * 255).astype(np.uint8, order='c')
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

        # TODO: handle case that no lut loaded.
        # TODO: Interpolation from app
        # self.make_interpolation_matrix(lut, INTER_TRILINEAR)

    @QtCore.Slot()
    def make_interpolation_matrix(self, lut, interpolation):
        if interpolation not in INTERPOLATORS:
            raise ValueError(f'Interpolation {interpolation} not supported.')
        # feature matrix with order of permutation: r, g, b

        pixels = self.img_base.reshape((self.img_base.shape[0] * self.img_base.shape[1], self.img_base.shape[2]))
        print('generating design matrix')
        # design_matrix_new = np.zeros((pixels_references.shape[0], size * size * size), pixels_references.dtype)
        data = np.ndarray((0,), dtype=pixels.dtype)
        indices_rows = np.ndarray((0,), dtype=int)
        indices_columns = np.ndarray((0,), dtype=int)

        lut_empty = colour.LUT3D(table=np.zeros((lut.size, lut.size, lut.size, 3), dtype=pixels.dtype), size=lut.size)
        indices_pixels = np.arange(pixels.shape[0])

        idx_col_design_matrix = 0
        for idx_r in tqdm(range(lut.size)):
            for idx_g in range(lut.size):
                for idx_b in range(lut.size):
                    lut_empty.table[idx_r, idx_g, idx_b] = 1.
                    weights_pixels = lut_empty.apply(
                        pixels,
                        interpolator=INTERPOLATORS[interpolation]
                    )[..., 0]
                    bool_weights = weights_pixels != 0
                    weights_pixels_non_zero = weights_pixels[bool_weights]
                    data = np.concatenate((data, weights_pixels_non_zero))
                    indices_rows = np.concatenate((indices_rows, indices_pixels[bool_weights]))
                    indices_columns = np.concatenate(
                        (indices_columns, np.full((weights_pixels_non_zero.shape[0],), idx_col_design_matrix)))
                    lut_empty.table[idx_r, idx_g, idx_b] = 0.
                    idx_col_design_matrix += 1

        result = csc_matrix(
            (data,
             (indices_rows, indices_columns)),
            shape=(pixels.shape[0], lut.size ** 3)
        )

        self.interpolation_matrix = result

    # @QtCore.Slot(QtGui.QMouseEvent)
    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        # TODO / FIXME: When proper image display scaling is implemented,
        #   the coords do not correspond to pixel coords directly anymore (maybe the don't even nof with display scaling?)
        #   hence, transorfmation must be performed. Is this done using qpainter?
        #

        # Fixme: Not possible to get relative position in widget?
        # pos_image = self.label_image.pos()
        pos_image = self.label_image.mapFromParent(self.label_image.pos())
        # print(pos_image)

        pos_pixel = [
            event.y() - pos_image.y(),
            event.x() - pos_image.x(),
        ]

        # print(pos_pixel)

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
        # if id_worker == self.queue_updates_image[-1][-1]:
        self.label_image.setPixmap(
            QtGui.QPixmap(image_updated)
        )

    @QtCore.Slot()
    def start_next_update(self):
        if self.queue_updates_image and not self.queue_updates_image[-1][0].isFinished():
            self.queue_updates_image = self.queue_updates_image[-1:]
            self.queue_updates_image[-1][0].start()
        else:
            self.queue_updates_image = []

    @QtCore.Slot(colour.LUT3D)
    def start_update_image(self, lut):
        # TODO: quit running thread.
        thread, worker = QtCore.QThread(), WorkerLut(self.img_base, lut)
        self.queue_updates_image.append((thread, worker, str(id(worker))))

        worker.moveToThread(thread)
        worker.finished.connect(thread.quit)
        # worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self.update_image_async)
        # thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.start_next_update)
        thread.started.connect(worker.run)

        if len(self.queue_updates_image) == 1:
            self.start_next_update()
