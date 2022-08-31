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

class MenuWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MenuWidget, self).__init__(parent)

        button_test = QPushButton('test')
        button_test2 = QPushButton('test2')

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

    @QtCore.Slot(Qt3DCore.QComponent)
    def update_image(self, emitter):
        # print('emit')
        image_transformed = emitter.lut.apply(self.img_base)
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