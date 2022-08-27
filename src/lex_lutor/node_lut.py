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
import sys

class NodeLut(Qt3DCore.QEntity):
    def __init__(self, indices_lut: tuple, coordinates: tuple, color_source: tuple, radius: int, parent=None):
        super(NodeLut, self).__init__(parent)
        self.indices_lut = indices_lut
        # TODO: handle display color space here.
        #   Color should be some color-managed object from colour.

        # TODO: slots for transformation. Set if trafo finished

        # Those coordinates are kept during ongoing transformation until transformation is finished
        self.coordinates_current = coordinates

        self.transform = Qt3DCore.QTransform(
                            translation=QtGui.QVector3D(
                                *coordinates
                            )
                        )
        self.color_source = QtGui.QColor(
                                *color_source,
                                255
                            )

        # TODO: must be changed in slot on movement!
        self.color_target = QtGui.QColor(
                                coordinates[0]*255,
                                coordinates[1]*255,
                                coordinates[2]*255,
                                255
                            )

        self.color_selected = QtGui.QColor(
            255,
            0,
            0,
            255
        )
        self.material = Qt3DExtras.QDiffuseSpecularMaterial(
                            ambient=self.color_source,
                            specular=QtGui.QColor(0, 0, 0, 0),
                            diffuse=QtGui.QColor(255, 255, 255, 255)
                        )

        self.mesh = Qt3DExtras.QCuboidMesh(xExtent=radius, yExtent=radius, zExtent=radius)
        self.addComponent(self.material)
        self.addComponent(self.transform)
        self.addComponent(self.mesh)

        # TODO: make signal for selected changed
        self.is_selected = False

        self.active = False

        self.picker = Qt3DRender.QObjectPicker(self.parentEntity())
        self.addComponent(self.picker)
        # self.picker.clicked.connect(self.parentEntity().slot_clicked)


    @QtCore.Slot(bool)
    def select(self, is_selected):
        self.material.setAmbient(self.color_selected if is_selected else self.color_source)
        self.is_selected = is_selected