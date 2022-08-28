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

FNS_TRAFO = {
    QtCore.Qt.Key_R: lambda node, distance: node.coordinates_current + QVector3D(distance, 0, 0),
    QtCore.Qt.Key_G: lambda node, distance: node.coordinates_current + QVector3D(0, distance, 0),
    QtCore.Qt.Key_B: lambda node, distance: node.coordinates_current + QVector3D(0, 0, distance),
}

class NodeLut(Qt3DCore.QEntity):
    def __init__(self, indices_lut: tuple, coordinates: QVector3D, color_source: QtGui.QColor, radius: int, parent=None):
        super(NodeLut, self).__init__(parent)
        self.indices_lut = indices_lut
        # TODO: handle display color space here.
        #   Color should be some color-managed object from colour.

        # TODO: slots for transformation. Set if trafo finished

        # TODO: coordinates etc. should be qvector.

        # Those coordinates are kept during ongoing transformation until transformation is finished
        self.coordinates_current = coordinates

        self.transform = Qt3DCore.QTransform(translation=coordinates)
        self.color_source = color_source

        # TODO: must be changed in slot on movement!
        self.color_target = QtGui.QColor(
                                coordinates.x()*255,
                                coordinates.y()*255,
                                coordinates.z()*255,
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

    @QtCore.Slot(int, float, float)
    def transform_dragging(self, mode, distance, weight):
        # TODO: always clip to domain
        print('trafo')
        distance_weighted = distance*weight
        coords_new = FNS_TRAFO[mode](self, distance_weighted)

        # TODO: clip to borders
        # TODO: Clipping must be reflected in the trafo fn, as it must handle color space correctly (
        #  e.g. pertain hue on clipping)

        self.transform.setTranslation(coords_new)
        # TODO: during dragging, this function calculates the new position based on the mode
        #   (which is mapped to a color transform function (e.g. Hue) of the current color space)
        #   and applies it to the transform.
        #   new position is calculated using distance, relative to coordinates_current.
        #   mode is a qt.key in code


    @QtCore.Slot()
    def transform_accept(self):
        # TODO: Transformation is ended: set coordinates_current to new transform coordinates
        self.coordinates_current = self.transform.translation()

    @QtCore.Slot()
    def transform_cancel(self):
        # TODO: Transformation is ended: reset transform to coordinates_current
        self.transform.setTranslation(self.coordinates_current)


    @QtCore.Slot(bool)
    def select(self, is_selected):
        self.material.setAmbient(self.color_selected if is_selected else self.color_source)
        self.is_selected = is_selected