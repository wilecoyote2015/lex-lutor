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
import colour
from lex_lutor.constants import HSV, HSL, HCL, color_spaces_components_transform, KEY_EXPOSURE


# TODO: instead of is_selected, have a selection weight.
#   this way, selection smoothing can be performed easily.

class NodeLut(Qt3DCore.QEntity):
    position_changed = QtCore.Signal(tuple, QVector3D)  # indices, coordinates
    mouse_hover_start = QtCore.Signal(tuple)
    mouse_hover_stop = QtCore.Signal(tuple)

    def __init__(self, indices_lut: tuple, coordinates_target: QVector3D, coordinates_source: QVector3D, radius: int,
                 lut_parent=None):
        super(NodeLut, self).__init__(lut_parent)
        self.indices_lut = indices_lut
        self.lut_parent = lut_parent
        # TODO: handle display color space here.
        #   Color should be some color-managed object from colour.

        # TODO: slots for transformation. Set if trafo finished

        # TODO: coordinates etc. should be qvector.

        # Those coordinates are kept during ongoing transformation until transformation is finished
        self.coordinates_current = coordinates_target
        self.coordinates_reset = coordinates_target
        self.coordinates_source = coordinates_source

        self.transform = Qt3DCore.QTransform(translation=coordinates_target)
        self.transform.translationChanged.connect(self.send_signal_position_changed)

        # TODO: must be changed in slot on movement!
        self.color_target = QtGui.QColor(
            coordinates_target.x() * 255,
            coordinates_target.y() * 255,
            coordinates_target.z() * 255,
                                255
                            )

        self.color_source = QtGui.QColor(
            coordinates_source.x() * 255,
            coordinates_source.y() * 255,
            coordinates_source.z() * 255,
            255
        )

        self.color_selected = QtGui.QColor(
            255,
            0,
            0,
            255
        )

        self.color_selected_base = QtGui.QColor(
            0,
            255,
            0,
            255
        )

        self.material = Qt3DExtras.QDiffuseSpecularMaterial(
            ambient=self.color_source,
            specular=QtGui.QColor(0, 0, 0, 0),
            diffuse=QtGui.QColor(255, 255, 255, 255),
        )

        self.mesh = Qt3DExtras.QCuboidMesh(xExtent=radius, yExtent=radius, zExtent=radius)
        self.addComponent(self.material)
        self.addComponent(self.transform)
        self.addComponent(self.mesh)

        # TODO: make signal for selected changed
        self.is_selected = False
        self.is_selected_base = False

        self.active = False

        self.picker = Qt3DRender.QObjectPicker(self.parentEntity())
        self.addComponent(self.picker)

        # FIXME: Problem: on quick transition between two nodes, left of previously hovered is always fired after entered
        #   of next one...
        self.picker.exited.connect(self.emit_mouse_hover_stop)
        self.picker.entered.connect(self.emit_mouse_hover_start)
        # self.picker.clicked.connect(self.parentEntity().slot_clicked)

    @QtCore.Slot(QVector3D)
    def send_signal_position_changed(self, coordinates):
        # print('Emit')
        self.position_changed.emit(self.indices_lut, coordinates)




    @QtCore.Slot()
    def accept_transform(self):
        # TODO: Transformation is ended: set coordinates_current to new transform coordinates
        self.coordinates_current = self.transform.translation()

    @QtCore.Slot()
    def cancel_transform(self):
        # TODO: Transformation is ended: reset transform to coordinates_current
        self.transform.setTranslation(self.coordinates_current)

    @QtCore.Slot(bool)
    def select(self, is_selected):
        self.is_selected = is_selected

        if not self.is_selected_base:
            self.material.setAmbient(self.color_selected if is_selected else self.color_source)

    @QtCore.Slot(bool)
    def select_base(self, is_selected):
        self.is_selected_base = is_selected

        self.material.setAmbient(self.color_selected_base if is_selected else (
            self.color_source if not self.is_selected else self.color_selected
        )
                                 )

    @QtCore.Slot()
    def emit_mouse_hover_stop(self):
        # print(f'left {self.indices_lut}')
        self.mouse_hover_stop.emit(self.indices_lut)

    @QtCore.Slot()
    def emit_mouse_hover_start(self):
        # print(f'Entered {self.indices_lut}')
        self.mouse_hover_start.emit(self.indices_lut)
