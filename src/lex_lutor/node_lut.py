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



class NodeLut(Qt3DCore.QEntity):
    position_changed = QtCore.Signal(tuple, QVector3D) # indices, coordinates
    def __init__(self, indices_lut: tuple, coordinates: QVector3D, color_source: QtGui.QColor, radius: int, lut_parent=None):
        super(NodeLut, self).__init__(lut_parent)
        self.indices_lut = indices_lut
        self.lut_parent= lut_parent
        # TODO: handle display color space here.
        #   Color should be some color-managed object from colour.

        # TODO: slots for transformation. Set if trafo finished

        # TODO: coordinates etc. should be qvector.

        # Those coordinates are kept during ongoing transformation until transformation is finished
        self.coordinates_current = coordinates

        self.transform = Qt3DCore.QTransform(translation=coordinates)
        self.transform.translationChanged.connect(self.send_signal_position_changed)
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

    @QtCore.Slot(QVector3D)
    def send_signal_position_changed(self, coordinates):
        # print('Emit')
        self.position_changed.emit(self.indices_lut, coordinates)

    @QtCore.Slot(int, float, float)
    def transform_dragging(self, mode, distance, weight):
        # TODO: always clip to domain: Value must be stuck at domain border,
        #   so one must find the point in movement path where at border.

        # TODO: Exposure. For this, first transformation must be into linear RGB.
        #   But how is exposure calculated then? Effect of exposure must not depend on linear color space
        #   choice. How is the transfer function handled in colour?

        # TODO: Linear space if upper case
        distance_weighted = distance*weight

        # if mode == KEY_EXPOSURE:
        #     coords_linear = self.transform_color_space(
        #         self.lut_parent.color_space,
        #         colour.gamma_function()
        #         self.coordinates_current
        #     )
        # else:
        color_space_transform, dimension_transform = color_spaces_components_transform[mode]
        coords_current_target_space = self.transform_color_space(
            self.lut_parent.color_space,
            color_space_transform,
            self.coordinates_current
        )

        components_vector_add = [1. if idx_ == dimension_transform else 0. for idx_ in range(3)]
        coords_new_target_space = coords_current_target_space + QVector3D(*components_vector_add) * distance_weighted

        # transform to target color space, modify the according component and then

        coords_new = self.transform_color_space(
            color_space_transform,
            self.lut_parent.color_space,
            coords_new_target_space
        )

        # TODO: clip to borders
        # TODO: Clipping must be reflected in the trafo fn, as it must handle color space correctly (
        #  e.g. pertain hue on clipping)

        self.transform.setTranslation(coords_new)
        # TODO: during dragging, this function calculates the new position based on the mode
        #   (which is mapped to a color transform function (e.g. Hue) of the current color space)
        #   and applies it to the transform.
        #   new position is calculated using distance, relative to coordinates_current.
        #   mode is a qt.key in code

    def transform_color_space(self, color_space_source, color_space_target, value_input: QVector3D):
        if color_space_target == color_space_source or color_space_target is None or color_space_source is None:
            return value_input

        input_array = np.asarray(value_input.toTuple())

        if color_space_source in (HSV, HSL, HCL):
            if color_space_target in (HSV, HSL, HCL):
                result = getattr(colour, f'{color_space_source}_to_{color_space_target}')(input_array)
            elif isinstance(color_space_target, colour.models.RGB_Colourspace):
                result = getattr(colour, f'{color_space_source}_to_RGB')(input_array)
        elif isinstance(color_space_source, colour.models.RGB_Colourspace):
            if color_space_target in (HSV, HSL, HCL):
                result = getattr(colour, f'RGB_to_{color_space_target}')(input_array)

            elif isinstance(color_space_target, colour.models.RGB_Colourspace):
                result =  colour.RGB_to_RGB(input_array, color_space_source, color_space_target)
            else: raise NotImplementedError
        else:
            raise NotImplementedError

        return QVector3D(*result.tolist())

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
        self.material.setAmbient(self.color_selected if is_selected else self.color_source)
        self.is_selected = is_selected