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
from lex_lutor.node_lut import NodeLut
from datetime import datetime, timedelta

# TODO: Add transformation that corresponds to exposure.
#   is there a transform function that transforms back to scene-referred linear space?
#   I hope so.
# TODO: Linear mode, where transformations are performed w.r.t. a linear RGB space?
#   For this, node value is first transformed into linear RGB and then to HSV etc.
#   then HSV -> linear -> lut color space.
#   Maybe this is triggered with upper case?
# TODO: Exposure! -> Transform to linear and then multiply value

# TODO: For table upscaling, see https://colour.readthedocs.io/en/develop/generated/colour.algebra.table_interpolation_trilinear.html#colour.algebra.table_interpolation_trilinear
#   Should be easy

# TODO: Chrominance should be bounded at zero (also saturation etc.) to prevent negative values and hence
#   color inversion!
#   best to introduce general bounds for trafo functions


class Lut3dEntity(Qt3DCore.QComponent):
    lut_changed = QtCore.Signal(colour.LUT3D)

    def __init__(self, lut, parent_gui):
        super().__init__()
        self.lut = None
        self.parent_gui = parent_gui
        self.mesh_node = None
        self.picker = Qt3DRender.QObjectPicker(self)
        self.nodes_lut = None
        self.color_space: colour.models.RGB_Colourspace = None

        self.load_lut(lut)

        # self.time_last_change = datetime.now()
        # self.timedelta_update = timedelta(milliseconds=100)

        # TODO: find a better way to block lut calc....
        #   use qtimer as long as dragging, so that update is performed on regular intervals?
        self.working = False


        # self.root_entity = None


    def get_values_lut_source(self, lut: colour.LUT3D):
        values_r_source = np.linspace(lut.domain[0,0], lut.domain[1,0], lut.size)
        values_g_source = np.linspace(lut.domain[0,1], lut.domain[1,1], lut.size)
        values_b_source = np.linspace(lut.domain[0,2], lut.domain[1,2], lut.size)

        return values_r_source, values_g_source, values_b_source



    # def load_lut_file(self, filepath):
    #     lut =


    def load_lut(self, lut: colour.LUT3D):
        self.lut = lut

        # TODO: 2 textures: source and target, that can be switched
        # TODO: Color map from lut space to display srgb

        values_r_source, values_g_source, values_b_source = self.get_values_lut_source(lut)

        radius = np.min(lut.domain[1] - lut.domain[0]) / lut.size / 5

        color_max= 255
        self.mesh_node = Qt3DExtras.QSphereMesh(rings=8, slices=8, radius=radius)


        nodes_lut = []
        for idx_r, value_r_source in enumerate(values_r_source):
            nodes_r = []
            for idx_g, value_g_source in enumerate(values_g_source):
                nodes_g = []
                for idx_b, value_b_source in enumerate(values_b_source):
                    entity_node = NodeLut(
                        (idx_r, idx_g, idx_b),
                        QVector3D(
                            lut.table[idx_r, idx_g, idx_b, 0],
                            lut.table[idx_r, idx_g, idx_b, 1],
                            lut.table[idx_r, idx_g, idx_b, 2],
                        ),
                        QtGui.QColor(
                            value_r_source * color_max,
                            value_g_source * color_max,
                            value_b_source * color_max,
                            255
                        ),
                        radius,
                        self
                    )
                    entity_node.picker.clicked.connect(self.slot_clicked)
                    entity_node.position_changed.connect(self.update_lut_node_changed)
                    self.parent_gui.cancel_transform.connect(entity_node.cancel_transform)
                    self.parent_gui.accept_transform.connect(entity_node.accept_transform)
                    nodes_g.append(entity_node)
                nodes_r.append(nodes_g)
            nodes_lut.append(nodes_r)
        self.nodes_lut = nodes_lut

    @QtCore.Slot(int, float)
    def transform_dragging(self, mode, distance):

        def fn(node: NodeLut):
            # TODO: calc weight based on selection center or something
            if node.is_selected:
                node.transform_dragging(mode, distance, 1.)

        # fn = node.
        self.iter_nodes(fn)

        self.lut_changed.emit(self.lut)

        # if datetime.now() - self.time_last_change > self.timedelta_update:
        #     self.time_last_change = datetime.now()

    @QtCore.Slot(tuple, QVector3D)
    def update_lut_node_changed(self, indices_node, coordinates_node):
        self.lut.table[indices_node] = np.asarray(coordinates_node.toTuple())
        # print(self.lut.table[indices_node])

    @QtCore.Slot()
    def slot_clicked(self, event):
        entity: NodeLut = event.entity()
        modifiers = event.modifiers()

        if event.button() == Qt3DRender.QPickEvent.LeftButton:
            if modifiers == Qt3DRender.QPickEvent.ShiftModifier:
                entity.select(not entity.is_selected)
            else:
                fn = lambda node: node.select(not node.is_selected and node is entity)
                self.iter_nodes(fn)
                # for nodes_r in self.nodes_lut:
                #     for nodes_g in nodes_r:
                #         for node in nodes_g:
                #             node.select(not node.is_selected and node is entity)

    def iter_nodes(self, fn, *args, **kwargs):
        for nodes_r in self.nodes_lut:
            for nodes_g in nodes_r:
                for node in nodes_g:
                    fn(node, *args, **kwargs)
