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

class Lut3dEntity(Qt3DCore.QComponent):
    def __init__(self, lut):
        super().__init__()
        self.lut = None
        self.mesh_node = None
        self.picker = Qt3DRender.QObjectPicker(self)
        self.nodes_lut = None

        self.load_lut(lut)

        # self.root_entity = None


    def get_values_lut_source(self, lut: colour.LUT3D):
        values_r_source = np.linspace(lut.domain[0,0], lut.domain[1,0], lut.size)
        values_g_source = np.linspace(lut.domain[0,1], lut.domain[1,1], lut.size)
        values_b_source = np.linspace(lut.domain[0,2], lut.domain[1,2], lut.size)

        return values_r_source, values_g_source, values_b_source

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
                        (
                            lut.table[idx_r, idx_g, idx_b, 0],
                            lut.table[idx_r, idx_g, idx_b, 1],
                            lut.table[idx_r, idx_g, idx_b, 2],
                        ),
                        (
                            value_r_source * color_max,
                            value_g_source * color_max,
                            value_b_source * color_max,
                        ),
                        radius,
                        self
                    )
                    entity_node.picker.clicked.connect(self.slot_clicked)
                    nodes_g.append(entity_node)
                nodes_r.append(nodes_g)
            nodes_lut.append(nodes_r)
        self.nodes_lut = nodes_lut

    @QtCore.Slot()
    def slot_clicked(self, event):
        entity: NodeLut = event.entity()
        modifiers = event.modifiers()

        if event.button() == Qt3DRender.QPickEvent.LeftButton:
            if modifiers == Qt3DRender.QPickEvent.ShiftModifier:
                entity.select(not entity.is_selected)
            else:
                for nodes_r in self.nodes_lut:
                    for nodes_g in nodes_r:
                        for node in nodes_g:
                            node.select(not node.is_selected and node is entity)
