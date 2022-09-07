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
    start_preview_weights = QtCore.Signal(colour.LUT3D)
    stop_preview_weights = QtCore.Signal(colour.LUT3D)

    def __init__(self, lut, parent_gui):
        super().__init__()
        self.lut = None
        self.parent_gui = parent_gui
        self.mesh_node = None
        self.picker = Qt3DRender.QObjectPicker(self)
        self.nodes_lut = None
        self.color_space: colour.models.RGB_Colourspace = None

        self.indices_node_preview_current = None

        self.load_lut(lut)

        self.start_preview_weights.connect(self.parent_gui.gui_parent.widget_menu.start_update_image)
        self.stop_preview_weights.connect(self.parent_gui.gui_parent.widget_menu.start_update_image)

        self.parent_gui.gui_parent.widget_menu.select_nodes_affecting_pixel.connect(self.select_nodes_by_source_colour_affecting)
        self.parent_gui.gui_parent.widget_menu.select_node_closest_pixel.connect(self.select_nodes_by_source_colour_closest)

        # self.time_last_change = datetime.now()
        # self.timedelta_update = timedelta(milliseconds=100)

        # TODO: find a better way to block lut calc....
        #   use qtimer as long as dragging, so that update is performed on regular intervals?

        # self.picker.entered.connect(self.slot_start_preview_weights)

        # self.root_entity = None


    def get_coordinates_lut_source(self, lut: colour.LUT3D):
        values_r_source = np.linspace(lut.domain[0,0], lut.domain[1,0], lut.size)
        values_g_source = np.linspace(lut.domain[0,1], lut.domain[1,1], lut.size)
        values_b_source = np.linspace(lut.domain[0,2], lut.domain[1,2], lut.size)

        return values_r_source, values_g_source, values_b_source



    # def load_lut_file(self, filepath):
    #     lut =

    def find_nodes_influencing_pixel(self, coordinates_pixel: QVector3D):
        result = []

        max_distances = (self.lut.domain[1] - self.lut.domain[0]) / (self.lut.size - 1)

        def fn(node: NodeLut):
            node_affects_pixel = np.all(
                np.abs(
                    (node.coordinates_source - coordinates_pixel).toTuple()
                ) < max_distances
            )
            if node_affects_pixel:
                result.append(node)

        self.iter_nodes(fn)

        return result

    def find_nearest_node_pixel(self, coordinates_pixel: QVector3D):
        distance_min = [np.inf]
        result = [None]

        def fn(node: NodeLut):
            distance = coordinates_pixel.distanceToPoint(node.coordinates_source)

            if distance < distance_min[0]:
                result[0] = node
                distance_min[0] = distance

        self.iter_nodes(fn)

        return result[0]


    def load_lut(self, lut: colour.LUT3D):
        if np.any(lut.domain != np.asarray([[0, 0, 0], [1, 1, 1]])):
            raise NotImplementedError

        self.lut = lut

        # TODO: 2 textures: source and target, that can be switched
        # TODO: Color map from lut space to display srgb

        coordinates_r_source, coordinates_g_source, coordinates_b_source = self.get_coordinates_lut_source(lut)

        radius = np.min(lut.domain[1] - lut.domain[0]) / lut.size / 5

        color_max= 255
        self.mesh_node = Qt3DExtras.QSphereMesh(rings=8, slices=8, radius=radius)


        nodes_lut = []
        for idx_r, value_r_source in enumerate(coordinates_r_source):
            nodes_r = []
            for idx_g, value_g_source in enumerate(coordinates_g_source):
                nodes_g = []
                for idx_b, value_b_source in enumerate(coordinates_b_source):
                    entity_node = NodeLut(
                        (idx_r, idx_g, idx_b),
                        QVector3D(
                            lut.table[idx_r, idx_g, idx_b, 0],
                            lut.table[idx_r, idx_g, idx_b, 1],
                            lut.table[idx_r, idx_g, idx_b, 2],
                        ),
                        QVector3D(
                            value_r_source,
                            value_g_source,
                            value_b_source,
                        ),
                        radius,
                        self
                    )
                    entity_node.picker.clicked.connect(self.slot_clicked)
                    # TODO: use one picker for all nodes...
                    # TODO: only enable hover whole holding shift?
                    entity_node.picker.setHoverEnabled(True)
                    # entity_node.mouse_hover_start.connect(self.parent_gui.gui_parent.widget_menu.slot_hover_node_start)
                    # entity_node.mouse_hover_stop.connect(self.parent_gui.gui_parent.widget_menu.slot_hover_node_stop)

                    # entity_node.mouse_hover_start.connect()
                    entity_node.mouse_hover_start.connect(self.slot_start_preview_weights)
                    entity_node.mouse_hover_stop.connect(self.slot_stop_preview_weights)

                    entity_node.position_changed.connect(self.update_lut_node_changed)
                    self.parent_gui.cancel_transform.connect(entity_node.cancel_transform)
                    self.parent_gui.accept_transform.connect(entity_node.accept_transform)
                    nodes_g.append(entity_node)
                nodes_r.append(nodes_g)
            nodes_lut.append(nodes_r)
        self.nodes_lut = nodes_lut

        self.lut_changed.emit(self.lut)

    @QtCore.Slot(tuple)
    def slot_start_preview_weights(self, indices_node):
        if self.parent_gui.mode_transform_current is None:
            lut_use = colour.LUT3D(
                colour.LUT3D.linear_table(self.lut.size) ** 2
            )

            lut_use.table = np.tile(np.mean(self.lut.table, axis=3)[..., np.newaxis], (1, 1, 1, 3))
            lut_use.table[indices_node] = [1., 0., 0.]

            modifiers = QGuiApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.Modifier.SHIFT:
                def fn(node: NodeLut):
                    if node.is_selected:
                        lut_use.table[node.indices_lut] = [1., 0., 0.]
                self.iter_nodes(fn)

            # print('start')
            self.indices_node_preview_current = indices_node
            self.start_preview_weights.emit(lut_use)

    @QtCore.Slot(tuple)
    def slot_stop_preview_weights(self, indices_node):
        if indices_node == self.indices_node_preview_current:
            self.stop_preview_weights.emit(self.lut)

    @QtCore.Slot()
    def toggle_select_all(self):
        some_nodes_selected = np.any(self.iter_nodes(lambda node: node.is_selected))

        self.iter_nodes(lambda node: node.select(not some_nodes_selected))

    @QtCore.Slot(int, float)
    def transform_dragging(self, mode, distance):

        def fn(node: NodeLut):
            # TODO: calc weight based on selection center or something
            if node.is_selected:
                node.transform_dragging(mode, distance, 1.)

        # fn = node.
        self.iter_nodes(fn)

        print(f'Move {distance}')

        self.lut_changed.emit(self.lut)

        # if datetime.now() - self.time_last_change > self.timedelta_update:
        #     self.time_last_change = datetime.now()

    @QtCore.Slot()
    def reset_selected_nodes(self):
        def fn(node: NodeLut):
            # TODO: currently, coords are reset to state where lut is loaded.
            #   reset to neutral color insted?
            if node.is_selected:
                node.transform.setTranslation(node.coordinates_reset)
                node.accept_transform()

        # fn = node.
        self.iter_nodes(fn)
        self.lut_changed.emit(self.lut)

    @QtCore.Slot()
    def select_nodes_by_source_colour_affecting(self, colour_float: QVector3D, expand_selection):
        nodes = self.find_nodes_influencing_pixel(colour_float)

        self.select_nodes(nodes, expand_selection, False)

    @QtCore.Slot()
    def select_nodes_by_source_colour_closest(self, colour_float: QVector3D, expand_selection):
        node = self.find_nearest_node_pixel(colour_float)

        self.select_nodes([node], expand_selection, False)

    @QtCore.Slot(tuple, QVector3D)
    def update_lut_node_changed(self, indices_node, coordinates_node):
        self.lut.table[indices_node] = np.asarray(coordinates_node.toTuple())
        # print(self.lut.table[indices_node])

    @QtCore.Slot()
    def slot_clicked(self, event):
        node: NodeLut = event.entity()
        modifiers = event.modifiers()

        if event.button() == Qt3DRender.QPickEvent.LeftButton and self.parent_gui.mode_transform_current is None:
            self.select_nodes(
                [node],
                modifiers == Qt3DRender.QPickEvent.ShiftModifier
                or modifiers == Qt3DRender.QPickEvent.ShiftModifier + Qt3DRender.QPickEvent.ControlModifier,
                True
            )

    def iter_nodes(self, fn, *args, **kwargs):
        results = []
        for nodes_r in self.nodes_lut:
            results_r = []
            for nodes_g in nodes_r:
                results_g = []
                for node in nodes_g:
                    results_g.append(fn(node, *args, **kwargs))
                results_r.append(results_g)
            results.append(results_r)
        return results

    def select_nodes(self, nodes, expand_selection, deselect_selected):
        if expand_selection:
            [node.select(not node.is_selected or not deselect_selected) for node in nodes]
        else:
            fn = lambda node: node.select((not node.is_selected or not deselect_selected) and node in nodes)
            self.iter_nodes(fn)