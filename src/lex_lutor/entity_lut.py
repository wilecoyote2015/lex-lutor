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
from lex_lutor.constants import HSV, HSL, HCL, color_spaces_components_transform, KEY_EXPOSURE
from datetime import datetime

from varname import nameof
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


class WorkerTransform(QObject):
    finished = Signal(list, np.ndarray)
    progress = Signal(int)

    def __init__(self, lut, mode, distance, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lut = lut
        self.mode = mode
        self.distance = distance

    def run(self, ):
        # TODO: Performance! This should also be threaded and use quere, as many transforms are triggered during
        #   dragging and block UI, especially if many nodes are selected.

        # TODO: Transform must be calculated here in vectorized form for all selected nodes.
        #   Calculating per node is too slow.

        # TODO: Exposure. For this, first transformation must be into linear RGB.
        #   But how is exposure calculated then? Effect of exposure must not depend on linear color space
        #   choice. How is the transfer function handled in colour?

        # TODO: Linear space if upper case
        t_0 = datetime.now()
        color_space_transform, dimension_transform = color_spaces_components_transform[self.mode]
        t_trafo_1 = datetime.now() - t_0

        # TODO: adapt below to vectorized
        # if (color_space_transform in (HSV, HSL, HCL) and dimension_transform in [0, 1]
        #         and self.coordinates_current.x() == self.coordinates_current.y() == self.coordinates_current.z()):
        #     # If target compomnent is related to color, but current node has no color, then nothing to do.
        #     return

        # TODO: get weights from nodes.

        try:
            def fn(node, result_):
                if node.is_selected:
                    result_.append(node)

            nodes_transform = self.lut.iter_nodes(fn)

            coordinates_nodes_current = np.asarray([node.coordinates_current.toTuple() for node in nodes_transform])
            t_coords_current = datetime.now() - t_0 - t_trafo_1

            # TODO
            weights = np.ones((coordinates_nodes_current.shape[0],))

            # Distance for each node
            distance_weighted = self.distance * weights

            coords_current_target_space = self.lut.transform_color_space(
                self.lut.color_space,
                color_space_transform,
                coordinates_nodes_current
            )
            t_coords_current_target_space = datetime.now() - t_coords_current - t_0 - t_trafo_1

            components_vector_add = np.asarray([1. if idx_ == dimension_transform else 0. for idx_ in range(3)])
            coords_new_target_space = coords_current_target_space + components_vector_add[np.newaxis, ...] * \
                                      distance_weighted[..., np.newaxis]
            # print(coords_new_target_space.toTuple())
            # TODO: respect domain!
            if color_space_transform in (HSV, HSL, HCL) and dimension_transform == 0:
                # print(coords_new_target_space.x())
                coords_new_target_space[..., 0] = np.mod(coords_new_target_space[..., 0], 1.)
            elif color_space_transform == HCL and dimension_transform == 1:
                # print(coords_new_target_space)
                # TODO: WRONG!
                coords_new_target_space[..., 1] = np.clip(coords_new_target_space[..., 0], 0, 2 / 3)
            elif color_space_transform == HSL and dimension_transform == 2:
                coords_new_target_space[..., 2] = self.lut.clip_l(coords_new_target_space)
                # coords_new_target_space.setZ(self.clip_l(*coords_new_target_space.toTuple()))
                pass
            else:
                coords_new_target_space = np.clip(coords_new_target_space, 0, 1)
            t_clip = datetime.now() - t_0 - t_coords_current - t_trafo_1 - t_coords_current_target_space

            # transform to target color space, modify the according component and then
            coords_new = self.lut.transform_color_space(
                color_space_transform,
                self.lut.color_space,
                coords_new_target_space
            )
            t_coords_new = datetime.now() - t_0 - t_coords_current - t_trafo_1 - t_coords_current_target_space - t_clip

            # TODO: clip to borders
            # TODO: Clipping must be reflected in the trafo fn, as it must handle color space correctly (
            #  e.g. pertain hue on clipping)
            # coors_new = np.reshape(coords_new, (self.lut.size, self.lut.size, self.lut.size, 3))

            self.finished.emit(nodes_transform, coords_new)



        except Exception as e:
            print(f'Error during transformation of node: \n {e}')


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

        # Nodes that represent the base selection before deriving by expansion by radius, hue etc.
        #   This selection contains all nodes that are either clicked directly or are picked in image preview.
        #   TODO: Draw them in other color than derived selection
        self.nodes_selection_base = set()

        self.indices_node_preview_current = None

        self.load_lut(lut)

        self.start_preview_weights.connect(self.parent_gui.gui_parent.widget_menu.start_update_image)
        self.stop_preview_weights.connect(self.parent_gui.gui_parent.widget_menu.start_update_image)

        self.parent_gui.gui_parent.widget_menu.select_nodes_affecting_pixel.connect(
            self.select_nodes_by_source_colour_affecting)
        self.parent_gui.gui_parent.widget_menu.select_node_closest_pixel.connect(
            self.select_nodes_by_source_colour_closest)

        self.parent_gui.cancel_transform.connect(self.cancel_transform)
        self.parent_gui.accept_transform.connect(self.accept_transform)

        self.preview_weights_on = False

        self.queue_updates_transform = []

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

        def fn(node: NodeLut, result_):
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

        def fn(node: NodeLut, result_):
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

                    # entity_node.position_changed.connect(self.update_lut_node_changed)
                    # self.parent_gui.cancel_transform.connect(entity_node.cancel_transform)
                    # self.parent_gui.accept_transform.connect(entity_node.accept_transform)
                    nodes_g.append(entity_node)
                nodes_r.append(nodes_g)
            nodes_lut.append(nodes_r)
        self.nodes_lut = nodes_lut

        self.lut_changed.emit(self.lut)

    @QtCore.Slot()
    def accept_transform(self):
        def fn(node, _):
            node.accept_transform()
            self.lut.table[node.indices_lut] = np.asarray(node.transform.translation().toTuple())

        self.iter_nodes(fn)

        self.lut_changed.emit(self.lut)

    @QtCore.Slot()
    def cancel_transform(self):
        def fn(node, _):
            node.cancel_transform()
            self.lut.table[node.indices_lut] = np.asarray(node.coordinates_current.toTuple())

        self.iter_nodes(fn)

        self.lut_changed.emit(self.lut)

    @QtCore.Slot(tuple)
    def slot_start_preview_weights(self, indices_node):
        if self.parent_gui.mode_transform_current is None:
            self.preview_weights_on = True
            lut_use = colour.LUT3D(
                colour.LUT3D.linear_table(self.lut.size) ** 2
            )

            lut_use.table = np.tile(np.mean(self.lut.table, axis=3)[..., np.newaxis], (1, 1, 1, 3))
            lut_use.table[indices_node] = [1., 0., 0.]

            modifiers = QGuiApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.Modifier.SHIFT:
                def fn(node: NodeLut, result_):
                    if node.is_selected:
                        lut_use.table[node.indices_lut] = [1., 0., 0.]

                self.iter_nodes(fn)

            # print('start')
            self.indices_node_preview_current = indices_node
            self.start_preview_weights.emit(lut_use)

    @QtCore.Slot()
    def select_nodes_derived(self, range_h, range_s, range_v, range_c, range_l):
        # each range is [min, max], where min must be negative and max must be positive.
        # Pixels in dense grid covering the selection range around each node's base coordinates in base selection
        # TODO: not all dimensions are orthogonal, as different color spaces are used!
        #   how to deal with this? E.g. if bandwidth of l is 0 but v is > 0, no nodes except base
        #   would be selected, as resulting ser of possible colors would be intersection of hsl and hsv subsets.
        #   maybe, use union instead of intersection? Does this make sense?
        #   then, the evenly spaced dummy pixels would be calculated first for hsv, then for hcl and then merged.
        pixels_dummy = []
        for node in self.nodes_selection_base:

    # get h, s, v, c, l coordinates of node by transforming from lut color space to hsv and hcl

    # get 3d grid of hsv dummy pixel values

    # get 3d grid of hcl dummy pixel values

    # backtransform both dummy pixel arrays to lut space

    # merge the arrays

    # select nearest for all pixels

    @QtCore.Slot(tuple)
    def slot_stop_preview_weights(self, indices_node):
        if self.preview_weights_on:
            if indices_node == self.indices_node_preview_current:
                self.preview_weights_on = False
                self.stop_preview_weights.emit(self.lut)

    @QtCore.Slot()
    def toggle_select_all(self):
        some_nodes_selected = np.any(self.iter_nodes(lambda node, result_: result_.append(node.is_selected)))

        nodes = self.iter_nodes(lambda node, result_: result_.append(node))

        if some_nodes_selected:
            self.deselect_nodes(nodes)
        else:
            self.select_nodes(nodes, False, False)

    @QtCore.Slot()
    def reset_selected_nodes(self):
        def fn(node: NodeLut, result_):
            # TODO: currently, coords are reset to state where lut is loaded.
            #   reset to neutral color insted?
            if node.is_selected:
                node.transform.setTranslation(node.coordinates_reset)
                node.accept_transform()
                self.lut.table[node.indices_lut] = np.asarray(node.coordinates_current.toTuple())

        # fn = node.
        self.iter_nodes(fn)
        self.lut_changed.emit(self.lut)

    def transform_color_space(self, color_space_source, color_space_target, input_array: np.ndarray):
        # TODO: ensuse vectorization (value_input must be ndarray) and move to entity_lut.
        if color_space_target == color_space_source or color_space_target is None or color_space_source is None:
            return input_array

        if color_space_source in (HSV, HSL, HCL):
            if color_space_target in (HSV, HSL, HCL):
                result = getattr(colour, f'{color_space_source}_to_{color_space_target}')(input_array)
            elif isinstance(color_space_target, colour.models.RGB_Colourspace):
                result = getattr(colour, f'{color_space_source}_to_RGB')(input_array)
            else:
                raise NotImplementedError
        elif isinstance(color_space_source, colour.models.RGB_Colourspace):
            if color_space_target in (HSV, HSL, HCL):
                result = getattr(colour, f'RGB_to_{color_space_target}')(input_array)

            elif isinstance(color_space_target, colour.models.RGB_Colourspace):
                result = colour.RGB_to_RGB(input_array, color_space_source, color_space_target)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return result

    @QtCore.Slot()
    def start_next_update_transform_dragging(self):
        if self.queue_updates_transform and not self.queue_updates_transform[-1][0].isFinished():
            self.queue_updates_transform = self.queue_updates_transform[-1:]
            self.queue_updates_transform[-1][0].start()
        else:
            self.queue_updates_transform = []

    @QtCore.Slot(int, float)
    def transform_dragging(self, mode, distance):
        # TODO: quit running thread.
        thread, worker = QtCore.QThread(), WorkerTransform(self, mode, distance)
        self.queue_updates_transform.append((thread, worker, str(id(worker))))

        worker.moveToThread(thread)
        worker.finished.connect(thread.quit)
        # worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self.apply_transform_dragging)
        # thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.start_next_update_transform_dragging)
        thread.started.connect(worker.run)

        if len(self.queue_updates_transform) == 1:
            self.start_next_update_transform_dragging()

    @QtCore.Slot(list, np.ndarray)
    def apply_transform_dragging(self, nodes, coordinates):
        for idx, node in enumerate(nodes):
            node.transform.setTranslation(
                QVector3D(*coordinates[idx])
            )
            self.lut.table[node.indices_lut] = coordinates[idx]

        self.lut_changed.emit(self.lut)

    def clip_l(self, coordinates_hsl):
        '''

                    C = (1-|2L-1|)S
                    m = L - C/2
                    C + m = (1 - |2L-1|)S / 2 + L

                    C- = (1 + 2L - 1) * S       ( L <= 0.5)
                    C+ = (1 - 2L + 1) * S       ( L > 0.5)


                    fall L <= 0.5:
                        C+m =  (1 + 2L - 1)S/2 +L = L * S + L
                            = L(1+S)
                    fall L > 0.5:
                        C+m =  (1 - 2L + 1)S/2 +L
                            = (2 - 2L) * S/2 + L
                            = (1 - L) * S + L
                            = S - LS + L
                            = S + (1-S) * L


                    A = 1 - |(H/60) % 2 - 1|
                    B = S * A
                    Z = S * (A - 1/2)
                    X = A * C
                    X + m = A * C - C/2 + L
                          = C (A - 1/2) + L
                    fall L <= 0.5:
                        X + m =  C- (A - 1/2) + L
                              = (1 + 2L - 1) * S * (A - 1/2) + L
                              = (1 + 2L - 1) * Z + L
                              = 2L * Z + L
                              = L ( 2 * Z + 1)
                    fall L > 0.5:
                        X+m = C+ (A - 1/2) + L
                            = (1 - 2L + 1) * Z + L
                            = (2 - 2L) * Z + L
                            = 2Z - 2LZ + L
                            = 2Z + L(1-2Z)

                    => C+m=0 wenn:num
                        L = 0
                        L = -S / (1-S) und L > 0.5 (L must be >= -S / (1+S) WHICH IS ALWAYS TRUE)

                    => X+m = 0 wenn:
                        L = 0
                        # ATTENTION: Hue is in range 0-1 instead of degree in colour! convert to degree!
                        L = - 2Z / (1-2Z) and L > 0.5 ( L darf nicht kleiner werden!)

                    => C+m=1 wenn:
                        L = 1 / (1+S) and L <= 0.5 (L darf nicht groesser werden)
                        L = 1 and L > 0.5 (L darf nicht griesser werden

                    => X+m = 1 wenn:
                        L = 1 / ( 2Z + 1) and L <= 0.5
                        L = (1-2Z) / (1-2Z) = 1  and L > 0.5  => NIE! (L darf nicht groesser werden)
                    '''

        h, s, l = coordinates_hsl[..., 0], coordinates_hsl[..., 1], coordinates_hsl[..., 2]

        h_deg = h * 360
        A = 1 - abs(np.mod(h_deg / 60, 2) - 1)
        Z = s * (A - 1 / 2)

        # TODO: vectorization!
        l_low = l <= 0.5
        lower = np.where(l_low, 0., np.maximum(
            -s / (1 - s),
            -2 * Z / (1 - 2 * Z)
        ))
        upper = np.where(l_low, np.minimum(
            1.,
            1 / (1 + s)
        ), np.minimum(
            1.,
            1.
        ))

        return np.clip(l, lower, upper)

    @QtCore.Slot()
    def select_nodes_by_source_colour_affecting(self, colour_float: QVector3D, expand_selection):
        # TODO: accept list of colours and vectorize
        nodes = self.find_nodes_influencing_pixel(colour_float)

        self.select_nodes(nodes, expand_selection, False)

    @QtCore.Slot()
    def select_nodes_by_source_colour_closest(self, colour_float: QVector3D, expand_selection):
        # TODO: accept list of colours and vectorize
        node = self.find_nearest_node_pixel(colour_float)

        self.select_nodes([node], expand_selection, False)

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
            for nodes_g in nodes_r:
                for node in nodes_g:
                    fn(node, results, *args, **kwargs)
        return results

    def select_nodes(self, nodes, expand_selection, deselect_selected, change_base_selection=True):
        if expand_selection:
            for node in nodes:
                if not node.is_selected or not deselect_selected:
                    node.select(True)
                    if change_base_selection:
                        self.nodes_selection_base.add(node)
                else:
                    node.select(False)
                    if change_base_selection:
                        self.remove_node_from_base_selection(node)
            # [node.select(not node.is_selected or not deselect_selected) for node in nodes]
        else:
            def fn(node, _):
                if (not node.is_selected or not deselect_selected) and node in nodes:
                    node.select(True)
                    if change_base_selection:
                        self.nodes_selection_base.add(node)
                else:
                    node.select(False)
                    if change_base_selection:
                        self.remove_node_from_base_selection(node)

            # fn = lambda node, _: node.select((not node.is_selected or not deselect_selected) and node in nodes)
            self.iter_nodes(fn)

    def deselect_nodes(self, nodes):
        for node in nodes:
            node.select(False)
            self.remove_node_from_base_selection(node)

    def remove_node_from_base_selection(self, node):
        if node in self.nodes_selection_base:
            self.nodes_selection_base.remove(node)
