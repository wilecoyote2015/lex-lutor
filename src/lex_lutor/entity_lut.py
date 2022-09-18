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
from lex_lutor.job_queue import JobQueue

from varname import nameof
# from PySide6 import Qt3DCore, Qt3DExtras, Qt3DInput, Qt3DRender
import sys
from lex_lutor.node_lut import NodeLut
from datetime import datetime, timedelta

# TODO: shortcut to toggle node hover selection mode between derived and non-derived.

# TODO: Handle press / release of shift while in preview

# TODO: While dragging sliders, update selection preview in 2d gui

# TODO: selection preview on node hover should take derived selection into account.
#   But for this, derived selection must become much faster.

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

        # try:
        def fn(node, result_):
            if node.is_selected:
                result_.append((node.indices_lut, node.coordinates_current.toTuple(), node.weight_selection))

        # print(self.lut.iter_nodes(fn))

        indices_nodes = []
        coordinates_nodes_current = []
        weights_nodes = []

        for indices_node, coordiantes_node, weight_node in self.lut.iter_nodes(fn):
            indices_nodes.append(indices_node)
            coordinates_nodes_current.append(coordiantes_node)
            weights_nodes.append(weight_node)

        # indices_nodes, coordinates_nodes_current, weights_nodes = zip(self.lut.iter_nodes(fn))

        if not indices_nodes:
            self.finished.emit([], [])

        coordinates_nodes_current = np.asarray(coordinates_nodes_current)
        # t_coords_current = datetime.now() - t_0 - t_trafo_1

        # TODO
        # print(weights_nodes)
        # print(self.distance)
        weights = np.asarray(weights_nodes)

        # Distance for each node
        distance_weighted = self.distance * weights

        coords_current_target_space = self.lut.transform_color_space(
            self.lut.color_space,
            color_space_transform,
            coordinates_nodes_current
        )
        # t_coords_current_target_space = datetime.now() - t_coords_current - t_0 - t_trafo_1

        components_vector_add = np.asarray([1. if idx_ == dimension_transform else 0. for idx_ in range(3)])
        coords_new_target_space = coords_current_target_space + components_vector_add[np.newaxis, ...] * \
                                  distance_weighted[..., np.newaxis]
        # print(coords_new_target_space.toTuple())
        # TODO: respect domain!
        if color_space_transform in (HSV, HSL, HCL) and dimension_transform == 0:
            # print(coords_new_target_space.x())
            coords_new_target_space[..., 0] = np.mod(coords_new_target_space[..., 0], 1.)
        elif color_space_transform == HCL and dimension_transform == 1:
            # FIXME: This is wrong.
            coords_new_target_space[..., 1] = np.clip(coords_new_target_space[..., 0], 0, 2 / 3)
        elif color_space_transform == HSL and dimension_transform == 2:
            coords_new_target_space[..., 2] = self.lut.clip_l(coords_new_target_space)
            # coords_new_target_space.setZ(self.clip_l(*coords_new_target_space.toTuple()))
            pass
        else:
            coords_new_target_space = np.clip(coords_new_target_space, 0, 1)
        # t_clip = datetime.now() - t_0 - t_coords_current - t_trafo_1 - t_coords_current_target_space

        # transform to target color space, modify the according component and then
        coords_new = self.lut.transform_color_space(
            color_space_transform,
            self.lut.color_space,
            coords_new_target_space
        )
        # t_coords_new = datetime.now() - t_0 - t_coords_current - t_trafo_1 - t_coords_current_target_space - t_clip

        # TODO: clip to borders
        # TODO: Clipping must be reflected in the trafo fn, as it must handle color space correctly (
        #  e.g. pertain hue on clipping)
        # coors_new = np.reshape(coords_new, (self.lut.size, self.lut.size, self.lut.size, 3))

        self.finished.emit(indices_nodes, coords_new)

        # except Exception as e:
        #     print(f'Error during transformation of node: \n {e}')


class WorkerGetNodesSelectionDerived(QObject):
    finished = Signal(list, list)
    progress = Signal(int)

    def __init__(self, lut, nodes_selection_base, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lut = lut
        self.nodes_selection_base = nodes_selection_base

    def run(self, ):
        nodes, weights = self.lut.get_nodes_select_derived(self.nodes_selection_base)
        self.finished.emit([node.indices_lut for node in nodes], weights)


class WorkerGetNodesSelectionDerivedHover(QObject):
    finished = Signal(list, list)
    progress = Signal(int)

    def __init__(self, lut, nodes_selection_base, nodes_other, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lut = lut
        self.nodes_selection_base = nodes_selection_base
        self.nodes_other = nodes_other
        # print(nodes_selection_base)
        # print(nodes_other)

    def run(self, ):
        # TODO / FIXME: handle double nodes! effective weight must be max!
        nodes_select_derived, weights_select_derived = self.lut.get_nodes_select_derived(self.nodes_selection_base)
        nodes_select_derived.extend(self.nodes_selection_base)
        weights_select_derived.extend([1.] * len(self.nodes_selection_base))
        nodes_select_derived.extend(self.nodes_other)
        weights_select_derived.extend([node.weight_selection for node in self.nodes_other])
        self.finished.emit(
            nodes_select_derived,
            weights_select_derived
        )


class Lut3dEntity(Qt3DCore.QComponent):
    lut_changed = QtCore.Signal(colour.LUT3D)
    start_preview_weights = QtCore.Signal(colour.LUT3D)
    stop_preview_weights = QtCore.Signal(colour.LUT3D)
    selection_base_changed = Signal()

    def __init__(self, lut, parent_gui):
        super().__init__()
        self.lut = None
        self.parent_gui = parent_gui
        self.mesh_node = None
        self.picker = Qt3DRender.QObjectPicker(self)
        self.nodes_lut = None
        self.color_space = self.parent_gui.gui_parent.widget_menu.color_space_lut
        # Nodes that represent the base selection before deriving by expansion by radius, hue etc.
        #   This selection contains all nodes that are either clicked directly or are picked in image preview.
        #   TODO: Draw them in other color than derived selection

        self.indices_node_preview_current = None

        self.load_lut(lut)

        self.start_preview_weights.connect(self.parent_gui.gui_parent.widget_menu.start_update_image)
        self.stop_preview_weights.connect(self.parent_gui.gui_parent.widget_menu.start_update_image)

        self.parent_gui.gui_parent.widget_menu.select_nodes_affecting_pixel.connect(
            self.select_nodes_by_source_colour_affecting_base)
        self.parent_gui.gui_parent.widget_menu.select_node_closest_pixel.connect(
            self.select_nodes_by_source_colour_closest_base)

        self.parent_gui.cancel_transform.connect(self.cancel_transform)
        self.parent_gui.accept_transform.connect(self.accept_transform)

        # TODO: this should be GUI code
        self.preview_weights_on_ = False
        self.preview_weights_always_on = False

        self.queue_updates_transform = JobQueue(
            WorkerTransform,
            self.apply_calculated_transorm_to_nodes_dragging_change
        )

        self.queue_nodes_derived_selection = JobQueue(
            WorkerGetNodesSelectionDerived,
            self.apply_selection_nodes_derived
        )

        self.queue_nodes_derived_selection_hover_pixel = JobQueue(
            WorkerGetNodesSelectionDerivedHover,
            self.start_preview_hover_pixel
        )

        self.hovering_pixels = False

        self.selection_base_changed.connect(self.select_nodes_derived)

        self.parent_gui.gui_parent.widget_menu.slider_h.valueChanged.connect(self.select_nodes_derived)
        self.parent_gui.gui_parent.widget_menu.slider_s_hsv.valueChanged.connect(self.select_nodes_derived)
        self.parent_gui.gui_parent.widget_menu.slider_v.valueChanged.connect(self.select_nodes_derived)
        self.parent_gui.gui_parent.widget_menu.slider_s_hsl.valueChanged.connect(self.select_nodes_derived)
        self.parent_gui.gui_parent.widget_menu.slider_l.valueChanged.connect(self.select_nodes_derived)

        self.parent_gui.gui_parent.widget_menu.slider_h.sliderPressed.connect(self.slot_start_preview_selection_slider)
        self.parent_gui.gui_parent.widget_menu.slider_s_hsv.sliderPressed.connect(
            self.slot_start_preview_selection_slider)
        self.parent_gui.gui_parent.widget_menu.slider_v.sliderPressed.connect(self.slot_start_preview_selection_slider)
        self.parent_gui.gui_parent.widget_menu.slider_s_hsl.sliderPressed.connect(
            self.slot_start_preview_selection_slider)
        self.parent_gui.gui_parent.widget_menu.slider_l.sliderPressed.connect(self.slot_start_preview_selection_slider)

        self.parent_gui.gui_parent.widget_menu.slider_h.sliderReleased.connect(self.slot_stop_preview_selection_slider)
        self.parent_gui.gui_parent.widget_menu.slider_s_hsv.sliderReleased.connect(
            self.slot_stop_preview_selection_slider)
        self.parent_gui.gui_parent.widget_menu.slider_v.sliderReleased.connect(self.slot_stop_preview_selection_slider)
        self.parent_gui.gui_parent.widget_menu.slider_s_hsl.sliderReleased.connect(
            self.slot_stop_preview_selection_slider)
        self.parent_gui.gui_parent.widget_menu.slider_l.sliderReleased.connect(self.slot_stop_preview_selection_slider)

        self.parent_gui.gui_parent.widget_menu.preview_pixel_hovered.connect(self.preview_selection_pixel)
        self.parent_gui.gui_parent.widget_menu.stop_preview_pixel_hovered.connect(self.slot_stop_hover_pixel)

        self.parent_gui.gui_parent.widget_menu.color_space_lut_changed.connect(self.set_color_space)

        self.lut_changed.emit(self.lut)

        # self.time_last_change = datetime.now()
        # self.timedelta_update = timedelta(milliseconds=100)

        # TODO: find a better way to block lut calc....
        #   use qtimer as long as dragging, so that update is performed on regular intervals?

        # self.picker.entered.connect(self.slot_start_preview_weights)

        # self.root_entity = None

    @property
    def preview_weights_on(self):
        return self.preview_weights_on_ or self.preview_weights_always_on

    @property
    def coordinates_lut_source(self):
        lut = self.lut
        values_r_source = np.linspace(lut.domain[0, 0], lut.domain[1, 0], lut.size)
        values_g_source = np.linspace(lut.domain[0, 1], lut.domain[1, 1], lut.size)
        values_b_source = np.linspace(lut.domain[0, 2], lut.domain[1, 2], lut.size)

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

    @QtCore.Slot()
    def set_color_space(self, color_space):
        self.color_space = color_space
        self.select_nodes_derived()

    @property
    def nodes_selection_base(self):
        def fn(node, result_):
            if node.is_selected_base:
                result_.append(node)

        return self.iter_nodes(fn)

    @property
    def nodes_selection(self):
        def fn(node, result_):
            if node.is_selected:
                result_.append(node)

        return self.iter_nodes(fn)

    def find_nearest_nodes_pixels(self, coordinates_pixels: np.ndarray):
        coordinate_axes_source = np.stack(self.coordinates_lut_source, axis=-1)

        indices_nearest = np.argmin(
            np.abs(
                coordinates_pixels[:, np.newaxis, ...] - coordinate_axes_source[np.newaxis, ...]
            ),
            axis=1
        )
        indices_nearest_unique = np.unique(indices_nearest, axis=0)

        nodes = [self.nodes_lut[indices[0]][indices[1]][indices[2]] for indices in indices_nearest_unique]

        return nodes


    def load_lut(self, lut: colour.LUT3D):
        if np.any(lut.domain != np.asarray([[0, 0, 0], [1, 1, 1]])):
            raise NotImplementedError

        self.lut = lut

        # TODO: 2 textures: source and target, that can be switched
        # TODO: Color map from lut space to display srgb

        coordinates_r_source, coordinates_g_source, coordinates_b_source = self.coordinates_lut_source

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
                    entity_node.mouse_hover_start.connect(self.slot_start_hover_node)
                    entity_node.mouse_hover_stop.connect(self.slot_stop_hover_node)

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
        if self.preview_weights_on:
            self.start_preview_with_selected_nodes()

    @QtCore.Slot()
    def cancel_transform(self):
        def fn(node, _):
            node.cancel_transform()
            self.lut.table[node.indices_lut] = np.asarray(node.coordinates_current.toTuple())

        self.iter_nodes(fn)

        self.lut_changed.emit(self.lut)
        if self.preview_weights_on:
            self.start_preview_with_selected_nodes()

    @property
    def indices_nodes_selected(self):
        def fn(node: NodeLut, result_):
            if node.is_selected:
                result_.append(node.indices_lut)

        return self.iter_nodes(fn)

    def make_lut_preview_selection(self, nodes, weigths_nodes):
        lut_use = colour.LUT3D(
            colour.LUT3D.linear_table(self.lut.size) ** 2
        )
        lut_use.table = np.tile(np.mean(self.lut.table, axis=3)[..., np.newaxis], (1, 1, 1, 3))
        for node, weight in zip(nodes, weigths_nodes):
            # print(weight)
            lut_use.table[node.indices_lut] = weight * np.asarray([1., 0., 0.]) + (1 - weight) * lut_use.table[
                node.indices_lut]

        return lut_use

    @QtCore.Slot(tuple)
    def slot_start_hover_node(self, indices_node):
        # TODO: consider derived selection:
        #   build simulated nodes of derived selection from nodes_preview and show them.

        # TODO: use worker queue for threading
        if self.parent_gui.mode_transform_current is None:
            self.preview_weights_on_ = True

            # indices_nodes_preview = [indices_node]
            node_hovered = self.nodes_lut[indices_node[0]][indices_node[1]][indices_node[2]]
            nodes_preview = [node_hovered]
            weights_nodes_preview = [1.]
            nodes_selection_derived, weights_nodes_selection_derived = self.get_nodes_select_derived(nodes_preview)

            nodes_preview.extend(nodes_selection_derived)
            weights_nodes_preview.extend(weights_nodes_selection_derived)

            # indices_nodes_preview.extend([node.indices_lut for node in nodes_selection_derived])

            modifiers = QGuiApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.Modifier.SHIFT:
                # TODO / FIXME: preview weight for already selected nodes must be max of already selected weight and new weight!!!
                nodes_preview.extend(self.nodes_selection)
            lut_use = self.make_lut_preview_selection(nodes_preview, weights_nodes_preview)

            # print('start')
            self.indices_node_preview_current = indices_node
            self.start_preview_weights.emit(lut_use)

    def start_preview_with_selected_nodes(self):
        nodes_selection = self.nodes_selection
        lut_use = self.make_lut_preview_selection(self.nodes_selection,
                                                  [node.weight_selection for node in nodes_selection])
        self.start_preview_weights.emit(lut_use)

    @QtCore.Slot()
    def slot_start_preview_selection_slider(self):
        self.preview_weights_on_ = True
        self.start_preview_with_selected_nodes()

    @QtCore.Slot()
    def slot_stop_preview_selection_slider(self):
        self.preview_weights_on_ = False
        self.stop_preview_weights_node_if_not_always_on()

    @property
    def size(self):
        return self.lut.size

    @QtCore.Slot()
    def toggle_preview_selection_always_on(self):
        if self.preview_weights_always_on:
            self.preview_weights_always_on = False
            self.stop_preview_weights.emit(self.lut)

        else:
            self.preview_weights_always_on = True
            self.start_preview_with_selected_nodes()

    def get_weights_distances(self, coords_base, coords_other, range_distance, index_hue=None):
        diff_min_max = range_distance[:, 1] - range_distance[:, 0]
        # TODO / FIXME: Handle nans in coordinates. Nans can occur e.g. for colorless nodes
        #   for hue or chroma (?) How shall those influence the calculated weights?
        #   Should they simply not influence weight calculation?

        abs_distances = np.abs(
            coords_base[:, np.newaxis, ...] - coords_other[np.newaxis, ...]
        )

        # hue is cyclic in [0,1].
        #   so, if distance is > 0.5, it is actually 1 - distance
        #   scale by 2 to norm distance to [0,1]
        if index_hue is not None:
            distance_hue_larger_05 = abs_distances[..., index_hue] > 0.5
            abs_distances[..., index_hue] = np.where(
                distance_hue_larger_05,
                -abs_distances[..., index_hue] + 1,
                abs_distances[..., index_hue],
            )
            # abs_distances[..., index_hue][distance_hue_larger_05] = 1. - abs_distances[..., index_hue][distance_hue_larger_05]
            # abs_distances[..., index_hue] = abs_distances[..., index_hue] * 2
        min_distances = np.min(
            abs_distances,
            axis=0
        )
        # FIXME: account for 0 in diff_min_max!
        # TODO / FIXME: Still all wrong!
        is_core = (min_distances <= range_distance[np.newaxis, :, 0]).astype(min_distances.dtype)
        tails = 1. - np.clip(
            (min_distances - range_distance[np.newaxis, :, 0]) / diff_min_max[np.newaxis, ...],
            0.,
            1.
        )
        # TODO: better handling of division by zero.
        tails[np.isnan(tails)] = is_core[np.isnan(tails)]
        weights_separate = tails

        weights = np.prod(weights_separate, axis=1)

        return weights

    def get_nodes_select_derived(self, nodes_selection_base):
        gui_2d = self.parent_gui.gui_parent.widget_menu
        band_h = gui_2d.slider_h.value()
        band_s_hsv = gui_2d.slider_s_hsv.value()
        band_v = gui_2d.slider_v.value()
        band_s_hsl = gui_2d.slider_s_hsl.value()
        band_l = gui_2d.slider_l.value()

        # Nothing to do if all nodes are already in base selection.
        if len(nodes_selection_base) == self.size ** 3 or not nodes_selection_base or band_h == (
        0, 0) and band_s_hsv == (0, 0) and band_v == (0, 0) and band_s_hsl == (0, 0) and band_l == (0, 0):
            return [], []
        else:
            coords_nodes_base = np.asarray(
                [node.coordinates_source.toTuple() for node in nodes_selection_base
                 ], dtype=np.float32)

            coords_nodes_other = np.asarray(self.iter_nodes(lambda node, result: result.append(
                node.coordinates_source.toTuple()) if not node.is_selected_base else None), dtype=np.float32)

            nodes_other = self.iter_nodes(
                lambda node, result: result.append(node) if not node.is_selected_base else None)

            range_hsv = np.asarray((band_h, band_s_hsv, band_v))
            range_hsl = np.asarray((band_h, band_s_hsl, band_l))

            coords_nodes_base_hsv = self.transform_color_space(
                self.color_space,
                HSV,
                coords_nodes_base
            )

            coords_nodes_other_hsv = self.transform_color_space(
                self.color_space,
                HSV,
                coords_nodes_other
            )

            # FIXME: HCL does not work properly -> Hue ring is always selected...
            #   because for some nodes, hcl coords are None? Also, there are negatice hues in hcl?
            coords_nodes_base_hsl = self.transform_color_space(
                self.color_space,
                HSL,
                coords_nodes_base
            )

            coords_nodes_other_hsl = self.transform_color_space(
                self.color_space,
                HSL,
                coords_nodes_other
            )

            weights_hsv = self.get_weights_distances(coords_nodes_base_hsv, coords_nodes_other_hsv, range_hsv, 0)
            weights_hsl = self.get_weights_distances(coords_nodes_base_hsl, coords_nodes_other_hsl, range_hsl, 0)

            weights = np.maximum(weights_hsv, weights_hsl)

            return nodes_other, list(weights)

    @QtCore.Slot()
    def select_nodes_derived(self):
        # Start a selection process.
        #   The actually selected nodes will correspond to the
        self.queue_nodes_derived_selection.start_job(self, self.nodes_selection_base)
        # nodes_select = self.get_nodes_select_derived(self.nodes_selection_base)

        # TODO: use the worker.
        #   On worker finish, call apply_selection_nodes_derived()

    @QtCore.Slot()
    def apply_selection_nodes_derived(self, indices_nodes_select, weights_nodes_select):
        # Remark: weights instead of nodes are used because nodes returned by the worker
        #   are copies from other thread.
        self.iter_nodes(lambda node, _: node.select(0.))
        nodes_select = [self.nodes_lut[indices_node[0]][indices_node[1]][indices_node[2]] for indices_node in
                        indices_nodes_select]
        nodes_select.extend(self.nodes_selection_base)
        weights_nodes_select.extend([1.] * len(self.nodes_selection_base))
        for node, weight in zip(nodes_select, weights_nodes_select):
            node.select(weight)

        if self.preview_weights_on:
            nodes_selected = self.nodes_selection
            self.start_preview_weights.emit(
                self.make_lut_preview_selection(nodes_selected, [node.weight_selection for node in nodes_selected]))

    @QtCore.Slot(tuple)
    def slot_stop_hover_node(self, indices_node):
        if self.preview_weights_on_:
            # If transiting mouse quickly between two nodes,
            # The event calling this slot is called after the start for next node.
            # Only stop preview if NOT already hovering over other node.
            if indices_node == self.indices_node_preview_current:
                self.preview_weights_on_ = False
                self.stop_preview_weights_node_if_not_always_on()

    @QtCore.Slot()
    def slot_stop_hover_pixel(self):
        self.hovering_pixels = False
        self.stop_preview_weights_node_if_not_always_on()

    def stop_preview_weights_node_if_not_always_on(self):
        if not self.preview_weights_on:
            self.stop_preview_weights.emit(self.lut)
        else:
            self.start_preview_with_selected_nodes()

    @QtCore.Slot()
    def toggle_select_all(self):
        some_nodes_selected = np.any(self.iter_nodes(lambda node, result_: result_.append(node.is_selected_base)))

        nodes = self.iter_nodes(lambda node, result_: result_.append(node))

        if some_nodes_selected:
            self.select_nodes_base([], False, False)
        else:
            self.select_nodes_base(nodes, False, False)

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
                # FIXME: strange values come out of hcl conversion
                result = getattr(colour, f'RGB_to_{color_space_target}')(input_array)

            elif isinstance(color_space_target, colour.models.RGB_Colourspace):
                result = colour.RGB_to_RGB(input_array, color_space_source, color_space_target)
            else:
                raise NotImplementedError
        else:
            print(color_space_source, color_space_target)
            # TODO / FIXME: seems Adobe RGB is no instance of RGB color space?
            raise NotImplementedError

        return result

    @QtCore.Slot()
    def preview_selection_pixel(self, value_pixel, expand_selection, select_closest):
        nodes_pixel = self.find_nearest_nodes_pixels(np.asarray(value_pixel.toTuple())[
                                                         np.newaxis, ...]) if select_closest else self.find_nodes_influencing_pixel(
            value_pixel)

        if self.preview_weights_always_on:
            nodes_preview = {*nodes_pixel}

            if expand_selection:
                nodes_preview = nodes_preview.union(self.nodes_selection)

            self.queue_nodes_derived_selection_hover_pixel.start_job(
                self,
                nodes_pixel,
                nodes_preview
            )
            self.hovering_pixels = True

        #     self.prev
        # pass

    def start_preview_hover_pixel(self, nodes, weights):
        # only of still hovering. Prevents update after mouse having left the image view
        # if queue job is finished after leaving and after job to preview currently selected nodes.
        if self.hovering_pixels:
            lut_use = self.make_lut_preview_selection(nodes, weights)
            self.start_preview_weights.emit(lut_use)
        else:
            pass

    @QtCore.Slot(int, float)
    def transform_dragging(self, mode, distance):
        self.queue_updates_transform.start_job(self, mode, distance)

    @QtCore.Slot(list, np.ndarray)
    def apply_calculated_transorm_to_nodes_dragging_change(self, indices_nodes, coordinates):
        for idx, indices_node in enumerate(indices_nodes):
            node = self.nodes_lut[indices_node[0]][indices_node[1]][indices_node[2]]
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
    def select_nodes_by_source_colour_affecting_base(self, colour_float: QVector3D, expand_selection):
        # TODO: accept list of colours and vectorize
        nodes = self.find_nodes_influencing_pixel(colour_float)

        self.select_nodes_base(nodes, expand_selection, False)

    @QtCore.Slot()
    def select_nodes_by_source_colour_closest_base(self, colour_float: QVector3D or np.ndarray, expand_selection):
        # TODO: accept list of colours and vectorize
        nodes = self.find_nearest_nodes_pixels(
            np.asarray(colour_float.toTuple())[np.newaxis, ...] if isinstance(colour_float, QVector3D) else colour_float
        )

        self.select_nodes_base(nodes, expand_selection, False)

    @QtCore.Slot()
    def slot_clicked(self, event):
        node: NodeLut = event.entity()
        modifiers = event.modifiers()

        if event.button() == Qt3DRender.QPickEvent.LeftButton and self.parent_gui.mode_transform_current is None:
            self.select_nodes_base(
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

    def select_nodes_base(self, nodes, expand_selection, deselect_selected):
        if expand_selection:
            for node in nodes:
                if not node.is_selected_base or not deselect_selected:
                    node.select_base(True)
                    # self.nodes_selection_base.add(node)
                    # if change_base_selection:

                else:
                    node.select_base(False)
                    # self.remove_node_from_base_selection(node)
            # [node.select(not node.is_selected or not deselect_selected) for node in nodes]
        else:
            def fn(node, _):
                if (not node.is_selected_base or not deselect_selected) and node in nodes:
                    node.select_base(True)
                else:
                    node.select_base(False)

            # fn = lambda node, _: node.select((not node.is_selected or not deselect_selected) and node in nodes)
            self.iter_nodes(fn)

        self.selection_base_changed.emit()

    # def deselect_nodes_base(self, nodes):
    #     for node in nodes:
    #         # node.select(False)
    #         self.remove_node_from_base_selection(node)
    #
    #     self.selection_base_changed.emit()
    #
    # def remove_node_from_base_selection(self, node):
    #     if node in self.nodes_selection_base:
    #         self.nodes_selection_base.remove(node)
