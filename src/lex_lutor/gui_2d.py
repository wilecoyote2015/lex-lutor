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
from lex_lutor.job_queue import JobQueue
from tqdm import tqdm
from scipy.sparse import csc_matrix
from lex_lutor.constants import color_spaces
from superqt import QRangeSlider
from typing import Generic, List, Sequence, Tuple, TypeVar, Union
from lex_lutor.curve_editor import Curve, CurveWidget

# TODO / FIXME: use base img color space lut everywhere where needed!

INTER_TRILINEAR = 'Trilinear'
INTER_TETRAHEDRAL = 'Tetrahedral'

INTERPOLATORS = {
    INTER_TRILINEAR: colour.algebra.table_interpolation_trilinear,
    INTER_TETRAHEDRAL: colour.algebra.table_interpolation_tetrahedral,
}


# TODO: double right click on image to deselect affecting nodes (respect ctrl!)

class LabelClickable(QtWidgets.QLabel):
    double_clicked = Signal(QtGui.QMouseEvent, tuple)
    pixel_hovered = Signal(QtGui.QMouseEvent, tuple)
    left = Signal(QtGui.QMouseEvent)

    def __init__(self, parent=None):
        super().__init__(parent)

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed,
        )
        sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy)

        self.pos_pixel_hovered_last = (0, 0)
        self.setMouseTracking(True)

    def get_pos_pixel_mouse(self, x_mouse, y_mouse):
        pos_image = self.mapFromParent(self.pos())
        # print(pos_image)

        pos_pixel = [
            x_mouse - pos_image.y(),
            y_mouse - pos_image.x(),
        ]
        return pos_pixel

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        # TODO: modifier
        pos_pixel = self.get_pos_pixel_mouse(event.x(), event.y())
        if self.pos_pixel_hovered_last != pos_pixel:
            self.pixel_hovered.emit(event, pos_pixel)
            self.pos_pixel_hovered_last = pos_pixel

    def hasHeightForWidth(self) -> bool:
        return True

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        pos_pixel = self.get_pos_pixel_mouse(event.x(), event.y())
        self.double_clicked.emit(event, pos_pixel)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.left.emit(event)

    def heightForWidth(self, width: int) -> int:
        # FIXME: does not work somehow.
        aspect_pixmap = self.pixmap().height() / self.pixmap().width()

        # return int(width * aspect_pixmap)
        return aspect_pixmap * width


class SliderFloat(QRangeSlider):
    # TODO: Pushable: left handle pushes right one and vice versa.
    # TODO: styling: bar left from first handle must be colored, too.
    # TODO: handles should be only half, so that range 0 still makes both handles draggable.
    max_ = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setRange(0, self.max_)

        self.setValue((0, 0))
        self.setBarIsRigid(False)
        self._singleStep = 0.

    # def setValue(self, value: int) -> None:
    #     self.valueChanged.emit(value / self.max_)

    def value(self):
        return (super().value()[0] / self.max_, super().value()[1] / self.max_)

    def setSliderPosition(self, pos: Union[float, Sequence[float]], index=None) -> None:
        """Set current position of the handles with a sequence of integers.

        If `pos` is a sequence, it must have the same length as `value()`.
        If it is a scalar, index will be
        """
        if isinstance(pos, (list, tuple)):
            val_len = len(self.value())
            if len(pos) != val_len:
                msg = f"'sliderPosition' must have same length as 'value()' ({val_len})"
                raise ValueError(msg)
            pairs = list(enumerate(pos))
        else:
            pairs = [(self._pressedIndex if index is None else index, pos)]

        indices_handles_dragged, positions = zip(*pairs)

        for idx, position in pairs:
            # bound to min/max value
            self._position[idx] = self._bound(position, idx)

        # the dragged handle may shift surrounding hanldes
        # for idx in range(len(self._position)):
        #     if idx < self._pressedIndex:

        for idx_handle_dragged, position_handle_dragged in pairs:
            for idx_other in range(len(self._position)):
                position_other = self._position[idx_other]
                if idx_other not in indices_handles_dragged:
                    if idx_other < idx_handle_dragged:
                        self._position[idx_other] = min(
                            position_handle_dragged - (idx_handle_dragged - idx_other) * self.singleStep(),
                            position_other
                        )
                    elif idx_other > idx_handle_dragged:
                        self._position[idx_other] = max(
                            position_handle_dragged + (idx_other - idx_handle_dragged) * self.singleStep(),
                            position_other
                        )

        self._doSliderMove()

    def _bound(self, value, index=None):
        if isinstance(value, (list, tuple)):
            return type(value)(self._bound(v) for v in value)
        pos = super()._bound(value)
        if index is not None:
            return np.clip(
                pos,
                self.singleStep() * index + self._minimum,
                self._maximum - self.singleStep() * (len(self._position) - 1 - index),
            )
        return self._type_cast(pos)

class WorkerLut(QObject):
    finished = Signal(QtGui.QImage, str)
    progress = Signal(int)

    def __init__(self, image_color_space_lut: np.ndarray, lut: colour.LUT3D, colour_space_lut, color_space_display,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_color_space_lut = image_color_space_lut
        self.lut = lut
        self.colour_space_lut = colour_space_lut
        self.color_space_display = color_space_display

    def run(self, ):
        """Long-running task."""
        image_transformed = self.lut.apply(self.image_color_space_lut)

        if not self.colour_space_lut == self.color_space_display:
            image_transformed = colour.models.RGB_to_RGB(image_transformed, self.colour_space_lut,
                                                         self.color_space_display)

        img_uint = (np.clip(image_transformed, 0, 1) * 255).astype(np.uint8, order='c')
        qimage = QtGui.QImage(
            img_uint,
            self.image_color_space_lut.shape[1],
            self.image_color_space_lut.shape[0],
            self.image_color_space_lut.shape[1] * 3,
            QtGui.QImage.Format_RGB888
        )

        self.finished.emit(qimage, str(id(self)))

        return qimage

class MenuWidget(QtWidgets.QWidget):
    select_nodes_affecting_pixel = Signal(QVector3D, bool)
    select_node_closest_pixel = Signal(QVector3D, bool)
    preview_pixel_hovered = Signal(QVector3D, bool, bool)
    stop_preview_pixel_hovered = Signal()
    color_space_lut_changed = Signal(colour.models.RGB_Colourspace)

    def __init__(self, parent=None):
        super(MenuWidget, self).__init__(parent)
        self.interpolation_matrix = None

        self.label_image = LabelClickable()
        self.label_image.setScaledContents(True)
        self.img_base = None
        self.img_base_colorspace_lut = None

        self.queue_updates_image = JobQueue(
            WorkerLut,
            self.update_image_async,
        )

        self.lut_last_update = colour.LUT3D.linear_table(3)

        menu = self.build_menu()

        layout = QVBoxLayout()
        layout.addWidget(self.label_image)
        # layout.addStretch(2)
        layout.addLayout(menu)

        self.setLayout(layout)

        self.label_image.double_clicked.connect(self.pixelDoubleClicked)
        self.label_image.pixel_hovered.connect(self.pixel_hovered)
        self.label_image.left.connect(self.pixel_hover_left)

    @property
    def color_space_image(self):
        return color_spaces[self.combo_color_space_image.currentText()]

    @property
    def color_space_lut(self):
        return color_spaces[self.combo_color_space_lut.currentText()]

    @property
    def color_space_display(self):
        return color_spaces[self.combo_color_space_display.currentText()]

    def build_menu(self):
        self.slider_h = SliderFloat(QtCore.Qt.Horizontal)
        self.slider_s_hsv = SliderFloat(QtCore.Qt.Horizontal)
        self.slider_v = SliderFloat(QtCore.Qt.Horizontal)
        self.slider_s_hsl = SliderFloat(QtCore.Qt.Horizontal)
        self.slider_l = SliderFloat(QtCore.Qt.Horizontal)

        self.layout_menu = QVBoxLayout()

        # self.layout_menu.addWidget(self.slider_h)
        # self.layout_menu.addWidget(self.slider_s_hsv)
        # self.layout_menu.addWidget(self.slider_v)
        # self.layout_menu.addWidget(self.slider_s_hsl)
        # self.layout_menu.addWidget(self.slider_l)

        self.layout_menu.addLayout(self.make_layout_with_label(self.slider_h, 'H'))
        self.layout_menu.addLayout(self.make_layout_with_label(self.slider_s_hsv, 'S_hsv'))
        self.layout_menu.addLayout(self.make_layout_with_label(self.slider_v, 'V'))
        self.layout_menu.addLayout(self.make_layout_with_label(self.slider_s_hsl, 'S_hsl'))
        self.layout_menu.addLayout(self.make_layout_with_label(self.slider_l, 'L'))

        self.combo_color_space_image = QtWidgets.QComboBox()
        self.combo_color_space_image.addItems(list(color_spaces.keys()))
        self.combo_color_space_image.setCurrentIndex(0)

        self.combo_color_space_lut = QtWidgets.QComboBox()
        self.combo_color_space_lut.addItems(list(color_spaces.keys()))
        self.combo_color_space_lut.setCurrentIndex(0)

        self.combo_color_space_display = QtWidgets.QComboBox()
        self.combo_color_space_display.addItems(list(color_spaces.keys()))
        self.combo_color_space_display.setCurrentIndex(0)

        # self.layout_menu.addWidget(self.combo_color_space_image)
        # self.layout_menu.addWidget(self.combo_color_space_lut)
        # self.layout_menu.addWidget(self.combo_color_space_display)
        self.layout_menu.addLayout(self.make_layout_with_label(self.combo_color_space_image, 'Image'))
        self.layout_menu.addLayout(self.make_layout_with_label(self.combo_color_space_lut, 'LUT'))
        self.layout_menu.addLayout(self.make_layout_with_label(self.combo_color_space_display, 'Display'))

        self.combo_color_space_lut.currentTextChanged.connect(self.slot_color_space_lut_changed)
        self.combo_color_space_image.currentTextChanged.connect(self.slot_color_space_image_changed)
        self.combo_color_space_display.currentTextChanged.connect(self.slot_color_space_display_changed)
        # self.combo_color_space_lut.currentTextChanged.connect(self.slot_color_space_lut_changed)

        self.curve_editor = CurveWidget(self, Curve())
        self.layout_menu.addWidget(self.curve_editor)

        # TODO: tabbed

        # TODO: update image display on color space change

        return self.layout_menu

    def set_img_base_colorspace_lut(self):
        self.img_base_colorspace_lut = colour.models.RGB_to_RGB(
            self.img_base,
            self.color_space_image,
            self.color_space_lut
        )

    def make_layout_with_label(self, widget, text_label):
        hbox = QHBoxLayout()

        label = QtWidgets.QLabel(text_label)

        hbox.addWidget(label)
        hbox.addWidget(widget)

        return hbox

    @QtCore.Slot(str)
    def load_image(self, path_image):

        # TODO: infer color space from image.
        #   https://ninedegreesbelow.com/photography/embedded-color-space-information.html
        image = colour.io.read_image(
            path_image
        )
        # TODO: dynamic image resize
        aspect = image.shape[0] / image.shape[1]
        width = 500
        height = int(width * aspect)
        self.img_base = cv2.resize(image, (width, height))
        # TODO: support non-rgb spaces
        self.set_img_base_colorspace_lut()
        # TODO:Color management. https://doc.qt.io/qt-6/qcolorspace.html
        img_color_space_display = colour.models.RGB_to_RGB(
            self.img_base,
            self.color_space_image,
            self.color_space_display
        )
        img_uint = (img_color_space_display * 255).astype(np.uint8, order='c')
        qimage = QtGui.QImage(
            img_uint,
            img_color_space_display.shape[1],
            img_color_space_display.shape[0],
            img_color_space_display.shape[1] * 3,
            QtGui.QImage.Format_RGB888
        )
        self.label_image.setPixmap(
            QtGui.QPixmap(qimage)

        )

    def get_value_pixel_mouse_space_lut(self, x_mouse, y_mouse):
        return self.img_base_colorspace_lut[
            y_mouse,
            x_mouse
        ]

    @QtCore.Slot(QtGui.QMouseEvent, tuple)
    def pixel_hovered(self, event: QtGui.QMouseEvent, pos_pixel) -> None:
        if self.img_base_colorspace_lut is None:
            return
        expand_selection = event.modifiers() in (
            QtCore.Qt.Modifier.SHIFT,
            QtCore.Qt.Modifier.SHIFT + QtCore.Qt.Modifier.CTRL,
        )
        select_closest = event.modifiers() in (
            QtCore.Qt.Modifier.CTRL,
            QtCore.Qt.Modifier.SHIFT + QtCore.Qt.Modifier.CTRL,
        )

        pixel_image = self.get_value_pixel_mouse_space_lut(*pos_pixel)

        self.preview_pixel_hovered.emit(QVector3D(*pixel_image), expand_selection, select_closest)

    @QtCore.Slot(QtGui.QMouseEvent)
    def pixel_hover_left(self, event):
        self.stop_preview_pixel_hovered.emit()

    @QtCore.Slot(QtGui.QMouseEvent, tuple)
    def pixelDoubleClicked(self, event: QtGui.QMouseEvent, pos_pixel) -> None:
        # TODO / FIXME: When proper image display scaling is implemented,
        #   the coords do not correspond to pixel coords directly anymore (maybe the don't even nof with display scaling?)
        #   hence, transorfmation must be performed. Is this done using qpainter?
        #

        # Fixme: Not possible to get relative position in widget?
        pixel_image = self.get_value_pixel_mouse_space_lut(*pos_pixel)

        # TODO: STRG makes only one
        expand_selection = event.modifiers() in (
            QtCore.Qt.Modifier.SHIFT,
            QtCore.Qt.Modifier.SHIFT + QtCore.Qt.Modifier.CTRL,
        )
        select_closest = event.modifiers() in (
            QtCore.Qt.Modifier.CTRL,
            QtCore.Qt.Modifier.SHIFT + QtCore.Qt.Modifier.CTRL,
        )

        # image is float, so pixel is coords.
        if select_closest:
            self.select_node_closest_pixel.emit(QVector3D(*pixel_image), expand_selection)
        else:
            self.select_nodes_affecting_pixel.emit(QVector3D(*pixel_image), expand_selection)

    @QtCore.Slot(QtGui.QImage, int)
    def update_image_async(self, image_updated, id_worker):
        # if id_worker == self.queue_updates_image[-1][-1]:
        self.label_image.setPixmap(
            QtGui.QPixmap(image_updated)
        )

    @QtCore.Slot()
    def slot_color_space_image_changed(self):
        self.set_img_base_colorspace_lut()
        self.start_update_image(self.lut_last_update)

    @QtCore.Slot()
    def slot_color_space_display_changed(self):
        self.start_update_image(self.lut_last_update)

    @QtCore.Slot()
    def slot_color_space_lut_changed(self):
        self.color_space_lut_changed.emit(self.color_space_lut)
        self.set_img_base_colorspace_lut()
        self.start_update_image(self.lut_last_update)

    @QtCore.Slot(colour.LUT3D)
    def start_update_image(self, lut):
        if self.img_base_colorspace_lut is None:
            return
        self.lut_last_update = lut
        self.queue_updates_image.start_job(self.img_base_colorspace_lut, lut, self.color_space_lut,
                                           self.color_space_display)
