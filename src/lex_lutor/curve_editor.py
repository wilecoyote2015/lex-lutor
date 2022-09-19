import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import (Property, QObject, QPropertyAnimation, Signal)
from PySide6.QtWidgets import QWidget, QPushButton, QGraphicsWidget, QHBoxLayout, QVBoxLayout, QApplication
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.QtGui import (QGuiApplication, QMatrix4x4, QQuaternion, QVector3D)
from PySide6.Qt3DInput import Qt3DInput
from PySide6.Qt3DRender import Qt3DRender
from scipy.interpolate import interp1d


# FROM https://discourse.panda3d.org/t/pyqt-curve-editor-curvefitter-example/15207/1
class Curve(QtCore.QObject):
    updated = Signal()
    """ Interface to the NURBS curve which also manages connecting the end of the
    curve with the beginning """

    def __init__(self, n_control_points=5, control_points=None):
        super().__init__(None)
        self._curve = None

        # Append some points to the border, to make sure the curve matches at
        # the edges
        self._border_points = 1

        # Curve color, used for displaying the curve
        self.color = (1, 1, 1)

        if control_points is not None and n_control_points != len(control_points):
            raise ValueError

        # # Control points, those are some demo values
        # # [point, coordinates]
        # if control_points is None:
        #     control_points = []
        #     coords_points_x = np.linspace(0, 1, n_control_points+1, endpoint=False)[1:, np.newaxis]
        #     coords_points_y = np.linspace(0, 1, n_control_points+1, endpoint=False)[1:, np.newaxis]
        #
        #     for idx in range(n_control_points):
        #

        self.control_points = np.tile(
            np.linspace(0, 1, n_control_points, endpoint=True)[..., np.newaxis],
            (1, 2)
        ) if control_points is None else control_points

        # Build the curve
        self.build_curve()

    def _get_cv_tangent(self, index):
        """ Returns the tangent of the nth point """
        pb = self._cv_points[(index - 1) % len(self._cv_points)]
        pm = self._cv_points[index]
        pa = self._cv_points[(index + 1) % len(self._cv_points)]

        get_diff = lambda p1, p2: QVector3D(p1[0] - p2[0], p1[1] - p2[1], 0)

        tangent_left = get_diff(pm, pb)
        tangent_right = get_diff(pa, pm)

        tangent_avg = (tangent_left + tangent_right) * 0.5
        return tangent_avg

    def build_curve(self):
        """ Rebuilds the curve based on the controll point values """

        self.interpolator_curve = interp1d(self.control_points[:, 0], self.control_points[:, 1], kind='quadratic',
                                           fill_value="extrapolate")
        self.updated.emit()

    def set_control_point(self, index, x_value, y_value):
        """ Updates the cv point at the given index """
        self.control_points[index] = np.asarray([x_value, y_value])
        self.build_curve()

    def get_y(self, x):
        """ Returns the value on the curve ranging whereas the offset should be
        from 0 to 1 (0 denotes the start of the curve). The returned value will
        be a value from 0 to 1 as well. """
        return np.min(np.max(self.interpolator_curve(x), 0), 1)

    def add_control_point(self, x, y):
        if not np.any(self.control_points[:, 0] == x):
            self.control_points = np.append(self.control_points, [[x, y]], axis=0)
            self.build_curve()

            return self.control_points.shape[0] - 1
        else:
            return None

    def remove_control_point(self, index):
        self.control_points = np.delete(self.control_points, index, 0)
        self.build_curve()


class CurveWidget(QWidget):
    curve_updated = Signal()
    """ This is a resizeable Widget which shows an editable curve which can
    be modified. """

    def __init__(self, parent, curve: Curve):
        """ Constructs the CurveWidget, we start with an initial curve """
        super().__init__(parent)

        # Append initial curve
        self.curve = curve

        # Widget render constants
        self.size_control_point = 3
        self._legend_border = 0
        self._bar_h = 0

        # Currently dragged control point, format is:
        # (CurveIndex, PointIndex, Drag-Offset (x,y))
        self._drag_point = None

        # Currently selected control point, format is:
        # (CurveIndex, PointIndex)
        self._selected_point = None

        self.curve.updated.connect(self.curve_updated.emit)

    def paintEvent(self, e):
        """ Internal QT paint event, draws the entire widget """
        qp = QtGui.QPainter()
        qp.begin(self)
        self._draw(qp)
        qp.end()

    def get_control_point_mouse(self, mouse_x, mouse_y):
        for cv_index, (x, y) in enumerate(self.curve.control_points):
            point_x, point_y = self.convert_curve_value_to_pixel_coordinate(x, y)
            # point_x = self._get_x_value_for(x)
            # point_y = self._get_y_value_for(y)
            if abs(point_x - mouse_x) < self.size_control_point + 4:
                if (abs(point_y - mouse_y)) < self.size_control_point + 4:
                    return cv_index, point_x, point_y

        return None, None, None

    def mouseDoubleClickEvent(self, event) -> None:
        self._drag_point = None
        self._selected_point = None

        mouse_pos = event.pos()
        mouse_x = mouse_pos.x() - self._legend_border
        mouse_y = mouse_pos.y()
        index_cv, x_point, y_point = self.get_control_point_mouse(mouse_x, mouse_y)
        if index_cv is not None and len(self.curve.control_points) > 4:
            self.curve.remove_control_point(index_cv)
            self.update()

    def mousePressEvent(self, event):
        """ Internal mouse-press handler """
        self._drag_point = None
        self._selected_point = None
        mouse_pos = event.pos()
        mouse_x = mouse_pos.x() - self._legend_border
        mouse_y = mouse_pos.y()

        index_cv, x_point, y_point = self.get_control_point_mouse(mouse_x, mouse_y)

        if index_cv is not None:
            drag_x_offset = x_point - mouse_x
            drag_y_offset = y_point - mouse_y

            self._drag_point = (index_cv, (drag_x_offset, drag_y_offset))
            self._selected_point = (index_cv)
        else:
            if self._drag_point is None:
                x_point, y_point = self.convert_pixel_coordinate_value_to_curve_value(mouse_x, mouse_y)
                index_cv = self.curve.add_control_point(x_point, y_point)
                if index_cv is not None:
                    self._drag_point = (index_cv, (0, 0))
                    self._selected_point = (index_cv)

        self.update()

    def mouseReleaseEvent(self, QMouseEvent):
        """ Internal mouse-release handler """
        self._drag_point = None

    def mouseMoveEvent(self, QMouseEvent):
        """ Internal mouse-move handler """
        # print("mouse moved:", QMouseEvent.pos())
        if self._drag_point is not None:
            mouse_x = QMouseEvent.pos().x() - self._drag_point[1][0] - self._legend_border
            mouse_y = QMouseEvent.pos().y() - self._drag_point[1][1] - self._bar_h

            # Convert to local coordinate
            local_x = max(0, min(1, mouse_x / float(self.width() - self._legend_border)))
            local_y = 1 - max(0, min(1, mouse_y / float(self.height() - self._legend_border - self._bar_h)))

            # Set new point data
            self.curve.set_control_point(self._drag_point[0], local_x, local_y)

            # Redraw curve
            self.update()

    def convert_curve_value_to_pixel_coordinate(self, x_curve, y_curve):
        x_pixel = max(0, min(1.0, x_curve)) * (self.width() - self._legend_border)
        y_pixel = max(0, min(1.0, 1.0 - y_curve)) * (self.height() - self._legend_border - self._bar_h) + self._bar_h

        return x_pixel, y_pixel

    def convert_pixel_coordinate_value_to_curve_value(self, x_pixel, y_pixel):
        x_curve = x_pixel / (self.width() - self._legend_border)
        y_curve = (self.height() - y_pixel - self._bar_h) / (self.height() - self._legend_border - self._bar_h)

        return x_curve, y_curve

    def _draw(self, painter):
        """ Internal method to draw the widget """

        canvas_width = self.width() - self._legend_border
        canvas_height = self.height() - self._legend_border - self._bar_h

        # Draw field background
        painter.setPen(QtGui.QColor(170, 170, 170))
        painter.setBrush(QtGui.QColor(230, 230, 230))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        num_vert_lines = 6  # 24 / 6 = 4, one entry per 4 hours
        line_spacing_x = (self.width() - self._legend_border) / 6.0
        line_spacing_y = (self.height() - self._legend_border - self._bar_h) / 20.0
        num_horiz_lines = int(np.ceil(canvas_height / float(line_spacing_y)) + 1)

        # Draw vertical lines
        painter.setPen(QtGui.QColor(200, 200, 200))
        for i in range(num_vert_lines):
            line_pos = i * line_spacing_x + self._legend_border
            painter.drawLine(line_pos, self._bar_h, line_pos, canvas_height + self._bar_h)

        # Draw horizontal lines
        painter.setPen(QtGui.QColor(200, 200, 200))
        for i in range(num_horiz_lines):
            line_pos = canvas_height - i * line_spacing_y + self._bar_h
            painter.drawLine(self._legend_border, line_pos, self.width(), line_pos)

            # Draw curve
        painter.setPen(QtGui.QColor(*self.curve.color))
        last_value = 0
        for i in range(canvas_width):
            rel_offset = i / (canvas_width - 1.0)
            curve_height = self.convert_curve_value_to_pixel_coordinate(rel_offset, self.curve.get_y(rel_offset))[1]

            if i == 0:
                last_value = curve_height

            painter.drawLine(self._legend_border + i - 1, last_value, self._legend_border + i, curve_height)
            last_value = curve_height

            # Draw the CV points of the curve
            painter.setBrush(QtGui.QColor(240, 240, 240))

            for cv_index, (x, y) in enumerate(self.curve.control_points):
                offs_x = x * canvas_width + self._legend_border
                offs_y = (1 - y) * canvas_height + self._bar_h

                if self._selected_point is not None and self._selected_point == cv_index:
                    painter.setPen(QtGui.QColor(255, 0, 0))
                else:
                    painter.setPen(QtGui.QColor(100, 100, 100))
                painter.drawRect(offs_x - self.size_control_point, offs_y - self.size_control_point,
                                 2 * self.size_control_point, 2 * self.size_control_point)


if __name__ == '__main__':
    import sys


    class Editor(QtWidgets.QMainWindow):

        def __init__(self):
            super().__init__()
            self.resize(400, 400)
            curve = Curve(5)
            widget = CurveWidget(self, curve)
            widget.setGeometry(40, 40, 500, 400)


    # Start application
    app = QApplication(sys.argv)
    editor = Editor()
    editor.show()
    app.exec_()
