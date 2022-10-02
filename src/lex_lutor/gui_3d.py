import colour
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import (Property, QObject, QPropertyAnimation, Signal)
from PySide6.QtWidgets import QWidget, QPushButton, QGraphicsWidget, QHBoxLayout, QVBoxLayout, QApplication
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.QtGui import (QGuiApplication, QMatrix4x4, QQuaternion, QVector3D, QCursor)
from PySide6.Qt3DInput import Qt3DInput
from PySide6.Qt3DRender import Qt3DRender
# from PySide6 import Qt3DCore, Qt3DExtras, Qt3DInput, Qt3DRender
import sys
from lex_lutor.entity_lut import Lut3dEntity
from lex_lutor.constants import color_spaces_components_transform
from lex_lutor.dialogs import get_color_space


# TODO: can the signals and slots be moved to entity_lut?


class OrbitTransformController(QObject):

    def __init__(self, parent):
        super().__init__(parent)
        self._target = None
        self._matrix = QMatrix4x4()
        self._radius = 1
        self._angle = 0

    def setTarget(self, t):
        self._target = t

    def getTarget(self):
        return self._target

    def setRadius(self, radius):
        if self._radius != radius:
            self._radius = radius
            self.updateMatrix()
            self.radiusChanged.emit()

    def getRadius(self):
        return self._radius

    def setAngle(self, angle):
        if self._angle != angle:
            self._angle = angle
            self.updateMatrix()
            self.angleChanged.emit()

    def getAngle(self):
        return self._angle

    def updateMatrix(self):
        self._matrix.setToIdentity()
        self._matrix.rotate(self._angle, QVector3D(0, 1, 0))
        self._matrix.translate(self._radius, 0, 0)
        if self._target is not None:
            self._target.setMatrix(self._matrix)

    angleChanged = Signal()
    radiusChanged = Signal()
    angle = Property(float, getAngle, setAngle, notify=angleChanged)
    radius = Property(float, getRadius, setRadius, notify=radiusChanged)

class CubeView(Qt3DExtras.Qt3DWindow):
    cancel_transform = QtCore.Signal()
    accept_transform = QtCore.Signal()

    def __init__(self, gui_parent):
        super().__init__()

        # self.camera()
        self.gui_parent = gui_parent

        self.root_entity: Qt3DCore.QEntity = None
        self.createScene()
        self.setRootEntity(self.root_entity)

        self.entity_lut: Lut3dEntity = None

        self.distance_last = None

        self.cancel_transform.connect(self.end_transform)
        self.accept_transform.connect(self.end_transform)

        self.modes_transform = (
            QtCore.Qt.Key_R,
            QtCore.Qt.Key_G,
            QtCore.Qt.Key_B,
        )
        self.mode_transform_current = None
        self.coordinates_mouse_event_start = QtCore.QPoint(0, 0)

    def delete_children(self, parent):
        for child in parent.childNodes():
            self.delete_children(child)
            child.deleteLater()

    def load_lut(self, lut: colour.LUT3D):
        # FIXME: with new lut, trigger new preview
        if self.entity_lut is not None:
            # self.root_entity.removeComponent(self.entity_lut)
            self.delete_children(self.entity_lut)
            self.entity_lut.deleteLater()

        self.entity_lut = Lut3dEntity(lut, self)
        self.root_entity.addComponent(self.entity_lut)

    @QtCore.Slot()
    def end_transform(self):
        self.mode_transform_current = None
        self.distance_last = None

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # TODO: cancel etc. must be invoked by proper signals that are emitted here.
        # TODO: Better move all this to entity_lut to reduce complexity
        key = event.key()
        modifiers = event.modifiers()

        print(modifiers)

        if key == QtCore.Qt.Key_Escape:
            self.cancel_transform.emit()
        if key == QtCore.Qt.Key_Enter:
            self.accept_transform.emit()
        if key == QtCore.Qt.Key_N and self.mode_transform_current is None:
            # TODO: use signal
            # TODO: Upper case N for reset to neutral color
            self.entity_lut.reset_selected_nodes()
        if key == QtCore.Qt.Key_A and event.modifiers() == QtCore.Qt.Modifier.SHIFT:
            self.entity_lut.toggle_select_all()
        # FIXME: for some reason, after using the curve, a click on 3d view must always be peformed
        #   before toggling is possible.
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.entity_lut.toggle_preview_selection_always_on()


        elif (key, int(modifiers)) in color_spaces_components_transform:
            # TODO: Better send signals to start and stop transform?
            def fn(node, _):
                node.coordinates_current_before_translation = node.coordinates_current
                # node.coordinates_without_base_adjustment_before_trafo_start = node.coordinates_without_base_adjustment

            self.entity_lut.iter_nodes(fn)
            self.mode_transform_current = (key, int(modifiers))
            self.coordinates_mouse_event_start = QCursor.pos()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.mode_transform_current is not None:
            if event.button() == QtGui.Qt.MouseButton.LeftButton:
                self.accept_transform.emit()
            elif event.button() == QtGui.Qt.MouseButton.RightButton:
                self.cancel_transform.emit()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        # TODO: trigger transform slot of all selected nodes (and later those that are in smooth transform radius)
        if self.mode_transform_current is not None:
            screen_size = self.screen().size()
            distance = -(self.coordinates_mouse_event_start.x() - event.x()) / screen_size.width() * 5
            if distance != self.distance_last:
                self.distance_last = distance
                self.entity_lut.transform_dragging(self.mode_transform_current, distance)



    @property
    def widget(self):
        widget = QtWidgets.QWidget.createWindowContainer(self)

        screenSize = self.screen().size()
        widget.setMinimumSize(QtCore.QSize(200, 100))
        widget.setMaximumSize(screenSize)

        return widget


    def createScene(self):
        self.root_entity = Qt3DCore.QEntity()
        self.defaultFrameGraph().setClearColor(QtGui.QColor(128, 128, 128))

        # Camera
        self.camera().lens().setPerspectiveProjection(90, 16 / 9, 0.1, 1000)
        self.camera().setPosition(QVector3D(1.2, 1.2, 1.2))
        self.camera().setViewCenter(QVector3D(0.5, 0.5, 0.5))

        # For camera controls
        self.camController = Qt3DExtras.QOrbitCameraController(self.root_entity)
        self.camController.setLinearSpeed(0)
        self.camController.setLookSpeed(180)
        self.camController.setCamera(self.camera())



if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = CubeView()

    lut = colour.LUT3D(colour.LUT3D.linear_table(9) ** (1 / 2.2))
    # w.load_lut(lut)


    # w.show()
    # w.resize(1200, 800)
    #
    # widget_3d = QtWidgets.QWidget.createWindowContainer(CubeView())
    # widget_3d.setFocusPolicy(QtCore.Qt.TabFocus)
    # widget_3d.show()
    # view = CubeView()
    # widget = view.widget

    # w = QtWidgets.QWidget()
    # layout = QHBoxLayout(w)
    # layout.addWidget(view.widget, 1)

    w.show()





    # view.resize(1200, 800)
    sys.exit(app.exec())