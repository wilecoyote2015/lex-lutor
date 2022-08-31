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

        self.modes_transform = (
            QtCore.Qt.Key_R,
            QtCore.Qt.Key_G,
            QtCore.Qt.Key_B,
        )
        self.mode_transform_current = None
        self.coordinates_mouse_event_start = QtCore.QPoint(0, 0)

    def load_lut(self, lut: colour.LUT3D):
        self.entity_lut = Lut3dEntity(lut, self)
        # TODO: remove old lut
        self.root_entity.addComponent(self.entity_lut)

        self.entity_lut.lut_changed.connect(self.gui_parent.widget_menu.start_update_image)

    #
    # def mousePressEvent(self, event: Qt3DInput.QMouseEvent) -> None:
    #     print(event.button())

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # TODO: cancel etc. must be invoked by proper signals that are emitted here.
        # TODO: Better move all this to entity_lut to reduce complexity
        key = event.key()

        if key == QtCore.Qt.Key_Escape:
            self.mode_transform_current = None
            self.cancel_transform.emit()
        if key == QtCore.Qt.Key_Enter:
            self.mode_transform_current = None
            self.accept_transform.emit()
        elif key in color_spaces_components_transform:
            # TODO: Better send signals to start and stop transform?
            self.mode_transform_current = key
            self.coordinates_mouse_event_start = QCursor.pos()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.mode_transform_current is not None:
            if event.button() == QtGui.Qt.MouseButton.LeftButton:
                self.mode_transform_current = None
                self.accept_transform.emit()
            elif event.button() == QtGui.Qt.MouseButton.RightButton:
                self.mode_transform_current = None
                self.cancel_transform.emit()
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        # TODO: trigger transform slot of all selected nodes (and later those that are in smooth transform radius)
        if self.mode_transform_current is not None:
            screen_size = self.screen().size()
            distance = (self.coordinates_mouse_event_start.x() - event.x()) / screen_size.width() * 2
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