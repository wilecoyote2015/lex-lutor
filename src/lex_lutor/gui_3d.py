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
from lex_lutor.entity_lut import Lut3dEntity

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
    def __init__(self):
        super().__init__()

        # self.camera()

        self.root_entity: Qt3DCore.QEntity = None
        self.createScene()
        self.setRootEntity(self.root_entity)

        self.entity_lut: Lut3dEntity = None

    def load_lut(self, lut: colour.LUT3D):
        self.entity_lut = Lut3dEntity(lut)
        self.root_entity.addComponent(self.entity_lut)


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

        self.nodes_lut = dict()
        self.lut = None

        # For camera controls
        self.camController = Qt3DExtras.QOrbitCameraController(self.root_entity)
        self.camController.setLinearSpeed(50)
        self.camController.setLookSpeed(180)
        self.camController.setCamera(self.camera())


