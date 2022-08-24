import colour
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


class MenuWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MenuWidget, self).__init__(parent)

        button_test = QPushButton('test')
        button_test2 = QPushButton('test2')

        layout = QVBoxLayout()
        layout.addWidget(button_test)
        layout.addWidget(button_test2)

        self.setLayout(layout)

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

        # Camera
        self.camera().lens().setPerspectiveProjection(45, 16 / 9, 0.1, 1000)
        self.camera().setPosition(QVector3D(0, 0, 40))
        self.camera().setViewCenter(QVector3D(0, 0, 0))

        # For camera controls
        self.createScene()
        self.camController = Qt3DExtras.QOrbitCameraController(self.rootEntity)
        self.camController.setLinearSpeed(50)
        self.camController.setLookSpeed(180)
        self.camController.setCamera(self.camera())

        self.setRootEntity(self.rootEntity)

        # input_ = Qt3DInput.QInputAspect()
        # self.registerAspect(input_)



        # self.widget = QtWidgets.QWidget.createWindowContainer(self)
        # screenSize = self.screen().size()
        # self.widget.setMinimumSize(QtCore.QSize(200, 100))
        # self.widget.setMaximumSize(screenSize)

        # input_ = Qt3DInput.QInputAspect()
        # self.registerAspect(input_)

    @property
    def widget(self):
        widget = QtWidgets.QWidget.createWindowContainer(self)

        screenSize = self.screen().size()
        widget.setMinimumSize(QtCore.QSize(200, 100))
        widget.setMaximumSize(screenSize)

        return widget

    def createScene(self):
        # Root entity
        self.rootEntity = Qt3DCore.QEntity()

        # Material
        self.material = Qt3DExtras.QPhongMaterial(self.rootEntity)

        # Torus
        self.torusEntity = Qt3DCore.QEntity(self.rootEntity)
        self.torusMesh = Qt3DExtras.QTorusMesh()
        self.torusMesh.setRadius(5)
        self.torusMesh.setMinorRadius(1)
        self.torusMesh.setRings(100)
        self.torusMesh.setSlices(20)

        self.torusTransform = Qt3DCore.QTransform()
        self.torusTransform.setScale3D(QVector3D(1.5, 1, 0.5))
        self.torusTransform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 45))

        self.torusEntity.addComponent(self.torusMesh)
        self.torusEntity.addComponent(self.torusTransform)
        self.torusEntity.addComponent(self.material)

        # Sphere
        self.sphereEntity = Qt3DCore.QEntity(self.rootEntity)
        self.sphereMesh = Qt3DExtras.QSphereMesh()
        self.sphereMesh.setRadius(3)

        self.sphereTransform = Qt3DCore.QTransform()
        self.controller = OrbitTransformController(self.sphereTransform)
        self.controller.setTarget(self.sphereTransform)
        self.controller.setRadius(20)

        self.sphereRotateTransformAnimation = QPropertyAnimation(self.sphereTransform)
        self.sphereRotateTransformAnimation.setTargetObject(self.controller)
        self.sphereRotateTransformAnimation.setPropertyName(b"angle")
        self.sphereRotateTransformAnimation.setStartValue(0)
        self.sphereRotateTransformAnimation.setEndValue(360)
        self.sphereRotateTransformAnimation.setDuration(10000)
        self.sphereRotateTransformAnimation.setLoopCount(-1)
        self.sphereRotateTransformAnimation.start()

        self.sphereEntity.addComponent(self.sphereMesh)
        self.sphereEntity.addComponent(self.sphereTransform)
        self.sphereEntity.addComponent(self.material)

        self.defaultFrameGraph().setClearColor(QtGui.QColor("#FF0000"))

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)


        self.window_3d = CubeView()
        # widget_3d = QtWidgets.QWidget.createWindowContainer(window_3d)

        # window_3d.createScene()
        # window_3d.show()


        # screenSize = window_3d.screen().size()
        # widget_3d.setMaximumSize(screenSize)
        # widget_3d.setMinimumSize(QtCore.QSize(200, 100))

        widget_menu = MenuWidget()

        layout = QHBoxLayout(self)
        layout.addWidget(self.window_3d.widget, 1)
        layout.addWidget(widget_menu)

        # self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MainWidget()
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