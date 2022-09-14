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
from lex_lutor.gui_3d import CubeView
from lex_lutor.gui_2d import MenuWidget

class MainWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)


        self.window_3d: CubeView = CubeView(gui_parent=self)
        self.window_3d_widget = self.window_3d.widget
        # widget_3d = QtWidgets.QWidget.createWindowContainer(window_3d)

        # window_3d.createScene()
        # window_3d.show()


        # screenSize = window_3d.screen().size()
        # widget_3d.setMaximumSize(screenSize)
        # widget_3d.setMinimumSize(QtCore.QSize(200, 100))

        self.widget_menu = MenuWidget(self)

        layout = QHBoxLayout(self)
        layout.addWidget(self.window_3d_widget, 1)
        layout.addWidget(self.widget_menu)


        # self.setLayout(layout)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        self.window_3d.keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        self.window_3d.keyReleaseEvent(event)