import os.path

import colour
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import (Property, QObject, QPropertyAnimation, Signal)
from PySide6.QtWidgets import QMainWindow, QWidget, QPushButton, QGraphicsWidget, QHBoxLayout, QVBoxLayout, QApplication
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.QtGui import (QGuiApplication, QMatrix4x4, QQuaternion, QVector3D)
from PySide6.Qt3DInput import Qt3DInput
from PySide6.Qt3DRender import Qt3DRender
# from PySide6 import Qt3DCore, Qt3DExtras, Qt3DInput, Qt3DRender
import sys
from lex_lutor.gui_3d import CubeView
from lex_lutor.gui_2d import MenuWidget
from os import path
from lex_lutor.dialogs import get_color_space


# TODO: support linear color spaces.
#   Seems color spaces must be implemented manually with according cctf


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # self.menubar.addAction(QtGui.QAction())

        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)
        self.create_menu_bar()

        self.directory_lut_last_opened = None
        self.directory_image_last_opened = None

        self.path_lut = None

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        menu_file = QtWidgets.QMenu('&File', self)
        menu_lut = QtWidgets.QMenu('&LUT', self)

        menu_bar.addMenu(menu_file)
        menu_bar.addMenu(menu_lut)

        self.action_new_lut = QtGui.QAction('&New LUT...', self)
        self.action_new_lut.triggered.connect(self.new_lut)
        # self.action_open_lut.setText('&Open LUT')
        menu_file.addAction(self.action_new_lut)

        self.action_open_lut = QtGui.QAction('&Open LUT...', self)
        self.action_open_lut.triggered.connect(self.open_lut)
        # self.action_open_lut.setText('&Open LUT')
        menu_file.addAction(self.action_open_lut)

        self.action_open_image = QtGui.QAction('&Open Image...', self)
        self.action_open_image.triggered.connect(self.open_image)
        # self.action_open_image.setText('&Open Image')
        menu_file.addAction(self.action_open_image)

        self.action_save_lut = QtGui.QAction('&Save LUT', self)
        self.action_save_lut.triggered.connect(self.save_lut)
        # self.action_open_image.setText('&Open Image')
        menu_file.addAction(self.action_save_lut)

        self.action_save_lut_as = QtGui.QAction('&Save LUT As...', self)
        self.action_save_lut_as.triggered.connect(self.save_lut_as)
        # self.action_open_image.setText('&Open Image')
        menu_file.addAction(self.action_save_lut_as)

        self.action_resample = QtGui.QAction('&Resample LUT...', self)
        self.action_resample.triggered.connect(self.resample_lut)
        # self.action_open_image.setText('&Open Image')
        menu_lut.addAction(self.action_resample)

        self.setMenuBar(menu_bar)

    def new_lut(self):
        size, _ = QtWidgets.QInputDialog.getInt(
            self,
            'Choose LUT size',
            'Choose the size of the new LUT. '
            'The Size is the number of nodes per axis. '
            'It is recommended to start with 5-9 and use an odd number.',
            5,
            3,
            16

        )
        lut = colour.LUT3D(colour.LUT3D.linear_table(size))
        self.path_lut = None
        self.main_widget.window_3d.load_lut(lut)

    def open_lut(self):
        path_lut, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            dir=os.path.dirname(self.path_lut) if self.path_lut is not None else None,
            filter='*.cube'
        )
        if path_lut:
            self.path_lut = path_lut
            lut = colour.read_LUT(path_lut)
            self.main_widget.window_3d.load_lut(lut)

    def save_lut(self):
        # TODO: consider different lut formats
        if self.path_lut is None:
            QtWidgets.QErrorMessage(self).showMessage(f'Please use Save LUT as... first.')
        else:
            colour.io.write_LUT_IridasCube(self.main_widget.window_3d.entity_lut.lut, self.path_lut)

    def save_lut_as(self):
        # TODO: consider different lut formats
        path_lut, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            dir=os.path.dirname(self.path_lut) if self.path_lut is not None else None,
            filter='*.cube'
        )

        path_save = os.path.splitext(path_lut)[0] + '.cube'

        # TODO: show dialog if exists
        print(path_save)
        if path_lut:
            self.path_lut = path_save
            colour.io.write_LUT_IridasCube(self.main_widget.window_3d.entity_lut.lut, path_save)

    def open_image(self):
        path_image, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            dir=self.directory_image_last_opened,
            # filter='*.cube'
        )
        print(path_image)
        if path_image:
            self.directory_image_last_opened = os.path.dirname(path_image)
            # lut = colour.read_LUT(path_lut)
            try:
                self.main_widget.widget_menu.load_image(path_image)
            except:
                QtWidgets.QErrorMessage(self).showMessage(f'File is not a valid image: {path_image}')

    def resample_lut(self):
        # TODO: interpolation
        size, _ = QtWidgets.QInputDialog.getInt(
            self,
            'Choose LUT size',
            'Choose the size to resample the LUT to.. '
            'The Size is the number of nodes per axis. ',
            5,
            3,
            16

        )
        entity_lut = self.main_widget.window_3d.entity_lut
        lut_old = entity_lut.lut
        # lut_new = entity_lut.lut.convert(type(entity_lut.lut), size=size)
        # domain = entity_lut.lut.domain

        table_lut_new = lut_old.apply(
            colour.LUT3D(colour.LUT3D.linear_table(size, domain=lut_old.domain)).table
        )
        lut_new = colour.LUT3D(table_lut_new, domain=lut_old.domain)

        # lut = colour.LUT3D(colour.LUT3D.linear_table(size))
        # self.path_lut = None
        self.main_widget.window_3d.load_lut(lut_new)

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

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # if event.key() == QtCore.Qt.Key.Key_Space:
        #     self.window_3d.entity_lut.toggle_preview_selection_always_on()
        # else:
        self.window_3d.keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        self.window_3d.keyReleaseEvent(event)