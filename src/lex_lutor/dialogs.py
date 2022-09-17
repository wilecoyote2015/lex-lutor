from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import (Property, QObject, QPropertyAnimation, Signal)
from PySide6.QtWidgets import QWidget, QPushButton, QGraphicsWidget, QHBoxLayout, QVBoxLayout, QApplication
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.QtGui import (QGuiApplication, QMatrix4x4, QQuaternion, QVector3D)
from PySide6.Qt3DInput import Qt3DInput
from PySide6.Qt3DRender import Qt3DRender
# from PySide6 import Qt3DCore, Qt3DExtras, Qt3DInput, Qt3DRender
from lex_lutor.constants import color_spaces


def get_color_space(parent, title):
    dialog = QtWidgets.QInputDialog()

    names_color_spaces = list(color_spaces.keys())
    name_color_space, ok = dialog.getItem(parent, title, '', names_color_spaces, 0, False)

    return color_spaces[name_color_space] if ok and name_color_space else None
