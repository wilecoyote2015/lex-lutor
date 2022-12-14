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
from lex_lutor.main_widget import MainWidget, MainWindow


# TODO: Node objects should store their index in the lut array.
# TODO: If a node's transfom is changed, this should be a signal that is connected to the window, delivering the index
#   of the node.The LUT itself should be a QObject class that has a slot for changed and changes its according value.
#   The LUT should then emit a signal that it has been changed, so that views like images can be updated.
#   REMARK: Triggering changed by each individual node may be problematic performance wise: Multiple concurrent signals
#   would trigger many rerenders if a selection of multiple nodes is changed.
#   maybe let the LUT entity have a signal (make the 3d lut node collection a qobject...) that indicates change.


# TODO: undo / redo!
if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MainWindow()

    lut = colour.LUT3D(colour.LUT3D.linear_table(9))
    w.main_widget.window_3d.load_lut(lut)
    w.show()

    # view.resize(1200, 800)
    sys.exit(app.exec())