import colour
from PySide6.QtCore import Qt


HSV = 'HSV'
HSL = 'HSL'
HCL = 'HCL'

# color space and color space dimension along which to move a node
color_spaces_components_transform = {
    Qt.Key_R: (None, 0),
    Qt.Key_G: (None, 1),
    Qt.Key_B: (None, 2),
    Qt.Key_H: (HSV, 0),
    Qt.Key_S: (HSL, 1),
    Qt.Key_V: (HSV, 2),
    Qt.Key_L: (HCL, 0),
    Qt.Key_C: (HCL, 1)

}