import colour
from PySide6.QtCore import Qt
from collections import OrderedDict

HSV = 'HSV'
HSL = 'HSL'
HCL = 'HCL'

KEY_EXPOSURE = Qt.Key_E

# color space and color space dimension along which to move a node
color_spaces_components_transform = {
    Qt.Key_R: (None, 0),
    Qt.Key_G: (None, 1),
    Qt.Key_B: (None, 2),
    Qt.Key_H: (HSV, 0),
    Qt.Key_S: (HSL, 1),
    Qt.Key_V: (HSV, 2),
    Qt.Key_L: (HSL, 2),
    Qt.Key_C: (HCL, 1)

}

color_spaces = OrderedDict(
    [(colorspace_.name, colorspace_) for colorspace_ in [
        colour.models.RGB_COLOURSPACE_sRGB,
        colour.models.RGB_COLOURSPACE_ADOBE_RGB1998,
        colour.models.RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB,
        colour.models.RGB_COLOURSPACE_BT709,
        colour.models.RGB_COLOURSPACE_BT2020,
        colour.models.RGB_COLOURSPACE_ACES2065_1,
        colour.models.RGB_COLOURSPACE_ACESCC,
        colour.models.RGB_COLOURSPACE_ACESCCT,
        colour.models.RGB_COLOURSPACE_ACESCG

    ]]
)
