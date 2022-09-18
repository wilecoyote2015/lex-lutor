import colour
from PySide6.QtCore import Qt
from collections import OrderedDict

HSV = 'HSV'
HSL = 'HSL'
HCL = 'HCL'

KEY_EXPOSURE = Qt.Key_E

# color space and color space dimension along which to move a node
color_spaces_components_transform = {
    (Qt.Key_R, Qt.NoModifier): (None, 0),
    (Qt.Key_G, Qt.NoModifier): (None, 1),
    (Qt.Key_B, Qt.NoModifier): (None, 2),
    (Qt.Key_H, Qt.NoModifier): (HSV, 0),
    (Qt.Key_S, Qt.NoModifier): (HSV, 1),
    (Qt.Key_V, Qt.NoModifier): (HSV, 2),
    (Qt.Key_S, int(Qt.Modifier.SHIFT)): (HSL, 1),
    (Qt.Key_L, Qt.NoModifier): (HSL, 2)

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
