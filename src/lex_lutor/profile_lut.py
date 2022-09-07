import colour
import cProfile

lut = colour.io.read_LUT('/home/bjoern/Pictures/hald-clut/HaldCLUT/own/provia_profiled_xt3.cube')

image = colour.io.read_image(
    '/home/bjoern/PycharmProjects/darktabe_hald_generator/samples/provia/DSCF0326.JPG'
)

cProfile.run('lut.apply(image)', sort='cumtime')
