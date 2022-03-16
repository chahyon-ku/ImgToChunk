import tkinter

import cv2
import numpy
from PIL import Image


image = Image.open('block_subset/oak_log.png')
image = numpy.array(image)
image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_NEAREST)
image = Image.fromarray(image)
cv2.imshow('abc', numpy.array(image))
cv2.waitKey()

grass_image = numpy.array(Image.open('block/grass_block_top.png'))
grass_image = cv2.resize(grass_image, (200, 200), interpolation=cv2.INTER_NEAREST)
grass_image = cv2.cvtColor(grass_image, cv2.COLOR_BGRA2GRAY)
print(grass_image.shape)
colored_grass_image = numpy.ones((200, 200, 3), dtype=numpy.uint8)
colored_grass_image[:, :, 0] = 0x50 * grass_image.astype(float) / 255.0
colored_grass_image[:, :, 1] = 0x7a * grass_image.astype(float) / 255.0
colored_grass_image[:, :, 2] = 0x32 * grass_image.astype(float) / 255.0
colored_grass_image = cv2.cvtColor(colored_grass_image, cv2.COLOR_RGB2BGRA)
print(colored_grass_image)
cv2.imshow('abc', colored_grass_image)
cv2.imwrite('block_subset/grass.png', colored_grass_image)
cv2.waitKey()
