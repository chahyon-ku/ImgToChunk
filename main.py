import math

from PIL import ImageTk, Image
import cv2
import numpy
from glumpy import glm
import tkinter


view = numpy.eye(4, dtype=float)
x_rot = 10
y_rot = 20
z_rot = 0
glm.rotate(view, x_rot, 1, 0, 0)
glm.rotate(view, y_rot, 0, 1, 0)
glm.rotate(view, z_rot, 0, 0, 1)
print(view)

def rot(x_rot, y_rot, z_rot):
    cx = math.cos(math.radians(x_rot))
    sx = math.sin(math.radians(x_rot))
    cy = math.cos(math.radians(y_rot))
    sy = math.sin(math.radians(y_rot))
    cz = math.cos(math.radians(z_rot))
    sz = math.sin(math.radians(z_rot))
    rot = numpy.array([[cy * cz, (sx * sy * cz) + (cx * sz), -(cx * sy * sz) + (sx * sz), 0],
                        [-(cy * sz), -(sx * sy * sz) + (cx * cz), (cx * sy * sz) + sx * cz, 0],
                        [sy, -(sx * cy), cx * cy, 0],
                        [0, 0, 0, 1]])
    return rot
new_view = numpy.eye(4) * rot(x_rot, 0, 0) * rot(0, y_rot, 0)
print(new_view)



# src_points = numpy.array([[853, 627], [890, 663], [918, 644], [961, 682]])
# dst_points = numpy.array([[0, 0], [0, 720], [1280, 720], [1280, 1440]])
src_points = numpy.array([[890, 663], [920, 645], [929, 705], [960, 683]])
dst_points = numpy.array([[0, 0], [200, 0], [0, 200], [200, 200]])
homography, status = cv2.findHomography(src_points, dst_points)
print(homography)


img = Image.open('data/1.png')
img = numpy.array(img)
h, w, c = img.shape
img = cv2.resize(img, (1280, 720))
img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
h, w, c = img.shape
print(img)

dirt_img = numpy.array(Image.open('block/dirt.png'))
dirt_img = cv2.cvtColor(dirt_img, cv2.COLOR_BGRA2RGB)
dirt_img = cv2.resize(dirt_img, (30, 30), interpolation=cv2.INTER_NEAREST)
# cv2.imshow('dirt_img', dirt_img)

result = cv2.matchTemplate(img, dirt_img, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(min_val, max_val)
result = result < 10000000
result = numpy.pad(result, [(0, dirt_img.shape[0] - 1)], 'constant', constant_values=0)
result = numpy.expand_dims(result, 2)
result = img * result
result = result.astype(numpy.uint8) * 255
# cv2.imshow('result', result)
# cv2.waitKey()


img_out = cv2.warpPerspective(img, homography, (400, 400))
# cv2.imshow('img_out', img_out)
# cv2.waitKey()

# img = cv2.GaussianBlur(img, (5, 5), 0, 0)
# img = cv2.Canny(img, 75, 125)

img_ori = img


model = numpy.eye(4, dtype=numpy.float32)
glm.translate(model, 0, 0, -10)
view = numpy.eye(4, dtype=numpy.float32)
roll = 0
yaw = 0
projection = glm.perspective(70, float(w) / h, 0.1, 100)

while True:
    img = img_ori.copy()
    # view = numpy.eye(4, dtype=numpy.float32)
    # glm.rotate(view, yaw, 0, 1, 0)
    # glm.rotate(view, roll, 1, 0, 0)
    cx = math.cos(math.radians(yaw))
    sx = math.sin(math.radians(yaw))
    cy = math.cos(math.radians(roll))
    sy = math.sin(math.radians(roll))
    cz = math.cos(math.radians(0))
    sz = math.sin(math.radians(0))
    view = numpy.array([[cy * cz, (sx * sy * cz) + (cx * sz), -(cx * sy * sz) + (sx * sz), 0],
                            [-(cy * sz), -(sx * sy * sz) + (cx * cz), (cx * sy * sz) + sx * cz, 0],
                            [sy, -(sx * cy), cx * cy, 0],
                            [0, 0, 0, 1]])
    view = rot(roll, 0, 0) @ rot(0, yaw, 0)

    vertices = numpy.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1],
                            [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    vertices = numpy.column_stack((vertices, numpy.ones((vertices.shape[0], 1)))) @ model @ view @ projection
    vertices = vertices[:, :2] / numpy.reshape(vertices[:, 3], (vertices.shape[0], 1))
    vertices = (vertices + 1) * numpy.array([w / 2, h / 2])
    indices = [[0, 1, 2, 3], [0, 1, 6, 5], [0, 3, 4, 5], [1, 6, 7, 2], [3, 4, 7, 2], [4, 5, 6, 7]]
    polygon = [numpy.array([vertices[indices[i]]]).astype(int) for i in range(len(indices))]
    cv2.polylines(img, polygon, True, 255)

    cv2.imshow('img', img)
    res = cv2.waitKeyEx(0)
    print(res)
    print(model)
    if res == ord('a'): # Left
        glm.translate(model, -0.1, 0, 0)
    elif res == ord('d'): # Right
        glm.translate(model, 0.1, 0, 0)
    elif res == ord('q'): # Up
        glm.translate(model, 0, -0.1, 0)
    elif res == ord('e'): # Down
        glm.translate(model, 0, 0.1, 0)
    elif res == ord('w'): # Up
        glm.translate(model, 0, 0, -0.1)
    elif res == ord('s'): # Down
        glm.translate(model, 0, 0, 0.1)
    elif res == ord('r'): # Up
        roll += 1
    elif res == ord('f'): # Down
        roll += -1
    elif res == ord('t'): # Up
        yaw += 1
    elif res == ord('g'): # Down
        yaw += -1