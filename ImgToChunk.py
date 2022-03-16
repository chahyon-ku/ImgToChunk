import math
import os

from glumpy import glm
from PIL import Image, ImageTk
import numpy
import tkinter
import cv2


def load_image(file_name, size):
    image = Image.open(file_name)
    image = numpy.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    image = Image.fromarray(image)
    photo_image = ImageTk.PhotoImage(image)
    return image, photo_image


class ImgToChunk():
    def __init__(self):
        super(ImgToChunk, self).__init__()
        self.window = tkinter.Tk(className='ImgToChunk')
        self.window.geometry('1600x900')
        self.window.bind('w', lambda event: self.key_fn('w'))
        self.window.bind('a', lambda event: self.key_fn('a'))
        self.window.bind('s', lambda event: self.key_fn('s'))
        self.window.bind('d', lambda event: self.key_fn('d'))
        self.window.bind('q', lambda event: self.key_fn('q'))
        self.window.bind('e', lambda event: self.key_fn('e'))
        self.window.bind('r', lambda event: self.key_fn('r'))
        self.window.bind('f', lambda event: self.key_fn('f'))
        self.window.bind('t', lambda event: self.key_fn('t'))
        self.window.bind('g', lambda event: self.key_fn('g'))


        self.image, self.photo_image = load_image('data/1.png', (1280, 720))
        self.image_canvas = tkinter.Canvas(self.window, width=1280, height=720)
        self.image_canvas.create_image((0, 0), image=self.photo_image, anchor='nw')
        self.image_canvas.place(x=10, y=10)
        self.image_canvas.bind("<Button-1>", self.image_click_fn)

        self.cube_x = 0
        self.cube_y = 0
        self.cube_z = -5
        self.yaw = 0
        self.roll = 0
        self.projection = glm.perspective(70, float(1280) / 720, 0.1, 100)
        self.cube_image = self.image
        self.cube_image_tk = self.photo_image


        self.grid_canvases = []
        self.grid_images = []
        self.grid_photo_images = []
        for row in range(6):
            self.grid_canvases.append([])
            self.grid_images.append([])
            self.grid_photo_images.append([])
            for col in range(2):
                grid_image, grid_photo_image = load_image('block/dirt.png', (125, 125))
                self.grid_images[row].append(grid_image)
                self.grid_photo_images[row].append(grid_photo_image)
                canvas = tkinter.Canvas(self.window, width=125, height=125)
                canvas.place(x=1300 + col * 125, y=0 + row * 125)
                canvas.create_image((0, 0), image=grid_photo_image, anchor='nw')
                self.grid_canvases[row].append(canvas)

        # self.x_line = None
        # self.y_line = None
        #
        # self.image_x_label = tkinter.Label(self.window, text='image x:')
        # self.image_x_label.place(x=1300, y=10)
        # self.image_x_label.config(font=("Courier", 15))
        # self.image_x_entry = tkinter.Entry(self.window)
        # self.image_x_entry.place(x=1400, y=10)
        # self.image_x_entry.config(font=("Courier", 15), width=10)
        #
        # self.image_y_label = tkinter.Label(self.window, text='image y:')
        # self.image_y_label.place(x=1300, y=35)
        # self.image_y_label.config(font=("Courier", 15))
        # self.image_y_entry = tkinter.Entry(self.window)
        # self.image_y_entry.place(x=1400, y=35)
        # self.image_y_entry.config(font=("Courier", 15), width=10)
        #
        # self.image_lines = []
        #
        # self.update_button = tkinter.Button(self.window, text='Update', width=10, command=self.update_button_fn)
        # self.update_button.place(x=1400, y=65)
        #
        # self.selected_entry_row = 0
        # self.selected_entry_col = 0
        # self.grid_entries = []
        # self.entry_grids = {}
        # for row in range(10):
        #     self.grid_entries.append([])
        #     for col in range(2):
        #         entry = tkinter.Entry(self.window)
        #         entry.place(x=1300 + 50 * col, y=65 + 25 * row)
        #         entry.config(font=("Courier", 15), width=4)
        #         entry.bind('<1>', self.entry_click_fn)
        #         entry.bind('<Enter>', self.update_button_fn)
        #         self.grid_entries[row].append(entry)
        #         self.entry_grids[entry] = (row, col)

        self.block_images = []
        for root, dir, files in os.walk('block_subset'):
            for file in files:
                path = os.path.join(root, file)
                block_image, block_image_tk = load_image(path, (200, 200))
                self.block_images.append(block_image)

        self.window.mainloop()

    def key_fn(self, key):
        if key == 'a':  # Left
            self.cube_x -= 0.1
        elif key == 'd':  # Right
            self.cube_x += 0.1
        elif key == 'q':  # Up
            self.cube_y -= 0.1
        elif key == 'e':  # Down
            self.cube_y += 0.1
        elif key == 'w':  # Up
            self.cube_z -= 0.1
        elif key == 's':  # Down
            self.cube_z += 0.1
        elif key == 'r':  # Up
            self.roll += 1
        elif key == 'f':  # Down
            self.roll += -1
        elif key == 't':  # Up
            self.yaw += 1
        elif key == 'g':  # Down
            self.yaw += -1

        view = numpy.eye(4, dtype=numpy.float32)
        glm.rotate(view, self.yaw, 0, 1, 0)
        glm.rotate(view, self.roll, 1, 0, 0)

        model = numpy.eye(4, dtype=numpy.float32)
        glm.translate(model, self.cube_x, self.cube_y, self.cube_z)

        vertices = numpy.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1],
                                [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
        vertices = numpy.column_stack((vertices, numpy.ones((vertices.shape[0], 1)))) @ model @ view @ self.projection
        vertices = vertices[:, :2] / numpy.reshape(vertices[:, 3], (vertices.shape[0], 1))
        vertices = (vertices + 1) * numpy.array([1280 / 2, 720 / 2])
        indices = [[0, 1, 2, 3], [0, 1, 6, 5], [0, 5, 4, 3], [1, 6, 7, 2], [3, 4, 7, 2], [5, 6, 7, 4]]
        polygons = [numpy.array([vertices[indices[i]]]).astype(int) for i in range(len(indices))]

        self.cube_image = numpy.array(self.image, numpy.uint8)
        cv2.polylines(self.cube_image, polygons, True, 255)
        self.cube_image = Image.fromarray(self.cube_image)
        self.cube_image_tk = ImageTk.PhotoImage(self.cube_image)
        self.image_canvas.delete('all')
        self.image_canvas.create_image((0, 0), image=self.cube_image_tk, anchor='nw')

        dst_points = numpy.array([[125, 125], [0, 125], [0, 0], [125, 0]])
        for row, polygon in enumerate(polygons):
            homography, status = cv2.findHomography(polygon, dst_points)
            tile_image = numpy.array(self.image, numpy.uint8)
            tile_image = cv2.warpPerspective(tile_image, homography, (125, 125))
            self.grid_images[row][0] = Image.fromarray(tile_image)
            self.grid_images[row][1] = self.get_most_similar_image(self.grid_images[row][0])
            self.grid_photo_images[row][0] = ImageTk.PhotoImage(self.grid_images[row][0])
            self.grid_photo_images[row][1] = ImageTk.PhotoImage(self.grid_images[row][1])
            self.grid_canvases[row][0].delete('all')
            self.grid_canvases[row][0].create_image((0, 0), image=self.grid_photo_images[row][0], anchor='nw')
            self.grid_canvases[row][1].delete('all')
            self.grid_canvases[row][1].create_image((0, 0), image=self.grid_photo_images[row][1], anchor='nw')


    def image_click_fn(self, event):
        self.image_x_entry.delete(0, tkinter.END)
        self.image_x_entry.insert(0, '{}'.format(event.x))
        self.image_y_entry.delete(0, tkinter.END)
        self.image_y_entry.insert(0, '{}'.format(event.y))
        if self.selected_entry_row is not None:
            self.grid_entries[self.selected_entry_row][0].delete(0, tkinter.END)
            self.grid_entries[self.selected_entry_row][0].insert(0, '{}'.format(event.x))
            self.grid_entries[self.selected_entry_row][1].delete(0, tkinter.END)
            self.grid_entries[self.selected_entry_row][1].insert(0, '{}'.format(event.y))
            self.selected_entry_row += 1
            if self.selected_entry_row > 3:
                self.selected_entry_row = 0
        if self.x_line:
            self.image_canvas.delete(self.x_line)
        if self.y_line:
            self.image_canvas.delete(self.y_line)
        self.x_line = self.image_canvas.create_line(event.x, event.y - 10, event.x, event.y + 10, fill='white', width=2)
        self.y_line = self.image_canvas.create_line(event.x - 10, event.y, event.x + 10, event.y, fill='white', width=2)
        self.update_button_fn()

    def entry_click_fn(self, event: tkinter.Event):
        if event.widget in self.entry_grids:
            self.selected_entry_row = self.entry_grids[event.widget][0]
            self.selected_entry_col = self.entry_grids[event.widget][1]
        print(self.selected_entry_row, self.selected_entry_col)

    def update_button_fn(self, event=None):
        src_points = []
        for row in range(4):
            src_points.append([])
            for col in range(2):
                s_val = self.grid_entries[row][col].get()
                if s_val.isdigit():
                    src_points[row].append(int(s_val))
                else:
                    return

        for image_line in self.image_lines:
            self.image_canvas.delete(image_line)
        self.image_lines = [self.image_canvas.create_line(src_points[0][0],
                                                          src_points[0][1],
                                                          src_points[1][0],
                                                          src_points[1][1],
                                                          fill='white', width=2),
                            self.image_canvas.create_line(src_points[1][0],
                                                          src_points[1][1],
                                                          src_points[2][0],
                                                          src_points[2][1],
                                                          fill='white', width=2),
                            self.image_canvas.create_line(src_points[2][0],
                                                          src_points[2][1],
                                                          src_points[3][0],
                                                          src_points[3][1],
                                                          fill='white', width=2),
                            self.image_canvas.create_line(src_points[3][0],
                                                          src_points[3][1],
                                                          src_points[0][0],
                                                          src_points[0][1],
                                                          fill='white', width=2)
                            ]

        src_points = numpy.array(src_points)
        dst_points = numpy.array([[0, 0], [0, 200], [200, 200], [200, 0]])
        homography, status = cv2.findHomography(src_points, dst_points)
        # self.trans_tile_canvas.delete('all')
        # self.trans_tile_image = cv2.warpPerspective(numpy.array(self.image), homography, (200, 200))
        # self.trans_tile_image = Image.fromarray(self.trans_tile_image)
        # self.trans_tile_photo_image = ImageTk.PhotoImage(self.trans_tile_image)
        # self.trans_tile_canvas.create_image((0, 0), image=self.trans_tile_photo_image, anchor='nw')
        #
        # self.pred_tile_canvas.delete('all')
        # self.pred_tile_image = self.get_most_similar_image(self.trans_tile_image)
        # self.pred_tile_photo_image = ImageTk.PhotoImage(self.pred_tile_image)
        # self.pred_tile_canvas.create_image((0, 0), image=self.pred_tile_photo_image, anchor='nw')

    def get_most_similar_image(self, trans_tile_image):
        min_result = math.inf
        min_image = None
        for i, block_image in enumerate(self.block_images):
            trans_tile_image2 = cv2.resize(numpy.array(trans_tile_image), (16, 16))
            block_image2 = cv2.resize(numpy.array(block_image), (16, 16))
            result = trans_tile_image2.astype(int) - block_image2
            result = numpy.sum(numpy.abs(result))
            #result = result * result
            #result = numpy.sum(result)
            # cv2.imshow('abc', numpy.array(trans_tile_image))
            # cv2.waitKey()
            # cv2.imshow('abc', numpy.array(block_image))
            # cv2.waitKey()
            if result < min_result:
                min_result = result
                min_image = block_image
        return min_image


img_to_chunk = ImgToChunk()
