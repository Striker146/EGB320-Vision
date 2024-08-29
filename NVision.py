import pickle
import time
from math import *

process_start_time = time.time()

import cv2

import numpy as np

standardWidth = 640
standardHeight = 480
res_scale = 1
width = standardWidth * res_scale
height = standardHeight * res_scale


def get_focal_length(pixel_size, distance, actual_width):
    return (pixel_size * distance) / actual_width


# 673
focalLength = get_focal_length(pixel_size=152 * res_scale, distance=300, actual_width=70)


class Obj_Args:
    def __init__(self, lower_hsv, higher_hsv, color_fill, color_outline, dim, name, lower_hsv_2=None, higher_hsv_2=None,
                 min_size=0, wall=False):

        self.lower_hsv = lower_hsv
        self.higher_hsv = higher_hsv

        if lower_hsv_2 is not None and higher_hsv_2 is not None:
            self.lower_hsv_2 = lower_hsv_2
            self.higher_hsv_2 = higher_hsv_2
            self.multi_hsv = True
        else:
            self.multi_hsv = False

        self.color_fill = color_fill
        self.color_outline = color_outline
        self.dim = dim
        self.name = name
        self.min_size = min_size
        self.wall = wall
        self.p_pipe_1, self.c_pipe_1 = Pipe()
        self.p_pipe_2, self.c_pipe_2 = Pipe()

class RoverWallSegment:
    def __init__(self,point,cam_height):
        self.point = point
        self.bearingx = round((self.point[0] - width / 2) * (65 / width), 2)
        self.bearingy = round((self.point[1] - height / 2) * (70 / height), 2)
        trueAngle = 90 - 15 - self.bearingy
        self.distance = round(cam_height * tan(radians(trueAngle)),2)


    def toString(self):
        stringass = f"d : {int(self.distance / 10)} cm, b : ({self.bearingx,self.bearingy}"
        return stringass






class RoverWall:
    def __init__(self,contour,color_fill,color_outline,name):
        self.contour = contour
        self.color_fill = color_fill
        self.color_outline = color_outline
        self.name = name
        self.wallsegments = self.get_segments(contour, height)

    def get_segments(self,contour,height):
        segments = []
        for count, point in enumerate(contour):
            if count % 5 == 0:
                a = point[0]
                segments.append(RoverWallSegment(point=a,cam_height=115))
        return segments





class RoverObject:
    # __slots__ = "contour","x","y","w","h","midx","midy","bearingx","bearingy","objectWidth","objectHeight","distance_w","distance_h","distance","distance_error","colorfill","coloroutline","name"
    def __init__(self, contour, color_fill, color_outline, object_dim, name):
        self.contour = contour
        x, y, w, h = cv2.boundingRect(contour)
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.midx = int(self.x + self.w / 2)
        self.midy = int(self.y + self.h / 2)
        self.bearingx = round((self.midx - width / 2) * (65 / width), 4)
        self.bearingy = round((self.midy - height / 2) * (65 / height), 4)
        self.objectWidth = object_dim[0]
        self.objectHeight = object_dim[1]
        distances = self.distance_to_camera()
        self.distance_w = int(distances[0])
        self.distance_h = int(distances[1])
        self.distance = int((self.distance_h + self.distance_w) / 2)
        self.distance_error = abs(int(self.distance_w - self.distance_h))
        self.color_fill = color_fill
        self.coloroutline = color_outline
        self.name = name

    def get_side_bearingx(self):
        a = (self.x - width / 2) * (65 / width)
        b = (self.x + self.w - width / 2) * (65 / width)
        return a, b

    def get_midx(self):
        return int(self.x + self.w / 2)

    def to_string(self):
        stringassembly = f"{self.name} : d {self.distance / 10} +- {self.distance_error / 10}cm  b ({self.bearingx},{self.bearingy})"
        debugString = f"mid ({self.midx}, {self.midy}), C ({self.x},{self.y}), w {self.w}, h {self.h} w dis {round(self.distance_w)} h dis {round(self.distance_h)}"
        return stringassembly, debugString

    def distance_to_camera(self):
        # compute and return the distance from the maker to the camera
        return ((self.objectWidth * focalLength) / self.w, (self.objectHeight * focalLength) / self.h)


class RoverObjects:
    # __slots__ = "rocks","obstacles","samples","landers"
    def __init__(self, rocks, obstacles, samples, landers, walls):
        self.rocks = rocks
        self.obstacles = obstacles
        self.samples = samples
        self.landers = landers
        self.walls = walls

    def draw_rover_wall(self,wall : RoverWall,drawn_frame):
        drawn_frame = cv2.drawContours(drawn_frame, [wall.contour], 0, wall.color_outline, 2)

        #drawn_frame = cv2.drawContours(drawn_frame, [wall.contour], 0, wall.color_fill, -1)

        color = (0, 255, 0)
        thickness = 9
        for count, segment in enumerate(wall.wallsegments):
            if segment.distance > 0:
                a = segment.point


                cirlcethickness = int(5)
                circlecolor = wall.color_fill
                circlepos = (a[0], a[1])
                cv2.circle(drawn_frame, circlepos, 2, circlecolor, cirlcethickness)
                circlepos = (a[0] + 10,a[1])
                textthickness = 1
                font = cv2.FONT_HERSHEY_TRIPLEX
                fontScale = 0.4

                cv2.putText(drawn_frame, segment.toString(), circlepos, font, fontScale, (0, 255, 255), textthickness,
                            cv2.LINE_AA)


        return drawn_frame

    def draw_rover_objects(self, drawn_frame, draw_contours=True, draw_rectangles=True, draw_dots=True,
                           draw_main_text=True, draw_debug_text=False, draw_fill=False,
                           fps=-1):

        all_rover_objs = [self.rocks, self.obstacles, self.samples, self.landers]
        # all_rover_objs = [self.samples]
        drawn_frame = cv2.drawMarker(drawn_frame, (round(width / 2), round(height / 2)), (0, 0, 255), cv2.MARKER_CROSS,
                                     int(30), int(5))

        for wall in self.walls:
            drawn_frame = self.draw_rover_wall(wall,drawn_frame)

        for rover_objs in all_rover_objs:
            for rover_obj in rover_objs:
                self.draw_rover_object(rover_obj, drawn_frame, draw_contours, draw_rectangles, draw_dots,
                                       draw_main_text,
                                       draw_debug_text, draw_fill=draw_fill)


        if fps != -1:
            cv2.putText(drawn_frame, "FPS : " + str(round(fps)), (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0),
                        1, cv2.FILLED)
        return drawn_frame

    def draw_rover_object(self, cnt, frame, draw_contours=False, draw_rectangles=False, draw_dots=False,
                          draw_main_text=False, draw_debug_text=False, draw_fill=False):
        if draw_contours:
            cv2.drawContours(frame, [cnt.contour], 0, (0, 0, 0), 2)

        if draw_fill:
            cv2.drawContours(frame, [cnt.contour], 0, cnt.color_fill, -1)

        if draw_rectangles:
            frame = cv2.rectangle(frame, (cnt.x, cnt.y), (cnt.x + cnt.w, cnt.y + cnt.h), (255, 255, 0))

        if draw_main_text or draw_debug_text:
            textthickness = 1
            font = cv2.FONT_HERSHEY_TRIPLEX
            fontScale = 0.4
            textcolor = (255, 255, 255)
            t1poistion = (cnt.midx - (100), cnt.midy + 15)
            t2poistion = (cnt.midx - (100), cnt.midy + 30)
            mainInfo, debugInfo = cnt.to_string()

        if draw_main_text:
            frame = cv2.putText(frame, mainInfo, t1poistion, font, fontScale, (0, 0, 0), textthickness * 2,
                                cv2.LINE_AA)
            frame = cv2.putText(frame, mainInfo, t1poistion, font, fontScale, textcolor, textthickness, cv2.LINE_AA)

        if draw_debug_text:
            frame = cv2.putText(frame, debugInfo, t2poistion, font, fontScale, (0, 0, 0), textthickness * 2,
                                cv2.LINE_AA)
            frame = cv2.putText(frame, debugInfo, t2poistion, font, fontScale, textcolor, textthickness,
                                cv2.LINE_AA)

        if draw_dots:
            cirlcethickness = int(5)
            cpoistion = (cnt.midx, cnt.midy)
            circlecolor = cnt.color_fill
            cv2.circle(frame, cpoistion, 2, circlecolor, cirlcethickness)
        return frame


def process_frame_pipe(obj_args: Obj_Args):
    while (True):
        hsv_serialized = obj_args.p_pipe_2.recv()
        if not hsv_serialized:
            break
        hsv = pickle.loads(hsv_serialized)
        start_time = time.time()
        if obj_args.wall:
            kernel = np.ones((10, 10), np.uint8)
        else:
            kernel = np.ones((6, 6), np.uint8)
        mask = cv2.inRange(hsv, obj_args.lower_hsv, obj_args.higher_hsv)
        if obj_args.multi_hsv:
            mask2 = cv2.inRange(hsv, obj_args.lower_hsv_2, obj_args.higher_hsv_2)
            mask = np.add(mask, mask2)

        # mask = cv2.erode(mask, kernel, iterations=1)
        # mask = cv2.dilate(mask, kernel, iterations=4)
        # mask = cv2.erode(mask, kernel, iterations=2)
        contours, hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rover_subset_objects = []
        for cnt in contours:
            rover_subset_objects.append(
                RoverObject(cnt, obj_args.color_fill, obj_args.color_outline, obj_args.dim, obj_args.name))
        end_time = time.time()
        seconds = round(end_time - start_time, 6)
        obj_args.c_pipe_1.send(rover_subset_objects)


def process_frame(hsv, obj_args: Obj_Args):
    if obj_args.wall:
        kernel = np.ones((18, 18), np.uint8)
    else:
        kernel = np.ones((12, 12), np.uint8)
    mask = cv2.inRange(hsv, obj_args.lower_hsv, obj_args.higher_hsv)
    if obj_args.multi_hsv:
        mask2 = cv2.inRange(hsv, obj_args.lower_hsv_2, obj_args.higher_hsv_2)
        mask = np.add(mask, mask2)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if obj_args.wall:
        contours = filter(lambda x: obj_args.min_size < cv2.contourArea(x), contours)

    rover_subset_objects = []
    for cnt in contours:
        if (obj_args.wall):
            obj = RoverWall(cnt, obj_args.color_fill, obj_args.color_outline,obj_args.name)
        else:
            obj = RoverObject(cnt, obj_args.color_fill, obj_args.color_outline, obj_args.dim, obj_args.name)
        rover_subset_objects.append(obj)
    if not obj_args.wall:
        rover_subset_objects.sort(key=lambda x: x.distance, reverse=False)
    return rover_subset_objects


from multiprocessing import pool
from multiprocessing import Process
from multiprocessing import Pipe
from multiprocessing import freeze_support

import copy

lower_blue = np.array([90, 70, 54])
higher_blue = np.array([130, 255, 255])

blue_args = Obj_Args(lower_hsv=lower_blue, higher_hsv=higher_blue, color_fill=(255, 0, 0),
                     color_outline=(255, 0, 0), dim=(70, 70), name="rock")

lower_green = np.array([40, 111, 60])
higher_green = np.array([70, 255, 255])
green_args = Obj_Args(lower_hsv=lower_green, higher_hsv=higher_green, color_fill=(0, 255, 0),
                      color_outline=(0, 255, 0), dim=(150, 150), name="obstacle")

lower_orange = np.array([0, 120, 50])
higher_orange = np.array([15, 255, 255])
lower_orange_2 = np.array([160, 120, 50])
higher_orange_2 = np.array([179, 255, 255])

orange_args = Obj_Args(lower_hsv=lower_orange, higher_hsv=higher_orange, color_fill=(0, 165, 255),
                       color_outline=(128, 128, 255), dim=(42.67, 42.67), name="sample", lower_hsv_2=lower_orange_2,
                       higher_hsv_2=higher_orange_2)

lower_yellow = np.array([20, 80, 130])
higher_yellow = np.array([31, 255, 255])
yellow_args = Obj_Args(lower_hsv=lower_yellow, higher_hsv=higher_yellow, color_fill=(255, 255, 255),
                       color_outline=(0, 0, 0), dim=(572, 54), name="lander")

lower_black = np.array([0, 0, 0])
higher_black = np.array([179, 255, 93])
black_args = Obj_Args(lower_hsv=lower_black, higher_hsv=higher_black, color_fill=(128, 128, 128),
                      color_outline=(255,255,255), dim=(572, 54), name="Wall",wall=True, min_size=10000)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, standardWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, standardHeight)
fps_list_pipe = []
fps_list_single = []
init_time = time.time()
frame_pool = pool.ThreadPool(processes=2)
args = [blue_args, green_args, orange_args, yellow_args, black_args]
jobs = []


def start_multiprocessing():
    for arg in args:
        p = Process(target=process_frame_pipe, args=(arg,))
        jobs.append(p)
        p.start()


def get_rover_objects(processed_frame, get_new_frame: bool = True, frame_name="frame", draw_cv=True):
    start_time = time.time()
    if get_new_frame:
        frame_thread = frame_pool.apply_async(cap.read)
    resized = cv2.resize(processed_frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # blur_hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

    rock_obj = process_frame(hsv, blue_args)
    obstacle_obj = process_frame(hsv, green_args)
    sample_obj = process_frame(hsv, orange_args)
    lander_obj = process_frame(hsv, yellow_args)
    vvv = process_frame(hsv, black_args)

    rover_objects = RoverObjects(rock_obj, obstacle_obj, sample_obj, lander_obj, vvv)

    if get_new_frame:
        _, new_frame = frame_thread.get()
    else:
        new_frame = processed_frame

    multi_end = time.time()

    fps_pipe = round(1 / (multi_end - start_time))

    fps_list_pipe.append(fps_pipe)

    fps_avg_multi = np.mean(fps_list_pipe)

    drawn_frame = rover_objects.draw_rover_objects(resized, fps=fps_avg_multi, draw_main_text=True, draw_contours=False,
                                                   draw_fill=True)
    if draw_cv:
        cv2.imshow(frame_name, drawn_frame)

    return rover_objects, new_frame, drawn_frame


def get_rover_objects_multi(processed_frame, get_new_frame: bool = True, frame_name="frame", draw_cv=True):
    try:
        start_time = time.time()
        if get_new_frame:
            frame_thread = frame_pool.apply_async(cap.read)
        resized = cv2.resize(processed_frame, (width, height), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        # hsv = cv2.GaussianBlur(hsv, (17, 17), 0)
        serialized_hsv = pickle.dumps(hsv)
        for arg in args:
            send_start = time.time()
            arg.c_pipe_2.send(serialized_hsv)
            close_time = time.time()

        rock_obj = blue_args.p_pipe_1.recv()
        obstacle_obj = green_args.p_pipe_1.recv()
        sample_obj = orange_args.p_pipe_1.recv()
        lander_obj = yellow_args.p_pipe_1.recv()

        rover_objects_piped = RoverObjects(rock_obj, obstacle_obj, sample_obj, lander_obj)

        if get_new_frame:
            _, new_frame = frame_thread.get()
        else:
            new_frame = processed_frame

        end_time = time.time()

        fps_pipe = round(1 / (end_time - start_time))

        fps_list_pipe.append(fps_pipe)

        fps_avg_multi = np.mean(fps_list_pipe)

        drawn_frame_piped = rover_objects_piped.draw_rover_objects(resized, fps=fps_avg_multi)
        if draw_cv:
            cv2.imshow(frame_name, drawn_frame_piped)

        return rover_objects_piped, new_frame, drawn_frame_piped

    except cv2.error:
        cap.release()
        cv2.destroyAllWindows()
