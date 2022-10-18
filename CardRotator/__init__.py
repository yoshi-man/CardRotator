"""
Created on Tue Oct 18 20:17:30 2022
@author: yoshi-man
"""

import os

import numpy as np
import math

import cv2
from PIL import Image, ImageDraw


class CardRotator():

    def __detect_file_pairs(self):
        valid_image_pairs = []

        input_files = os.listdir(self.input_folder)
        input_images = list(filter(lambda x: (x[-3:] == 'jpg'), input_files))

        for image in input_images:
            file_is_front_image = ("_front.jpg" in image)
            file_has_back_image = (f"{image[:-10]}_back.jpg" in input_images)

            if file_is_front_image and file_has_back_image:
                valid_image_pairs.append(f"{image[:-10]}")

        return valid_image_pairs

    def __init__(self, input_folder: str, frames: int = 240, speed: int = 60, buffer_px: int = 100, zoom_factor: int = 50, verbose: bool = True):
        """Initialise a rotator object. Use .run(output_path: str) to start generating. 

        Keyword arguments:
        input_folder: str   -- absolute path of where the input images resides
        frames: int         -- number of frames, controls the smoothness (default 240)
        speed: int          -- speed of rotation, play around with this number (default 60)
        buffer_px: int      -- the black frame to extend the image by (default 100)
        zoom_factor: int    -- affects the field of view, play around with this (default 50)
        verbose: bool       -- if you'd like to have printed statements (default True)
        """
        self.input_folder = input_folder
        self.file_names = self.__detect_file_pairs()
        self.frames = frames
        self.buffer_px = buffer_px
        self.zoom_factor = zoom_factor
        self.verbose = verbose

        if verbose:
            print(f"Valid Image Pairs Detected: {len(self.file_names)}")

    def __generate_frames(self, file_name: str, output_path: str):
        temp_folder_exists = os.path.exists(f"{output_path}/temp")

        if not temp_folder_exists:
            os.makedirs(f"{output_path}/temp")

        for i in range(0, self.frames):
            angle_increment = 2*math.pi/self.frames
            angle = angle_increment * i

            # between 90 degrees and 270 degrees, we will use the BACK face, else use the front face
            face = "back" if (angle >= math.pi/2 and angle <=
                              3*math.pi/2) else "front"

            img = cv2.imread(f"{self.input_folder}/{file_name}_{face}.jpg")

            if face == "back":
                img = cv2.flip(img, 1)

            length, width, _ = img.shape

            # add a buffer black frame of buffer_px size, so when it rotates, our image remains in frame
            img = cv2.copyMakeBorder(img, self.buffer_px, self.buffer_px, self.buffer_px,
                                     self.buffer_px, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

            # pinpoint the coordinates of the original corners, this is used to trace the trajectory depending on angle
            bottom_left = [self.buffer_px, self.buffer_px+length]
            bottom_right = [self.buffer_px+width, self.buffer_px+length]
            top_right = [self.buffer_px+width, self.buffer_px]
            top_left = [self.buffer_px, self.buffer_px]

            # now trace where the four corners will be at depending on the angle
            trans_bottom_left = [self.buffer_px + width*math.pow(
                math.sin(angle/2), 2), bottom_left[1] + self.zoom_factor*math.sin(angle)]
            trans_bottom_right = [bottom_right[0] - width*math.pow(
                math.sin(angle/2), 2), bottom_right[1] - self.zoom_factor*math.sin(angle)]
            trans_top_right = [top_right[0] - width*math.pow(
                math.sin(angle/2), 2), self.buffer_px + self.zoom_factor*math.sin(angle)]
            trans_top_left = [self.buffer_px + width*math.pow(
                math.sin(angle/2), 2), self.buffer_px - self.zoom_factor*math.sin(angle)]

            input_pts = np.float32(
                [bottom_left, bottom_right, top_right, top_left])
            output_pts = np.float32(
                [trans_bottom_left, trans_bottom_right, trans_top_right, trans_top_left])

            # use perspective transform to fit our image into the transformed points
            M = cv2.getPerspectiveTransform(input_pts, output_pts)
            out = cv2.warpPerspective(
                img, M, (width+2*self.buffer_px, length+2*self.buffer_px), flags=cv2.INTER_LINEAR)

            cv2.imwrite(f"{output_path}/temp/{file_name}_{str(i)}.png",
                        out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def __combine_frames(self, file_name: str, output_path: str):
        frame_images = []

        for f in range(self.frames):
            temp_frame = Image.open(
                f"{output_path}/temp/{file_name}_{str(f)}.png")
            frame_images.append(temp_frame)

        im1 = frame_images[0]
        im1.info['duration'] = self.speed
        im1.save(f"{output_path}/{file_name}.gif", save_all=True,
                 append_images=frame_images[1:], loop=0, optimize=True, quality=80)

        for f in range(self.frames):
            if os.path.exists(f"{output_path}/temp/{file_name}_{str(f)}.png"):
                os.remove(f"{output_path}/temp/{file_name}_{str(f)}.png")

    def run(self, output_path: str):
        if self.verbose:
            print(
                f"Generating gifs for {len(self.file_names)} pairs of images...")
        for file_name in self.file_names:
            if self.verbose:
                print(f"Generating GIF for {file_name}...", end='\r')
            self.__generate_frames(file_name, output_path)
            self.__combine_frames(file_name, output_path)
            if self.verbose:
                print(f"Generating GIF for {file_name}...Done.")
