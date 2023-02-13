#!/usr/bin/env python3
# Copyright (c) 2021 Burak Can
# Copyright (c) 2022 ACS research group, Institute of Computer Science, University of Tartu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""This module captures an image from the camera"""

from typing import Dict, List, Tuple, Union
from queue import Queue
import queue
import time
import threading
import os
import tempfile
import re

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import dlib

from idcard_face_match.face_compare import get_face_locations, face_distance_to_conf
import face_recognition
import idcard_face_match.globals as globals
import pgi
pgi.require_version('GdkPixbuf', '2.0')
from pgi.repository import GdkPixbuf


# check if cupy is available
cupy = True
try:
    import cupy as cp
except:
    cupy = False

def to_gpu(a):
    return cp.asarray(a) if cupy else a

def from_gpu(a):
    return cp.asnumpy(a) if cupy else a

# class to show timing
class Timer():
    def __init__(self):
        self.s = self.beginning = time.time()
        self.output = '--'*20+'\n'

    def snap(self, text):
        self.output+= '%s: %.5f sec\n' % (text, (time.time()-self.s))
        self.s = time.time()

    def total(self, display=False):
        total_time = (time.time()-self.beginning)
        self.output+= 'Total: %.5f sec\n' % (total_time)
        self.output+= '--'*20
        if display:
            print(self.output)
        return total_time

# bufferless VideoCapture - captures frames in a thread and returns the latest frame
class VideoCapture:

    def __init__(self, name, resolution, mjpg=True):
        self.cap = cv2.VideoCapture(name, cv2.CAP_V4L)

        if not self.cap.isOpened():
            print("[-] VideoCapture(): Cannot open camera /dev/video%s" % (name))
            exit()

        if mjpg: # we need to use 'MJPG' to get max camera resolution in max frame rate
            print("[+] VideoCapture(): setting camera image format to MJPG")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))

        # properties must be read and set before the capturing starts
        if resolution:
            width, height = resolution
            print("[+] VideoCapture(): setting camera resolution to %sx%s" % (width, height))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


        # There are two strategies for JPG frame decoding:
        # If 'decode_in_main_thread' is set to 'True', the grabbed frames will not be decoded
        # by the frame grabbing thread grabber(). This will speed up the frame grabbing thread
        # as it will not spend time on frame decoding. However, the decoding will have to be done
        # in the main camera thread when the read() method is called inside the GUI frame construction loop.
        # If the CPU has fast cores that can perform frame decoding in the grabber() thread
        # maintaining 30 FPS then the 'decode_in_main_thread' should be set to 'False'.
        # Otherwise - for a single core CPU, it makes sense to continuously grab raw MJPG frames, but convert to RGB (decode)
        # in the read() method only those frames that will be processed/displayed in the GUI.
        self.decode_in_main_thread = False # assuming multi-core CPU
        if self.decode_in_main_thread:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        # frame rate (preferably, should be 30)
        val = self.cap.get(cv2.CAP_PROP_FPS)
        print("[+] VideoCapture(): CAP_PROP_FPS =", val)

        # image format (likely 'YUYV'/'MJPG', preferably should be MJPG)
        val = self.cap.get(cv2.CAP_PROP_FOURCC)
        print("[+] VideoCapture(): CAP_PROP_FOURCC =", int(val).to_bytes(4, byteorder='little').decode())

        # read the first frame to make sure that self.frame is not empty
        ret, self.frame = self.cap.read()

        # start the capturing thread
        self.t = threading.Thread(target=self.grabber)
        self.t.daemon = True
        self.t.start()

    # continuously grab frames as soon as they are available
    def grabber(self):

        current_second = 0
        current_second_frame_cnt = 0
        self.previous_second_frame_cnt = 0

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                print("[-] VideoCapture(): failure to grab frame")
                break

            # add the read frame to the frame count
            second = int(time.time())
            if current_second!=second:
                self.previous_second_frame_cnt = current_second_frame_cnt
                current_second = second
                current_second_frame_cnt = 1
            else:
                current_second_frame_cnt+= 1

    # retrieve the latest raw frame and decode it to RGB
    def read(self):

        if self.decode_in_main_thread:
            return cv2.imdecode(self.frame, cv2.IMREAD_UNCHANGED)

        return self.frame # assume that this stores already decoded frame

    # return number of frames read in the previous second
    def get_frame_count(self):
        return self.previous_second_frame_cnt

# generate a message box with text
def generate_msgbox_image(text, scale_factor, error=False, text_opacity=100):

    error_box_svg = """<svg width="424" height="96" viewBox="0 0 424 96" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="424" height="96" rx="48" fill="#F66D5F"/>
<path d="M 368 68 C 379 68 388 59 388 48 C 388 37 379 28 368 28 C 357 28 348 37 348 48 C 348 59 357 68 368 68 Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M 362.34 53.6598 L 373.66 42.3398" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M 373.66 53.6598 L 362.34 42.3398" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

    activity_box_svg = """<svg width="457" height="96" viewBox="0 0 457 96" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="457" height="96" rx="48" fill="white"/>
<path d="M 400.5 68 C 411.5 68 420.5 59 420.5 48 C 420.5 37 411.5 28 400.5 28 C 389.5 28 380.5 37 380.5 48 C 380.5 59 389.5 68 400.5 68 Z" stroke="#4DC85F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M 392 47.9998 L 397.66 53.6598 L 409 42.3398" stroke="#4DC85F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

    opacity_hex = bytes([255*text_opacity//100]).hex()
    if error:
        box_svg = error_box_svg
        color = '#ffffff'+opacity_hex
        base_length = 295 # length of the text for which the box has been generated
    else:
        box_svg = activity_box_svg
        color = '#4D3531'+opacity_hex
        base_length = 327 # length of the text for which the box has been generated

    # calculate the text length
    font_size = 31
    font = ImageFont.truetype('resources/PlusJakartaSans-SemiBold.ttf', font_size)
    image = Image.fromarray(np.zeros((0, 0), np.uint8))
    draw = ImageDraw.Draw(image)
    text_length = int(draw.textlength(text, font))
    base_length_difference = text_length - base_length

    # resize SVG appropriately
    # find all 3-digit numbers (x coordinates) in the SVG definition and adjust using base_length_difference
    for s in re.findall(r"[ \"]\d\d\d[ \.\"]", box_svg):
        box_svg = box_svg.replace(s, s[0] + str(int(s[1:4])+base_length_difference) + s[-1])

    # write the resized SVG to tempfile and load the file as an opencv image
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(box_svg)
            tmp.close()
            img = load_svg(path, scale_factor)
    finally:
        os.remove(path)

    # put text on a separate image and then compose together
    image = Image.fromarray(img)
    font = ImageFont.truetype('resources/PlusJakartaSans-SemiBold.ttf', int(font_size*scale_factor))
    text_image = Image.new('RGBA', image.size, (255,255,255,0))
    draw = ImageDraw.Draw(text_image)
    draw.text((34*scale_factor, 67*scale_factor), text, color, font=font, anchor="ld")
    combined = Image.alpha_composite(image, text_image)
    image = np.array(combined)
#    imgbytes = cv2.imencode(".png", image)[1].tobytes()
#    open("/tmp/box.png", 'wb').write(imgbytes)
    return image


# generate progress bar images for each percent (0 - 100)
def generate_progress_bar_images(w, h, scale_factor):

    w = int(w*scale_factor)
    h = int(h*scale_factor)

    ##
    ## Create a circle (in gradient coloring) with transparent background
    ##
    # generate a square image colored using vertical gradient
    gradient = Image.new('RGB', (w, h), "#E9EBF3") # start color
    top = Image.new('RGB', (w, h), "#ffffff") # end color
    mask = Image.new('L', (w, h))
    mask_data = []
    for y in range(h):
        mask_data.extend([int(255 * (y / h))] * w)
    mask.putdata(mask_data)
    gradient.paste(top, (0, 0), mask)
    # convert to opencv numpy array
    gradient = np.array(gradient)
    gradient = gradient[:, :, ::-1].copy()
    # draw a circle on the alpha channel
    # we draw a much bigger circle and then scale it down to obtain the antialiasing effect
    # (see https://stackoverflow.com/questions/14350645/is-there-an-antialiasing-method-for-python-pil)
    factor = 10
    alpha = Image.new('L', [h*factor,w*factor], 0)
    draw = ImageDraw.Draw(alpha)
    pie_thickness = int(6*scale_factor)
    draw.pieslice([(pie_thickness*factor,pie_thickness*factor), (h*factor-pie_thickness*factor,w*factor-pie_thickness*factor)], 0, 360, fill=255)
    alpha = alpha.resize((w, h), resample=Image.ANTIALIAS)
    # apply alpha mask (circle) on the colored square to obtain a circle in gradient color
    gradient_circle = np.dstack((gradient, np.array(alpha)))
    # set all transparent pixels to black (this is needed as we will later sum pixel values when underlaying the image)
    gradient_circle[gradient_circle[...,3]==0]=[0,0,0,0]

    ##
    ## Create a bigger circle (white) with transparent background
    ##
    # create a square image filled with white color
    white = np.zeros((h, w, 3), np.uint8)
    white[:] = (255,255,255)
    # draw a pie on the alpha channel
    alpha = Image.new('L', [h*factor,w*factor], 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([(0,0), (h*factor,w*factor)], 0, 360, fill=255)
    alpha = alpha.resize((w, h), resample=Image.ANTIALIAS)
    # apply alpha mask (pie) on the colored square to obtain pie in 7D859D color
    white_circle = np.dstack((white, np.array(alpha)))
    # set all transparent pixels to black (this is needed as we will later sum pixel values when underlaying the image)
    white_circle[white_circle[...,3]==0]=[0,0,0,0]

    ##
    ## Create a pie (in 7D859D color) with transparent background
    ##
    # create a square image filled with 7D859D color
    rgb7D859D = np.zeros((h, w, 3), np.uint8)
    rgb7D859D[:] = (157,133,125) # RGB color 7D859D

    images = {}

    print("[+] generate_progress_bar_images(): drawing progress bars...")
    for i in range(0,1001):

        percent = i/10

        # draw a pie on the alpha channel
        alpha = Image.new('L', [h*factor,w*factor], 0)
        draw = ImageDraw.Draw(alpha)
        start_angle = -90
        end_angle = -90 + 360/100*percent
        draw.pieslice([(0,0), (h*factor,w*factor)], start_angle, end_angle, fill=255)
        alpha = alpha.resize((w, h), resample=Image.ANTIALIAS)
        # apply alpha mask (pie) on the colored square to obtain pie in 7D859D color
        pie = np.dstack((rgb7D859D, np.array(alpha)))
        # set all transparent pixels to black (this is needed as we will later sum pixel values when underlaying the image)
        pie[pie[...,3]==0]=[0,0,0,0]

        ##
        ## Overlay the "gradient circle" image over the "pie" image overlayed over the "white circle"
        ##
        result = blend_images(background=white_circle, foreground=pie)
        result = blend_images(background=result, foreground=gradient_circle)

        # set all partly opaque pixels to fully opaque
#        opaq = result[result[..., 3] > 0]
#        opaq[...,3] = 255
#        result[result[..., 3] > 0] = opaq

        images[percent] = result


    return images


# generate colored percentage images for each match percent (0 - 100)
def generate_match_percent_images(img, color, scale_factor):

    # Unfortunately, freetype functionality will be available only in OpenCV release 5.0: [https://github.com/opencv/opencv-python/issues/305]
    # ft = cv2.freetype.createFreeType2()
    # therefore, for now using PIL
    font = ImageFont.truetype('resources/PlusJakartaSans-SemiBold.ttf', int(20*scale_factor))

    images = []
    print("[+] generate_match_percent_images(): drawing percent box images...")
    for percent in range(0,101):
        text = '%s%%' % (percent)
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        draw.text((img.shape[1]//2, img.shape[0]//2), text, color, font=font, anchor="mm")
        result = np.array(image)
        images.append(result)
    return images


# add fully opaque alpha channel to the image
def add_alpha(img):
    if img.shape[2] < 4:
        if type(img)==np.ndarray:
            ones = np.ones((img.shape[0], img.shape[1], 1), dtype = img.dtype) * 255
        else: # assume cupy
            ones = cp.ones((img.shape[0], img.shape[1], 1), dtype = img.dtype) * 255
        img = np.concatenate(
            [
                img,
                ones
            ],
            axis = 2,
        )
    return img

# remove alpha channel from the image
def remove_alpha(img):
    return img[..., :3]

# blend two transparent images (of same dimensions) together
def blend_images(background, foreground):
    t = Timer()

    # transparency factor for each pixel from 0 to 1
    mask_foreground = np.divide(foreground[..., 3:], 255, dtype=np.float32)
    mask_background = np.divide(background[..., 3:], 255, dtype=np.float32)
    t.snap("scale mask")

    # scale intensity of each RGB component according to foreground's transparency factor
    foreground = np.maximum(mask_foreground, 1.0-mask_background)*foreground[..., :3] # do not adjust pixels in foreground, that were fully transparent in background
    t.snap("scale intensity1")
    background = (1.0-mask_foreground)*background[..., :3]
    t.snap("scale intensity2")

    # add foreground and background pixel values together
    # we should use cv2.add() here because pixel values can overflow 255
    # however, we use np.add() with the following workaround:
    # (https://stackoverflow.com/questions/29611185/avoid-overflow-when-adding-numpy-arrays)
    result = np.add(foreground.astype(np.uint8), background.astype(np.uint8))
    result[result < background.astype(np.uint8)] = 255
    t.snap("np.add()")
    alpha = np.maximum(mask_foreground, mask_background)*255 # retain transparency for pixels that were transparent in both images
    t.snap("set alpha")
    result = np.concatenate([result, alpha], axis = 2)
    t.snap("np.concatenate()")
    t.total(display=False)
    return result

# underlays a smaller image (background) under a bigger image (foreground) taking into account transparency of both images
def underlay_image(background, foreground_orig, x, y):

    # case when the background falls outside the image borders
    if x < 0:
        background = background[:, abs(x):]
        x = 0
    if y < 0:
        background = background[abs(y):, :]
        y = 0

    if x + background.shape[1] > foreground_orig.shape[1]:
        crop_right = (x + background.shape[1]) - foreground_orig.shape[1]
        background = background[:, :-crop_right]

    if y + background.shape[0] > foreground_orig.shape[0]:
        crop_bottom = (y + background.shape[0]) - foreground_orig.shape[0]
        background = background[:-crop_bottom, :]

    # if the background falls completely outside the image borders - skip blending
    if not background.shape[0] or not background.shape[1]:
        return foreground_orig

    h, w = background.shape[0], background.shape[1]
    foreground = foreground_orig[y:y+h, x:x+w] # extract the region that will be blended
    if background.shape[2] < 4: # handle backgrounds that do not have transparency layer (e.g., image from camera)
        background = add_alpha(background)
    result = blend_images(background, foreground)
    foreground_orig[y:y+h, x:x+w] = result # set the region
    return foreground_orig

# underlays a smaller image (background) under a bigger image (foreground) taking into account transparency of both images
def underlay_image_optimized(background, foreground_orig, x, y):

    t = Timer()
    h, w = background.shape[0], background.shape[1]
    foreground = foreground_orig[y:y+h, x:x+w] # extract the region that will be blended
    t.snap("extract region")

    # transparency factor for each pixel from 0 to 1
    mask_foreground = np.divide(foreground[..., 3:], 255, dtype=np.float32) # float32 is faster than the higher precision float64
    t.snap("scale mask")

    # scale intensity of each RGB component according to foreground's transparency factor
    foreground = np.multiply(mask_foreground, foreground[..., :3])
    t.snap("scale intensity1")

    beta = 1.0-mask_foreground
    t.snap("scale intensity2")
    background = np.multiply(beta, background[..., :3])
    t.snap("scale intensity2")

    # add foreground and background pixel values together
    # we should use cv2.add() here, because pixel values can overflow 255
    # but in practice data passed to this method is safe from overflows
    result = np.add(foreground.astype(np.uint8), background.astype(np.uint8))
    t.snap("np.add()")

#    alpha = np.ones((h, w, 1), dtype=np.uint8)*255 # fully opaque
#    t.snap("set alpha")
#    result = np.concatenate([result, alpha], axis = 2)
#    t.snap("np.concatenate()")
#    foreground_orig[y:y+h, x:x+w] = result # set the region
    foreground_orig[y:y+h, x:x+w, :3] = result # set the region
    t.snap("set region")
    t.total(display=False)
    return foreground_orig

# overlays a smaller image (foreground) on top of a bigger image (background) taking into account transparency of both images
def overlay_image(foreground, background_orig, x, y):

    # case when the foreground falls outside the image borders
    if x < 0:
        foreground = foreground[:, abs(x):]
        x = 0
    if y < 0:
        foreground = foreground[abs(y):, :]
        y = 0

    if x + foreground.shape[1] > background_orig.shape[1]:
        crop_right = (x + foreground.shape[1]) - background_orig.shape[1]
        foreground = foreground[:, :-crop_right]

    if y + foreground.shape[0] > background_orig.shape[0]:
        crop_bottom = (y + foreground.shape[0]) - background_orig.shape[0]
        foreground = foreground[:-crop_bottom, :]

    # if the foreground falls completely outside the image borders - skip blending
    if not foreground.shape[0] or not foreground.shape[1]:
        return background_orig

    h, w = foreground.shape[0], foreground.shape[1]
    background = background_orig[y:y+h, x:x+w] # extract the region that will be blended
    if background.shape[2] < 4: # handle backgrounds that do not have transparency layer (e.g., image from camera)
        background = add_alpha(background)
    result = blend_images(background, foreground)
    if background_orig.shape[2] < 4:
        result = remove_alpha(result)
    background_orig[y:y+h, x:x+w] = result # set the region
    return background_orig

# scales SVG to OpenCV image in the required resolution
def load_svg(filename, scale_factor, enlarge=0, alpha=True):

    # the design SVG files have been generated for the window size 1080x1920 (16:9 in portrait mode)
    # scale up or down according to the requested window size
    pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
    file_height = pixbuf.get_height()
    #print("[+] load_svg(%s): file_height: %s " % (filename, file_height))

    # enlarging is used to work around scaling defects
    # enlarging is not necessary if scaling is not applied
    if scale_factor == 1:
        enlarge = 0
    pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(filename, -1, file_height*scale_factor+enlarge)

    # convert to PNG and then to OpenCV image
    _, png_bytes = pixbuf.save_to_bufferv("png", [], [])
    png_np = np.frombuffer(bytes(png_bytes), dtype=np.uint8)
    if not alpha: # this option not used in practice
        return cv2.imdecode(png_np, cv2.IMREAD_COLOR)
    return cv2.imdecode(png_np, cv2.IMREAD_UNCHANGED)

def process_reference_face_jpg(jpg_content, scale_factor):
    # load the JPG facial image

    #jpg_content = open('../photo_artefact/1.jpg','rb').read()
    jpg_as_np = np.frombuffer(jpg_content, dtype=np.uint8)
    face = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)

    #imgbytes = cv2.imencode(".png", face)[1].tobytes()
    #open("/tmp/id_orig.png", 'wb').write(imgbytes)

    # certain cardholder facial photos (that have been submitted by the cardholder in the PPA online application environment) have an additional grey background
    # here we are trying to detect the background and crop the image accordingly
    h, w = face.shape[0], face.shape[1]
    crop_left, crop_right, crop_top, crop_bottom = 0, 0, 0, 0

    # a heuristic to detect background
    def is_background_line(rgb_line, crop_cnt):
        # background has color RGB(192,192,192), but can have some variations due to JPG encoding artefacts
        tolerance = 5
        if crop_cnt: tolerance = 10 # relax tolerance if previous lines were detected to have background value
        matches = 0

        for rgb in rgb_line:
            rgb = list(rgb)
            if abs(rgb[0]-192) <= tolerance and abs(rgb[1]-192) <= tolerance and abs(rgb[2]-192) <= tolerance:
                matches+=1

        # removing the line if it contains more than 70% pixels containing background pixel values
        match_perc = matches*100/rgb_line.shape[0]
        print("[+] is_background_line(): confidence that the line contains background: %s%%, cropped lines: %d" % (int(match_perc), crop_cnt))
        if match_perc > 70:
            return True
        return False

    while is_background_line(face[:, crop_left], crop_left):
        crop_left+=1
    while is_background_line(face[:, w-1-crop_right], crop_right):
        crop_right+=1
    while is_background_line(face[crop_top, :], crop_top):
        crop_top+=1
    while is_background_line(face[h-1-crop_bottom, :], crop_bottom):
        crop_bottom+=1
    if crop_left+crop_right+crop_top+crop_bottom:
        print("[+] process_reference_face_jpg(): photo in photo detected! Cropping left[%s], right[%s], top[%s], bottom[%s]" % (crop_left, crop_right, crop_top, crop_bottom))
        face = face[crop_top:h-1-crop_bottom, crop_left:w-1-crop_right]


    # take facial measurements on the non-modified image
    face_loc = get_face_locations(face)
    face_encoded = face_recognition.face_encodings(face[:, :, ::-1], face_loc, 1, "large")[0]


    h, w = face.shape[0], face.shape[1]
    if h!=w and h > w:
        # the problem is that eMRTD facial images are not square but usually 480x640
        # hence we make the image square by stretching the sides of the image
        # this, hopefully, makes the person's shoulders wider without affecting the head
        print("[+] process_reference_face_jpg(): input resolution %sx%s" % (w, h))
        difference = h-w
        print("[+] process_reference_face_jpg(): the width must be enlarged by %s pixels to make it square" % (difference))

        # sides of the image that will be streched, should be 50% away from the margin of the detected face (to avoid stretching ears)
        right, left = face_loc[0][1], face_loc[0][3]
        left_pixels = left//2
        right_pixels = (w-right)//2

        # the sides should be stretched to the required width proportionally to their width (i.e., a larger side gets stretched more)
        stretch_left = int((difference + left_pixels + right_pixels) * (left_pixels / (left_pixels + right_pixels)))
        stretch_right = (difference + left_pixels + right_pixels) - stretch_left
        print("[+] process_reference_face_jpg(): stretching the leftmost %s pixels to %s pixels" % (left_pixels, stretch_left))
        print("[+] process_reference_face_jpg(): stretching the rightmost %s pixels to %s pixels" % (right_pixels, stretch_right))
        left_part = face[:, :left_pixels]
        center_part = face[:, left_pixels:w-right_pixels]
        right_part = face[:, w-right_pixels:]

        # perform stretching
        left_part = cv2.resize(left_part, (stretch_left, h))
        right_part = cv2.resize(right_part, (stretch_right, h))

        # concatenate the parts together to make a square image
        img = np.zeros([h,h,3],dtype=np.uint8)
        x_offsets = [left_part.shape[1], center_part.shape[1], right_part.shape[1]]
        img[:, 0:x_offsets[0]] = left_part
        img[:, x_offsets[0]:x_offsets[0]+x_offsets[1]] = center_part
        img[:, x_offsets[0]+x_offsets[1]:x_offsets[0]+x_offsets[1]+x_offsets[2]] = right_part
        face = img

        #imgbytes = cv2.imencode(".png", face)[1].tobytes()
        #open("/tmp/id_adjusted.png", 'wb').write(imgbytes)

    # scale the image to the required size
    face = cv2.resize(face, (int(185*scale_factor), int(185*scale_factor)))

    # draw a circle with transparent pixels outside the circle
    h, w = face.shape[0], face.shape[1]
    lum_img = Image.new('L',[h,w] ,0)
    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(0,0),(h,w)],0,360,fill=255)
    lum_img_arr = np.array(lum_img)
    face = np.dstack((face, lum_img_arr))
    # set all transparent pixels to black (this is needed as we will later sum pixel values when underlaying the image)
    face[face[...,3]==0]=[0,0,0,0]
    face_rounded = face

    #imgbytes = cv2.imencode(".png", face_rounded)[1].tobytes()
    #open("/tmp/id_round.png", 'wb').write(imgbytes)

    return face_rounded, face_encoded

# add name of the document holder to the appropriate position on the overlay
def put_document_holders_name(img, text, scale_factor):
    # Unfortunately, freetype functionality will be available only in OpenCV release 5.0: [https://github.com/opencv/opencv-python/issues/305]
    # ft = cv2.freetype.createFreeType2()
    # therefore, for now using PIL
    font = ImageFont.truetype('resources/PlusJakartaSans-SemiBold.ttf', int(24*scale_factor))
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    draw.text((int(410*scale_factor), int(296*scale_factor)), text, (158, 158, 158), font=font, anchor="rd")
    overlay = np.array(image)
    return overlay

# time adjustment formula for the ease-in / ease-out effect
# (https://math.stackexchange.com/questions/121720/ease-in-out-function/121755#121755)
def ease(t):
    param = 1.5 # can be adjusted
    sqr = t*t
    return sqr/(param*(sqr-t)+1)

# camera reading and display thread
def camera(
    q_main: Queue,
    q_camera: Queue,
    camera_id: int,
    camera_resolution: int,
    camera_rotate: int,
    screen_width: int,
    show_fps: bool,
    fullscreen: bool,
    camera_size: int,
) -> None:
    """
    Continuously capture from camera and draw the screen output
    """

    # wait until the smart card thread is ready (not really necessary)
    while q_camera.get()[0]!='main thread ready':
        pass

    # must return 'True' to make use of GPU
    print("[+] camera(): dlib with CUDA:", dlib.DLIB_USE_CUDA)
    print("[+] camera(): CUDA-enabled cupy:", cupy)

    # window size (must be 16:9 in portrait mode)
    window_width, window_height = screen_width, int(screen_width/(9/16))

    scale_factor = window_height/1920
    print("[+] camera(): working in resolution %sx%s (scale factor: %s)" % (window_width, window_height, scale_factor))

    print("[+] camera(): using /dev/video%s for camera input" % (camera_id))
    cap = VideoCapture(camera_id, camera_resolution, mjpg=True)
    time.sleep(3)

    # read a single camera frame to perform calculations
    frame = cap.read()

    if camera_rotate:
        print("[+] camera(): Camera input frame rotated %s degrees clockwise" % (camera_rotate))
        camera_rotate_param = {90:cv2.ROTATE_90_CLOCKWISE, 180:cv2.ROTATE_180, 270:cv2.ROTATE_90_COUNTERCLOCKWISE}[camera_rotate]
        frame = cv2.rotate(frame, camera_rotate_param)

    # calculate the size of the camera frame to have the same aspect ratio as the visible camera area in the UI (950x1456)
    camera_width, camera_height = frame.shape[1], frame.shape[0]
    print("[+] camera(): Camera input frame: %sx%s" % (camera_width, camera_height))
    camera_width_old, camera_height_old = camera_width, camera_height
    if camera_width > camera_height:
        camera_width_old, camera_width = camera_width, int(camera_height*950/1456)
    else:
        camera_height_old, camera_height = camera_height, int(camera_width*1456/950)
        # if we would have to increase height to obtain the required aspect ratio, then reduce width instead
        if camera_height > camera_height_old:
            camera_height = camera_height_old
            camera_width_old, camera_width = camera_width, int(camera_height*950/1456)
    print("[+] camera(): Camera frame cropped: %sx%s" % (camera_width, camera_height))

    if camera_size < 100:
        camera_width = int(camera_width * camera_size/100)
        camera_height = int(camera_height * camera_size/100)
        print("[+] camera(): Camera frame cropped due to --camera_size parameter: %sx%s" % (camera_width, camera_height))

    # calculate the new dimmensions of the camera frame by making sure that the center part of the original camera frame is preserved
    width_diff_half = (camera_width_old-camera_width)//2
    height_diff_half = (camera_height_old-camera_height)//2
    cropped_w_start, cropped_w_end = 0+width_diff_half, camera_width_old-width_diff_half
    cropped_h_start, cropped_h_end = 0+height_diff_half, camera_height_old-height_diff_half

    # load SVG UI design components
    overlay = load_svg("resources/overlay.svg", scale_factor)
    bottom_panel = load_svg("resources/bottom.svg", scale_factor, enlarge=1)
    match_no = to_gpu(load_svg("resources/match_no.svg", scale_factor))
    match_yes = to_gpu(load_svg("resources/match_yes.svg", scale_factor))
    match_blank = to_gpu(load_svg("resources/match_blank.svg", scale_factor))
    match_no_percent = load_svg("resources/match_no_percent.svg", scale_factor)
    match_yes_percent = load_svg("resources/match_yes_percent.svg", scale_factor)
    downloaded_ellipse = load_svg("resources/downloaded_ellipse.svg", scale_factor)
    data_successfully_downloaded = load_svg("resources/data_successfully_downloaded.svg", scale_factor)
    card_not_supported = to_gpu(load_svg("resources/card_not_supported.svg", scale_factor))
    card_read_error = to_gpu(load_svg("resources/card_read_error.svg", scale_factor))

    # generate some of the UI components
    progress_bar_images = generate_progress_bar_images(196, 196, scale_factor)
    match_no_percent_images = generate_match_percent_images(match_no_percent, (95,109,246), scale_factor)
    match_yes_percent_images = generate_match_percent_images(match_yes_percent, (255,255,255), scale_factor)

    # activity messages (display text can be changed here)
    activities = {}
    activities['activity:establishing secure channel'] = 'Establishing secure channel...'
    activities['activity:verifying data authenticity'] = 'Verifying data authenticity...'
    activities['activity:verifying chip authenticity'] = 'Verifying chip authenticity...'
    activities['activity:downloading facial image'] = 'Downloading facial image...'
    activities['activity:online validity check'] = 'Online validity check...'

    # generate an image box for each activity
    for key in activities.keys():
        text = activities[key]
        activities[key] = []
        for opacity in range(100,79,-1):
            activities[key]+= [generate_msgbox_image(text, scale_factor, error=False, text_opacity=opacity)]

    # error messages (display text can be changed here)
    errors = {}
    errors['error:card communication failure'] = 'Card communication failure'
    errors['error:BAC key failure'] = 'BAC key construction failed'
    errors['warning:data authenticity check failed'] = 'Data authenticity check failed'
    errors['warning:chip authentication failure'] = 'Chip authentication failed'
    errors['warning:document expired'] = 'Document is expired'
    errors['warning:online check failed'] = 'Online check failed'

    # generate an image box for each error/warning
    for key in errors.keys():
        errors[key] = generate_msgbox_image(errors[key], scale_factor, error=True)

    # remember the base overlay as we will later compose different overlays based on the events received
    base_overlay = np.copy(overlay)

    # place the rounded reference face in the empty circle
    face_pos_x = 448
    face_pos_y = 184
    sample_face_rounded, sample_face_encoded = process_reference_face_jpg(open("resources/jaak-kristjan.jpg", 'rb').read(), scale_factor)
    overlay = underlay_image(sample_face_rounded, overlay, int(face_pos_x*scale_factor), int(face_pos_y*scale_factor))
    reference_face_encoded = sample_face_encoded

    # place the name of the face holder next to the face
    sample_name = "Sample ID card photo"
    name = "" # will be filled with the cardholder's name
    overlay = put_document_holders_name(overlay, sample_name, scale_factor)

    # remember the overlay of the matching against the ID card sample as we will later reuse it once the ID card is removed
    sample_overlay = np.copy(overlay)

    # add bottom panel
    bottom_pos_x = 65
    bottom_pos_y = 1394
    overlay = underlay_image(bottom_panel, overlay, int(bottom_pos_x*scale_factor), int(bottom_pos_y*scale_factor))

    # positions for the match button (will be used inside the loop)
    match_pos_x = 680
    match_pos_y = 240

    # positions for the progress bar (will be used inside the loop)
    progress_pos_x = 442
    progress_pos_y = 180

    # positions for the activity bar (will be used inside the loop)
    error_pos_x = int(1080*scale_factor/2)
    error_pos_y = int(1270*scale_factor)
    activity_pos_x = int(1080*scale_factor/2)
    activity_pos_y = int(1670*scale_factor)

    loading_active = False
    matching_active = True
    last_progress_value = -1

    face_match_threshold = 0.5 # default is 0.6 (smaller is more strict)
    bottom_panel_sliding_time = 1.0 # sliding up/down time for the bottom panel
    sliding_active = False
    time_perf_stats = time.time() # time when the performance stats were printed
    current_activity = ""
    activity_start = "" # timestamp of when the current progress activity started
    activity_fade_cycle_time = 1.0 # time of one fade-out and fade-in cycle for progress box text
    error = ""
    warnings = []

    # processing and camera frame rate
    fps_font = ImageFont.truetype('resources/PlusJakartaSans-SemiBold.ttf', int(20*scale_factor))
    fps = 0
    camera_fps = 0

    bottom_panel = to_gpu(bottom_panel)
    overlay = to_gpu(overlay)
    sample_overlay = to_gpu(sample_overlay)

    window_title = "ID card face match"
    if fullscreen:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

    while True:

        t = Timer()
        frame = cap.read()
        t.snap('Read   camera frame')

        if camera_rotate: # rotate if required
            frame = cv2.rotate(frame, camera_rotate_param)
            t.snap('Rotate   camera frame')

        camera_frame = cv2.flip(frame, 1) # to have the mirror effect
        t.snap('Flip   camera frame')

        camera_frame = np.copy(camera_frame)
        t.snap('Copy   camera frame')

        # perform cropping and resizing
        # XXX - check if there is a significant difference whether the face detection and encoding is performed on the original or scaled-up image
        camera_frame = camera_frame[cropped_h_start:cropped_h_end, abs(cropped_w_start):cropped_w_end]
        t.snap('Crop   camera frame')
        camera_frame = cv2.resize(camera_frame, (int(952*scale_factor), int(1456*scale_factor)))
        t.snap('Resize camera frame')

        face_matched = False

        # find faces, calculate similarity and draw boxes around faces
        if matching_active:

            # find faces in the image
            face_locations = get_face_locations(camera_frame)
            t.snap('face matching: get_face_locations')

            # return the 128-dimension face encoding for each face in the image (uses GPU if dlib compiled with CUDA support)
            face_encodings = face_recognition.face_encodings(camera_frame, face_locations, 1, "large")
            t.snap('face matching: obtain face encodings')
            # compare a list of face encodings to a known face encoding and get a euclidean distance for each comparison face
            distances = face_recognition.face_distance(face_encodings, reference_face_encoded)
            t.snap('face matching: calculate face distances')
            for i in range(len(face_locations)):
                top, right, bottom, left = face_locations[i]

                confidence_percent = int(face_distance_to_conf(distances[i], face_match_threshold)*100)

                if distances[i] <= face_match_threshold:
                    color = (95, 200, 77)
                    percent_image = match_yes_percent_images[confidence_percent]
                    face_matched = True
                else:
                    color = (95, 109, 246)
                    percent_image = match_no_percent_images[confidence_percent]

                thickness = int(8*scale_factor)
                cv2.rectangle(camera_frame, (left, top), (right, bottom), color, thickness)
                camera_frame = overlay_image(percent_image, camera_frame, left+(right-left)//2-percent_image.shape[1]//2, bottom+thickness-percent_image.shape[0]//2)

            t.snap('face matching: draw rectangle and put text on camera frame')

        camera_frame = to_gpu(camera_frame)
        frame_shown = np.copy(overlay)
        t.snap('camera frame and shown frame: to gpu')

        # up/down sliding of the bottom panel
        if sliding_active:

            tdiff = time.time() - sliding_start

            if sliding_direction == 'down':

                if tdiff > bottom_panel_sliding_time:
                    sliding_active = False
                    if not loading_active: # loading not active only if 'Unknown card' or 'Card read error'
                        if error == 'Unknown card':
                            # display unknown card error
                            overlay = overlay_image(card_not_supported, overlay, activity_pos_x - card_not_supported.shape[1]//2, activity_pos_y)
                        elif error == 'Card read error':
                            # display card read error
                            overlay = overlay_image(card_read_error, overlay, activity_pos_x - card_read_error.shape[1]//2, activity_pos_y)
                else:
                    sliding_factor = tdiff/bottom_panel_sliding_time
                    sliding_factor = ease(sliding_factor) # applying ease-out/in animation effect
                    y_offset = 416*sliding_factor
                    frame_shown = underlay_image(bottom_panel, frame_shown, int(bottom_pos_x*scale_factor), int((bottom_pos_y+y_offset)*scale_factor))

            elif sliding_direction == 'up':

                if tdiff > bottom_panel_sliding_time:
                    sliding_active = False
                    matching_active = True
                    overlay = underlay_image(bottom_panel, overlay, int(bottom_pos_x*scale_factor), int(bottom_pos_y*scale_factor))

                sliding_factor = 1-tdiff/bottom_panel_sliding_time
                sliding_factor = ease(sliding_factor) # applying ease-out/in animation effect
                y_offset = 416*sliding_factor
                if y_offset < 0: y_offset = 0

                frame_shown = underlay_image(bottom_panel, frame_shown, int(bottom_pos_x*scale_factor), int((bottom_pos_y+y_offset)*scale_factor))
            t.snap('perform sliding action')

        # put the prepared camera frame under the overlay to produce frame that will be displayed
        frame_shown = underlay_image_optimized(camera_frame, frame_shown, int(65*scale_factor), int(353*scale_factor))
        t.snap('underlay_image(): put camera frame under overlay')

        # add match status icon that corresponds to the face matching result
        if not matching_active:
            frame_shown = overlay_image(match_blank, frame_shown, int(match_pos_x*scale_factor), int(match_pos_y*scale_factor))
        elif face_matched:
            frame_shown = overlay_image(match_yes, frame_shown, int(match_pos_x*scale_factor), int(match_pos_y*scale_factor))
        else:
            frame_shown = overlay_image(match_no, frame_shown, int(match_pos_x*scale_factor), int(match_pos_y*scale_factor))
        t.snap('overlay_image(): overlaying match status icon')

        # display FPS on screen if required
        if show_fps:
            image = Image.fromarray(from_gpu(frame_shown))
            draw = ImageDraw.Draw(image)
            text = 'FPS: %0.1f/%d' % (fps, camera_fps)
            draw.text((int(20*scale_factor), int(50*scale_factor)), text, (0, 0, 0), font=fps_font, anchor="ld")
            frame_shown = to_gpu(np.array(image))
            t.snap('text(): draw FPS')

#        imgbytes = cv2.imencode(".png", frame_shown)[1].tobytes()
#        open("/tmp/output.png", 'wb').write(imgbytes)
#        exit()

        # window has been closed
        if not cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE):
            cv2.destroyAllWindows()
            q_main.put("exit")
            return

        cv2.imshow(window_title, from_gpu(frame_shown))
        code = cv2.pollKey()
        if code == 27: # ESC - exit
            cv2.destroyAllWindows()
            q_main.put("exit")
            return
        elif code in [102,70]: # 'f'/'F' - fullscreen
            cv2.destroyAllWindows()
            # toggle fullscreen mode
            if fullscreen:
                print("[+] camera(): switching to non-fullscreen mode")
                cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
                fullscreen = False
            else:
                print("[+] camera(): switching to fullscreen mode")
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                fullscreen = True

        t.snap('show image on screen')

        # check for events and update output image accordingly
        if q_camera.qsize():
            queue_element = q_camera.get()
            print("[+] camera(): Event from the queue:", queue_element if queue_element[0]!='ID image' else queue_element[0])

            # document holder's photo has been downloaded from the chip
            if queue_element[0] == 'ID image':
                # reset overlay to base overlay
                overlay = np.copy(base_overlay)
                # add document holder's face
                try:
                    reference_face_rounded, reference_face_encoded = process_reference_face_jpg(queue_element[1], scale_factor)
                    overlay = underlay_image(reference_face_rounded, overlay, int(face_pos_x*scale_factor), int(face_pos_y*scale_factor))
                except:
                    print("[-] camera(): a bug: failed loading document holder's facial photo!")
                # add document holder's name
                overlay = put_document_holders_name(overlay, name, scale_factor)
                matching_active = True
                loading_active = False
                last_progress_value = -1
                globals.bytes_read = 0
                globals.progress = 0

                # show warnings
                for i in range(len(warnings)):
                    overlay = overlay_image(errors[warnings[i]], overlay, activity_pos_x - errors[warnings[i]].shape[1]//2, activity_pos_y-int(i*140*scale_factor))
                overlay = to_gpu(overlay)

            # document holder's name has been read from the personal data file
            elif queue_element[0] == 'ID name':
                # remember document holder's name
                name = queue_element[1]

            # the card has been removed
            elif queue_element[0] == 'Disconnect':
                # reset overlay to sample overlay
                overlay = np.copy(sample_overlay)
                reference_face_encoded = sample_face_encoded
                matching_active = False
                loading_active = False
                last_progress_value = -1
                globals.bytes_read = 0
                globals.progress = 0
                sliding_active = True
                sliding_start = time.time()
                sliding_direction = 'up'
                current_activity = ""
                error = ""
                warnings = []

            # a valid card has been inserted
            elif queue_element[0].startswith('Valid card'):
                matching_active = False
                loading_active = True
                sliding_active = True
                sliding_start = time.time()
                sliding_direction = 'down'
                current_activity = 'activity:establishing secure channel'
                activity_start = time.time()
                overlay = np.copy(base_overlay)
                overlay = put_document_holders_name(overlay, "Loading data...", scale_factor)
                overlay = overlay_image(progress_bar_images[0], overlay, int(progress_pos_x*scale_factor), int(progress_pos_y*scale_factor))
                overlay = to_gpu(overlay)

            # an invalid card or unreadable card has been inserted
            elif queue_element[0] in ['Unknown card','Card read error']:
                error = queue_element[0]
                matching_active = False
                sliding_active = True
                sliding_start = time.time()
                sliding_direction = 'down'
                overlay = np.copy(base_overlay)
                overlay = put_document_holders_name(overlay, "Loading data...", scale_factor)
                overlay = overlay_image(progress_bar_images[0], overlay, int(progress_pos_x*scale_factor), int(progress_pos_y*scale_factor))
                overlay = to_gpu(overlay)

            # process activities
            elif queue_element[0].startswith("activity:"):
                current_activity = queue_element[0]
                #activity_start = time.time()

            # process errors
            elif queue_element[0].startswith("error:"):
                error = queue_element[0]
                # make sure that the condition below (for updating the progress bar) is satisfied and the error box is shown
                if last_progress_value >=99:
                    last_progress_value-= 1
                else:
                    last_progress_value+= 1

            # process warnings (will be shown after loading completes or an error is raised)
            elif queue_element[0].startswith("warning:"):
                if queue_element[0] not in warnings: # skip the same type of warnings
                    warnings.append(queue_element[0])


        if loading_active and not sliding_active:
            # reset overlay to base overlay and update progress bar
            overlay = np.copy(base_overlay)
            overlay = put_document_holders_name(overlay, "Loading data...", scale_factor)
            overlay = to_gpu(overlay)

            # print out current progress if progress changed
            if globals.progress != last_progress_value:
                last_progress_value = globals.progress
                print("[+] camera(): setting progress bar value to:", last_progress_value)

            if last_progress_value > 99:
                progress_image = downloaded_ellipse
                overlay = overlay_image(to_gpu(data_successfully_downloaded), overlay, activity_pos_x - data_successfully_downloaded.shape[1]//2, activity_pos_y)
            else:
                progress_image = progress_bar_images[last_progress_value]

                # display activity (the bottom panel must have slided off)
                if current_activity:

                    # activity text fade-in and fade-out effect
                    tdiff = (time.time() - activity_start) % activity_fade_cycle_time

                    # reverse fading direction if we are in the second half of the cycle
                    activity_fade_cycle_half_time = activity_fade_cycle_time/2
                    if tdiff >= activity_fade_cycle_half_time:
                        tdiff = abs(tdiff % -activity_fade_cycle_half_time)

                    tdiff_ratio = tdiff/activity_fade_cycle_half_time

                    opacities_cnt = len(activities[current_activity])
                    opacity_index = int(opacities_cnt*ease(tdiff_ratio))
                    overlay = overlay_image(to_gpu(activities[current_activity][opacity_index]), overlay, activity_pos_x - activities[current_activity][opacity_index].shape[1]//2, activity_pos_y)

                # display error (the bottom panel must have slided off)
                if error:
                    overlay = overlay_image(to_gpu(errors[error]), overlay, activity_pos_x - errors[error].shape[1]//2, activity_pos_y-int(140*scale_factor))


            overlay = overlay_image(to_gpu(progress_image), overlay, int(progress_pos_x*scale_factor), int(progress_pos_y*scale_factor))




        t.snap('process events and update progress bar')

        # print performance stats every second
        if show_fps and time.time() > time_perf_stats + 1:
            time_perf_stats = time.time()
            time_total = t.total(display=True)
            fps = 1/time_total
            camera_fps = cap.get_frame_count()
            print("[+] Processing frame rate: %0.2f fps" % (fps))
            print("[+] Camera grab frame rate: %s fps" % (camera_fps))
