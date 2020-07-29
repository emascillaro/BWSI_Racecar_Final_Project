"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Final Project
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

from enum import IntEnum


# Separate 
class State(IntEnum):
    start = 0
    line_following = 1
    cone_slaloming = 2
    wall_parking = 3

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 30
MAX_DIST_DIF = 30

# A crop window for the floor directly in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# Colors, stored as a pair (hsv_min, hsv_max)
#BLUE = ((90, 50, 50), (120, 255, 255))  # The HSV range for the color blue
# TODO (challenge 1): add HSV ranges for other colors
BLUE = ((90, 255, 255), (120, 255, 255))  # The HSV range for the color blue
RED = ((0, 255, 255), (10, 255, 255))
GREEN = ((50, 255, 255), (70, 255, 255))
PURPLE = ((130,255,255), (140,255,255))
ORANGE = ((10,255,255), (20,255,255))
YELLOW = ((20,255,255), (40,255,255))

## Realsene Camera ##
NEON_GREEN_CONE = ((40, 100, 100),(70, 255, 255))
NEON_ORANGE_CONE = ((150, 90, 90),(179, 255, 255))
RED_TAPE = ((150, 90, 90),(179, 255, 255))
YELLOW_TAPE = ((10, 90, 90),(40, 255, 255))
GREEN_TAPE = ((60, 60, 60),(80, 255, 255))

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

PRIORITY = ["G", "R", "Y"]

LEFT_POINT = (rc.camera.get_height() // 2, int(rc.camera.get_width() * 1 / 4))
RIGHT_POINT = (rc.camera.get_height() // 2, int(rc.camera.get_width() * 3 / 4))

# The Kernel size to use when measuring distance at these points
KERNEL_SIZE = 11

########################################################################################
# Functions
########################################################################################


def update_contour():
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    global contour_center
    global contour_area
    
    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # TODO (challenge 1): Search for multiple tape colors with a priority order
        # (currently we only search for blue)

        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

        # Find all of the blue contours
        contours = rc_utils.find_contours(image, BLUE[0], BLUE[1])

        # Select the largest contour
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_center)

        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        rc.display.show_color_image(image)


def start():
    """
    This function is run once every time the start button is pressed
    """
    global speed
    global angle
    global cur_state
    global prevangle

    cur_state = State.line_following
    # Initialize variables
    speed = 0
    angle = 0
    prevangle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

def checkRed(image):
    global RED
    contours = rc_utils.find_contours(image, RED[0], RED[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    return contour

def checkBlue(image):
    global BLUE
    contours = rc_utils.find_contours(image, BLUE[0], BLUE[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    return contour

def checkGreen(image):
    global GREEN
    contours = rc_utils.find_contours(image, NEON_GREEN_CONE[0], NEON_GREEN_CONE[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    return contour

def checkOrange(image):
    global ORANGE
    contours = rc_utils.find_contours(image, NEON_ORANGE_CONE[0], NEON_ORANGE_CONE[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    return contour

def checkYellow(image):
    global YELLOW
    contours = rc_utils.find_contours(image, YELLOW_TAPE[0], YELLOW_TAPE[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    return contour


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global cur_state
    global PRIORITY
    global prevangle
    # Get all images
    image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()
    ###### Line Following State ######
    if cur_state == State.line_following:
        if image is None:
            contour_center = None
            contour_area = 0
        else:
            # Crop the image to the floor directly in front of the car
            image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

            colorContours = []
            contour = None
            colorContours = []
            red = checkRed(image)
            green = checkGreen(image)
            blue = checkBlue(image)
            yellow = checkYellow(image)
            
            for priority in PRIORITY:
                if priority == "Y" and yellow is not None:
                    colorContours.append(yellow)
                elif priority == "R" and red is not None:
                    colorContours.append(red)
                elif priority == "G" and green is not None:
                    colorContours.append(green)
            
            if not colorContours:
                angle = prevangle
                contour = None
            else:
                contour = colorContours[0]
            
            if contour is not None:
                # Calculate contour information
                contour_center = rc_utils.get_contour_center(contour)
                contour_area = rc_utils.get_contour_area(contour)

                # Draw contour onto the image
                rc_utils.draw_contour(image, contour)
                rc_utils.draw_circle(image, contour_center)
#change
            else:
                contour_center = None
                contour_area = 0
                
            if contour_center is not None:
                angle = rc_utils.remap_range(contour_center[1], 0, rc.camera.get_width(), -1, 1, True)
                angle = rc_utils.clamp(angle, -1,1)
                prevangle = angle

            # Display the image to the screen
            rc.display.show_color_image(image)


    ##### Cone Slaloming State ######
    elif cur_state == State.cone_slaloming:
        print("cone slaloming")
    



    ###### Wall Parking State ######
    elif cur_state == State.wall_parking:
        print("Wall Parking")
        
        # Get distance at 1/4, 2/4, and 3/4 width
        center_dist = rc_utils.get_depth_image_center_distance(depth_image)
        left_dist = rc_utils.get_pixel_average_distance(depth_image, LEFT_POINT, KERNEL_SIZE)
        right_dist = rc_utils.get_pixel_average_distance(depth_image, RIGHT_POINT, KERNEL_SIZE)
        
        print("distance", center_dist)

        # Get difference between left and right distances
        dist_dif = left_dist - right_dist
        print("dist_dif", dist_dif)

        # Remap angle
        angle = rc_utils.remap_range(dist_dif, -MAX_DIST_DIF, MAX_DIST_DIF, -1, 1, True)
        
        if abs(dist_dif) > 1:
            print("entered")
            angle = rc_utils.remap_range(dist_dif, -MAX_DIST_DIF, MAX_DIST_DIF, -1, 1, True)
            if center_dist > 20:
                speed = 0.5
            elif center_dist < 21 and center_dist > 10:
                speed = rc_utils.remap_range(center_dist, 20, 10, 0.5, 0)
                speed = rc_utils.clamp(speed, 0, 0.5)
            else:
                speed = 0
            print("speed", speed)
            rc.drive.set_speed_angle(speed, angle)
        else:
            # stop moving             
            rc.drive.stop()
    print("angle", angle)
    print("speed", speed)
    rc.drive.set_speed_angle(0.5, angle)
        
    
    
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
