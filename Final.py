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

class Mode(IntEnum):
    red_align = 0  # Approaching a red cone to pass
    blue_align = 1  # Approaching a blue cone to pass
    red_pass = 2  # Passing a red cone (currently out of sight to our left)
    blue_pass = 3  # Passing a blue cone (currently out of sight to our right)
    red_find = 4  # Finding a red cone with which to align
    blue_find = 5  # Finding a blue cone with which to align
    red_reverse = 6  # Aligning with a red cone, but we are too close so must back up
    blue_reverse = 7  # Aligning with a blue cone, but we are too close so must back up
    no_cones = 8  # No cones in sight, inch forward until we find one

''''''

# Colors, stored as a pair (hsv_min, hsv_max)
BLU = ((100, 150, 150), (120, 255, 255))  # The HSV range for the color blue
REDD = ((170, 50, 50), (10, 255, 255))  # The HSV range for the color blue

# Speeds
MAX_ALIGN_SPEED = 0.8
MIN_ALIGN_SPEED = 0.4
PASS_SPEED = 0.5
FIND_SPEED = 0.2
REVERSE_SPEED = -0.2
NO_CONES_SPEED = 0.4

# Times
REVERSE_BRAKE_TIME = 0.25
SHORT_PASS_TIME = 1.0
LONG_PASS_TIME = 1.2

# Cone finding parameters
MIN_CONTOUR_AREA = 100
MAX_DISTANCE = 250
REVERSE_DISTANCE = 50
STOP_REVERSE_DISTANCE = 60

CLOSE_DISTANCE = 30
FAR_DISTANCE = 120

# >> Variables
cur_mode = Mode.no_cones
count = 0
red_center = None
red_distance = 0
prev_red_distance = 0
blue_center = None
blue_distance = 0
prev_blue_distance = 0

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
# BLUE = ((90, 50, 50), (120, 255, 255))  # The HSV range for the color blue
# TODO (challenge 1): add HSV ranges for other colors
#BLUE = ((90, 255, 255), (120, 255, 255))  # The HSV range for the color blue
RED = ((175, 100, 100), (5, 255, 255)) 
BLUE = ((90, 100, 100), (120, 255, 255))    # The HSV range for the color blue
GREEN = ((40, 100, 100), (70, 255, 255))
PURPLE = ((130,255,255), (140,255,255))
ORANGE = ((10,100,100), (20,255,255))
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

counter = 0
count = 0

PRIORITY = ["R", "Y", "G"]

LEFT_POINT = (rc.camera.get_height() // 2, int(rc.camera.get_width() * 1 / 4))
RIGHT_POINT = (rc.camera.get_height() // 2, int(rc.camera.get_width() * 3 / 4))

# The Kernel size to use when measuring distance at these points
KERNEL_SIZE = 11

########################################################################################
# Functions
########################################################################################

cones_done = False
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
'''
def checkBlue(image):
    global BLUE
    contours = rc_utils.find_contours(image, BLUE[0], BLUE[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    return contour
'''
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
    global cones_done
    global cur_mode
    global counter
    # Get all images
    image = rc.camera.get_color_image()


    #cur_state == State.cone_slaloming
    corners, ids = rc_utils.get_ar_markers(image)
    length = len(corners)
    if length > 0:
        id = 300
        index = 0
        for idx in range(0, len(ids)):
            if ids[idx] < id:
                id = ids[idx]
                index = idx
        TL = corners[index][0][0]
        TR = corners[index][0][1]
        BL = corners[index][0][3]
        area = (abs(TL[0]-TR[0]) + abs(TL[1]-TR[1])) * (abs(TL[0]-BL[0]) + abs(TL[1]-BL[1]))

        print(id[0], area)
        
        if id[0] == 32 and area > 1900:
            if cur_state is not State.cone_slaloming:
                cur_mode = Mode.no_cones
                counter = 0
            cur_state = State.cone_slaloming
            print("State: ", cur_state)
        elif id[0] == 236 and area > 850:
            cur_state = State.wall_parking
            print("State: ", cur_state)
        
    depth_image = rc.camera.get_depth_image()
    ###### Line Following State ######
    if cur_state == State.line_following:
        if image is None:
            contour_center = None
        else:
            # Crop the image to the floor directly in front of the car
            image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

            colorContours = []
            contour = None
            colorContours = []
            red = checkRed(image)
            green = checkGreen(image)
            #blue = checkBlue(image)
            yellow = checkYellow(image)
            
            for priority in PRIORITY:
                if priority == "Y" and yellow is not None:
                    colorContours.append(yellow)
                    print("yellow")
                elif priority == "R" and red is not None:
                    colorContours.append(red)
                    print("red")
                elif priority == "G" and green is not None:
                    colorContours.append(green)
                    print("green")
            
            if not colorContours:
                angle = prevangle
                contour = None
            else:
                contour = colorContours[0]
            
            if contour is not None:
                # Calculate contour information
                contour_center = rc_utils.get_contour_center(contour)

                # Draw contour onto the image
                rc_utils.draw_contour(image, contour)
                rc_utils.draw_circle(image, contour_center)
            #change
            else:
                contour_center = None
                
            if contour_center is not None:
                angle = rc_utils.remap_range(contour_center[1], 0, rc.camera.get_width(), -1, 1, True)
                angle = rc_utils.clamp(angle, -1,1)
                prevangle = angle

            # Display the image to the screen
            rc.display.show_color_image(image)


    ##### Cone Slaloming State ######
    elif cur_state == State.cone_slaloming:
        print("cone slaloming")
        update_cones()
    

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
    rc.drive.set_speed_angle(0.6, angle)

CROP_TOP_HALF = ((0,0), (rc.camera.get_height() //2, rc.camera.get_width()))
def find_cones():
    """
    Find the closest red and blue cones and update corresponding global variables.
    """
    global red_center
    global red_distance
    global prev_red_distance
    global blue_center
    global blue_distance
    global prev_blue_distance

    prev_red_distance = red_distance
    prev_blue_distance = blue_distance

    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()

    if color_image is None or depth_image is None:
        red_center = None
        red_distance = 0
        blue_center = None
        blue_distance = 0
        print("No image found")
        return

    # Search for red cone
    contours = rc_utils.find_contours(color_image, ORANGE[0], ORANGE[1])
    contour = rc_utils.get_largest_contour(contours, 30)

    if contour is not None:
        red_center = rc_utils.get_contour_center(contour)
        red_distance = rc_utils.get_pixel_average_distance(depth_image, red_center)

        # Only use count it if the cone is less than MAX_DISTANCE away
        if red_distance <= 250:
            rc_utils.draw_contour(color_image, contour, rc_utils.ColorBGR.red.value)
            rc_utils.draw_circle(color_image, red_center, rc_utils.ColorBGR.red.value)
        else:
            red_center = None
            red_distance = 0
    else:
        red_center = None
        red_distance = 0

    # Search for the blue cone
    contours2 = rc_utils.find_contours(color_image, GREEN[0], GREEN[1])
    contour2 = rc_utils.get_largest_contour(contours2, 30)

    if contour2 is not None:
        blue_center = rc_utils.get_contour_center(contour2)
        blue_distance = rc_utils.get_pixel_average_distance(depth_image, blue_center)

        # Only use count it if the cone is less than MAX_DISTANCE away
        if blue_distance <= 250:
            rc_utils.draw_contour(color_image, contour2, rc_utils.ColorBGR.blue.value)
            rc_utils.draw_circle(
                color_image, blue_center, rc_utils.ColorBGR.blue.value
            )
        else:
            blue_center = None
            blue_distance = 0
    else:
        blue_center = None
        blue_distance = 0

    rc.display.show_color_image(color_image)

def update_cones():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global cur_mode
    global counter
    global angle
    global speed
    global red_distance
    global red_center
    global blue_distance
    global blue_center
    global cones_done
    global cur_state

    rc.drive.set_max_speed(0.25)

    find_cones()
    
    # Align ourselves to smoothly approach and pass the red cone while it is in view
    if cur_mode == Mode.red_align:
        # Once the red cone is out of view, enter Mode.red_pass
        if (
            red_center is None
            or red_distance == 0
            or red_distance - prev_red_distance > CLOSE_DISTANCE
        ):
            if 0 < prev_red_distance < FAR_DISTANCE:
                counter = max(1.5, counter)
                cur_mode = Mode.red_pass
            else:
                cur_mode = Mode.no_cones

        # If it seems like we are not going to make the turn, enter Mode.red_reverse
        elif (
            red_distance < REVERSE_DISTANCE
            and red_center[1] > rc.camera.get_width() // 4
        ):
            counter = REVERSE_BRAKE_TIME
            cur_mode = Mode.red_reverse

        # Align with the cone so that it gets closer to the left side of the screen
        # as we get closer to it, and slow down as we approach
        else:
            goal_point = rc_utils.remap_range(
                red_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                0,
                rc.camera.get_width() // 4,
                True,
            )

            angle = rc_utils.remap_range(
                red_center[1], goal_point, rc.camera.get_width() // 2, 0, 1
            )
            angle = rc_utils.clamp(angle, -1, 1)

            speed = rc_utils.remap_range(
                red_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                MIN_ALIGN_SPEED,
                MAX_ALIGN_SPEED,
                True,
            )

    elif cur_mode == Mode.blue_align:
        if (
            blue_center is None
            or blue_distance == 0
            or blue_distance - prev_blue_distance > CLOSE_DISTANCE
        ):
            if 0 < prev_blue_distance < FAR_DISTANCE:
                counter = max(1.5, counter)
                cur_mode = Mode.blue_pass
            else:
                cur_mode = Mode.no_cones
        elif (
            blue_distance < REVERSE_DISTANCE
            and blue_center[1] < rc.camera.get_width() * 3 // 4
        ):
            counter = REVERSE_BRAKE_TIME
            cur_mode = Mode.blue_reverse
        else:
            goal_point = rc_utils.remap_range(
                blue_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                rc.camera.get_width(),
                rc.camera.get_width() * 3 // 4,
                True,
            )

            angle = rc_utils.remap_range(
                blue_center[1], goal_point, rc.camera.get_width() // 2, 0, -1
            )
            angle = rc_utils.clamp(angle, -1, 1)

            speed = rc_utils.remap_range(
                blue_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                MIN_ALIGN_SPEED,
                MAX_ALIGN_SPEED,
                True,
            )

    # Curve around the cone at a fixed speed for a fixed time to pass it
    if cur_mode == Mode.red_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, -0.5)
        speed = PASS_SPEED
        counter -= rc.get_delta_time()

        # After the counter expires, enter Mode.blue_align if we see the blue cone,
        # and Mode.blue_find if we do not
        if counter <= 0:
            cur_mode = Mode.blue_align if blue_distance > 0 else Mode.blue_find

    elif cur_mode == Mode.blue_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, 0.5)
        speed = PASS_SPEED

        counter -= rc.get_delta_time()
        if counter <= 0:
            cur_mode = Mode.red_align if red_distance > 0 else Mode.red_find

    # If we know we are supposed to be aligning with a red cone but do not see one,
    # turn to the right until we find it
    elif cur_mode == Mode.red_find:
        angle = 1
        speed = FIND_SPEED
        if red_distance > 0:
            cur_mode = Mode.red_align

    elif cur_mode == Mode.blue_find:
        angle = -1
        speed = FIND_SPEED
        if blue_distance > 0:
            cur_mode = Mode.blue_align

    # If we are not going to make the turn, reverse while keeping the cone in view
    elif cur_mode == Mode.red_reverse:
        if counter >= 0:
            counter -= rc.get_delta_time()
            speed = -1
            angle = 1
        else:
            angle = -1
            speed = REVERSE_SPEED
            if (
                red_distance > STOP_REVERSE_DISTANCE
                or red_center[1] < rc.camera.get_width() // 10
            ):
                counter = LONG_PASS_TIME
                cur_mode = Mode.red_align

    elif cur_mode == Mode.blue_reverse:
        if counter >= 0:
            counter -= rc.get_delta_time()
            speed = -1
            angle = 1
        else:
            angle = 1
            speed = REVERSE_SPEED
            if (
                blue_distance > STOP_REVERSE_DISTANCE
                or blue_center[1] > rc.camera.get_width() * 9 / 10
            ):
                counter = LONG_PASS_TIME
                cur_mode = Mode.blue_align

    # If no cones are seen, drive forward until we see either a red or blue cone
    elif cur_mode == Mode.no_cones:
        angle = 0
        speed = NO_CONES_SPEED

        if red_distance > 0 and blue_distance == 0:
            cur_mode = Mode.red_align
        elif blue_distance > 0 and red_distance == 0:
            cur_mode = Mode.blue_align
        elif blue_distance > 0 and red_distance > 0:
            cur_mode = (
                Mode.red_align if red_distance < blue_distance else Mode.blue_align
            )

    print(
        f"Mode: {cur_mode.name}, red_distance: {red_distance:.2f} cm, blue_distance: {blue_distance:.2f} cm, speed: {speed:.2f}, angle: {angle:2f}"
    )
    
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
