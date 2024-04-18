import cv2 as cv
import numpy as np

colors = {"RED" : (0,0,255),
          "YELLOW" : (0,255,255),
          "GREEN" : (0,255,0),
          "WHITE" : (255,255,255)}

def rescale(frame, scale=0.75):
    """_summary_

    Args:
        frame (_type_): cv2 frame
        scale (float, optional): _description_. Defaults to 0.75.

    Returns:
        _type_: rescales the image to the scale value and returns the image
    """
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# fit curve by points
def fit_curve(frame,points):
    """_summary_

    Args:
        frame (_type_): image
        points (_type_): points within the image that correspond to a certain color

    Returns:
        _type_: polynomial curve parameters and curve points
    """
    y, x = points
    fit_points = []
    if len(x) > 0:
        curve_params = np.polyfit(np.array(y), np.array(x), 2)
        curve = np.poly1d(curve_params)
        for h in range(0, frame.shape[0]):
            w = curve(h).astype(int)
            if w >= frame.shape[1] or w < 0:
                continue
            fit_points.append((w, h))
    return curve_params, np.array(fit_points)

def draw_points(points, image, color):
    """_summary_

    Args:
        points (_type_): curve points returned from fit_curve
        image (_type_): image on which the curve is drawn
        color (_type_): color value
    """
    for i, point in enumerate(points):
        w, h = point
        # draw average mid lane
        if color == colors.get("WHITE",(0,0,0)) and i%5==0:
            continue
        image[h][w] = color

def calculate_curvature(curve_params, point):
    """_summary_

    Args:
        curve_params (_type_): curve parameters returned from fit_curve
        point (_type_): given point

    Returns:
        _type_: _description_
    """
    a, b, c = curve_params
    first_derivative = 2 * a * point + b * point
    second_derivative = 2 * a
    radius = np.sqrt((1 + first_derivative**2)**3) / np.abs(second_derivative)
    return radius


def pipeline(frame):
    temp = frame.copy()
    # canvas for detected lane
    detected_lane = np.zeros((frame.shape), dtype='uint8')

    # find white dash line
    gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    _, masked_right = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    h, w = masked_right.shape
    masked_right[0:h, int(0.9 * w):] = 0
    masked_right[0:h, :int(0.75 * w)] = 0
    right_lane = np.where(masked_right == 255)

    # white lane
    white_lane_curve_param, white_points = fit_curve(frame,right_lane)
    white_curvature = calculate_curvature(white_lane_curve_param, int(temp.shape[0] / 2))
    draw_points(white_points, detected_lane, colors.get("GREEN",(0,0,0)))

    # find yellow line
    hsv = cv.cvtColor(temp, cv.COLOR_BGR2HSV)
    lower = np.array([20, 120, 120], dtype="uint8")
    upper = np.array([30, 255, 255], dtype="uint8")
    masked_yellow = cv.inRange(hsv, lower, upper)
    yellow_lane = np.where(masked_yellow == 255)
    # or left lane
    _, masked_left = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)
    h, w = masked_left.shape
    masked_left[0:h, :int(0.15 * w)] = 0
    masked_left[0:h, int(0.5 * w):] = 0
    left_lane = np.where(masked_left == 255)

    # yellow lane/left lane
    yellow_lane_curve_param, yellow_points = fit_curve(frame,yellow_lane) #left_lane
    yellow_curvature = calculate_curvature(yellow_lane_curve_param, int(temp.shape[0] / 2))
    draw_points(yellow_points, detected_lane, colors.get("YELLOW",(0,0,0)))

    # average lane
    mid_lane = np.where(cv.cvtColor(detected_lane, cv.COLOR_BGR2GRAY) != 0)
    mid_lane_curve_param, mid_lane_points = fit_curve(frame, mid_lane)
    average_curvature = calculate_curvature(mid_lane_curve_param, int(temp.shape[0] / 2))
    average_points = np.array(mid_lane_points)
    draw_points(average_points, detected_lane, colors.get("WHITE",(0,0,0)))

    lane_img = cv.cvtColor(cv.bitwise_or(masked_right, masked_yellow), cv.COLOR_GRAY2BGR)

    points = np.vstack((yellow_points, np.flip(white_points, 0)))
    lane_coverage = detected_lane.copy()
    cv.fillPoly(lane_coverage, [points], (102, 255, 178))

    result  = {"lane_image": lane_img,
               "detected_lane": detected_lane,
               "lane_coverage": lane_coverage,
               "white_curve": white_points,
               "white_lane_curvature": white_curvature,
               "yellow_curve": yellow_points,
               "masked_yellow_lane": hsv,
               "yellow_lane_curvature": yellow_curvature,
               "average_curvature": average_curvature,
               "gray_scale_image": gray}

    return result

def ROI():
    p1 = (420, 349)
    p2 = (571, 349)
    p3 = (212, 446)
    p4 = (740, 446)
    return p1,p2,p3,p4

# draw region of interest
def draw_ROI(frame):
    temp = frame.copy()
    p1,p2,p3,p4 = ROI()
    cv.line(temp, p1, p2, colors.get("RED",(0,0,0)), 3)
    cv.line(temp, p2, p4, colors.get("RED",(0,0,0)), 3)
    cv.line(temp, p3, p4, colors.get("RED",(0,0,0)), 3)
    cv.line(temp, p1, p3, colors.get("RED",(0,0,0)), 3)
    cv.circle(temp, p1, 10, colors.get("RED",(0,0,0)), -1)
    cv.circle(temp, p2, 10, colors.get("RED",(0,0,0)), -1)
    cv.circle(temp, p3, 10, colors.get("RED",(0,0,0)), -1)
    cv.circle(temp, p4, 10, colors.get("RED",(0,0,0)), -1)
    return temp

def birds_eye_view(frame):
    temp = frame.copy()
    p1,p2,p3,p4 = ROI()
    h, w = frame.shape[:2]
    pts1 = np.float32([p1, p2, p3, p4])
    pts2 = np.float32([[int(w/10), int(h/10)], [int(w*9/10), int(h/10)], [int(w/10), int(h*9/10)], [int(w*9/10), int(h*9/10)]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(temp, M, (w, h))
    return M, dst

