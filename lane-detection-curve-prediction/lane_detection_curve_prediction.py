from pathlib import Path
from print_utils import Printer
from image_utils import rescale, birds_eye_view, pipeline, colors
import numpy as np
import cv2 as cv

def main(fname = './challenge.mp4'):
    assert Path(fname).is_file()
    cap = cv.VideoCapture(fname)
    if not cap.isOpened():
        Printer.red("Unable to open video file.")
        return

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # rescale the image to be smaller
            video_frame_rescaled = rescale(frame)
            h, w = video_frame_rescaled.shape[:2]
            canvas = np.zeros((int(h * 1.2), w, 3) , dtype='uint8')
            canvas[:h, :w] = video_frame_rescaled

            # find the topview
            transform, birds_eye_view = birds_eye_view(video_frame_rescaled)
            birds_eye_view_rescaled = rescale(birds_eye_view, scale = 0.2)
            small_h, small_w = birds_eye_view_rescaled.shape[:2]

            # render top view
            canvas[h:h+small_h, :small_w] = birds_eye_view_rescaled
            result = pipeline(birds_eye_view_rescaled)

            # gray_scale
            result["gray_scale_image"] = result["gray_scale_image"].reshape((result["gray_scale_image"].shape[0], result["gray_scale_image"].shape[1], 1))
            canvas[h:h+small_h, small_w:int(small_w * 2)] = result["gray_scale_image"]

            # lane image
            canvas[h:h+small_h, int(small_w * 2):int(small_w * 3)] = result["lane_image"]

            # detected_lane
            canvas[h:h+small_h, int(small_w * 3):int(small_w * 4)] = result["detected_lane"]

            # detected_yellow_masked_lane
            canvas[h:h+small_h, int(small_w * 4):] = result["masked_yellow_lane"]


            # project curve back to original image
            result_big = pipeline(topview)
            recovered_lane_coverage = cv.warpPerspective(result_big["lane_coverage"], np.linalg.inv(transform), (w, h))
            coverage_region = np.where(recovered_lane_coverage != (0, 0, 0))
            projected_final_image = cv.addWeighted(video_frame_rescaled, 1, recovered_lane_coverage, 0.4, 0.0)
            canvas[:h, :w] = projected_final_image

            # text
            # white_lane_text_center = (int(0.9*w)+3,int(0.03*h))
            # yellow_lane_text_center = (int(0.9*w)+6,int(0.03*h))
            # average_lane_text_center = (int(0.9*w)+9,int(0.03*h))
            turn_text_center = (int(0.9*w),int(0.03*h))

            # white_lane_text = "White Lane Curvature: {}".format(int(result["white_lane_curvature"]))
            # yellow_lane_text = "Yellow Lane Curvature: {}".format(int(result["yellow_lane_curvature"]))
            # average_lane_text = "Average Curvature: {}".format(int(result["average_curvature"]))
            turn_text = "Turning right" if int(result["average_curvature"]) > 0 else "Turning left"

            # cv.putText(canvas, white_lane_text, white_lane_text_center, cv.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, thickness=1)
            # cv.putText(canvas, yellow_lane_text, yellow_lane_text_center, cv.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, thickness=1)
            # cv.putText(canvas, average_lane_text, average_lane_text_center, cv.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, thickness=1)
            cv.putText(canvas, turn_text, turn_text_center, cv.FONT_HERSHEY_SIMPLEX, 0.4, colors.get("RED",(0,0,0)), thickness=1)

            cv.imshow('Lane Detection and Curve Prediction', canvas)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            break


    cap.release()

if __name__ == '__main__':
   main()
