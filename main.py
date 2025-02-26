import logging
from multiprocessing import Queue
from sys import argv
from typing import Tuple

import cv2
from numpy import ndarray

from Detector import Detector
from Display import Display

SAMPLE_VIDEO_URL = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"

logging.basicConfig(level=logging.DEBUG)


def capture_frame(cap: cv2.VideoCapture) -> Tuple[bool, ndarray]:
    """Capture a single frame from the stream,
    convert all colors to grayscale and blur it.

    :return: A single captured frame from the video stream and its rate.
    """

    ret, frame = cap.read()
    first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return ret, cv2.GaussianBlur(first_frame, (21, 21), 0)


def main(video_url: str):
    """Accepts a URL link to a video and capture it using OpenCV.
        The main function creates 2 sub processes: Detector and Display,
        Detector checks for changes in the frames,
        Display draws a square around the found changes,
        adds a timestamp and displays the frame to the user.

    :param video_url: A valid URL link to a video.
    """

    cap = cv2.VideoCapture(video_url)

    # If the video link is invalid for some reason exit with code 1
    if not cap.isOpened():
        logging.error('Fail to open video url.')
        exit(1)

    ret, first_frame = capture_frame(cap)

    # Create a Queue object in order to send the frame to the Detector instance,
    #  and another Queue for the Detector to communicate with the Display instance.
    frame_queue = Queue()
    display_queue = Queue()

    detector = Detector(first_frame, frame_queue, display_queue)
    display = Display(display_queue)

    detector.start()
    display.start()

    # Iterate each frame in the video and send it to the Detector,
    #  once all the frames are sent, send a 'done' string to the close the
    #  both the Detector and Display instances.
    while True:
        try:
            ret, frame = capture_frame(cap)
        except cv2.error:
            break

        if not ret:
            break

        frame_queue.put(frame)

    frame_queue.put("done")

    detector.join()
    display.join()

    logging.info("Main finished.")


if __name__ == "__main__":
    # If no video url passed in commandline use a default sample one.

    try:
        video_url = argv[1]
    except IndexError:
        video_url = SAMPLE_VIDEO_URL

    main(video_url)
