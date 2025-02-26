import logging
from multiprocessing import Process, Queue

import cv2
import imutils
from numpy import ndarray


class Detector(Process):
    def __init__(self,
                 first_frame: ndarray,
                 frame_queue: Queue,
                 display_queue: Queue):
        super().__init__()

        self.first_frame = first_frame
        self.frame_queue = frame_queue
        self.display_queue = display_queue

    def run(self):
        """Run until `done` is passed to `display_queue` from the main process,
            for each frame pulled from the queue use OpenCV to detect changes
            from `first_frame`, pass all changes to `Display` for processing."""

        while True:
            frame = self.frame_queue.get()

            if isinstance(frame, str) and frame == "done":
                self.display_queue.put("done")
                logging.info("Detector shutting down.")

                break
            elif isinstance(frame, ndarray):
                frame_delta = cv2.absdiff(self.first_frame, frame)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

                thresh = cv2.dilate(thresh,
                                    None,
                                    iterations=2)
                cnts = cv2.findContours(thresh.copy(),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                for c in cnts:
                    if cv2.contourArea(c) < 500:
                        continue

                    x, y, w, h = cv2.boundingRect(c)

                    self.display_queue.put((frame, x, y, w, h))
