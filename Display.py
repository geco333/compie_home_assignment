import logging
from datetime import datetime
from multiprocessing import Process, Queue

import cv2
from numpy import ndarray


class Display(Process):
    def __init__(self, display_queue: Queue):
        super().__init__()

        self.display_queue = display_queue

    def run(self):
        while True:
            frame_data = self.display_queue.get()

            if isinstance(frame_data, str) and frame_data == "done":
                logging.info("Display shutting down.")
                break
            elif isinstance(frame_data, tuple):
                logging.debug("Display")

                frame, x, y, w, h = frame_data

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame,
                            datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                            (10, frame.shape[0] - 690),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            1)

                blur = frame[y:y + h, x:x + w]
                blur = cv2.GaussianBlur(blur, (17, 17), 30)

                frame[y:y + blur.shape[0], x:x + blur.shape[1]] = blur

                self.show_detection(frame)

    def show_detection(self, frame: ndarray):
        """Displays the frame with all detections and a timestamp.
            the user must press 'q' or 'k' to close the display.

        :param frame: The frame to display.
        """

        cv2.imshow("video", frame)

        while key := cv2.waitKey(0) not in [ord('q'), ord('k')]:
            if key == ord('q'):
                break
