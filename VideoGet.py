from threading import Thread
import cv2
from queue import Queue
from math import ceil


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0, webcamStream=False):
        self.src = src
        self.webcamStream = webcamStream
        self.stream = cv2.VideoCapture(src)
        stream_fps = int(self.stream.get(cv2.CAP_PROP_FPS))
        stream_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stream_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"FPS : {stream_fps},  Height: {stream_height},  Width: {stream_width}")
        # self.frame_to_skip = ceil(stream_fps / 12)
        self.frame_to_skip = 1
        (self.grabbed, self.frame) = self.stream.read()
        self.queue = Queue(1000)
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        count_skip_frame = 0
        while not self.stopped:
            count_skip_frame += 1
            (self.grabbed, self.frame) = self.stream.read()
            if not self.queue.full():
                if not self.grabbed:
                    self.stop()
                    return
                if count_skip_frame % self.frame_to_skip == 0:
                    self.queue.put(self.frame)

    def read(self):
        if self.src == 0 or self.webcamStream:
            return self.frame
        else:
            return self.queue.get()

    def more(self):
        if self.src == 0 or self.webcamStream:
            return True
        else:
            return self.queue.qsize() > 0

    def stop(self):
        self.stopped = True
