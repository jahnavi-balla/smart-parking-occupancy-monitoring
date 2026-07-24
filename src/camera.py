"""
camera.py
---------
Wraps OpenCV's VideoCapture so the rest of the app doesn't need to know
about OpenCV's quirks (raw indices, release() calls, isOpened() checks, etc).

Works with:
    - A webcam (pass an integer index, e.g. 0 for the default camera)
    - A video file (pass a file path string, e.g. "videos/sample.mp4")
"""

import cv2


class CameraError(Exception):
    """Raised when the camera/video source can't be opened or read from."""
    pass


class Camera:
    """
    A thin, friendly wrapper around cv2.VideoCapture.

    Usage:
        with Camera(source=0) as cam:
            for frame in cam.frames():
                ...
    """

    def __init__(self, source=0, width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None  # will hold the cv2.VideoCapture object once opened

    def open(self):
        """
        Opens the video source. Raises CameraError with a clear message
        if it fails, instead of letting the program crash with a cryptic
        OpenCV error.
        """
        self.cap = cv2.VideoCapture(self.source)

        # Ask the camera to use a specific resolution. This can silently
        # fail on some hardware, which is fine - we just get the default.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise CameraError(
                f"Could not open video source '{self.source}'. "
                "Check that a webcam is connected (or that the video "
                "file path is correct) and that no other app is using it."
            )
        return self

    def frames(self):
        """
        A generator that yields frames one at a time until the source
        runs out (end of video file) or is interrupted.

        Using a generator means main.py can simply do:
            for frame in cam.frames():
                ...
        without worrying about the read()/ret checking boilerplate.
        """
        if self.cap is None:
            raise CameraError("Camera.open() must be called before frames().")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                # This happens naturally at the end of a video file,
                # or if a webcam gets disconnected mid-stream.
                break

            yield frame

    def release(self):
        """Frees the camera/video resource. Always safe to call."""
        if self.cap is not None:
            self.cap.release()

    # --- Context manager support: enables the `with Camera(...) as cam:` syntax ---
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        # Returning False means: don't suppress any exception that occurred.
        return False