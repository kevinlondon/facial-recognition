# coding:utf-8
import time
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
import cv2
import pybind_example

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

WINDOW_SIZE = 10


class DetectLayout(BoxLayout):

    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = 'IMG_{}.png'.format(timestr)
        camera.export_to_png(filename)
        print("Captured as {}".format(filename))


class KivyCamera(Image):

    def __init__(self, fps=30, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.actual_fps = []

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return frame

    def frame_to_texture(self, frame):
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return image_texture

    def update_fps(self):
        self.actual_fps.append(Clock.get_fps())
        stats = pybind_example.rolling_stats(self.actual_fps, WINDOW_SIZE)
        fpses = [stat[0] for stat in stats]
        self.fps = str(self.actual_fps[-1])

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        if self.detect:
            self.detect_faces(frame)

        if self.green_mode:
            frame[:, :, 2] = 0
            frame[:, :, 0] = 0

        self.update_fps()

        # display image from the texture
        self.texture = self.frame_to_texture(frame)


class CamApp(App):

    def build(self):
        self.layout = DetectLayout()
        return self.layout

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.layout.ids['camera'].capture.release()


if __name__ == '__main__':
    CamApp().run()
