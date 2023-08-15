import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np

from ultralytics import YOLO

class MainApp(App):
    def build(self):
        self.model = YOLO("./models/best.pt")
        self.model.conf = 0.40
        self.capture = cv2.VideoCapture(0)

        layout = BoxLayout(orientation='vertical')

        # Video feed display
        self.video_feed = Image()
        layout.add_widget(self.video_feed)

        # Engagement score display
        self.engagement_label = Label(text="Engagement Score: 0", size_hint=(1, 0.1))
        layout.add_widget(self.engagement_label)

        # Schedule the update function
        Clock.schedule_interval(self.update, 1.0/30.0)

        return layout

    import cv2

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use the model to predict bounding boxes
            results = self.model.predict(frame_rgb)
            
            # Draw bounding boxes on the frame
            for result in results:
                if result.boxes:
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Convert the frame to texture
            texture = self.frame_to_texture(frame_rgb)
            self.video_feed.texture = texture


    def frame_to_texture(self, frame):
        """Convert an RGB frame to a Kivy texture."""
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        return texture

if __name__ == '__main__':
    MainApp().run()
