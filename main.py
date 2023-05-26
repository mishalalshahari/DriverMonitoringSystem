from kivymd.app import MDApp
from kivymd.uix.label import MDLabel,MDIcon
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDFlatButton,MDRectangleFlatButton,MDFloatingActionButton,MDRoundFlatButton
from kivymd.uix.dialog import MDDialog
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.image import Image

import numpy as np
import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import winsound
import threading
Window.size=(300,500)

class App(MDApp,Image):

    def start(self):
        self.run = True
        self.isbeep = False
        def calculate_EAR(eye):

            y1 = dist.euclidean(eye[1], eye[5])
            y2 = dist.euclidean(eye[2], eye[4])

            x1 = dist.euclidean(eye[0], eye[3])

            EAR = (y1 + y2) / x1
            return EAR

        maxcy = countyawn = 0
        maxb = countblink = 0
        blink_thresh = 0.45
        succ_frame = 2
        count_frame = 0
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        def cal_yawn(shape):
            top_lip = shape[50:53]
            top_lip = np.concatenate((top_lip, shape[61:64]))

            low_lip = shape[56:59]
            low_lip = np.concatenate((low_lip, shape[65:68]))

            top_mean = np.mean(top_lip, axis=0)
            low_mean = np.mean(low_lip, axis=0)

            distance = dist.euclidean(top_mean, low_mean)
            return distance

        cam = cv2.VideoCapture(0)


        face_model = dlib.get_frontal_face_detector()
        landmark_model = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        yawn_thresh = 35
        ptime = 0
        while self.run:
            suc, frame = cam.read()

            if not suc:
                break

            ctime = time.time()
            fps = int(1 / (ctime - ptime))
            ptime = ctime
            cv2.putText(frame, f'FPS:{fps}', (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 200, 0), 3)

            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_model(img_gray)
            for face in faces:

                shapes = landmark_model(img_gray, face)
                shape = face_utils.shape_to_np(shapes)

                lefteye = shape[L_start: L_end]
                righteye = shape[R_start:R_end]

                left_EAR = calculate_EAR(lefteye)
                right_EAR = calculate_EAR(righteye)

                avg = (left_EAR + right_EAR) / 2

                if avg < blink_thresh:
                    count_frame += 1
                    countblink += 1
                else:
                    if count_frame >= succ_frame:
                        cv2.putText(frame, f'blinked {countblink},max: {maxb}', (30, 30),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        maxb = max(maxb, countblink)
                        countblink = 0
                        if maxb >= 30:
                            winsound.Beep(2500, 3000)
                            self.isbeep = True
                            maxb = 0
                            continue
                    else:
                        count_frame = 0

                lip = shape[48:60]

                lip_dist = cal_yawn(shape)
                if lip_dist > yawn_thresh:
                    # cv2.putText(frame, f'User Yawning! count{countyawn}, max{maxcy}',(frame.shape[1]//2 - 170 ,frame.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),2)
                    countyawn += 1
                    maxcy = max(maxcy, countyawn)

                    if countyawn == 5:
                        countyawn = 0
                        winsound.Beep(2500, 3000)
                        self.isbeep = True


        cam.release()
        cv2.destroyAllWindows()

    def build(self):
        self.theme_cls.primary_palette="Teal"
        self.theme_cls.theme_style="Dark"
        screen = Screen()
        label = MDLabel(text="Hello there, Hope you're doing well.",halign='center',theme_text_color='Custom',text_color="#FFFFFF",
                        font_style='H6',pos_hint={'center_x':0.5, 'center_y':0.6})
        icon_label = MDIcon(icon='language-python',halign='center')

        btn_flat = MDRoundFlatButton(text='Start Monitoring',pos_hint={'center_x':0.5, 'center_y':0.4},on_release=self.monitor)
        screen.add_widget(label)
        screen.add_widget(btn_flat)
        return screen

    def monitor(self,obj):

        close_button = MDFlatButton(text="Stop Monitoring",on_release=self.close_dialog)
        self.dialog = MDDialog(title="Cool,",text="You're being monitored!",size_hint=(0.7,1),buttons=[close_button])
        self.dialog.open()
        self.thread = threading.Thread(target = self.start)
        self.thread.start()

    def close_dialog(self,obj):
        self.dialog.dismiss()
        self.run = False
        self.thread.stop()

App().run()