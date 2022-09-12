import os
import time
import joblib
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtGui
from myGUI import Ui_MainWindow
import sys
import numpy as np
import cv2
from GestureScore.utils import *
from GestureScore.body_part_angle import *
import mediapipe as mp
from GestureScore.types_of_exercise import TypeOfExercise
from GestureTrack.sample_pose import *
from GestureTrack.sample_pose2d import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class ScoreThread(QThread):
    sinOut = pyqtSignal(QImage)
    scoreSignal = pyqtSignal(str)

    def __init__(self, mw, exercise_type):
        super(ScoreThread, self).__init__()
        self.cond = QWaitCondition()
        self._isPause = False
        self.mutex = QMutex()
        self.mw = mw
        self.exercise_type = exercise_type

    def pause(self):
        self._isPause = True

    def run(self):
        prevTime = 0
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            counter = 0  # movement of exercise
            status = True  # state of move
            avg_score = 0
            self.mutex.lock()
            while self.mw.cap.isOpened():
                ret, frame = self.mw.cap.read()
                nchannel = frame.shape[2]

                frame = cv2.resize(frame, (1200, 680), interpolation=cv2.INTER_AREA)
                # recolor frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                # make detection
                results = pose.process(frame)
                # recolor back to BGR
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    counter, status, avg_score = TypeOfExercise(landmarks).calculate_exercise(
                        self.exercise_type, counter, status, avg_score)
                except:
                    pass

                TypeOfExercise(landmarks).score_table(self.exercise_type, counter, status, avg_score, self._isPause)
                self.scoreSignal.emit(str(avg_score))

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255),
                                           thickness=2,
                                           circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,255,0),
                                           thickness=2,
                                           circle_radius=2),
                )
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 6)

                frameHeight = frame.shape[0]
                frameWidth = frame.shape[1]
                a = self.mw.ui.video.size()
                if a.width() / frameWidth < a.height() / frameHeight:
                    scaleFactor = a.width() / frameWidth
                else:
                    scaleFactor = 1.0 * a.height() / frameHeight

                timg = cv2.resize(frame, (int(scaleFactor * frame.shape[1]), int(scaleFactor * frame.shape[0])))
                timg = cv2.cvtColor(timg, cv2.COLOR_BGR2RGB)
                limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel * timg.shape[1],
                                      QtGui.QImage.Format_RGB888)
                self.mw.ui.video.setPixmap(QtGui.QPixmap(limage))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                if self._isPause:
                    break
                    #self.cond.wait(self.mutex)
            cv2.destroyAllWindows()
            #self.msleep(1000)
            #self.mutex.unlock()


class TrackThread(QThread):
    sinImage = pyqtSignal(QImage)

    def __init__(self, mw):
        super(TrackThread, self).__init__()
        self.cond = QWaitCondition()
        self.mw = mw

    def run(self):
        time.sleep(1)
        model_complexity = 1
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.5
        enable_segmentation = False
        segmentation_score_th = 0.5
        use_brect = True
        prevTime = 0

        pose = mp_pose.Pose(
            # upper_body_only=upper_body_only,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        while self.mw.cap.isOpened():
            ret, image = self.mw.cap.read()
            image = cv.flip(image, 1)  # ミラー表示
            debug_image = copy.deepcopy(image)

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = pose.process(image)

            if enable_segmentation and results.segmentation_mask is not None:
                mask = np.stack((results.segmentation_mask,) * 3,
                                axis=-1) > segmentation_score_th
                bg_resize_image = np.zeros(image.shape, dtype=np.uint8)
                bg_resize_image[:] = (0, 255, 0)
                debug_image = np.where(mask, debug_image, bg_resize_image)
            if results.pose_landmarks is not None:
                brect = calc_bounding_rect(debug_image, results.pose_landmarks)
                debug_image = draw_landmarks(
                    debug_image,
                    results.pose_landmarks,
                )
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(debug_image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (175, 65, 84), 6)

            frameHeight = debug_image.shape[0]
            frameWidth = debug_image.shape[1]
            a = self.mw.ui.video.size()
            if a.width() / frameWidth < a.height() / frameHeight:
                scaleFactor = a.width() / frameWidth
            else:
                scaleFactor = 1.0 * a.height() / frameHeight

            timg = cv2.resize(debug_image,
                              (int(scaleFactor * debug_image.shape[1]), int(scaleFactor * debug_image.shape[0])))
            timg = cv2.cvtColor(timg, cv2.COLOR_BGR2RGB)
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], timg.shape[2] * timg.shape[1],
                                  QtGui.QImage.Format_RGB888)
            self.mw.ui.video.setPixmap(QtGui.QPixmap(limage))
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break


class PlotThread(QThread):
    def __init__(self, filename):
        super(PlotThread, self).__init__()
        self.cond = QWaitCondition()
        self.filename = filename

    def run(self):
        if self.filename != "None":
            cmd = "python GestureTrack\sample_pose.py --device " + self.filename + " --plot_world_landmark"
        else:
            cmd = "python GestureTrack\sample_pose.py --plot_world_landmark"
        os.system(cmd)


class Plot2dThread(QThread):
    def __init__(self, mw, bg_path):
        super(Plot2dThread, self).__init__()
        self.cond = QWaitCondition()
        self._isPause = False
        self.mutex = QMutex()
        self.mw = mw
        self.bg_path = bg_path

    def pause(self):
        self._isPause = True

    def run(self):
        static_image_mode = False
        model_complexity = 1
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.5
        rev_color = False

        self.mw.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.mw.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

        pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # 色指定
        if rev_color:
            color = (255, 255, 255)
            bg_color = (100, 33, 3)
        else:
            color = (100, 33, 3)
            bg_color = (255, 255, 255)

        while True:
            ret, image = self.mw.cap.read()

            image = cv.flip(image, 1)
            debug_image01 = copy.deepcopy(image)
            debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
            cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]), bg_color, thickness=-1)

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks is not None:
                # 描画
                debug_image01 = draw_landmarks(
                    debug_image01,
                    results.pose_landmarks,
                )
                debug_image02 = draw_stick_figure(
                    debug_image02,
                    results.pose_landmarks,
                    color=color,
                    bg_color=bg_color,
                )

            frameHeight = debug_image01.shape[0]
            frameWidth = debug_image01.shape[1]
            a = self.mw.ui.video.size()
            if a.width() / frameWidth < a.height() / frameHeight:
                scaleFactor = a.width() / frameWidth
            else:
                scaleFactor = 1.0 * a.height() / frameHeight

            timg = cv2.resize(debug_image01,
                              (int(scaleFactor * debug_image01.shape[1]), int(scaleFactor * debug_image01.shape[0])))
            timg = cv2.cvtColor(timg, cv2.COLOR_BGR2RGB)
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], timg.shape[2] * timg.shape[1],
                                  QtGui.QImage.Format_RGB888)
            self.mw.ui.video.setPixmap(QtGui.QPixmap(limage))

            debug_image02 = cv2.resize(debug_image02, (
            int(scaleFactor * debug_image02.shape[1]), int(scaleFactor * debug_image02.shape[0])))
            # debug_image02.resize(400, 260)
            cv.imshow('Pose 2D', debug_image02)
            key = cv.waitKey(1)
            if self._isPause:
                break
        cv2.destroyAllWindows()


class SegmentThread(QThread):
    def __init__(self, mw, bg_path):
        super(SegmentThread, self).__init__()
        self.cond = QWaitCondition()
        self._isPause = False
        self.mutex = QMutex()
        self.mw = mw
        self.bg_path = bg_path

    def pause(self):
        self._isPause = True

    def run(self):
        model_selection = 0
        score_th = 0.1

        if self.bg_path is not None:
            bg_image = cv.imread(self.bg_path)
        else:
            bg_image = None

        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection)

        while True:
            ret, image = self.mw.cap.read()

            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = selfie_segmentation.process(image)

            # 描画 ################################################################
            mask = np.stack((results.segmentation_mask,) * 3, axis=-1) >= score_th

            if bg_image is None:
                bg_resize_image = np.zeros(image.shape, dtype=np.uint8)
                bg_resize_image[:] = (0, 255, 0)
            else:
                bg_resize_image = cv.resize(bg_image, (image.shape[1], image.shape[0]))
            debug_image = np.where(mask, debug_image, bg_resize_image)

            frameHeight = debug_image.shape[0]
            frameWidth = debug_image.shape[1]
            a = self.mw.ui.video.size()
            if a.width() / frameWidth < a.height() / frameHeight:
                scaleFactor = a.width() / frameWidth
            else:
                scaleFactor = 1.0 * a.height() / frameHeight

            timg = cv2.resize(debug_image,
                              (int(scaleFactor * debug_image.shape[1]), int(scaleFactor * debug_image.shape[0])))
            timg = cv2.cvtColor(timg, cv2.COLOR_BGR2RGB)
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], timg.shape[2] * timg.shape[1],
                                  QtGui.QImage.Format_RGB888)
            self.mw.ui.video.setPixmap(QtGui.QPixmap(limage))

            if self._isPause:
                break


class SafetyTread(QThread):
    def __init__(self, mw):
        super(SafetyTread, self).__init__()
        self.cond = QWaitCondition()
        self._isPause = False
        self.mutex = QMutex()
        self.mw = mw

    def pause(self):
        self._isPause = True

    def run(self):
        pose_knn = joblib.load('GestutreSafety/Model/PoseKeypoint.joblib')
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        prevTime = 0
        keyXYZ = [
            "nose_x",
            "nose_y",
            "nose_z",
            "left_eye_inner_x",
            "left_eye_inner_y",
            "left_eye_inner_z",
            "left_eye_x",
            "left_eye_y",
            "left_eye_z",
            "left_eye_outer_x",
            "left_eye_outer_y",
            "left_eye_outer_z",
            "right_eye_inner_x",
            "right_eye_inner_y",
            "right_eye_inner_z",
            "right_eye_x",
            "right_eye_y",
            "right_eye_z",
            "right_eye_outer_x",
            "right_eye_outer_y",
            "right_eye_outer_z",
            "left_ear_x",
            "left_ear_y",
            "left_ear_z",
            "right_ear_x",
            "right_ear_y",
            "right_ear_z",
            "mouth_left_x",
            "mouth_left_y",
            "mouth_left_z",
            "mouth_right_x",
            "mouth_right_y",
            "mouth_right_z",
            "left_shoulder_x",
            "left_shoulder_y",
            "left_shoulder_z",
            "right_shoulder_x",
            "right_shoulder_y",
            "right_shoulder_z",
            "left_elbow_x",
            "left_elbow_y",
            "left_elbow_z",
            "right_elbow_x",
            "right_elbow_y",
            "right_elbow_z",
            "left_wrist_x",
            "left_wrist_y",
            "left_wrist_z",
            "right_wrist_x",
            "right_wrist_y",
            "right_wrist_z",
            "left_pinky_x",
            "left_pinky_y",
            "left_pinky_z",
            "right_pinky_x",
            "right_pinky_y",
            "right_pinky_z",
            "left_index_x",
            "left_index_y",
            "left_index_z",
            "right_index_x",
            "right_index_y",
            "right_index_z",
            "left_thumb_x",
            "left_thumb_y",
            "left_thumb_z",
            "right_thumb_x",
            "right_thumb_y",
            "right_thumb_z",
            "left_hip_x",
            "left_hip_y",
            "left_hip_z",
            "right_hip_x",
            "right_hip_y",
            "right_hip_z",
            "left_knee_x",
            "left_knee_y",
            "left_knee_z",
            "right_knee_x",
            "right_knee_y",
            "right_knee_z",
            "left_ankle_x",
            "left_ankle_y",
            "left_ankle_z",
            "right_ankle_x",
            "right_ankle_y",
            "right_ankle_z",
            "left_heel_x",
            "left_heel_y",
            "left_heel_z",
            "right_heel_x",
            "right_heel_y",
            "right_heel_z",
            "left_foot_index_x",
            "left_foot_index_y",
            "left_foot_index_z",
            "right_foot_index_x",
            "right_foot_index_y",
            "right_foot_index_z"
        ]
        res_point = []

        with mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while self.mw.cap.isOpened():
                success, image = self.mw.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                if results.pose_landmarks:
                    for index, landmarks in enumerate(results.pose_landmarks.landmark):
                        # print(index, landmarks.x, landmarks.y, landmarks.z)
                        res_point.append(landmarks.x)
                        res_point.append(landmarks.y)
                        res_point.append(landmarks.z)
                    shape1 = int(len(res_point) / len(keyXYZ))
                    res_point = np.array(res_point).reshape(shape1, len(keyXYZ))
                    pred = pose_knn.predict(res_point)
                    res_point = []
                    if pred == 0:
                        cv2.putText(image, "Fall", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
                    else:
                        cv2.putText(image, "Normal", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # Flip the image horizontally for a selfie-view display.
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (175, 65, 84), 6)
                # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                frameHeight = image.shape[0]
                frameWidth = image.shape[1]
                a = self.mw.ui.video.size()
                if a.width() / frameWidth < a.height() / frameHeight:
                    scaleFactor = a.width() / frameWidth
                else:
                    scaleFactor = 1.0 * a.height() / frameHeight

                timg = cv2.resize(image, (int(scaleFactor * image.shape[1]), int(scaleFactor * image.shape[0])))
                timg = cv2.cvtColor(timg, cv2.COLOR_BGR2RGB)
                limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], timg.shape[2] * timg.shape[1],
                                      QtGui.QImage.Format_RGB888)
                self.mw.ui.video.setPixmap(QtGui.QPixmap(limage))
                if self._isPause:
                    break


class myMainWindow(QMainWindow):
    signalImage = pyqtSignal(QImage)

    def __init__(self):
        super(myMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.image = None
        self.cap = None
        self.exercise_type = None
        self.track_type = 0
        self.func = 0
        # self.btn_ext.clicked.connect(self.exit)
        self.ui.btn_file.clicked.connect(self.openfile)
        self.ui.btn_camera.clicked.connect(self.opencam)
        self.ui.pull_up.clicked.connect(self.pullup)
        self.ui.sit_up.clicked.connect(self.situp)
        self.ui.push_up.clicked.connect(self.pushup)
        self.ui.squat_up.clicked.connect(self.squat)
        self.ui.walk_dan.clicked.connect(self.walk)
        self.ui.human_3d.clicked.connect(self.human3d)
        self.ui.human_2d.clicked.connect(self.human2d)
        self.ui.hand_3d.clicked.connect(self.hand3d)
        self.ui.hand_3d.setEnabled(False)
        self.ui.human_seg.clicked.connect(self.humanseg)
        self.ui.tabWidget.currentChanged[int].connect(self.function)
        self.setMinimumSize(700, 1100)

        self.ui.label_11.setText(
            '''<a style="font-family: Roman; color: #0000FF; font-size: 12pt;  text-decoration: none" href="https://github.com/sastpg"> Github</a>''')
        self.ui.label_11.setOpenExternalLinks(True)

    def openfile(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Select Video", "", "All Files(*)")
        self.cap = cv2.VideoCapture(filename)

        self.thread0 = ScoreThread(self, self.exercise_type)
        self.thread1 = TrackThread(self)
        self.thread2 = SafetyTread(self)
        self.thread3 = PlotThread(filename)
        self.thread4 = SegmentThread(self, None)
        self.thread5 = Plot2dThread(self, None)

        # self.btn_ok.clicked.connect(self.t.resume)
        # self.thread.scoreSignal.connect(self.Change)
        # self.t.sinOut.connect(self.updatalabel)
        if self.func == 0:
            if self.exercise_type is None:
                QMessageBox.warning(self, 'Warning', 'Exercise type is not chosen!')
            else:
                self.thread0.start()
                self.ui.btn_pause.clicked.connect(self.thread0.pause)
        elif self.func == 1:
            if self.track_type == 0:
                QMessageBox.warning(self, 'Warning', 'Track type is not chosen!')
            if self.track_type == 1:
                self.thread1.start()
                self.thread3.start()
            elif self.track_type == 2:
                self.thread5.start()
                self.ui.btn_pause.clicked.connect(self.thread5.pause)
            elif self.track_type == 4:
                self.thread4.start()
                self.ui.btn_pause.clicked.connect(self.thread4.pause)
        elif self.func == 2:
            self.thread2.start()
            self.ui.btn_pause.clicked.connect(self.thread2.pause)

    def opencam(self):
        self.cap = cv2.VideoCapture(0)
        self.thread0 = ScoreThread(self, self.exercise_type)
        self.thread1 = TrackThread(self)
        self.thread2 = SafetyTread(self)
        self.thread3 = PlotThread("None")
        self.thread4 = SegmentThread(self, None)
        self.thread5 = Plot2dThread(self, None)
        # self.t.scoreSignal.connect(self.Change)
        if self.func == 0:
            if self.exercise_type is None:
                QMessageBox.warning(self, 'Warning', 'Exercise type is not chosen!')
            else:
                self.thread0.start()
                self.ui.btn_pause.clicked.connect(self.thread0.pause)
        elif self.func == 1:
            if self.track_type == 0:
                QMessageBox.warning(self, 'Warning', 'Track type is not chosen!')
            if self.track_type == 1:
                self.thread1.start()
                self.thread3.start()
            elif self.track_type == 2:
                self.thread5.start()
                self.ui.btn_pause.clicked.connect(self.thread5.pause)
            elif self.track_type == 4:
                self.thread4.start()
                self.ui.btn_pause.clicked.connect(self.thread4.pause)
        elif self.func == 2:
            self.thread2.start()
            self.ui.btn_pause.clicked.connect(self.thread2.pause)

    def function(self, index):
        self.func = index

    def passImage(self, image):
        self.image = image
        self.signalImage.emit(self.image)

    def exit(self):
        self.close()

    def Change(self, msg):
        self.label.setText("Score : " + str(msg))

    def pullup(self):
        self.exercise_type = "pull-up"

    def situp(self):
        self.exercise_type = "sit-up"

    def pushup(self):
        self.exercise_type = "push-up"

    def walk(self):
        self.exercise_type = "walk"

    def squat(self):
        self.exercise_type = "squat"

    def human3d(self):
        self.track_type = 1

    def human2d(self):
        self.track_type = 2

    def hand3d(self):
        self.track_type = 3

    def humanseg(self):
        self.track_type = 4


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_gui = myMainWindow()
    my_gui.show()
    sys.exit(app.exec_())
