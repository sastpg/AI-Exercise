import numpy as np
from GestureScore.body_part_angle import BodyPartAngle
from GestureScore.utils import *


class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def push_up(self, counter, status, avg_score):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_left_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2

        standard = [45, 170]
        standard_sum = 2 * sum(standard)

        if status:
            if avg_arm_angle < 70:
                counter += 1
                status = False
            avg_score = 0
        else:
            if avg_arm_angle > 160:
                status = True
            left_arm_score = (1 - abs((self.angle_of_the_left_arm() - standard[0]) / standard_sum)) * 100
            right_arm_score = (1 - abs((self.angle_of_the_right_arm() - standard[0]) / standard_sum)) * 100
            left_leg_score = (1 - abs((self.angle_of_the_left_leg() - standard[1]) / standard_sum)) * 100
            right_leg_score = (1 - abs((self.angle_of_the_right_leg() - standard[1]) / standard_sum)) * 100
            avg_score = (left_arm_score + right_arm_score + left_leg_score + right_leg_score) / 4

        return [counter, status, avg_score]

    # def push_up_method_2():

    def pull_up(self, counter, status, avg_score):
        nose = detection_body_part(self.landmarks, "NOSE")
        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        avg_shoulder_y = (left_elbow[1] + right_elbow[1]) / 2

        standard = [30, 45]
        standard_sum = 2 * sum(standard)

        if status:
            if nose[1] > avg_shoulder_y:
                counter += 1
                status = False
            left_arm_score = (1 - abs((self.angle_of_the_left_arm() - standard[0]) / standard_sum)) * 100
            right_arm_score = (1 - abs((self.angle_of_the_right_arm() - standard[0]) / standard_sum)) * 100
            left_shoulder_score = (1 - abs((self.angle_of_the_left_shoulder() - standard[1]) / standard_sum)) * 100
            right_shoulder_score = (1 - abs((self.angle_of_the_right_shoulder() - standard[1]) / standard_sum)) * 100
            avg_score = (left_arm_score + right_arm_score + left_shoulder_score + right_shoulder_score) / 4
        else:
            if nose[1] < avg_shoulder_y:
                status = True
            avg_score = 0

        return [counter, status, avg_score]

    def squat(self, counter, status, avg_score):
        left_leg_angle = self.angle_of_the_right_leg()
        right_leg_angle = self.angle_of_the_left_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2

        standard = [45, 50]
        standard_sum = 2 * sum(standard)

        if status:
            if avg_leg_angle < 70:
                counter += 1
                status = False
            avg_score = 0
        else:
            if avg_leg_angle > 160:
                status = True
            left_leg_score = (1 - abs((self.angle_of_the_left_leg() - standard[0]) / standard_sum)) * 100
            right_leg_score = (1 - abs((self.angle_of_the_right_leg() - standard[0]) / standard_sum)) * 100
            abdomen_score = (1 - abs((self.angle_of_the_abdomen() - standard[1]) / standard_sum)) * 100
            avg_score = (left_leg_score + right_leg_score + abdomen_score) / 3

        return [counter, status, avg_score]

    def walk(self, counter, status):
        right_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")

        if status:
            if left_knee[0] > right_knee[0]:
                counter += 1
                status = False

        else:
            if left_knee[0] < right_knee[0]:
                counter += 1
                status = True

        return [counter, status]

    def sit_up(self, counter, status, avg_score):
        angle = self.angle_of_the_abdomen()

        standard = [45, 60]
        standard_sum = 2 * sum(standard)

        if status:
            if angle < 55:
                counter += 1
                status = False
            avg_score = 0
        else:
            if angle > 105:
                status = True
            abdomen_score = (1 - abs((self.angle_of_the_abdomen() - standard[0]) / standard_sum)) * 100
            left_leg_score = (1 - abs((self.angle_of_the_left_leg() - standard[1]) / standard_sum)) * 100
            right_leg_score = (1 - abs((self.angle_of_the_right_leg() - standard[1]) / standard_sum)) * 100
            avg_score = (abdomen_score + left_leg_score + right_leg_score) / 3

        return [counter, status, avg_score]

    def calculate_exercise(self, exercise_type, counter, status, avg_score):
        if exercise_type == "push-up":
            counter, status, avg_score = TypeOfExercise(self.landmarks).push_up(
                counter, status, avg_score)
        elif exercise_type == "pull-up":
            counter, status, avg_score = TypeOfExercise(self.landmarks).pull_up(
                counter, status, avg_score)
        elif exercise_type == "squat":
            counter, status, avg_score = TypeOfExercise(self.landmarks).squat(
                counter, status, avg_score)
        elif exercise_type == "walk":
            counter, status = TypeOfExercise(self.landmarks).walk(
                counter, status)
        elif exercise_type == "sit-up":
            counter, status, avg_score = TypeOfExercise(self.landmarks).sit_up(
                counter, status, avg_score)

        return [counter, status, avg_score]

    def score_table(self, exercise, counter, status, avg_score, isPause):
        score_table = cv2.imread("./images/score_table.png")
        cv2.putText(score_table, "Activity : " + exercise.replace("-", " "),
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2,
                    cv2.LINE_AA)
        cv2.putText(score_table, "Counter : " + str(counter), (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
        cv2.putText(score_table, "Status : " + str(status), (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
        if exercise == "push-up":
            cv2.putText(score_table, "Score : " + str(avg_score), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left arm : " + str(self.angle_of_the_left_arm()), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right arm : " + str(self.angle_of_the_right_arm()), (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left leg : " + str(self.angle_of_the_left_leg()), (10, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right leg : " + str(self.angle_of_the_right_leg()), (10, 570),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)

        elif exercise == "pull-up":
            cv2.putText(score_table, "Score : " + str(avg_score), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left arm : " + str(self.angle_of_the_left_arm()), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right arm : " + str(self.angle_of_the_right_arm()), (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left shoulder : " + str(self.angle_of_the_left_shoulder()), (10, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right shoulder : " + str(self.angle_of_the_right_shoulder()), (10, 570),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)

        elif exercise == "squat":
            cv2.putText(score_table, "Score : " + str(avg_score), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left leg : " + str(self.angle_of_the_left_leg()), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right leg : " + str(self.angle_of_the_right_leg()), (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of abdomen : " + str(self.angle_of_the_abdomen()), (10, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
        elif exercise == "walk":
            cv2.putText(score_table, "right_knee : " + str(detection_body_part(self.landmarks, "RIGHT_KNEE")), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "left_knee : " + str(detection_body_part(self.landmarks, "LEFT_KNEE")), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
        elif exercise == "sit-up":
            cv2.putText(score_table, "Score : " + str(avg_score), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left leg : " + str(self.angle_of_the_left_leg()), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right leg : " + str(self.angle_of_the_right_leg()), (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of abdomen : " + str(self.angle_of_the_abdomen()), (10, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)

        cv2.imshow("Score Table", score_table)

