import cv2
import numpy as np

from src.service.neuron_service import NeuronConsumer


class CameraService:

    def __init__(self, neuron_consumer: NeuronConsumer) -> None:
        self.neuron_consumer = neuron_consumer
        self.cap = cv2.VideoCapture(0)  # Захват вебки
        self.eye_cascade = cv2.CascadeClassifier('resources/haarcascade_eye.xml')  # Штука для распознавания глаз

    def start(self) -> None:
        while True:
            ret, img = self.cap.read()  # 1. Получить кадр с вебки
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 2. В ч/б

            eyes_img, eyes_coords = self.__get_eyes(gray_img)  # 3. Получить изображение глаз (145,60) и его координаты

            if eyes_img is not None:
                gaze_direction = self.neuron_consumer.guess_something(eyes_img)  # 4. Отдать его нейронке
                self.__draw_rectangle(img, eyes_coords,
                                      gaze_direction)  # 5. Отрисовать на основном изображении результат
                cv2.imshow("camera", img)  # 6. Показать основное изображение

            if cv2.waitKey(10) == 27:  # Выход по Esc
                break

    def __get_eyes(self, full_gray_image: np.ndarray) -> tuple:
        eyes = self.__detect_eyes(full_gray_image)  # Получить координаты глаз

        if (len(eyes) == 2) and abs(eyes[0][1] - eyes[1][1]) < 10:
            return self.__cut_original_image(full_gray_image, eyes)  # Вырезать их из основного изображения
        else:
            return None, None

    def __detect_eyes(self, full_gray_image: np.ndarray) -> np.ndarray:
        return self.eye_cascade.detectMultiScale(
            full_gray_image,  #
            scaleFactor=1.2,  # Ищем глаза в области с лицом
            minNeighbors=10,
            # minSize=(5, 5),
        )

    @staticmethod
    def __cut_original_image(full_image: np.ndarray, eyes: np.ndarray) -> tuple:
        min_y = min(eyes[0][1], eyes[1][1])
        max_h = max(eyes[0][3], eyes[1][3])
        min_x = min(eyes[0][0], eyes[1][0])
        max_x = max(eyes[0][0], eyes[1][0])
        max_w = max(eyes[0][2], eyes[1][2])

        cut_img = full_image[min_y:min_y + max_h, min_x:max_x + max_w]  # Подготовка изображения для нейросети
        return cv2.resize(cut_img, (145, 60)), (min_y, max_h, min_x, max_x, max_w)

    @staticmethod
    def __draw_rectangle(full_image: np.ndarray, eyes_coords: tuple, gaze_direction: str) -> None:
        min_y, max_h, min_x, max_x, max_w = eyes_coords
        rectangle_x = min_x
        rectangle_y = min_y
        rectangle_width = max_x + max_w
        rectangle_height = min_y + max_h

        if gaze_direction == "to_cam":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(full_image, gaze_direction, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA, False)
        cv2.rectangle(full_image, (rectangle_x, rectangle_y), (rectangle_width, rectangle_height), color, 2)
