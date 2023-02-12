import random
import numpy as np
import cv2

from keras import utils, Sequential
from keras.layers import Dense, Flatten


class NeuronTeacher:

    def train_the_network(self, network_name: str) -> None:
        x_train, y_train, x_test, y_test = self.__get_sample()  # 1. Получить выборку
        self.__train(network_name, (x_train, y_train), (x_test, y_test))  # 2. Обучить по ней нейросеть

    def __get_sample(self) -> np.ndarray:
        # 1. Номера изображений в resources/saved_to...
        train_var_start = 0  # Включительно
        train_var_end = 3901  # Не включительно
        test_var_start = 3901  # Включительно
        test_var_end = 4001  # Не включительно

        # 2. Получить порядок изображений в выборке и y-составляющую выборки (0 или 1)
        numbers_arr, numbers_test_arr, y_train, y_test = self.__get_y_sample((train_var_start, train_var_end),
                                                                             (test_var_start, test_var_end))
        # 3. Получить по порядку изображений x-составляющую выборки (сами изображения)
        x_train, x_test = self.__get_x_sample((numbers_arr, numbers_test_arr))

        return x_train, y_train, x_test, y_test

    @staticmethod
    def __get_y_sample(train_dimension: tuple, test_dimension: tuple) -> tuple:
        train_var_start, train_var_end = train_dimension
        test_var_start, test_var_finish = test_dimension

        numbers_arr = []  # Массив с <img_number><1/0>. 1 - в камеру, 0 - в экран
        numbers_test_arr = []  # Массив с <img_number><1/0>. 1 - в камеру, 0 - в экран

        # Такой прием нужен для хранения в одном массиве изображений с одинаковыми номерами
        for i in range(train_var_start, train_var_end):
            numbers_arr.append(i * 10 + 1)
            numbers_arr.append(i * 10)

        for i in range(test_var_start, test_var_finish):
            numbers_test_arr.append(i * 10 + 1)
            numbers_test_arr.append(i * 10)

        random.shuffle(numbers_arr)
        random.shuffle(numbers_test_arr)

        y_train = np.array(numbers_arr)
        y_test = np.array(numbers_test_arr)
        y_train = y_train % 10  # Получили массив с 1 и 0 для тестовой и основной выборки
        y_test = y_test % 10

        return numbers_arr, numbers_test_arr, y_train, y_test

    @staticmethod
    def __get_x_sample(numbers_arrays: tuple) -> tuple:
        numbers_arr, numbers_test_arr = numbers_arrays

        img_arr = []
        img_test_arr = []
        for i in numbers_arr:
            if i % 10 == 1:
                img = cv2.imread(f'resources/saved_to_cam/img{i // 10}.jpg')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_arr.append(img)
            else:
                img = cv2.imread(f'resources/saved_to_screen/img{i // 10}.jpg')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_arr.append(img)

        x_train = np.array(img_arr)

        for i in numbers_test_arr:
            if i % 10 == 1:
                img = cv2.imread(f'resources/saved_to_cam/img{i // 10}.jpg')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_test_arr.append(img)
            else:
                img = cv2.imread(f'resources/saved_to_screen/img{i // 10}.jpg')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_test_arr.append(img)

        x_test = np.array(img_test_arr)

        return x_train, x_test

    @staticmethod
    def __train(network_name: str, train_sample: tuple, test_sample: tuple) -> None:
        x_train, y_train = train_sample
        x_test, y_test = test_sample

        x_train = x_train / 255  # Нормализация
        x_test = x_test / 255

        y_train_cat = utils.to_categorical(y_train, 2)  # Приведение выборки в вектор [0, 1] / [1, 0]
        y_test_cat = utils.to_categorical(y_test, 2)

        model = Sequential([
            Flatten(input_shape=(60, 145, 1)),
            Dense(50, activation='relu'),
            Dense(50, activation='relu'),
            Dense(50, activation='relu'),
            Dense(2, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

        model.evaluate(x_test, y_test_cat)
        model.save(f"resources/networks/{network_name}")
