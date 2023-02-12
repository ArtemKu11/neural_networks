import random
from statistics import mean

import numpy as np

from keras import utils, Sequential
from keras.layers import Dense, Flatten

from src.datacls.temperature_list import TemperatureList


class NeuronTeacher:

    def __init__(self) -> None:
        self.dataset_name = ""
        self.max_temperature = None
        self.min_temperature = None
        self.avg_temperature = None

    def train_the_network(self, network_name: str, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        x_train, y_train, x_test, y_test = self.__get_sample()  # 1. Получить выборку
        self.__resolve_max_and_min((x_train, y_train))
        self.__train(network_name, (x_train, y_train), (x_test, y_test))  # 2. Обучить по ней нейросеть

    def __get_sample(self) -> tuple:
        sample = self.__get_all_sample()
        return self.__split_sample(sample)

    def __get_all_sample(self) -> list[TemperatureList]:
        f = open(f"resources/{self.dataset_name}", "r")
        temperature_list1 = TemperatureList([], None)
        temperature_list2 = TemperatureList([], None)
        temperature_list3 = TemperatureList([], None)
        temperature_list4 = TemperatureList([], None)
        temperature_list5 = TemperatureList([], None)
        temperature_list6 = TemperatureList([], None)

        sample = []

        for index, line in enumerate(f):
            line_list = line.split(",")
            temperature = int(line_list[2])

            if temperature_list1.length() == 3:
                temperature_list1.next_value = temperature - temperature_list1.temperature_list[2]
                sample.append(temperature_list1)
                temperature_list1 = TemperatureList([], None)
            temperature_list1.append(temperature)

            if index > 0:
                if temperature_list2.length() == 3:
                    temperature_list2.next_value = temperature - temperature_list2.temperature_list[2]
                    sample.append(temperature_list2)
                    temperature_list2 = TemperatureList([], None)
                temperature_list2.append(temperature)

            if index > 1:
                if temperature_list3.length() == 3:
                    temperature_list3.next_value = temperature - temperature_list3.temperature_list[2]
                    sample.append(temperature_list3)
                    temperature_list3 = TemperatureList([], None)
                temperature_list3.append(temperature)

            # if index > 2:
            #     if temperature_list4.length() == 5:
            #         temperature_list4.next_value = temperature - temperature_list4.temperature_list[4]
            #         sample.append(temperature_list4)
            #         temperature_list4 = TemperatureList([], None)
            #     temperature_list4.append(temperature)
            #
            # if index > 3:
            #     if temperature_list5.length() == 6:
            #         temperature_list5.next_value = temperature - temperature_list5.temperature_list[2]
            #         sample.append(temperature_list5)
            #         temperature_list5 = TemperatureList([], None)
            #     temperature_list5.append(temperature)
            #
            # if index > 4:
            #     if temperature_list6.length() == 6:
            #         temperature_list6.next_value = temperature - temperature_list6.temperature_list[2]
            #         sample.append(temperature_list6)
            #         temperature_list6 = TemperatureList([], None)
            #     temperature_list6.append(temperature)

        f.close()
        return sample

    def __split_sample(self, sample: tuple) -> tuple:
        random.shuffle(sample)
        test_sample_length = round(len(sample) * 0.1)
        test_sample = sample[0:test_sample_length]
        train_sample = sample[test_sample_length:]
        x_train, y_train = self.__get_as_np_array(train_sample)
        x_test, y_test = self.__get_as_np_array(test_sample)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def __get_as_np_array(sample: list[TemperatureList]) -> tuple:
        x_sample = []
        y_sample = []
        for temperature_list in sample:
            if abs(temperature_list.next_value) > 1:
                continue
            x_sample.append(temperature_list.temperature_list)
            y_sample.append(temperature_list.next_value)
        return np.array(x_sample), np.array(y_sample)

    def __resolve_max_and_min(self, sample: tuple):
        x_train, x_test = sample
        self.min_temperature = min(x_train.min(), x_test.min())
        self.max_temperature = max(x_train.max(), x_test.max())
        print(f"Макс. температура: {self.max_temperature}")
        print(f"Мин. температура: {self.min_temperature}")

    def __train(self, network_name: str, train_sample: tuple, test_sample: tuple) -> None:
        x_train, y_train = train_sample
        x_test, y_test = test_sample

        total = self.max_temperature + abs(self.min_temperature)

        x_train = (x_train + abs(self.min_temperature)) / total  # Нормализация
        x_test = (x_test + abs(self.min_temperature)) / total

        y_train = y_train + 1
        y_test = y_test + 1

        y_train_cat = utils.to_categorical(y_train, 3)  # Приведение выборки в вектор [0, 1] / [1, 0]
        y_test_cat = utils.to_categorical(y_test, 3)


        model = Sequential([
            Flatten(input_shape=(3,)),
            Dense(50, activation='relu'),
            Dense(10, activation='relu'),
            Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train_cat, batch_size=32, epochs=20, validation_split=0.2)

        model.evaluate(x_test, y_test_cat)
        model.save(f"resources/networks/{network_name}")
