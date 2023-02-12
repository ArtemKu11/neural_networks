import numpy as np
from keras.models import load_model


class NeuronConsumer:
    def __init__(self, network_name: str) -> None:
        self.model = load_model(f"resources/networks/{network_name}")

    def guess_something(self, temperatures: tuple) -> int:

        neuron_temperatures = np.array(temperatures)
        neuron_temperatures = (neuron_temperatures + 29) / 60
        neuron_temperatures = np.expand_dims(neuron_temperatures, axis=0)
        neuron_temperatures = np.expand_dims(neuron_temperatures, axis=2)

        prediction = self.model.predict(neuron_temperatures, verbose=0)
        max_value = np.argmax(prediction[0])

        return temperatures[len(temperatures) - 1] + max_value - 1
