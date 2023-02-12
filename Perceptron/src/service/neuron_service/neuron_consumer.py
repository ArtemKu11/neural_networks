import numpy as np
from keras.models import load_model


class NeuronConsumer:
    def __init__(self, network_name: str) -> None:
        self.model = load_model(f"resources/networks/{network_name}")

    def guess_something(self, gray_eye_image: np.ndarray) -> str:

        to_neuron_image = gray_eye_image / 255

        to_neuron_image = np.expand_dims(to_neuron_image, axis=0)
        to_neuron_image = np.expand_dims(to_neuron_image, axis=3)

        prediction = self.model.predict(to_neuron_image, verbose=0)

        first_digit = prediction[0, 0]
        second_digit = prediction[0, 1]

        if first_digit > second_digit:
            return 'to_screen'
        else:
            return 'to_cam'
