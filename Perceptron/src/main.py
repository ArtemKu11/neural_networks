from service import CameraService, NeuronConsumer, NeuronTeacher


if __name__ == '__main__':
    network_name = "perceptron1.h5"

    # NeuronTeacher().train_the_network(network_name)

    CameraService(NeuronConsumer(network_name)).start()
