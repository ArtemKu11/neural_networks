from parsers import DatasetParser
from service import NeuronTeacher, NeuronConsumer

if __name__ == '__main__':
    network_name = "test_network.h5"

    # parser = DatasetParser(result_file_name="moscow_temperature.csv")
    # parser.parse(startswith="Europe,Russia,,Moscow")

    # NeuronTeacher().train_the_network(network_name, "moscow_temperature.csv")

    print(NeuronConsumer(network_name).guess_something((0, 1, 0)))
