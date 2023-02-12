class DatasetParser:
    def __init__(self, result_file_name: str) -> None:
        self.result_file_name = result_file_name

    def parse(self, startswith: str):
        startswith = startswith.lower()
        f = open("resources/city_temperature.csv", "r")
        f2 = open(f"resources/{self.result_file_name}", "w")
        for line in f:
            if line.lower().startswith(startswith):
                line_list = line.split(",")
                celsius = self.__to_celsius(line_list[7][:-1])
                line_list.pop()
                year = line_list.pop()
                if celsius == -73:
                    continue
                line_list.append(str(celsius))
                line_list.append(year + "\n")
                f2.write(",".join(line_list[4:]))
        f.close()
        f2.close()

    def __to_celsius(self, fahrenheit_value: str):
        float_fahrenheit_value = float(fahrenheit_value)
        celsius_value = (float_fahrenheit_value - 32) / 1.8
        # celsium = celsium * 10
        # celsium = math.trunc(celsium)
        # return celsium / 10
        return round(celsius_value)
