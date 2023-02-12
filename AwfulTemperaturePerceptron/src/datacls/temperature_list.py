from dataclasses import dataclass


@dataclass
class TemperatureList:
    temperature_list: list[int]
    next_value: int

    def __gt__(self, other):
        return max(self.temperature_list) > max(other.temperature_list)

    def __add__(self, other):
        return self.next_value + other

    def __radd__(self, other):
        if other == 0:
            return self.next_value
        else:
            return self.__add__(other)

    def length(self):
        return len(self.temperature_list)

    def append(self, value: int):
        return self.temperature_list.append(value)
