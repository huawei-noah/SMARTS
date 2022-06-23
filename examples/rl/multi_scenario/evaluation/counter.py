import functools


@functools.total_ordering
class Counter:
    def __init__(self, initial=0):
        self.value = initial

    def __int__(self):
        return int(self.value)

    def __eq__(self, other):
        return int(self) == other

    def __ne__(self, other):
        return int(self) != other

    def __lt__(self, other):
        return int(self) < other

    def __add__(self, other):
        return int(self) + other

    def increment(self, amount=1):
        self.value += amount
