class CustomDict(dict):
    def __init__(self, **kwargs):
        mapping = dict(**kwargs)
        mapping = self._ensure_types(mapping)
        super().__init__(mapping)

    def __setitem__(self, key, value):
        if key not in self.keys():
            raise AttributeError(f"Cannot add new key {key}.")

        res = self._ensure_types({key: value})
        return super().__setitem__(key, res[key])

    def __add__(self, other: "CustomDict"):
        for key, value in other.items():
            self[key] += value
        return self

    def __sub__(self, other: "CustomDict"):
        for key, value in other.items():
            self[key] -= value
        return self

    def __mul__(self, other: "CustomDict"):
        for key, value in other.items():
            self[key] *= value
        return self

    def __truediv__(self, other: "CustomDict"):
        for key, value in other.items():
            self[key] /= value
        return self

    def __delitem__(self, key):
        raise AttributeError(f"Cannot delete individual keys after object instantiation.")

    def update(self, **kwargs):
        new = dict(**kwargs)
        diff = set(new.keys()).difference(set(self.keys()))
        if diff:
            raise AttributeError(f"Cannot add new keys {diff}.")

        res = self._ensure_types(new)
        super().update(res)

    def _ensure_types(self, mapping):
        result = {}
        for key, value in mapping.items():
            if not isinstance(key, str):
                message = f"Expected key to be of type `str`, got key {key} of type {type(key)}."
                raise TypeError(message)

            if isinstance(value, float):
                pass
            elif isinstance(value, int):
                value = float(value)
            else:
                raise TypeError(
                    f"Expected value to be of type `int` or `float`, got key {key} with value of type {type(value)}."
                )

            result[key] = value

        return result
