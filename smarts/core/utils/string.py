import math


def truncate(str_, length, separator="..."):
    if len(str_) <= length:
        return str_

    start = math.ceil((length - len(separator)) / 2)
    end = math.floor((length - len(separator)) / 2)
    return f"{str_[:start]}{separator}{str_[len(str_) - end:]}"
