def pretty_dict(d, indent=0):
    """Pretty the output format of a dictionary.

    Parameters
    ----------
    d
        dict, the input dictionary instance.
    indent
        int, indent level, non-negative.
    Returns
    -------
    res
        str, the output string
    """

    res = ""
    for k, v in d.items():
        res += "\t" * indent + str(k)
        if isinstance(v, dict):
            res += "\n" + pretty_dict(v, indent + 1)
        else:
            res += ": " + str(v) + "\n"
    return res
