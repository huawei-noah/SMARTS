import yaml
import json
import re

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u"tag:yaml.org,2002:float",
    re.compile(
        u"""^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list(u"-+0123456789."),
)

# to preserve exponential numbers
def load_yaml(path):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
        json_data = json.dumps(data)
    return yaml.load(json_data, Loader=loader)
