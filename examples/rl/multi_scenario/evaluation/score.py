from evaluation.metric import Metric


class Score:
    def __init__(self):
        self._results = {
            "rules": [],
            "goal": [],
            "human_likeness": [],
            "time": [],
        }

    def add(self, metric):
        for component in self._results.keys():
            func = globals()[f"_{component}"]
            self._results[component].append(func(costs))

    def compute(self):


def _safety(self, metric):
    collisions  
    off_road
    wrong_way
    on_shoulder
    dist_to_obstacles


     
    "dist_to_obstacles": lambda _: _distance_to_obstacles,
    "jerk": lambda _: _jerk,
    "lane_center_offset": lambda _: _lane_center_offset,
    "on_shoulder": lambda _: _on_shoulder,
    "reached_goal": lambda _: _reached_goal(),
    "steering_rate": lambda _: _steering_rate(),
    "velocity_offset": lambda _: _velocity_offset,
    "yaw_rate": lambda _: _yaw_rate,

def _completion(metric):
    copleted = [agent["reached_goal"] for agent in metric["costs"].values()]
    completed = 1 if all(completed) else 0
    completed = sum(completed)
    mean_completed = metric["episodes"]
    return mean_completed


def _human_likeness(metric):
    ds = [ [agent["lane_center_offset"]
        for agent in costs.values()]
        print(ds,"ssssssssssssssssssss")
    return sum(ds)

    lane_center_offset
    steering_rate
    yaw_rate
    jerk
    off_route: bool


def _time(metric):
    metric[]steps


 
    off_route: bool

def _average(old_ave:float, newcurrent:float, ):