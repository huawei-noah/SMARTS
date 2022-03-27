def create_linear_schedule(start_val, end_val, num_steps):
    start_val = float(start_val)
    end_val = float(end_val)
    num_steps = float(num_steps)
    # assuming steps start at 0
    def f(step):
        step = min(step, num_steps - 1)
        return start_val + step * (end_val - start_val) / (num_steps - 1)

    return f


def linear_schedule(step, start_val, end_val, num_steps):
    start_val = float(start_val)
    end_val = float(end_val)
    num_steps = float(num_steps)
    step = min(step, num_steps - 1)
    return_val = start_val + step * (end_val - start_val) / (num_steps - 1)
    return_val = max(return_val, min(start_val, end_val))
    return_val = min(return_val, max(start_val, end_val))
    return return_val
