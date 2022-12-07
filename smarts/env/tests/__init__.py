from ray import tune


class Trainable(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.score = 0

    def step(self):  # This is called iteratively.
        self.score += 1
        return {"score": self.score}
