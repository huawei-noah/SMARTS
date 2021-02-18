# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import matplotlib.pyplot as plt
import numpy as np


def radar_plots(values, labels, features, title):
    plt.style.use("ggplot")

    N = len(features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # close the radar plots
    values = np.concatenate((values.T, [values.T[0]])).T
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    for i, label in enumerate(labels):
        ax.plot(angles, values[i], "o-", linewidth=2, label=label)
        # fill color
        ax.fill(angles, values[i], alpha=0.25, label=label)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, features)
    min_v = np.min(values)
    max_v = np.max(values)
    ax.set_ylim(min_v - 0.2, max_v + 0.2)
    plt.title(title)
    ax.grid(True)
    plt.legend(labels, loc="best", bbox_to_anchor=(0.5, 0.0, -0.54, 1.15))
    plt.show()


if __name__ == "__main":
    radar_plots(
        values=np.random.randn(5, 5),
        labels=["PPO", "DQN", "MADDPG", "A2C", "DPG"],
        features=[
            "Safety",
            "Agility",
            "Stability",
            "Control Diversity",
            "Cut-In Ratio",
        ],
        title="Behavior Analysis",
    )
