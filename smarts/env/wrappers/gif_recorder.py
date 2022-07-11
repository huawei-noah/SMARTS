import os

from moviepy.editor import *
import gym.envs
import shutil


class GifRecorder:
    def __init__(self, dir, env):
        self.dir = dir
        self.env = env

        try:
            os.mkdir(self.dir)
        except:
            pass

        self.dir_name = None
        if "/" not in dir:
            self.dir_name = dir
        else:
            last_index = dir.rindex("/")
            self.dir_name = dir[-(len(dir) - last_index - 1) :]

    def capture_frame(self, step_num):
        image = self.env.render(mode="rgb_array")

        with ImageClip(image) as image_clip:
            image_clip.save_frame(f"{self.dir}/{self.dir_name}_{step_num}.jpeg")

    def generate_gif(self):
        with ImageSequenceClip(self.dir, fps=10) as clip:
            clip.write_gif(f"videos/{self.dir_name}.gif")
        clip.close()

    def close_recorder(self):
        try:
            shutil.rmtree(self.dir)
        except:
            pass
