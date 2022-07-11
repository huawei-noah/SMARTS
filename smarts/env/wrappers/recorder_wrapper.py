import os

from moviepy.editor import *
import gym
import gym.envs

from smarts.env.wrappers.gif_recorder import GifRecorder


class RecorderWrapper(gym.Wrapper):
    def __init__(self, dir, env):

        try:
            os.mkdir("videos")
        except:
            pass

        super().__init__(env)
        # assert "rgb_array" in env.metadata.get("render_modes", [])
        self.dir = "videos/" + dir
        self.gif_recorder = None
        self.recording = False
        self.current_frame = -1

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        if self.recording == False:
            self.start_recording()

        return observations

    def start_recording(self):
        if self.gif_recorder is None:
            self.gif_recorder = GifRecorder(self.dir, self.env)
        self.gif_recorder.capture_frame(self.next_frame_id())
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        if self.recording == True:
            self.gif_recorder.capture_frame(self.next_frame_id())

        return observations, rewards, dones, infos

    def next_frame_id(self):
        self.current_frame += 1
        return self.current_frame

    def close(self):
        self.gif_recorder.close_recorder()

    def __del__(self):
        self.gif_recorder.close_recorder()

    def close_recorder(self):
        self.gif_recorder.generate_gif()
        self.gif_recorder.close_recorder()
