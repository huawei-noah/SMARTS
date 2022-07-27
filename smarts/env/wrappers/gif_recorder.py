# MIT License
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import os
import sys

try:
    from moviepy.editor import ImageClip
    from moviepy.editor import ImageSequenceClip
except (ImportError, ModuleNotFoundError):
    print(sys.exc_info())
    print(
        "You may not have installed the [gym] dependencies required to capture the video. Install them first using the command `pip install -e .[gym]` at the source directory."
    )
import shutil
import time
from pathlib import Path


class GifRecorder:
    """
    Use images(rgb_array) to create a gif file.
    """

    def __init__(self, dir, env):
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        self.dir = dir + "_" + timestamp_str
        self.env = env

        try:
            os.mkdir(self.dir)
        except:
            pass

        self._dir_name = str(Path(dir).name)

    def capture_frame(self, step_num, image):
        """
        Create image according to the rgb_array and store it with step number in the destinated folder
        """
        with ImageClip(image) as image_clip:
            image_clip.save_frame(f"{self.dir}/{self._dir_name}_{step_num}.jpeg")

    def generate_gif(self):
        """
        Use the images in the same folder to create a gif file.
        """
        with ImageSequenceClip(self.dir, fps=10) as clip:
            clip.write_gif(f"videos/{self._dir_name}.gif")
        clip.close()

    def close_recorder(self):
        """
        close the recorder by deleting the image folder.
        """
        try:
            shutil.rmtree(self.dir)
        except:
            pass
