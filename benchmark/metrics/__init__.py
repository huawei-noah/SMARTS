# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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

from abc import ABCMeta, abstractmethod


class MetricHandler(metaclass=ABCMeta):
    def __init__(self):
        self._logs_mapping = None

    @property
    def logs_mapping(self):
        raise NotImplementedError

    @abstractmethod
    def log_step(self, **kwargs):
        """ Called at each time step to log the step information """
        pass

    @abstractmethod
    def show_plots(self, **kwargs):
        """ Do visualization """
        pass

    @abstractmethod
    def write_to_csv(self, csv_dir):
        """ Write logs to csv files """
        pass

    @abstractmethod
    def read_logs(self, csv_dir):
        """ Read logs from local csv files"""
        pass

    @abstractmethod
    def compute(self, **kwargs):
        """ Analysis with given metrics """
        pass
