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

import logging

import rospy

# Note:  these ROS logging utils will require installing the SMARTS package with the "[ros]" extensions.


class LogToROSHandler(logging.Handler):
    """
     Logging Handler that converts python logging levels to rospy logger levels.
     Adapted from a solution found here:
         https://gist.github.com/ablakey/4f57dca4ea75ed29c49ff00edf622b38
     Referenced in a thread here:
         https://github.com/ros/ros_comm/issues/1384

    This handler can be added to any normal logging.Logger object using
        logger.addHandler(LogToROSHandler())
    """

    level_map = {
        logging.DEBUG: rospy.logdebug,
        logging.INFO: rospy.loginfo,
        logging.WARNING: rospy.logwarn,
        logging.ERROR: rospy.logerr,
        logging.CRITICAL: rospy.logfatal,
    }

    def emit(self, record):
        try:
            self.level_map[record.levelno]("%s: %s" % (record.name, record.msg))
        except KeyError:
            rospy.logerr(
                "unknown log level %s LOG: %s: %s"
                % (record.levelno, record.name, record.msg)
            )


def log_everything_to_ROS(level=None):
    """
    Calling this will add a LogToROSHandler handler object to the root logger.
    Any logger that propagates its messages to the root handler (most do by default)
    will also have its messages logged via the rospy logging topics.

    If level is passed, the the root logger level will be set to
    this for all non-ROS messages.

    NOTE:  In order to avoid an infinite recursion, the `propagate` property
    will be set to `False` on any existing loggers whose name starts with "ros".
    (All of the rospy loggers start with this string.)
    """
    root = logging.getLogger(None)
    for logger_name, logger in root.manager.loggerDict.items():
        if logger_name.startswith("ros"):
            logger.propagate = False
    ros_handler = LogToROSHandler()
    if level is not None:
        ros_handler.setLevel(level)
    root.addHandler(ros_handler)
