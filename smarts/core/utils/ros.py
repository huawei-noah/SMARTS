import logging
import rospy


# Note:  use of these utilities may require installing the SMARTS package with the "[ros]" extentions.


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
