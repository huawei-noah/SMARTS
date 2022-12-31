.. _faq:

Frequently asked questions
==========================

This is a list of frequently asked questions.  Feel free to suggest new entries!

1. What do I need for rendering images in SMARTS?
    Need ``x11`` or a dummy of that rendering system on Linux. We recommend a decent GPU if you are training an Agent that makes use of observation's depending on the rendering pipeline such as TopDownRGB, OGM, or DriveableGridMap.

2. Where can I find debug logs?
    In most cases SMARTS debug logs are located at ``~/.smarts``. These can be helpful to diagnose problems.

3. Exception: Could not open window.
    This may be due to some old dependencies of ``Panda3D``. Try the following instructions to solve it.

    .. code-block:: bash

        # Set DISPLAY 
        $ vim ~/.bashrc
        $ export DISPLAY=":1"
        $ source ~/.bashrc

        # Set xorg server
        $ sudo wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf
        $ sudo /usr/bin/Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ./xdummy.log -config /etc/X11/xorg.conf $DISPLAY & 0
