.. _faq:

Frequently asked questions
==========================

This is a list of frequently asked questions.  Feel free to suggest new entries!

1. What do I need for rendering images in SMARTS?
    Need ``x11`` or a dummy of that rendering system on Linux. We recommend a decent GPU if you are training an Agent that makes use of observation's depending on the rendering pipeline such as TopDownRGB, OGM, or DriveableGridMap.

2. Where can I find debug logs?
    In most cases SMARTS debug logs are located at ``~/.smarts``. These can be helpful to diagnose problems.

3. Exception: Could not open display. (Ubuntu)
    This may be due to needing a display to render with a ``GL`` renderer backend setting. Try the following instructions to solve it.

    .. code-block:: bash

        # Set DISPLAY, can be as needed
        $ echo export DISPLAY=":1" >> ~/.bashrc
        $ source ~/.bashrc

        # Do once: Install x11 dummy which allows creating a fake display
        $ sudo apt-get install -y xserver-xorg-video-dummy x11-apps
        # Potentially the following if you need software rendering:
        # sudo apt-get install -y mesa-utils

        # Do once: set xorg server
        $ sudo wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf

        # Do as needed:
        $ sudo /usr/bin/Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ./xdummy.log -config /etc/X11/xorg.conf $DISPLAY &


    Note that ``mesa-utils`` installs ``llvm``, which is one option out of several that emulate ``OpenGL`` using software. ``llvm`` is not needed if a GPU is available.

4. Custom rendering and Obfuscation maps show completely blank. (Ubuntu)
    This is due to needing ``OpenGL`` to render using scripts. If you have a GPU make sure ``OpenGL`` is installed and the GPU has the necessary drivers for rendering. 
    
    See the previous question if you need software rendering.

5. The simulation keeps crashing on connection in ``SumoTrafficSimulation``. How do I fix this?
    This is likely due to using large scale parallelization. You will want to use the centralized management server. See :ref:`centralized_traci_management`.
