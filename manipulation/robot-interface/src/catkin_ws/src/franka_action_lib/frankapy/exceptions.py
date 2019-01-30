class FrankaArmCommException(Exception):
    """ Communication failure. Usually occurs due to timeouts.
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

class FrankaArmCollisionException(Exception):
    """ Failure of control, typically due to a kinematically unreachable pose.
    """

    def __init__(self, cmd, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
        self.cmd = cmd

    def __str__(self):
        return "Command resulted in a collision! Got params {}".format(self.cmd)