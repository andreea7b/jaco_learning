from prpy.base.manipulator import Manipulator


class ArchieManipulator(Manipulator):
    def __init__(self):
        Manipulator.__init__(self)
        self.controller = self.GetRobot().GetController()
