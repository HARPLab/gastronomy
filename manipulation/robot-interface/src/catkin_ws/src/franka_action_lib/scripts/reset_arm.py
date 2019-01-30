from frankapy import FrankaArm

if __name__ == '__main__':
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()