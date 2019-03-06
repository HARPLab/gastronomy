import rospy
from enum import Enum

from std_msgs.msg import Bool, Float64

class State(Enum):
  MOVING_ARM = 0
  FOLLOWING_TRAJECTORY = 1
 
food_acquired_topic = "/food_acquired" # std_msgs/Bool from PlayTapoTrajectory

def wait(state):
  with waitLogicDictionary[state]() as waitLogic:
    waitLogic.wait()
  return

class WaitLogic(object):
  def __enter__(self):
    return self
  def __exit__(self, exc_type, exc_value, traceback):
    # if your extension creates ros subscribers, be sure to overwrite this
    pass

  # this method subscribes to any relevant ros topics
  # blocks, and returns a new state when the robot should transition
  # from the current state to a new state.
  # This class should be extended by any logic implementation for each of the different states
  def wait(self):
    rospy.logerr("This method shouldn't be called from this base class, but should be implemented in the child class")

# by inheriting from this class, you get access to a self.topic_true attribute
# which will turn to True after the input topic_string publishes a True 
class TopicBasedWaitLogic(WaitLogic):
  def __init__(self, topic_string, topic_type):
    self.topic_string = topic_string
    self.topic_type = topic_type
  
  def __enter__(self):
    self.last_topic_value = None
    self.listen_for_topic = rospy.Subscriber(self.topic_string, self.topic_type, self.update_last_topic_value)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.listen_for_topic.unregister()

  def update_last_topic_value(self, message):
    self.last_topic_value = message.data

class DistanceBasedWaitLogic(TopicBasedWaitLogic):
  def __init__(self, distance_to_goal_topic = "/distance_to_target", epsilon=0.001):
    self.epsilon = epsilon
    # super nicely generates a self.last_topic_value attribute for us.
    super(DistanceBasedWaitLogic, self).__init__(distance_to_goal_topic, Float64)
  
  def wait(self):
    r = rospy.Rate(10) # 10Hz
    while (self.last_topic_value is None 
           or self.last_topic_value > self.epsilon):
      r.sleep()

class FollowTrajectoryWaitLogic(TopicBasedWaitLogic):
  def __init__(self):
    # super nicely generates a self.last_topic_value attribute for us.
    super(FollowTrajectoryWaitLogic, self).__init__(food_acquired_topic, Bool)

  def wait(self):
    rospy.logwarn("Picking up food")
    r = rospy.Rate(10) # 10Hz
    while self.last_topic_value is None or not self.last_topic_value:
      r.sleep()

# a dictionary that gives you the logic class constructor associated with each state
waitLogicDictionary = {
                     State.MOVING_ARM : DistanceBasedWaitLogic,
                     State.FOLLOWING_TRAJECTORY : FollowTrajectoryWaitLogic,
                   }
