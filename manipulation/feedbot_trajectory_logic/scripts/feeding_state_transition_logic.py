import rospy
from enum import Enum

from std_msgs.msg import Bool, Float64
from spoon_perception.srv import ObjectSpoon, ObjectSpoonResponse
from geometry_msgs.msg import PointStamped

class State(Enum):
  MOVE_TO_PLATE = 1
  PICK_UP_FOOD = 2
  PREPARE_FOR_MOUTH = 3
  MOVE_TO_MOUTH = 4
  WAIT_IN_MOUTH = 5
  PREPARE_FOR_PLATE = 6
  WAIT_FOR_SPOON_CALIBRATION = 10

distance_to_goal_topic = "/distance_to_target" # std_msgs/Float64
food_acquired_topic = "/food_acquired" # std_msgs/Bool from PlayTapoTrajectory
object_in_spoon_service_name = "/detect_object_spoon" # service to detect whether there is an object in spoon

hist_corr_threshold = 0.5

class TransitionLogic(object):
  def __enter__(self):
    return self
  def __exit__(self, exc_type, exc_value, traceback):
    # if your extension creates ros subscribers, be sure to overwrite this
    pass

  # this method subscribes to any relevant ros topics
  # blocks, and returns a new state when the robot should transition
  # from the current state to a new state.
  # This class should be extended by any logic implementation for each of the different states
  def wait_and_return_next_state(self):
    rospy.logerr("This method shouldn't be called from this base class, but should be implemented in the child class")

# by inheriting from this class, you get access to a self.topic_true attribute
# which will turn to True after the input topic_string publishes a True 
class TopicBasedTransitionLogic(TransitionLogic):
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

class DistanceBasedTransitionLogic(TopicBasedTransitionLogic):
  def __init__(self, epsilon, next_state):
    self.epsilon = epsilon
    self.next_state = next_state
    super(DistanceBasedTransitionLogic, self).__init__(distance_to_goal_topic, Float64)
  
  def wait_and_return_next_state(self):
    rospy.logwarn("within class " + self.__class__.__name__)
    start_time = rospy.get_time();
    r = rospy.Rate(10) # 10Hz
    while (self.last_topic_value is None 
           or self.last_topic_value > self.epsilon):
      r.sleep()
    return self.next_state


class PickUpStateTransitionLogic(TopicBasedTransitionLogic):
  def __init__(self):
    # super nicely generates a self.topic_true attribute for us.
    super(PickUpStateTransitionLogic, self).__init__(food_acquired_topic, Bool)
    # what we go to next depends on whether there is food in the spoon
    # however, if we're simulating the spoon we make a dummy call to service
    if rospy.get_param('~simulate_spoon'):
      self._check_spoon = lambda:ObjectSpoonResponse(0,"dummy")
    else:
      self._check_spoon = rospy.ServiceProxy(object_in_spoon_service_name, ObjectSpoon)

  def wait_and_return_next_state(self):
    rospy.logwarn("Picking up food")
    r = rospy.Rate(10) # 10Hz
    while self.last_topic_value is None or not self.last_topic_value:
      r.sleep()
    check_spoon_response = self._check_spoon()
    if check_spoon_response.histCorr < hist_corr_threshold:
      if rospy.get_param('~follow_mouth'):
        rospy.wait_for_messge(rospy.get_param("~mouth_point_topic"), PointStamped)
      return State.PREPARE_FOR_MOUTH
    return State.PICK_UP_FOOD

class WaitInMouthStateTransitionLogic(TransitionLogic):
  def wait_and_return_next_state(self):
    rospy.sleep(4)
    return State.PREPARE_FOR_PLATE

class SpoonCalibrationStateTransitionLogic(TransitionLogic):
  def wait_and_return_next_state(self):
    rospy.logwarn("Waiting for %s service to come up" % object_in_spoon_service_name)
    if not rospy.get_param('~simulate_spoon'):
      rospy.wait_for_service(object_in_spoon_service_name)
    return State.MOVE_TO_PLATE 

class MoveToPlateStateTransitionLogic(DistanceBasedTransitionLogic):
  def __init__(self):
    # super nicely generates a self.topic_true attribute for us.
    super(MoveToPlateStateTransitionLogic, self).__init__(0.02, State.PICK_UP_FOOD)

class PrepareForMouthStateTransitionLogic(DistanceBasedTransitionLogic):
  def __init__(self):
    # super nicely generates a self.topic_true attribute for us.
    super(PrepareForMouthStateTransitionLogic, self).__init__(0.02, State.MOVE_TO_MOUTH)
 
class MoveToMouthStateTransitionLogic(DistanceBasedTransitionLogic):
  def __init__(self):
    # super nicely generates a self.topic_true attribute for us.
    super(MoveToMouthStateTransitionLogic, self).__init__(0.02, State.WAIT_IN_MOUTH)

  #def wait_and_return_next_state(self):
    #if rospy.get_param('~just_follow_mouth', False):
    #  while(not rospy.is_shutdown()):
    #    rospy.sleep(10)
  #  return super(MoveToMouthStateTransitionLogic, self).wait_and_return_next_state(distance_to_goal_topic, Float64)

class PrepareForPlateStateTransitionLogic(DistanceBasedTransitionLogic):
  def __init__(self):
    # super nicely generates a self.topic_true attribute for us.
    super(PrepareForPlateStateTransitionLogic, self).__init__(0.02, State.MOVE_TO_PLATE)

# a dictionary that gives you the logic class constructor associated with each state
transitionLogicDictionary = {
                     State.MOVE_TO_PLATE   : MoveToPlateStateTransitionLogic,
                     State.PICK_UP_FOOD    : PickUpStateTransitionLogic,
                     State.PREPARE_FOR_MOUTH: PrepareForMouthStateTransitionLogic,
                     State.MOVE_TO_MOUTH: MoveToMouthStateTransitionLogic,
                     State.WAIT_IN_MOUTH: WaitInMouthStateTransitionLogic,
                     State.PREPARE_FOR_PLATE: PrepareForPlateStateTransitionLogic,
                     State.WAIT_FOR_SPOON_CALIBRATION : SpoonCalibrationStateTransitionLogic,
                   }
