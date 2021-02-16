#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String, Int32
import json
from time import sleep
from ipdb import set_trace
from threading import Thread, Lock
import sys,signal

print_status = False

class Sensor_Aggregator():
    def __init__(self, robot, human_input, num_tables):
        self.robot = robot
        self.human_input = human_input
        self.num_tables = num_tables
        self.kitchen = 3
        self.interrupted = False
        self.pub_action = rospy.Publisher('ActionStatus', String, queue_size=10)
        if not self.robot:
            self.executeAction = rospy.Subscriber('CobotExecuteAction4Table', String, self.action_execution)
            # self.table_service_sub = rospy.Subscriber("table_service", Int32, self.hand_raise)
            rospy.Subscriber("ActionStatus", String, self.action_status)
        else:
            rospy.Subscriber("ActionStatus", String, self.action_status)
            # self.table_service_sub = rospy.Subscriber("table_service", Int32, self.hand_raise)

        self.mutex = Lock() 
        self.success = False
        self.hand_raises = {}
        self.unknown_tables_state = []
        self.num_action_executions = 0
    
    def hand_raise(self, data):
        try:
            self.mutex.acquire()
            table = data.data
            if table == -1:
                self.hand_raises = {}
            else:
                found = False
                for t in self.hand_raises:
                    if t == table:
                        found = True
                if not found:
                    self.hand_raises[table] = 1
                    if not self.robot:
                        if table not in self.unknown_tables_state:
                            self.unknown_tables_state.append(table)  
                else:
                    self.hand_raises[table] += 1
                    if self.hand_raises[table] == 10:
                        if table not in self.unknown_tables_state:
                            self.unknown_tables_state.append(table)                   

        finally:
            self.mutex.release()

    def action_execution(self, data):
        # print (data.data)
        action = json.loads(data.data)
        print ("---- action name: ", action["name"], " ----")
        self.action = action
        # set_trace()
        self.num_action_executions += 1
        sleep(1)
        action["status"] = 0
        if len(self.hand_raises) == 0:
            if not action['req_ack']:
                action["status"] = 0 ## action is done
            else:
                # action["status"] = 2 ## action requires acknowledgement
                # while (action['id'] != 4 and action["table"] not in self.hand_raises) or (action["id"] == 4 and self.kitchen not in self.hand_raises):
                #     print ("waiting for action to be complete")
                #     sleep(1)
                action["status"] = 0
                
        else:
            action["status"] = 1 ## action incomplete
            # self.x = int(input("x: "))
            # self.y = int(input("y: "))
            # self.interrupted = True
        
        self.pub_action.publish(json.dumps(action))
        # self.success = True

    def action_status (self, data):
        self.interrupted = False
        action = json.loads(data.data)
        if (action["status"] == 0):
            self.success = True
        elif (action["status"] == 1):
            self.x = int(input("x: "))
            self.y = int(input("y: "))
            self.interrupted = True
            self.success = True
            print ("ACTION EXECUTION INTERRUPTED")
        elif (action["status"] == 2):
            while (action['id'] != 4 and action["table"] not in self.hand_raises) or (action["id"] == 4 and self.kitchen not in self.hand_raises):
                print ("waiting for action to be complete")
                print (self.hand_raises)
                sleep(0.1)
            action["status"] = 0
            self.hand_raises.clear()
            self.pub_action.publish(json.dumps(action))
            self.success = True

    def set_cur_state(self, data):
        self.cur_state_data = json.loads(data.data)
        self.cur_table_id = self.cur_state_data['table']
        # print (self.interrupted, self.human)

    def publish(self):
        self.pub_states = []
        for table_id in range(self.num_tables):
            self.pub_states.append(rospy.Publisher('StateDataTable' + str(table_id), String, queue_size=50))
            rospy.Subscriber('CurrentStateTable' + str(table_id), String, self.set_cur_state)
        
        rospy.init_node('talker', anonymous=True)

        rate = rospy.Rate(10) # 10hz        

        while not rospy.is_shutdown():
            if self.success:
                # self.cur_table_id = input("Table id: ")
                self.mutex.acquire()
                count = 0
                while count < (self.num_tables):
                    self.cur_state_data = None
                    while self.cur_state_data is None:
                        sleep(0.1)

                    if print_status:
                        print (self.cur_state_data)
                        print (self.action)
                    hand_raise = self.cur_state_data["hand_raise"]
                    self.x = self.cur_state_data["x"]
                    self.y = self.cur_state_data["y"]
                    temp = self.human_input
                    if print_status:
                        if self.num_action_executions == 1: ############# hack
                            print ("---- Follow the model ----")
                            self.human_input = False

                    if self.interrupted or (self.human_input and hand_raise == 1) or (self.action['type'] == 2 and self.action['table'] == self.cur_state_data["table"]):
                        # set_trace()
                        if self.human_input:
                            if self.action['type'] != 2:
                                if print_status and self.num_action_executions == 2: ############# hack
                                    # print ("---- Do not follow the model ----")
                                    follow_model = False
                                else:
                                    get_follow_model = input("please enter y if you would like to follow the model and n otherwise (y or n):")
                                    follow_model = (get_follow_model == "y")                                
                            else:
                                follow_model = False
                        else:
                            follow_model = not(i in self.unknown_tables_state)

                        if not follow_model:
                            if self.action['type'] == 2:
                                print ("---- Human waiter questions ----")
                            else:
                                print ("---- Do not follow the model ----")
                            if print_status:   
                                print("Table " + str(self.cur_state_data["table"]))
                            obs = self.cur_state_data
                            obs["explicit"] = 0

                            obs["x"] = self.x; obs["y"] = self.y
                            food = 0
                            water = 0
                            cooking_status = 0
                            current_request = 0

                            if hand_raise == 1:
                                # print ("1:want menu, 2:ready to order, 3:want food, 4:eating/want drinks, 5:drinking/want bill, 6:cash ready, 7:cash collected, 8:clean table")
                                
                                # print ("-- customer satisfaction --", "0 to 5")
                                # customer_satisfaction = int(input("customer satisfaction (0<=sat<=5): "))
                                if self.action['type'] != 2:
                                    # print ("-- customer request --", "1:want menu, 2:ready to order, 3:want food, 4:eating, 5:want drinks, 6:drinking, 7:want bill, 8:cash ready, 9:cash collected, 10:clean table")
                                    ''' customer_request = int(input("current request (>0): "))
                                    explicit = int(input("explicit (== 1 or 0): "))
                                    if customer_request == 1 or customer_request == 2: 
                                        current_request = customer_request
                                    elif customer_request == 3: 
                                        current_request = 3
                                        food = 0
                                        print ("-- cooking status --", "1:just started, 2:half ready, 3:ready to be served")
                                        cooking_status = int(input("cooking status: ")) - 1
                                        if cooking_status == 2:
                                            obs["time_since_food_ready"] = int(input("time since food ready: "))
                                            obs["food_picked_up"] = bool(input("picked food up: ") == "1")

                                    elif customer_request == 4: 
                                        cooking_status == 2
                                        current_request = 4
                                        print ("-- food --", "1:just served, 2:half eaten, 3:eaten")
                                        food = int(input("food (>0): "))

                                    elif customer_request == 5:
                                        current_request = 4
                                        food = 3
                                        water = 0

                                    elif customer_request == 6: 
                                        current_request = 5
                                        print ("-- water --", "1:just served, 2:half drank, 3:drank")
                                        water = int(input("water (>0): "))

                                    elif customer_request == 7:
                                        current_request = 5
                                        water = 3
                                        food = 3

                                    elif customer_request > 7:
                                        current_request = customer_request - 2
                                        water = 3 

                                
                                    obs["hand_raise"] = hand_raise
                                    obs["current_request"] = current_request
                                    obs["cooking_status"] = cooking_status
                                    obs["food"] = food
                                    obs["water"] = water
                                    obs["explicit"] = explicit '''

                                    obs["explicit"] = 0
                                    if print_status and self.num_action_executions == 2: ############# hack
                                        print ("---- Observed Unhappy ----")
                                        obs["emotion"] = 0 
                                    else:
                                        emotion = int(input("unhappy (0), neutral (1), happy (2), unknown (3): "))
                                        obs["emotion"] = emotion 
                                    obs["have_bread"] = 0 

                                else:
                                    # set_trace()
                                    answer = int(input("answer (yes=y,no=n): ") == "y")
                                    obs["answer"] = answer

                            self.pub_states[self.cur_table_id].publish(json.dumps(obs))
                        else:
                            print ("---- Follow the model ----")
                            obs = self.cur_state_data
                            self.pub_states[self.cur_table_id].publish(json.dumps(obs))

                    else:
                        print ("---- Follow the model ----")
                        self.pub_states[self.cur_table_id].publish(json.dumps(self.cur_state_data))
                        if self.num_action_executions == 1:
                            self.human_input = temp
                        # print ("published")

                    count += 1

                self.hand_raises = {}
                self.unknown_tables_state.clear()
                self.success = False
                self.interrupted = False
                self.mutex.release()
            else:
                rate.sleep()


def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)

if __name__ == '__main__':
    try:
        signal.signal(signal.SIGINT, signal_handler)
        robot = False
        human_input = True
        num_tables = 2
        Sensor_Aggregator(robot, human_input, num_tables).publish()
    except rospy.ROSInterruptException:
        pass