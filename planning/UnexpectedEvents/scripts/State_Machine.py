import numpy as np
from time import sleep
import time
import copy
import threading
from pdb import set_trace
import sys

import gym
from pomdp_tasks_env import POMDPTasks
from Qlearning import Q_learning
import rl
import pylab as plt
import signal
from draw_env import *

robot = None
# random.seed(80) # was 80

random = None
reset_random = None

state_num = 0
MAX_TIME = None
# MAX_TIME = 10
global_time = 0
render_trace = True
VI = False
print_status = True
vrep = False
# goals = [[0,0],[2,2],[0,2]]
# goals = [[5,5],[2,7],[7,8],[3,2],[8,4],[6,1],[4,9],[1,4]]
goals = [[2,8],[8,5],[2,2],[5,5],[8,8],[5,2],[2,5],[5,8],[8,2],[10,10],[0,0],[0,10],[10,0]]
ROS = False
if vrep:
    from vrep_render import Vrep_Restaurant
if ROS:
    import rospy

def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)

def create_state_machines(time, table, robot):
    global state_num
    state_num = 0
    state_machines = dict()

    seat_sm = seating_SM(time, table, robot)
    state_machines[seat_sm.name] = seat_sm

    order_sm = ordering_SM(time, table, robot)
    state_machines[order_sm.name] = order_sm

    wait_for_food_sm = waiting_for_food_SM(time, table, robot)
    state_machines[wait_for_food_sm.name] = wait_for_food_sm

    get_bill_sm = get_bill_SM(time, table, robot)
    state_machines[get_bill_sm.name] = get_bill_sm

    return state_machines


def seating_SM(time, table, robot):
    sm = State_Machine("seating_SM", table, robot)
    talking1 = State("talking1", time, table)
    have_menu = State("have the menu", time, table)
    reading = State("reading the menu", time, table)
    talking2 = State("talking2", time, table)
    decided = State("decided what to eat", time, table)
    talking3 = State("talking3", time, table)
    ready = State("ordering_SM", time, table)
    # done_sm = State("done_SM", time, table)

    Human_Action("H", sm.root, talking1, table, 0.9)
    Human_Action("H", have_menu, reading, table, 0.95)
    Human_Action("H", have_menu, talking2, table, 0.05)
    Human_Action("H", reading, talking2, table, 0.5)
    Human_Action("H", talking2, reading, table, 1)
    Human_Action("H", reading, decided, table, 0.5)
    Human_Action("H", decided, talking3, table, 0.9)
    Sensing_Action(robot, "sense", decided, ready, table, 0.1)
    Sensing_Action(robot, "sense", talking3, ready, table, 1)
    # Sensing_Action(robot, "sense", decided, done_sm, table, 0.1)
    # Sensing_Action(robot, "sense", talking3, done_sm, table, 1)

    Execution_Action(robot, "exec", sm.root, have_menu, table, 0.1)
    Execution_Action(robot, "exec", talking1, have_menu, table, 1)

    return sm


def ordering_SM(time, table, robot):
    sm = State_Machine("ordering_SM", table, robot)
    robot_at_table = State("robot is at the table", time, table)
    asked = State("robot asked what do you like to have", time, table)
    request_food = State("I like to have ...", time, table)
    q_recommendation = State("what do you recommend?", time, table)
    q_specials = State("what are today's specials?", time, table)
    r_recommendation = State("robot gives recommendations", time, table)
    done_ordering = State("done with order", time, table)
    waiting_for_food_sm = State("waiting_for_food_SM", time, table)

    Human_Action("H", asked, request_food, table, 1/3)
    Human_Action("H", asked, q_recommendation, table, 1/3)
    Human_Action("H", asked, q_specials, table, 1/3)
    Human_Action("H", r_recommendation, request_food, table, 1)
    Sensing_Action(robot, "sense", request_food, done_ordering, table, 1)
    Sensing_Action(robot, "sense",done_ordering, asked, table, 0.5) ## conditionals
    Sensing_Action(robot, "sense", done_ordering, waiting_for_food_sm, table, 0.5)  ## conditionals

    Execution_Action(robot, "exec", sm.root, robot_at_table, table, 1)
    Sensing_Action(robot, "exec", robot_at_table, asked, table, 1)
    Sensing_Action(robot, "exec", q_recommendation, r_recommendation, table, 1)
    Sensing_Action(robot, "exec", q_specials, r_recommendation, table, 1)

    return sm


def waiting_for_food_SM(time, table, robot):
    sm = State_Machine("waiting_for_food_SM", table, robot)
    food_half_ready = State("food is half ready", time, table)
    food_ready = State("food is ready", time, table)
    brings_food = State("robot brings the food", time, table)
    eating_sm = State("eating SM", time, table)
    drinking_sm = State("drinking SM", time, table)
    brings_water = State("robot brings water", time, table)
    done_eating = State("people are done eating", time, table)
    get_bill_sm = State("get_bill_SM", time, table)

    Human_Action("H", brings_food, eating_sm, table, 1)
    Human_Action("H", eating_sm, eating_sm, table, 1/4)
    Human_Action("H", eating_sm, drinking_sm, table, 1/2)
    Human_Action("H", drinking_sm, eating_sm, table, 1/4)
    Human_Action("H", drinking_sm, done_eating, table, 1/4)
    Human_Action("H", brings_water, drinking_sm, table, 1)
    Human_Action("H", done_eating, get_bill_sm, table, 1)
    Human_Action("H", eating_sm, done_eating, table, 1/4) ## conditionals

    Sensing_Action(robot, "sense", sm.root, food_half_ready, table, 1)
    Sensing_Action(robot, "sense", food_half_ready, food_ready, table, 1)
    

    Execution_Action(robot, "exec", drinking_sm, brings_water, table, 1/4)
    Execution_Action(robot, "exec", food_ready, brings_food, table, 1)
    Execution_Action(robot, "exec", drinking_sm, drinking_sm, table, 1/4)

    return sm

def get_bill_SM(time, table, robot):
    sm = State_Machine("get_bill_SM", table, robot)
    waiting_for_bill = State("human waiting for the bill", time, table)
    has_bill = State("human has the bill", time, table)
    place_cards = State("human places the credit card", time, table)
    get_cards = State("robot gets the cards", time, table)
    bring_receipt = State("robot brings back the cards and the receipt", time, table)
    sign_receipt = State("people sign and take their cards", time, table)
    left = State("people leave", time, table)
    get_receipt = State("robot gets the receipts", time, table)
    done_sm = State("done_SM", time, table)

    Human_Action("H", has_bill, place_cards, table, 1)
    Human_Action("H", bring_receipt, sign_receipt, table, 1)

    Sensing_Action(robot, "sense", sign_receipt, left, table, 1/2)

    Sensing_Action(robot, "sense",sm.root, waiting_for_bill, table, 1)
    Execution_Action(robot, "sense", place_cards, get_cards, table, 1)

    Execution_Action(robot, "exec", waiting_for_bill, has_bill, table, 1)
    Execution_Action(robot, "exec", get_cards, bring_receipt, table, 1)
    Execution_Action(robot, "exec", sign_receipt, get_receipt, table, 1/2)
    Execution_Action(robot, "exec", left, get_receipt, table, 1)
    Execution_Action(robot, "exec", get_receipt, done_sm, table, 1)

    return sm

class Robot:
    def __init__(self, restaurant):
        self.tasks = list()
        self.mutex = threading.Lock()
        self.done = False
        self.restaurant = restaurant
        self.features = None

        self.reset()

    def add_task(self, task):
        try:
            self.mutex.acquire()
            self.tasks.append(task)
        finally:
            self.mutex.release()

    def remove_task(self,task):
        try:
            self.mutex.acquire()
            self.tasks.remove(task)
        finally:
            self.mutex.release()

    def pop_task(self, task):
        try:
            self.mutex.acquire()
            self.tasks.remove(task)
        finally:
            self.mutex.release()

    def select_task(self,tasks):
        global print_status
        # global render_trace
        # if len(self.tasks) == 3:
        #     if render_trace:
        #         set_trace()
        #         render_trace = False
        #     self.reset()
        if print_status:
            print ("robot start: ", int(self.get_feature("x").value),int(self.get_feature("y").value))
        global MAX_TIME, global_time, VI
        # env = gym.make('Discrete-Restaurant-v0')
        envs = POMDPTasks(self.restaurant, tasks,self, MAX_TIME,VI)

        if VI:
            pass
        else:
            selected_task = 0
        
        task = tasks[selected_task]
        # print ("****************************************************************")
        # print ("robot start: ", int(self.get_feature("x").value),int(self.get_feature("y").value))
        # print(task.table.get_prefix() + " executed ", int(task.get_feature("x").value),int(task.get_feature("y").value), "--", task.begin.name, ",", task.end.name)
        # print ("****************************************************************")
        self.tasks.remove(task)
        return task

    def run(self):
        global MAX_TIME, global_time
        self.done = False
        self.curr_task = None
        
        while not self.done:
            self.mutex.acquire() 

            if self.tasks.__len__() > 0:               
                self.planned_tasks = list(self.tasks)
                self.restaurant.render()
                task = self.select_task(self.planned_tasks)
                self.curr_task = task
                # _,_,num_steps = task.execute(render=True)
                task.done = True
                self.curr_task = None
                # global_time += num_steps
                self.restaurant.render()
            
            self.mutex.release()
            if self.tasks.__len__() == 0:   
                # dum1 = State("dum1", 0, self.restaurant.dummy_table)
                # dum2 = State("dum2", 0, self.restaurant.dummy_table)
                # dummy_task = Execution_Action(self, "exec", dum1, dum2, self.restaurant.dummy_table, 1)
                # self.curr_task = None
                # _,_,_ = dummy_task.execute(render=True)
                sleep(0.1)
                self.restaurant.render()

            
    def reset(self, pos=None):
        global reset_random
        # if self.features == None:
        self.features = []
        if pos is None:
            # reset_random.randint(0,11)
            # reset_random.randint(0,11)
            
            self.features.append(Feature("x", "discrete", False, 0, 10, 1, reset_random.randint(0,11))) ## 10 by 10 grid
            self.features.append(Feature("y", "discrete", False, 0, 10, 1, reset_random.randint(0,11)))
            # self.features.append(Feature("x", "discrete", False, 0, 10, 1, 3)) ## 10 by 10 grid
            # self.features.append(Feature("y", "discrete", False, 0, 10, 1, 4))
        else:
            reset_random.randint(0,11)
            reset_random.randint(0,11)

            self.features.append(Feature("x", "discrete", False, 0, 10, 1, pos[0])) ## 10 by 10 grid
            self.features.append(Feature("y", "discrete", False, 0, 10, 1, pos[1]))

    def set(self, features):
        self.features = []
        
        for f in features:
            self.features.append(f)

    def get_features (self):
        return self.features

    def get_feature (self, name):
        for feature in self.features:
            if feature.name == name:
                return feature

    def set_feature (self, name, value):
        for feature in self.features:
            if feature.name == name:
                feature.value = value

class Action:
    def __init__(self, name, begin, end, table, transition_prob = 0.5):
        self.name = table.get_prefix() + " " + name
        self.begin = begin
        self.end = end
        self.table = table
        self.transition_prob = transition_prob
        self.done = False
        begin.add_child(self, transition_prob)
        end.add_parent(self)

    def print(self):
        print ("*** action: ", self.name)
        self.begin.print()
        self.end.print()

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


class Feature:
    def __init__(self, name, typee, time_based, low, high, discretization, value, observable=True, dependent=False):
        self.name = name
        self.type = typee
        self.time_based = time_based
        self.low = low
        self.high = high
        self.discretization = discretization
        self.value = value
        self.observable = observable
        self.dependent = dependent

    def feature_value(self, time, table):
        global MAX_TIME, random
        if self.time_based:
            if table.patience is None:
                table.patience = random.randint(low = MAX_TIME, high = MAX_TIME, size = 1)

            new_value = int(np.power(time*(1.0/table.patience)-self.value,2))
            if new_value < self.low:
                return self.low
            elif new_value > self.high:
                return self.high
            else:
                return new_value
        else:
            return self.value

    def set_value (self, new_value):
        self.value = new_value

class Robot_Action (Action):
    def __init__(self, robot, name, begin, end, table, transition_prob):
        super().__init__(name, begin, end, table, transition_prob)
        self.table = table
        self.robot = robot

    def feature_values(self, time):
        return self.table.feature_values()

    def get_feature (self, name):
        return self.table.get_feature(name)

    def set_feature (self, name, value):
        self.table.set_feature(name, value)

    def reset (self):
        self.table.reset()

    def get_features (self):
        return self.table.get_features()

    def simulate (self, task_features, task_feature_names, robot_features, robot_feature_names, time=None):
        new_state = []
        new_state_robot = []
        robot_state = []
        count = 0
        position_features = []
        for feature in self.get_features():
            if feature.time_based:
                new_state.append(self.get_feature(task_feature_names[count]).feature_value(time, self.table))
            else:
                new_value = task_features[count]
                new_state.append(new_value)
            count += 1 

        count = 0
        for feature in self.robot.get_features():
            if not feature.time_based:
                task_count = 0
                for name in task_feature_names:
                    if robot_feature_names[count] == name:
                        new_value = task_features[task_count]
                    task_count += 1
                new_state_robot.append(new_value)
                position_features.append((robot_feature_names[count], robot_features[count], new_value))

            count += 1 

        num_steps = self.get_num_steps(position_features)
        return new_state, new_state_robot, num_steps

    def execute(self, render=False, time=None):
        new_state = []
        new_state_robot = []
        robot_state = []
        count = 0
        position_features = []
        for feature in self.get_features():
            if feature.time_based and time is not None:
                new_val = feature.feature_value(time, self.table)
                feature.set_value(new_val)
                new_state.append(new_val)
            elif not feature.time_based:
                value = self.get_feature(feature.name).value
                new_state.append(value)
            count += 1 

        count = 0
        for feature in self.robot.get_features():
            if not feature.time_based:            
                new_value = self.get_feature(feature.name).value
                new_state_robot.append(new_value)
                position_features.append((feature.name, self.robot.get_feature(feature.name).value, new_value))
                if not render:
                    self.robot.set_feature(feature.name, new_value)

            count += 1 

        if render:
            prev_x = position_features[0][1]
            x = position_features[0][2]
            prev_y = position_features[1][1]
            y = position_features[1][2]
            self.table.render()
            if not vrep:
                while (prev_x != x and not (self.table == self.table.restaurant.dummy_table and self.robot.tasks.__len__() != 0)):
                    dir_x = np.sign(x-prev_x)
                    prev_x = prev_x + dir_x
                    self.robot.set_feature("x", prev_x)
                    self.table.render()
                while (prev_y != y and not (self.table == self.table.restaurant.dummy_table and self.robot.tasks.__len__() != 0)):
                    dir_y = np.sign(y-prev_y)
                    prev_y = prev_y + dir_y
                    self.robot.set_feature("y", prev_y)
                    self.table.render()
            else:
                ##self.table.restaurant.vrep_sim.go_to ((prev_x,prev_y),(x,y),self.table.id)
                pass
            

        num_steps = self.get_num_steps(position_features)
        return new_state, new_state_robot, num_steps

    def get_num_steps(self, position_features):
        prev_x = position_features[0][1]
        x = position_features[0][2]
        prev_y = position_features[1][1]
        y = position_features[1][2]
        # prev_theta = position_features[2][1]
        # theta = position_features[2][2]

        num_steps = np.abs(x-prev_x) + np.abs(y-prev_y)
        return num_steps

class Sensing_Action (Robot_Action):
    def __init__(self, robot, name, begin, end, table, transition_prob = 0.5):
        super().__init__(robot, name, begin, end, table, transition_prob)

    def print(self):
        print("sensing action name: ", self.name)
        self.begin.print()
        self.end.print()

class Execution_Action (Robot_Action):
    def __init__(self, robot, name, begin, end, table, transition_prob = 0.5):
        super().__init__(robot, name, begin, end, table, transition_prob)

    def print(self):
        print("execution action name: ", self.name)
        self.begin.print()
        self.end.print()

class Human_Action (Action):
    def __init__(self, name, begin, end, table, transition_prob = 0.5):
        super().__init__(name, begin, end, table, transition_prob)

class State:
    def __init__(self, name, time=2, table = ""):
        global state_num
        ## each state can have different properties...
        self.name = name
        self.parents = list()
        self.children = list()
        self.transition_probs = list()
        self.time = time
        self.table = table
        self.number = state_num
        state_num += 1

    def add_child(self, child, trans_prob):
        self.children.append(child)
        self.transition_probs.append(trans_prob)

    def add_parent(self, parent):
        self.parents.append(parent)

    def print(self):
        print (self.table.get_prefix() + " state: ",self.name, " #", self.number)
        


    def execute_on_robot (self, table, edge):        
        if isinstance(edge, Execution_Action):
            edge.done = False
            self.table.robot.add_task(edge)
            while (not edge.done):
                sleep(0.1)

            num_reqs = table.get_feature ("num_past_requests")
            num_reqs.set_value(num_reqs.value+1)
            print(edge.table.get_prefix() + " executed ", edge.begin.name, ",", edge.end.name)

            req = table.get_feature ("current_request")
            t_req = table.get_feature ("time_since_hand_raise")
            hand_raise = table.get_feature ("hand_raise")
            satisfaction = table.get_feature ("customer_satisfaction")

            req.set_value(0)
            t_req.set_value(0)
            hand_raise.set_value(0)
            satisfaction.set_value(0)
            table.time_hand_raise = None

        return edge.end

    def execute (self, table):
        global global_time, random
        
        edge = None
        req = table.get_feature ("current_request")
        t_req = table.get_feature ("time_since_hand_raise")
        hand_raise = table.get_feature ("hand_raise")
        if "drinking SM" in self.name:
            water = table.get_feature ("water")
            if water.value == 0:
                req.set_value(4)
                t_req.set_value(0)
                hand_raise.set_value(1)
                table.time_hand_raise = global_time
                for e in self.children:
                    if isinstance(e, Execution_Action):
                        edge = e
                        next_state = edge.end
                        break
            else:
                water.set_value(water.value-1)
                food = table.get_feature ("food")
                if food.value == food.high:
                    for e in self.children:
                        if "people are done eating" in e.end.name:
                            edge = e
                            next_state = edge.end
                            break
                else:  
                    for e in self.children:
                        if "eating SM" in e.end.name:
                            edge = e
                            next_state = edge.end
                            break
        elif "robot brings water" in self.name:
            water = table.get_feature ("water")
            water.set_value(water.high)
            req.set_value(0)
            t_req.set_value(0)
            hand_raise.set_value(0)
            table.time_hand_raise = None          
            
            edge = random.choice(self.children, p=self.transition_probs) 
            next_state = edge.end     
                        
        elif "eating SM" in self.name:
                food = table.get_feature ("food")
                if food.value == food.high:
                    for e in self.children:
                        if "people are done eating" in e.end.name:
                            edge = e
                            next_state = edge.end
                            break
                else:
                    food.set_value(food.value+1)
                    edge = random.choice(self.children, p=self.transition_probs) 
                    next_state = edge.end     
        else:     
            edge = random.choice(self.children, p=self.transition_probs) 
            next_state = edge.end     

            food = table.get_feature("food")

            if "have the menu" in next_state.name:                
                req.set_value(1)
                t_req.set_value(0)
                hand_raise.set_value(1)
                table.time_hand_raise = global_time
            elif "ordering_SM" in self.name:
                req.set_value(2)
                t_req.set_value(0)
                hand_raise.set_value(1)
                table.time_hand_raise = global_time

            elif "waiting_for_food_SM" in self.name:
                req.set_value(3)
                t_req.set_value(0)
                hand_raise.set_value(1)
                table.time_hand_raise = global_time
                food.set_value(0)
            elif "food is half ready" in self.name:
                food.set_value(food.value+1)
                req.set_value(3)
                hand_raise.set_value(1)
            elif "food is ready" in self.name:
                food.set_value(food.value+1)  
                table.time_food_ready = global_time  
                req.set_value(3)
                hand_raise.set_value(1)
            elif "robot brings the food" in self.name:
                food.set_value(food.value+1)  
                table.time_food_ready=None  
                req.set_value(0)
                hand_raise.set_value(0)

                
            elif "human waiting for the bill" in self.name:
                req.set_value(5)
                t_req.set_value(0)
                hand_raise.set_value(1)
                table.time_hand_raise = global_time
            elif "human places the credit card" in self.name:
                req.set_value(6)
                t_req.set_value(0)
                hand_raise.set_value(1)
                table.time_hand_raise = global_time
            elif "robot brings back the cards and the receipt" in next_state.name:
                req.set_value(7)
                t_req.set_value(0)
                hand_raise.set_value(1)
                table.time_hand_raise = global_time
            elif "robot gets the receipts" in next_state.name:
                req.set_value(8)
                t_req.set_value(0)
                hand_raise.set_value(1)
                table.time_hand_raise = global_time
            else:
                req.set_value(0)
                hand_raise.set_value(0)
                table.time_hand_raise = None



        table.print_features()
        sleep(self.time)
        next_state = self.execute_on_robot(table, edge)
        
    
        return next_state, edge 

class State_Machine:
    def __init__(self, name, table, robot):
        self.name = name
        self.root = State(name, time=0, table=table)
        self.table = table
        self.robot = robot

    def run(self, table):
        global global_time
        done = False
        self.current_state = self.root
        hist_num = 0
        self.print()
        while not self.table.restaurant.done:
            self.current_state.print()
            if table.get_feature("hand_raise").value == 1:
                t_req = table.get_feature ("time_since_hand_raise")
                t_req.set_value(global_time-table.time_hand_raise)
            else:
                t_req = table.get_feature ("time_since_hand_raise")
                t_req.set_value(0)
                table.time_hand_raise=None 

            if table.get_feature("food").value == 2:
                t_req = table.get_feature ("time_since_food_ready")
                t_req.set_value(global_time-table.time_food_ready)
            else:
                t_req = table.get_feature ("time_since_food_ready")
                t_req.set_value(0)
                table.time_food_ready=None 

            table.update_satisfaction()


            if self.current_state is None or self.current_state.children.__len__() == 0:
                sleep(self.current_state.time)
                break

            next_state, edge = self.current_state.execute(table)
            self.current_state = next_state
            table.histories.append(History(hist_num,table.start_time,edge.end))
            hist_num += 1

    def print (self):
        print ( "** " + self.table.get_prefix() + " state machine name: ", self.name)

class Human:
    def __init__(self, name, id):
        self.name = name
        self.id =  id

class History:
    def __init__(self, id, start_time, state):
        self.id = id
        self.time_passed = time.time() - start_time
        self.state = state

    def print(self):
        print (str(self.id) + " " + str(self.state.number) + "," + str(self.time_passed))

class Table:
    def __init__(self, restaurant, id, robot, fake=False):
        global random
        self.num_humans = 0
        self.time = 0
        self.fake = fake
        self.restaurant = restaurant
        self.id = id
        self.start_time = time.time()
        self.time_hand_raise = None
        self.time_food_ready = None
        self.humans = set()
        self.histories = list()
        self.robot = robot
        if not self.fake:
            self.num_humans = random.randint(5)
            self.time = random.randint(5)/5.0
            self.state_machines = create_state_machines(self.time, self, robot)
        self.patience = None
        for h in range(self.num_humans):
            self.humans.add(Human(str(h),str(h)))

        self.reset_all()

    def run(self):
        self.curr_state_machine = self.state_machines['seating_SM']
        while (not self.restaurant.done):
            self.curr_state_machine.run(self)
            if self.curr_state_machine.current_state.name == "done_SM":
                break
            else:
                self.curr_state_machine = self.state_machines[self.curr_state_machine.current_state.name]

        # print ("*** history of table " + str(self.id) + " ***")
        # for h in self.histories:
        #     h.print()

    def get_prefix(self):
        return "Table " + str(self.id) + ": "

    def feature_values(self, time):
        new_values = [] 
        for feature in self.features:
            new_values.append(feature.feature_value(time,self.table))
        return new_values

    def get_feature (self, name):
        for feature in self.features:
            if feature.name == name:
                return feature

    def set_feature (self, name, value):
        for feature in self.features:
            if feature.name == name:
                feature.value = value

    def get_features (self):
        return self.features

    def print_features(self):
        features_value = []
        for feature in self.features:
            features_value.append (feature.value)

        print (features_value)

    def reset_all (self):
        global random
        self.features = []
        if not self.fake:
            global goals
            # self.initialization_feature_range = (6, 9)
            # self.features.append(Feature("attention", "discrete", True, 0, MAX_TIME, 1, \
            #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * random.random_sample() + self.initialization_feature_range[0])))
            # self.features.append(Feature("urgency", "discrete", True, 0, MAX_TIME, 1, \

            #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * random.random_sample() + self.initialization_feature_range[0])))
            # self.features.append(Feature("completion", "discrete", True, 0, MAX_TIME, 1, \
            #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * random.random_sample() + self.initialization_feature_range[0])))
            
            # self.features.append(Feature("theta", "discrete", False, 0, 11, 1, random.randint(0,12))) # theta # * 30 degree

            
            # self.features.append(Feature("urgency", "discrete", True, 0, MAX_TIME, 1, 0)) # increases based on time from request

            # self.features.append(Feature("x", "discrete", False, 0, 10, 1, goals[self.id][0])) 
            # self.features.append(Feature("y", "discrete", False, 0, 10, 1, goals[self.id][1]))

            self.goal_x = goals[self.id][0]
            self.goal_y = goals[self.id][1]

            # self.features.append(Feature("next_request", "discrete", False, 0, 8, 1, 0, observable=False, dependent=False))
            ## just started, half-ready, ready
            self.features.append(Feature("cooking_status", "discrete", False, 0, 2, 1, 0, observable=True, dependent=False))
            ## ready-hot, ready-almost-hot, ready-cold
            self.features.append(Feature("time_since_food_ready", "discrete", False, 0, MAX_TIME-1, 1, 0, observable=True, dependent=True))
            ## not-served, served-full, served-half, served-empty
            self.features.append(Feature("water", "discrete", False, 0, 3, 1, 0, observable=True, dependent=False)) ## full, empty, half
            ## not-served, served-full, served-half, served-empty
            self.features.append(Feature("food", "discrete", False, 0, 3, 1, 0, observable=True, dependent=False))

            self.features.append(Feature("time_since_served", "discrete", False, 0, MAX_TIME-1, 1, 0, observable=True, dependent=True))

            self.features.append(Feature("hand_raise", "discrete", False, 0, 1, 1, 0, observable=True, dependent=False)) 
            ## just raised, half, late
            self.features.append(Feature("time_since_hand_raise", "discrete", False, 0, MAX_TIME-1, 1, 0, observable=True, dependent=True))
            self.features.append(Feature("food_picked_up", "discrete", False, 0, 1, 1, 0, observable=True, dependent=True))
            # self.features.append(Feature("num_past_requests", "discrete", False, 0, 9, 1, 0))

            # no_req, want_menu, ready_to_order, want_food, want_water, want_bill, get_cards, want_cards_back, done_table
            self.features.append(Feature("current_request", "discrete", False, 0, 8, 1, 0, observable=True, dependent=False))
            self.features.append(Feature("customer_satisfaction", "discrete", False, 0, 5, 1, 0, observable=False, dependent=False))
            
        else:
            # self.features.append(Feature("next_request", "discrete", False, 0, 8, 1, 0, observable=False, dependent=False))
            ## just started, half-ready, ready
            self.features.append(Feature("cooking_status", "discrete", False, 0, 2, 1, 0, observable=True, dependent=False))
            ## ready-hot, ready-almost-hot, ready-cold
            self.features.append(Feature("time_since_food_ready", "discrete", False, 0, MAX_TIME-1, 1, 0, observable=True, dependent=True))
            self.features.append(Feature("water", "discrete", False, 0, 2, 1, 0, observable=True, dependent=False)) ## full, empty, half
            ## served-full, served-half, served-empty
            self.features.append(Feature("food", "discrete", False, 0, 2, 1, 0, observable=True, dependent=False))

            self.features.append(Feature("time_since_served", "discrete", False, 0, MAX_TIME-1, 1, 0, observable=True, dependent=True))

            self.features.append(Feature("hand_raise", "discrete", False, 0, 1, 1, 0, observable=True, dependent=False)) 
            ## just raised, half, late
            self.features.append(Feature("time_since_hand_raise", "discrete", False, 0, MAX_TIME-1, 1, 0, observable=True, dependent=True))
            self.features.append(Feature("food_picked_up", "discrete", False, 0, 1, 1, 0, observable=True, dependent=True))
            # self.features.append(Feature("num_past_requests", "discrete", False, 0, 9, 1, 0))

            # no_req, want_menu, ready_to_order, want_food, want_water, want_bill, get_cards, want_cards_back, done_table
            self.features.append(Feature("current_request", "discrete", False, 0, 8, 1, 0, observable=True, dependent=False)) ## also change "get_possible_obss" in pomdp_client
            self.features.append(Feature("customer_satisfaction", "discrete", False, 0, 5, 1, 0, observable=False, dependent=False))

    def reset (self):
        # self.get_feature("attention").set_value ( \
        #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * random.random_sample() + self.initialization_feature_range[0]))
        # self.get_feature("urgency").set_value ( \
        #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * random.random_sample() + self.initialization_feature_range[0]))
        # self.get_feature("completion").set_value ( \
        #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * random.random_sample() + self.initialization_feature_range[0]))
        pass
    def render(self):
        self.restaurant.render()

    def update_satisfaction(self):
        pass

class Restaurant:
    def __init__(self,seed,num_tables,horizon,greedy,simple,model,no_op,run_on_cobot,hybrid,deterministic,hybrid_3T,shani_baseline,hierarchical_baseline):
        global random, MAX_TIME, global_time, vrep, VI, reset_random, goals
        if seed is not None and seed != -1:
            random = np.random.RandomState(seed)
            reset_random = np.random.RandomState(seed+10)
        else:
            random = np.random.RandomState()
            reset_random = np.random.RandomState()
            seed = None


        if run_on_cobot:
            goals = [[6,2],[6,6],[3,7]]

        if vrep:
            self.vrep_sim = Vrep_Restaurant()
        self.done = False
        self.tables = list()
        self.threads = list()
        self.num_tables = num_tables

        MAX_TIME = 5 ##self.num_tables * 6

        self.robot = Robot(self)
        self.robot_thread = threading.Thread(target=self.robot.run, daemon=True, args=())
        # self.dummy_table = Table(self, 10000, self.robot, fake=True)
        for t in range(self.num_tables):
            table = Table(self, t, self.robot)
            self.tables.append(table)
            self.threads.append(threading.Thread(target=table.run, daemon=True, args=()))

        for table in self.tables:
            talking1 = State("talking1", table.time, table)
            have_menu = State("have the menu", table.time, table)
            robot_task = Execution_Action(self.robot, "exec", talking1, have_menu, table, 1.0)

            self.robot.add_task(robot_task)

        print ("seed: ", seed, "#tables", self.num_tables, "horizon: ", horizon, "greedy: ", greedy, "simple: ", simple, \
            "model: ", model, "no_op: ", no_op, "hybrid: ", hybrid, "deterministic: ",deterministic, "hybrid_3T: ", hybrid_3T, \
            "shani_baseline: ", shani_baseline, "hierarchical_baseline: ", hierarchical_baseline)
        envs = POMDPTasks(self, list(self.robot.tasks),self.robot, seed, random, reset_random, horizon, greedy, simple, model, \
            no_op, run_on_cobot, hybrid, deterministic, hybrid_3T, shani_baseline, hierarchical_baseline)

        # set_trace()


        # rng = list(range(self.num_tables))
        # random.shuffle(rng)

        # self.robot_thread.start()
        # for t in rng:
        #     self.threads[t].start()

        # try:
        #     if ROS:
        #         rospy.init_node('sony', anonymous=True)
        #         rate = rospy.Rate(10) # 10hz
        #     while(not self.done): ## and not rospy.is_shutdown()):
        #         self.done = True
        #         for t in rng:
        #             if self.threads[t].isAlive():
        #                 self.done = False
        #                 break
        #         if ROS:
        #             rate.sleep()
        #         else:
        #             time.sleep(0.1)
        #         global_time += 1
        #         # print ("global_time: ", global_time)


        # finally:
        #     self.done = True
        #     self.robot.done = True
        #     if vrep:
        #         self.vrep_sim.done()

        # for t in self.threads:
        #     t.join()


    def print(self):
        for t in range(self.num_tables):
            print(t)

    def render(self, mode='human', close=False):
        global vrep
        if not vrep:
            """ Viewer only supports human mode currently. """
            global render_trace
            plt.clf()
            margin = 0.2
            x_low = self.robot.get_feature("x").low - margin
            x_high = self.robot.get_feature("x").high + margin
            y_low = self.robot.get_feature("y").low - margin
            y_high = self.robot.get_feature("y").high + margin
            coords = [x_low,x_high,y_low,y_high]
            drawTables(coords, self.tables)
            drawRobot(self.robot, 0.5)
            if self.robot.curr_task is not None:
                drawTasks(self.robot.tasks + [self.robot.curr_task])
            else:
                drawTasks(self.robot.tasks)
            # plt.plot(self.points[:,0],self.points[:,1],'.')

            plt.axis('equal')
            plt.axis(coords)

            # drawPath(self.poses, self.counter)
            plt.show(block=False)
            plt.pause(0.5)

            if render_trace:
                set_trace() 
                render_trace = False
            ## plt.close()

            # if self.finish_render:
            #     plt.show(block=False)
            #     plt.pause(0.0000000001)

            # if close:
            #     plt.close()
        else:
            self.vrep_render()

    def vrep_render (self):
        self.vrep_sim.render(self.robot, self.tables, self.done)

def main():
    # print command line arguments
    # set_trace()
    signal.signal(signal.SIGINT, signal_handler)
    no_op = False
    hybrid = False
    deterministic = False
    hybrid_3T = False
    shani_baseline = False
    hierarchical_baseline = False
    run_on_cobot = False
    if "no_op" in str(sys.argv[6]):
        no_op = True
    if "hybrid" in str(sys.argv[6]):
        hybrid = True
        if "hybrid_3T" in str(sys.argv[6]):
            hybrid_3T = True
    if "deterministic" in str(sys.argv[6]):
        deterministic = True
    if "shani" in str(sys.argv[6]):
        shani_baseline = True
    if "H_POMDP" in str(sys.argv[6]):
        hierarchical_baseline = True
    if len(sys.argv) > 7 and ("robot" in str(sys.argv[7])):
        run_on_cobot = True

    rst = Restaurant(seed=int(sys.argv[1]),num_tables=int(sys.argv[2]),horizon=int(sys.argv[3]),greedy=bool(sys.argv[4] == "True")\
        ,simple=bool(sys.argv[5] == "True"), model=str(sys.argv[6]), no_op=no_op, run_on_cobot=run_on_cobot, hybrid=hybrid, deterministic=deterministic\
        , hybrid_3T=hybrid_3T, shani_baseline=shani_baseline, hierarchical_baseline=hierarchical_baseline)

if __name__ == "__main__":
    main()
