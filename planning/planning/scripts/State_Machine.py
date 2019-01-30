import numpy as np
from time import sleep
import time
import copy
import threading
from pdb import set_trace

import gym
from discrete_tasks_env import DiscreteTasks
from Qlearning import Q_learning
import rl
import pylab as plt

from draw_env import *

robot = None
np.random.seed(3)
state_num = 0
MAX_TIME = 10
global_time = 0

VI = False

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
    Execution_Action(robot, "exec", robot_at_table, asked, table, 1)
    Execution_Action(robot, "exec", q_recommendation, r_recommendation, table, 1)
    Execution_Action(robot, "exec", q_specials, r_recommendation, table, 1)

    return sm


def waiting_for_food_SM(time, table, robot):
    sm = State_Machine("waiting_for_food_SM", table, robot)
    food_ready = State("food is ready", time, table)
    brings_food = State("robot brings the food", time, table)
    eating_sm = State("eating SM", time, table)
    drinking_sm = State("drinking SM", time, table)
    done_eating = State("people are done eating", time, table)
    get_bill_sm = State("get_bill_SM", time, table)

    Human_Action("H", brings_food, eating_sm, table, 1)
    Human_Action("H", eating_sm, eating_sm, table, 1/3)
    Human_Action("H", eating_sm, drinking_sm, table, 1/3)
    Human_Action("H", drinking_sm, eating_sm, table, 1/2)
    Human_Action("H", drinking_sm, drinking_sm, table, 1/2)
    Human_Action("H", done_eating, get_bill_sm, table, 1)

    Sensing_Action(robot, "sense", sm.root, food_ready, table, 1)
    Sensing_Action(robot, "sense",eating_sm, done_eating, table, 1/3) ## conditionals

    Execution_Action(robot, "exec", food_ready, brings_food, table, 1)

    return sm

def get_bill_SM(time, table, robot):
    sm = State_Machine("get_bill_SM", table, robot)
    wants_bill = State("human called the robot for the bill", time, table)
    waiting_for_bill = State("human waiting for the bill", time, table)
    has_bill = State("human has the bill", time, table)
    place_cards = State("human places the credit card", time, table)
    get_cards = State("robot gets the cards", time, table)
    took_cards = State("robot took the cards", time, table)
    bring_receipt = State("robot brings back the cards and the receipt", time, table)
    sign_receipt = State("people sign and take their cards", time, table)
    left = State("people leave", time, table)
    get_receipt = State("robot gets the receipts", time, table)
    done_sm = State("done_SM", time, table)

    Human_Action("H", sm.root, wants_bill, table, 1/2)
    Human_Action("H", has_bill, place_cards, table, 1)
    Human_Action("H", bring_receipt, sign_receipt, table, 1)

    Sensing_Action(robot, "sense", sign_receipt, left, table, 1/2)

    Sensing_Action(robot, "sense", wants_bill, waiting_for_bill, table, 1)
    Sensing_Action(robot, "sense",sm.root, waiting_for_bill, table, 1/2)
    Sensing_Action(robot, "sense", place_cards, get_cards, table, 1)

    Execution_Action(robot, "exec", waiting_for_bill, has_bill, table, 1)
    Execution_Action(robot, "exec", get_cards, took_cards, table, 1)
    Execution_Action(robot, "exec", took_cards, bring_receipt, table, 1)
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

        self.reset()

    def add_task(self, task):
        self.mutex.acquire()
        try:
            self.tasks.append(task)
        finally:
            self.mutex.release()

    def remove_task(self,task):
        self.mutex.acquire()
        try:
            self.tasks.remove(task)
        finally:
            self.mutex.release()

    def pop_task(self, task):
        self.mutex.acquire()
        try:
            self.tasks.remove(task)
        finally:
            self.mutex.release()

    def select_task(self,tasks):
        print ("robot start: ", int(self.get_feature("x").value),int(self.get_feature("y").value))
        global MAX_TIME, global_time, VI
        # env = gym.make('Discrete-Restaurant-v0')
        if VI is not None and (len(tasks) > 1):
            initial_features = copy.deepcopy(self.get_features())
            env = DiscreteTasks(tasks,self, MAX_TIME, VI)

            num_train_episodes = 50000
            num_test_episodes = 10
            
            print ("env name: ",env.name)
            print("# of actions: ",env.nA)
            print("# of states: ", env.nS)
            if not VI:
                print ("**************************************")
                print ("Q-learning started.")
                agent = Q_learning(env, env.nA, env.nS, epsilon=0.1, gamma=0.95, alpha=0.05, exploration_policy="epsilon_greedy_w_decay")
                agent.train(num_train_episodes,num_test_episodes)
                val_func = agent.get_V()
                policy = agent.get_policy()
                Q = agent.get_Q()
                print ("Q-learning ended.")
                print ("**************************************")
            else:
                print ("**************************************")
                print ("Value iteration started.")
                val_func, iterations, policy, Q = rl.value_iteration(env, gamma=0.95)
                print ("num of iterations: "+str(iterations))
                print ("Value iteration ended.")
                print ("**************************************")
            t0_x = tasks[0].get_feature("x").value
            t0_y = tasks[0].get_feature("y").value
            t1_x = tasks[1].get_feature("x").value
            t1_y = tasks[1].get_feature("y").value
            print ("task0: ", tasks[0].table.id, " ",t0_x,t0_y,"task1: ", tasks[1].table.id, " ",t1_x,t1_y)
            if len(tasks) ==3:
                t2_x = tasks[2].get_feature("x").value
                t2_y = tasks[2].get_feature("y").value
                print ("task0: ", tasks[0].table.id, " ",t0_x,t0_y,"task1: ", tasks[1].table.id, " ",t1_x,t1_y,"task2: ", tasks[2].table.id, " ",t2_x,t2_y)

            self.set(initial_features)
            policy_reshaped =  policy.reshape(env.state_space_dim)
            if len(tasks) == 2:
                selected_task = policy_reshaped[0,0,t0_x,t0_y,t1_x,t1_y,int(self.get_feature("x").value),int(self.get_feature("y").value)]
            if len(tasks) == 3:
                selected_task = policy_reshaped[0,0,0,t0_x,t0_y,t1_x,t1_y,t2_x,t2_y,int(self.get_feature("x").value),int(self.get_feature("y").value)]

            # if len(tasks) ==2:
            #     new_val_func = val_func.reshape(env.state_space_dim)
            #     print ("(0,0)", policy.reshape(env.state_space_dim)[0,0,t0_x,t0_y,t1_x,t1_y,int(self.get_feature("x").value),int(self.get_feature("y").value)])
            #     print ("Q(0,0),0", Q.reshape(env.state_space_dim+(2,))[0,0,t0_x,t0_y,t1_x,t1_y,int(self.get_feature("x").value),int(self.get_feature("y").value),0])
            #     print ("Q(0,0),1", Q.reshape(env.state_space_dim+(2,))[0,0,t0_x,t0_y,t1_x,t1_y,int(self.get_feature("x").value),int(self.get_feature("y").value),1])
            #     rl.show_value_function(env,new_val_func[0,0,t0_x,t0_y,t1_x,t1_y,:,:])

            # if len(tasks) ==3:
            #     new_val_func = val_func.reshape(env.state_space_dim)
            #     print ("(0,0)", policy.reshape(env.state_space_dim)[0,0,0,t0_x,t0_y,t1_x,t1_y,t2_x,t2_y,int(self.get_feature("x").value),int(self.get_feature("y").value)])
            #     print ("Q(0,0),0", Q.reshape(env.state_space_dim+(3,))[0,0,0,t0_x,t0_y,t1_x,t1_y,t2_x,t2_y,int(self.get_feature("x").value),int(self.get_feature("y").value),0])
            #     print ("Q(0,0),1", Q.reshape(env.state_space_dim+(3,))[0,0,0,t0_x,t0_y,t1_x,t1_y,t2_x,t2_y,int(self.get_feature("x").value),int(self.get_feature("y").value),1])
            #     print ("Q(0,0),2", Q.reshape(env.state_space_dim+(3,))[0,0,0,t0_x,t0_y,t1_x,t1_y,t2_x,t2_y,int(self.get_feature("x").value),int(self.get_feature("y").value),2])
            #     rl.show_value_function(env,new_val_func[0,0,0,t0_x,t0_y,t1_x,t1_y,t2_x,t2_y,:,:])

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
        while not self.done:
            if self.tasks.__len__() > 0:
                self.mutex.acquire()
                self.planned_tasks = list(self.tasks)
                task = self.select_task(self.planned_tasks)
                self.curr_task = task
                _,_,num_steps = task.execute(render=True)
                task.done = True
                self.curr_task = None
                global_time += num_steps
                self.restaurant.render()
                self.mutex.release()
            else:
                sleep(0.1)

    def reset(self):
        self.features = []
        # self.features.append(Feature("x", "discrete", False, 0, 10, 1, np.random.randint(0,11))) ## 10 by 10 grid
        # self.features.append(Feature("y", "discrete", False, 0, 10, 1, np.random.randint(0,11)))
        self.features.append(Feature("x", "discrete", False, 0, 6, 1, np.random.randint(0,7))) ## 10 by 10 grid
        self.features.append(Feature("y", "discrete", False, 0, 6, 1, np.random.randint(0,7)))
        # self.features.append(Feature("theta", "discrete", False, 0, 11, 1, np.random.randint(0,12))) # theta # * 30 degree

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
    def __init__(self, name, typee, time_based, low, high, discretization, value = 1):
        self.name = name
        self.type = typee
        self.time_based = time_based
        self.low = low
        self.high = high
        self.discretization = discretization
        self.value = value

    def feature_value(self, time, table):
        global MAX_TIME
        if self.time_based:
            if table.patience is None:
                table.patience = np.random.randint(low = MAX_TIME, high = MAX_TIME, size = 1)

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
            while (prev_x != x):
                dir_x = np.sign(x-prev_x)
                prev_x = prev_x + dir_x
                self.robot.set_feature("x", prev_x)
                self.table.render()
            while (prev_y != y):
                dir_y = np.sign(y-prev_y)
                prev_y = prev_y + dir_y
                self.robot.set_feature("y", prev_y)
                self.table.render()

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

class Execution_Action (Robot_Action):
    def __init__(self, robot, name, begin, end, table, transition_prob = 0.5):
        super().__init__(robot, name, begin, end, table, transition_prob)

    def print(self):
        print("execution action name: ", self.name)

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
        print (self.table.get_prefix() + " state name: ",self.name, " #", self.number)

class State_Machine:
    def __init__(self, name, table, robot):
        self.name = name
        self.root = State(name, time=0, table=table)
        self.table = table
        self.robot = robot

    def run(self, table):
        done = False
        self.current_state = self.root
        hist_num = 0
        self.print()
        while not self.table.restaurant.done:
            self.current_state.print()
            if self.current_state is None or self.current_state.children.__len__() == 0:
                sleep(self.current_state.time)
                break
            sleep(self.current_state.time)
            edge = np.random.choice(self.current_state.children, p=self.current_state.transition_probs)
            if isinstance(edge, Robot_Action):
                self.robot.add_task(edge)
                while (not edge.done):
                    sleep(0.1)
                print(edge.table.get_prefix() + " executed ", edge.begin.name, ",", edge.end.name)
            self.current_state = edge.end
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
    def __init__(self, restaurant, id, robot):
        self.restaurant = restaurant
        self.id = id
        self.start_time = time.time()
        self.humans = set()
        self.histories = list()
        self.num_humans = np.random.randint(5)
        self.time = np.random.randint(5)/5.0
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

    def reset_all (self):
        self.features = []
        # self.initialization_feature_range = (6, 9)
        # self.features.append(Feature("attention", "discrete", True, 0, MAX_TIME, 1, \
        #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * np.random.random_sample() + self.initialization_feature_range[0])))
        # self.features.append(Feature("urgency", "discrete", True, 0, MAX_TIME, 1, \
        #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * np.random.random_sample() + self.initialization_feature_range[0])))
        # self.features.append(Feature("completion", "discrete", True, 0, MAX_TIME, 1, \
        #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * np.random.random_sample() + self.initialization_feature_range[0])))

        # self.features.append(Feature("x", "discrete", False, 0, 10, 1, np.random.randint(0,11))) ## 10 by 10 grid
        # self.features.append(Feature("y", "discrete", False, 0, 10, 1, np.random.randint(0,11)))
        self.features.append(Feature("x", "discrete", False, 0, 6, 1, np.random.randint(0,7))) ## 10 by 10 grid
        self.features.append(Feature("y", "discrete", False, 0, 6, 1, np.random.randint(0,7)))
        # self.features.append(Feature("theta", "discrete", False, 0, 11, 1, np.random.randint(0,12))) # theta # * 30 degree

    def reset (self):
        # self.get_feature("attention").set_value ( \
        #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * np.random.random_sample() + self.initialization_feature_range[0]))
        # self.get_feature("urgency").set_value ( \
        #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * np.random.random_sample() + self.initialization_feature_range[0]))
        # self.get_feature("completion").set_value ( \
        #     int((self.initialization_feature_range[1] - self.initialization_feature_range[0]) * np.random.random_sample() + self.initialization_feature_range[0]))
        pass
    def render(self):
        self.restaurant.render()

class Restaurant:
    def __init__(self):
        self.done = False
        self.tables = list()
        self.threads = list()
        self.num_tables = 2
        self.robot = Robot(self)
        self.robot_thread = threading.Thread(target=self.robot.run, daemon=True, args=())
        for t in range(self.num_tables):
            table = Table(self, t, self.robot)
            self.tables.append(table)
            self.threads.append(threading.Thread(target=table.run, daemon=True, args=()))

        rng = list(range(self.num_tables))
        np.random.shuffle(rng)
        print (rng)
        self.robot_thread.start()
        for t in rng:
            self.threads[t].start()

        while(not self.done):
            self.done = True
            for t in rng:
                if self.threads[t].isAlive():
                    self.done = False
                    break
            sleep(1)

        self.done = True
        self.robot.done = True

        for t in self.threads:
            t.join()


    def print(self):
        for t in range(self.num_tables):
            print(t)

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """

        plt.clf()
        margin = 0.5
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
        # plt.close()

        # if self.finish_render:
        #     plt.show(block=False)
        #     plt.pause(0.0000000001)

        # if close:
        #     plt.close()

rst = Restaurant()
