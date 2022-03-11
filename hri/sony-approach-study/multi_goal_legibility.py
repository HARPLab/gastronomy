import table_path_code as resto
import approach_path_planner as ap
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import cv2


# Set of tests for multi-goal legibility
test_scenarios = []
file_prefix = 'multigoal-paths/multi-'

#TODO add buffer points

# Triangle
#length, width
dim = (1000, 1000)
start = (500, 500)
# goals = [(600, 0), (600,600), (0, 300)]
goals = [(800, 200), (800,800), (0, 500)]

l1 = "triangle"
r1 = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=dim)

# Two and one
dim = (800, 800)
start = (400, 400)
goals = [(300, 300), (300, 500), (500, 300)]

l2 = 'two_and_one'
r2 = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=dim)

# Additional
dim = (800, 800)
start = (400, 400)
goals = [(300, 300), (300, 500), (500, 500), (500, 300)]

l3 = 'square'
r3 = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=dim)

dim = (1000, 1000)
start = (200, 500)
# goals = [(600, 0), (600,600), (0, 300)]
goals = [(700, 500)]

l4 = "single_goal"
r4 = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=dim)

dim = (1000, 1000)
start = (300, 500)
# goals = [(600, 0), (600,600), (0, 300)]
goals = [(800, 200), (800,800)]

l5 = "side-by-side"
r5 = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=dim)

dim = (1000, 1000)
start = (200, 500)
# goals = [(600, 0), (600,600), (0, 300)]
goals = [(400, 500), (800,500)]

l6 = "in-a-line"
r6 = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=dim)

test_scenarios = [[l1, r1], [l2, r2], [l3, r3], [l4, r4], [l5, r5], [l6, r6]]




for ri in range(len(test_scenarios)):
	label, scenario = test_scenarios[ri]
	label = file_prefix + label
	ap.select_paths_and_draw(scenario, label)