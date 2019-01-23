import rosbag
import ipdb

bag = rosbag.Bag('/home/mohit/try_1/cutting_rosbag_2019-01-18-18-27-25.bag')
ipdb.set_trace()
for topic, msg, t in bag.read_messages(topics=['chatter', 'numbers']):
    print(msg)
    bag.close()
