#!/usr/bin/env python

from __future__ import print_function

"""Create a recording with arbitrary duration.

PySoundFile (https://github.com/bastibe/PySoundFile/) has to be installed!

"""
import argparse
import tempfile
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
args = parser.parse_args()

try:
    import rospy
    import soundfile as sf
    from sounddevice_ros.msg import AudioInfo, AudioData
    import numpy  # Make sure NumPy is loaded before it is used in the callback
    assert numpy  # avoid "imported but unused" message (W0611)


    if args.filename is None:
        args.filename = tempfile.mktemp(prefix='delme_rec_unlimited_',
                                        suffix='.wav', dir='')
    
    rospy.init_node('sounddevice_ros_subscriber')

    audio_info_msg = rospy.wait_for_message('/audio_info', AudioInfo)
    
    sample_rate = audio_info_msg.sample_rate
    num_channels = audio_info_msg.num_channels
    subtype = args.subtype
    if audio_info_msg.subtype:
        subtype = audio_info_msg.subtype

    # Make sure the file is opened before recording anything:
    with sf.SoundFile(args.filename, mode='x', samplerate=sample_rate,
                      channels=num_channels, subtype=subtype) as file:

        def callback(msg):
            file.write(numpy.asarray(msg.data).reshape((-1,num_channels)))

        audio_sub = rospy.Subscriber('/audio', AudioData, callback)

        rospy.spin()

except KeyboardInterrupt:
    print('\nRecording finished: ' + repr(args.filename))
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))