#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

def draw_square():
    # Start a new ROS node named 'turtle_square'
    rospy.init_node('turtle_square', anonymous=True)
    
    # Create a publisher to send velocity commands
    pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    
    # Create a Twist message to hold velocity commands
    vel_msg = Twist()
    
    # Set a rate of 1 Hz (1 cycle per second)
    rate = rospy.Rate(1)
    
    # Repeat 4 times to draw a square
    for _ in range(4):
        # Move forward
        vel_msg.linear.x = 2.0  # Speed
        vel_msg.angular.z = 0.0  # No rotation
        
        # Publish the velocity command
        pub.publish(vel_msg)
        
        # Wait for 1 second
        rate.sleep()
        
        # Stop moving
        vel_msg.linear.x = 0.0
        pub.publish(vel_msg)
        
        # Turn 90 degrees
        vel_msg.angular.z = 1.57  # ~90 degrees in radians
        pub.publish(vel_msg)
        
        # Wait for 1 second
        rate.sleep()
        
        # Stop turning
        vel_msg.angular.z = 0.0
        pub.publish(vel_msg)

draw_square()