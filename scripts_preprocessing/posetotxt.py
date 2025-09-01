import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.time import Time

import os
import time

class PoseSaver(Node):
    def __init__(self, timeout_sec=5.0):
        super().__init__('pose_saver')
        self.pose_file = open('pose.txt', 'w')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/davis/left/pose',
            self.listener_callback,
            10
        )
        self.get_logger().info('Subscribed to /davis/left/pose')
        self.last_msg_time = self.get_clock().now()
        self.timeout_duration = Duration(seconds=timeout_sec)
        self.create_timer(1.0, self.check_timeout)

    def listener_callback(self, msg):
        # Save the timestamp of the last received message
        self.last_msg_time = self.get_clock().now()

        # Format: timestamp x y z qx qy qz qw
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = msg.pose
        line = f"{t:.9f} {pose.position.x:.6f} {pose.position.y:.6f} {pose.position.z:.6f} " \
               f"{pose.orientation.x:.6f} {pose.orientation.y:.6f} {pose.orientation.z:.6f} {pose.orientation.w:.6f}\n"
        self.pose_file.write(line)
        self.pose_file.flush()

    def check_timeout(self):
        now = self.get_clock().now()
        if now - self.last_msg_time > self.timeout_duration:
            self.get_logger().info('No pose messages received for timeout duration. Shutting down.')
            rclpy.shutdown()

    def destroy_node(self):
        self.pose_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PoseSaver(timeout_sec=5.0)  # Stop if no message for 5 seconds
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
