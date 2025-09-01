#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from dvs_msgs.msg import EventArray
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class DVSToImageConverter(Node):
    def __init__(self):
        super().__init__('dvs_to_image_converter')
        
        # Parameters
        self.image_width = 346   # DAVIS346 width
        self.image_height = 240  # DAVIS346 height
        self.accumulation_time = 0.033  # 33ms (30 FPS equivalent)
        
        # Event accumulators for both cameras
        self.left_event_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.right_event_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.combined_event_image = np.zeros((self.image_height, self.image_width * 2, 3), dtype=np.uint8)
        self.last_reset_time = None
        
        # ROS setup
        self.bridge = CvBridge()
        
        # Subscribers for both left and right
        self.left_event_sub = self.create_subscription(
            EventArray,
            '/davis/left/events',
            self.left_event_callback,
            10
        )
        
        self.right_event_sub = self.create_subscription(
            EventArray,
            '/davis/right/events',
            self.right_event_callback,
            10
        )
        
        # Publishers
        self.left_image_pub = self.create_publisher(Image, '/davis/left/events_image', 10)
        self.right_image_pub = self.create_publisher(Image, '/davis/right/events_image', 10)
        self.combined_image_pub = self.create_publisher(Image, '/davis/combined/events_image', 10)
        
        # Timer for periodic image publishing
        self.timer = self.create_timer(self.accumulation_time, self.publish_images)
        
        # OpenCV window setup for direct display
        self.show_windows = True  # Set to False to disable OpenCV windows
        if self.show_windows:
            cv2.namedWindow('Left Events', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Right Events', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Combined Events', cv2.WINDOW_AUTOSIZE)
        
        self.get_logger().info('DVS Stereo to Image converter started - UPDATED VERSION')
        self.get_logger().info('Subscribed to left and right event topics')
    
    def left_event_callback(self, msg):
        """Process incoming left DVS events"""
        # Remove debug logging to reduce spam
        self._process_events(msg, self.left_event_image)
    
    def right_event_callback(self, msg):
        """Process incoming right DVS events"""
        # Remove debug logging to reduce spam  
        self._process_events(msg, self.right_event_image)
    
    def _process_events(self, msg, event_image):
        """Common event processing logic"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if self.last_reset_time is None:
            self.last_reset_time = current_time
        
        for event in msg.events:
            x, y = int(event.x), int(event.y)
            
            # Boundary check
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                if event.polarity:
                    # Positive events -> Red channel
                    event_image[y, x, 2] = min(255, event_image[y, x, 2] + 50)
                    event_image[y, x, 0] = 0  # Clear blue
                    event_image[y, x, 1] = 0  # Clear green
                else:
                    # Negative events -> Blue channel
                    event_image[y, x, 0] = min(255, event_image[y, x, 0] + 50)
                    event_image[y, x, 2] = 0  # Clear red
                    event_image[y, x, 1] = 0  # Clear green
    
    def publish_images(self):
        """Publish accumulated event images and reset"""
        current_time = self.get_clock().now().to_msg()
        
        # Publish left image
        if self.left_event_image is not None:
            ros_image_left = self.bridge.cv2_to_imgmsg(self.left_event_image, encoding='bgr8')
            ros_image_left.header.stamp = current_time
            self.left_image_pub.publish(ros_image_left)
        
        # Publish right image
        if self.right_event_image is not None:
            ros_image_right = self.bridge.cv2_to_imgmsg(self.right_event_image, encoding='bgr8')
            ros_image_right.header.stamp = current_time
            self.right_image_pub.publish(ros_image_right)
        
        # Create and publish combined side-by-side image
        self.combined_event_image[:, :self.image_width] = self.left_event_image
        self.combined_event_image[:, self.image_width:] = self.right_event_image
        
        ros_image_combined = self.bridge.cv2_to_imgmsg(self.combined_event_image, encoding='bgr8')
        ros_image_combined.header.stamp = current_time
        self.combined_image_pub.publish(ros_image_combined)
        
        # Display images in OpenCV windows if enabled
        if self.show_windows:
            # Only show if there are events to display
            if np.any(self.left_event_image):
                cv2.imshow('Left Events', self.left_event_image)
            if np.any(self.right_event_image):
                cv2.imshow('Right Events', self.right_event_image)
            if np.any(self.combined_event_image):
                cv2.imshow('Combined Events', self.combined_event_image)
            
            # Process OpenCV events (required for display)
            cv2.waitKey(1)
        
        # Decay the images (optional - creates trailing effect)
        self.left_event_image = (self.left_event_image * 0.9).astype(np.uint8)
        self.right_event_image = (self.right_event_image * 0.9).astype(np.uint8)
        
        # Or completely reset (uncomment for sharp frames)
        # self.left_event_image.fill(0)
        # self.right_event_image.fill(0)

class DVSImageSaver(Node):
    """Alternative: Save images to disk for both cameras"""
    def __init__(self):
        super().__init__('dvs_image_saver')
        
        # Parameters
        self.image_width = 346
        self.image_height = 240
        self.save_interval = 0.1  # Save every 100ms
        self.output_dir = "/tmp/dvs_images/"
        
        # Create output directories
        import os
        os.makedirs(f"{self.output_dir}left/", exist_ok=True)
        os.makedirs(f"{self.output_dir}right/", exist_ok=True)
        os.makedirs(f"{self.output_dir}combined/", exist_ok=True)
        
        # Event accumulators
        self.left_event_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.right_event_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.frame_count = 0
        
        # Subscribers
        self.left_event_sub = self.create_subscription(
            EventArray,
            '/davis/left/events',
            self.left_event_callback,
            10
        )
        
        self.right_event_sub = self.create_subscription(
            EventArray,
            '/davis/right/events',
            self.right_event_callback,
            10
        )
        
        # Timer for saving images
        self.timer = self.create_timer(self.save_interval, self.save_images)
        
        self.get_logger().info(f'DVS Stereo Image Saver started, saving to {self.output_dir}')
    
    def left_event_callback(self, msg):
        """Process incoming left DVS events"""
        self._process_events(msg, self.left_event_image)
    
    def right_event_callback(self, msg):
        """Process incoming right DVS events"""
        self._process_events(msg, self.right_event_image)
    
    def _process_events(self, msg, event_image):
        """Common event processing logic"""
        for event in msg.events:
            x, y = int(event.x), int(event.y)
            
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                if event.polarity:
                    # Positive events -> Red
                    event_image[y, x] = [0, 0, 255]  # BGR format
                else:
                    # Negative events -> Blue
                    event_image[y, x] = [255, 0, 0]  # BGR format
    
    def save_images(self):
        """Save current event images"""
        if np.any(self.left_event_image) or np.any(self.right_event_image):
            # Save left image
            if np.any(self.left_event_image):
                filename_left = f"{self.output_dir}left/frame_{self.frame_count:06d}.png"
                cv2.imwrite(filename_left, self.left_event_image)
            
            # Save right image
            if np.any(self.right_event_image):
                filename_right = f"{self.output_dir}right/frame_{self.frame_count:06d}.png"
                cv2.imwrite(filename_right, self.right_event_image)
            
            # Save combined side-by-side image
            combined_image = np.hstack([self.left_event_image, self.right_event_image])
            filename_combined = f"{self.output_dir}combined/frame_{self.frame_count:06d}.png"
            cv2.imwrite(filename_combined, combined_image)
            
            self.frame_count += 1
            
            # Reset images
            self.left_event_image.fill(0)
            self.right_event_image.fill(0)

def main(args=None):
    rclpy.init(args=args)
    
    # Choose one:
    # Option 1: Publish images as ROS topics
    node = DVSToImageConverter()
    
    # Option 2: Save images to disk
    # node = DVSImageSaver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()