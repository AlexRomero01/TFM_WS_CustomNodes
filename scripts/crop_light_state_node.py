#!/usr/bin/env python3
"""
crop_light_state_node.py
=========================
ROS 2 node that classifies whether the camera is currently viewing a
sun-exposed or shaded crop area based on the brightness histogram of
the incoming RGB frame.

Classification rule:
  If the fraction of pixels above `bright_pixel_threshold` intensity
  exceeds `bright_fraction_threshold`, the state is "sun"; otherwise
  "shade".

Subscriptions:
    /camera/camera/color/image_raw   sensor_msgs/Image  (bgr8)

Publications:
    crop_light_state   std_msgs/String   ("sun" | "shade")
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np


class CropLightStateNode(Node):
    def __init__(self):
        super().__init__('crop_light_state_node')

        # Subscriber to the RGB camera from RealSense
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)

        # Publisher for crop light state ("sun" or "shade")
        self.publisher_ = self.create_publisher(String, 'crop_light_state', 10)

        self.bridge = CvBridge()

        # Decision parameters (adjustable during testing)
        self.bright_pixel_threshold = 130  # pixels above this intensity are considered "bright"
        self.bright_fraction_threshold = 0.35  # minimum fraction of bright pixels to classify as "sun"

        self.get_logger().info("CropLightStateNode (histogram-based) started.")

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Compute histogram of intensity values
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()  # normalize to probability distribution

        # Compute fraction of pixels above the brightness threshold
        bright_fraction = hist[self.bright_pixel_threshold:].sum()

        # Decide state based on bright fraction
        if bright_fraction > self.bright_fraction_threshold:
            state = "sun"
        else:
            state = "shade"

        # Publish result
        msg_out = String()
        msg_out.data = state
        self.publisher_.publish(msg_out)

        self.get_logger().info(
            f"Light state: {state} (bright_fraction={bright_fraction:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CropLightStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
