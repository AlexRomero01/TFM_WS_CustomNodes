#!/usr/bin/env python3
"""
utm_base_link_xy.py
====================
ROS 2 node that reads the UTM → base_link TF transform at 1 Hz and
publishes the robot's UTM (x, y, z) position as a PointStamped message.

This provides a convenient metric position topic for the MQTT publisher
and CSV logger without requiring consumers to query the TF tree directly.

Publications:
    /tf_utm_baselink   geometry_msgs/PointStamped  (UTM frame, 1 Hz)
"""
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from rclpy.time import Time
import rclpy.duration
from geometry_msgs.msg import PointStamped


class SimpleTFReader(Node):
    def __init__(self):
        super().__init__('simple_tf_reader')

        # TF listener setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher for the new topic
        self.tf_pub = self.create_publisher(PointStamped, '/tf_utm_baselink', 10)

        # Timer to check TF every second
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1 Hz

        self.get_logger().info("SimpleTFReader started. Publishing on /tf_utm_baselink")

    def timer_callback(self):
        try:
            now = Time()
            transform = self.tf_buffer.lookup_transform(
                'utm',
                'base_link',
                now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            # Log info for debugging
            self.get_logger().info(f"x: {x:.3f}, y: {y:.3f}, z: {z:.3f}")

            # Publish as PointStamped
            msg = PointStamped()
            msg.header.frame_id = 'utm'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x = x
            msg.point.y = y
            msg.point.z = z
            self.tf_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f"TF not available: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTFReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
