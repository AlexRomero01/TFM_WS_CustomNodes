#!/usr/bin/env python3
"""
RANSAC.py
=========
ROS 2 node that fits a ground plane to each incoming depth frame using
RANSAC (via Open3D) and publishes the result for downstream nodes.

Subscriptions:
    /camera/camera/depth/image_rect_raw   sensor_msgs/Image      (16UC1, mm)
    /camera/camera/depth/camera_info      sensor_msgs/CameraInfo

Publications:
    /ransac_ground_mask      sensor_msgs/Image      (mono8) — ground pixels
    /ransac_plants_mask      sensor_msgs/Image      (mono8) — non-ground pixels
    /ransac_filtered_cloud   sensor_msgs/PointCloud2         — non-ground cloud
    /ransac_plane            std_msgs/Float32MultiArray      — [a, b, c, d] plane coefficients
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float32MultiArray
from sensor_msgs_py import point_cloud2
import numpy as np
import cv2
from cv_bridge import CvBridge
import open3d as o3d


class RANSACGroundRemoval(Node):
    def __init__(self):
        super().__init__('ransac_ground_removal')

        # RANSAC parameters
        self.ransac_distance_threshold = 0.05  # metres (5 cm inlier tolerance)
        self.ransac_n = 3                       # minimum points to fit the model
        self.num_iterations = 1000
        self.ground_angle_threshold = 15        # degrees from vertical

        # Subscriptions
        self.sub_depth = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        self.sub_info = self.create_subscription(
            CameraInfo,
            '/camera/camera/depth/camera_info',
            self.info_callback,
            10
        )

        # Publications
        self.pub_ground_mask = self.create_publisher(
            Image, '/ransac_ground_mask', 10)
        self.pub_plants_mask = self.create_publisher(
            Image, '/ransac_plants_mask', 10)
        self.pub_filtered_cloud = self.create_publisher(
            PointCloud2, '/ransac_filtered_cloud', 10)
        self.pub_plane_coeffs = self.create_publisher(
            Float32MultiArray, '/ransac_plane', 10)

        self.bridge = CvBridge()

        # Fallback intrinsics (RealSense D457 at 640×480).
        # Overwritten as soon as a CameraInfo message arrives.
        self.camera_info = CameraInfo()
        self.camera_info.header.frame_id = 'camera_depth_optical_frame'
        self.camera_info.height = 480
        self.camera_info.width = 640
        self.camera_info.distortion_model = 'plumb_bob'
        self.camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_info.k = [
            391.92132568359375, 0.0, 323.88165283203125,
            0.0, 391.92132568359375, 240.40322875976562,
            0.0, 0.0, 1.0
        ]
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_info.p = [
            391.92132568359375, 0.0, 323.88165283203125, 0.0,
            0.0, 391.92132568359375, 240.40322875976562, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        self.frame_count = 0

        self.get_logger().info(
            f"RANSACGroundRemoval node started\n"
            f"  Distance threshold : {self.ransac_distance_threshold} m\n"
            f"  Num iterations     : {self.num_iterations}"
        )

    def info_callback(self, msg: CameraInfo):
        """Store camera intrinsics received from the camera driver."""
        self.camera_info = msg

    def depth_callback(self, msg: Image):
        """Back-project the depth image to a point cloud and run RANSAC ground removal."""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if self.camera_info is None:
                self.get_logger().warn("Camera info not available yet")
                return

            pcd = self.depth_to_pointcloud(depth_image, self.camera_info)

            if len(pcd.points) == 0:
                self.get_logger().warn("Empty point cloud generated")
                return

            # Fit ground plane with RANSAC
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.ransac_distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )

            [a, b, c, d] = plane_model

            # Verify that the detected plane is roughly horizontal.
            # The normal vector's Z component should be dominant for a ground plane.
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            z_angle = np.degrees(np.arcsin(abs(normal[2])))

            # Distance from the camera origin to the ground plane: |d| / ||normal||
            plane_normal_magnitude = np.sqrt(a**2 + b**2 + c**2)
            ground_depth = abs(d) / plane_normal_magnitude

            points = np.asarray(pcd.points)
            ground_points = points[inliers]
            avg_z_depth = np.mean(ground_points[:, 2]) if len(ground_points) > 0 else 0

            self.get_logger().info(
                f"Frame {self.frame_count}: "
                f"plane normal angle from vertical: {z_angle:.1f}°, "
                f"ground depth (plane distance): {ground_depth:.3f} m, "
                f"ground depth (avg z): {avg_z_depth:.3f} m, "
                f"inliers: {len(inliers)}/{len(pcd.points)}"
            )

            # Build ground mask by re-projecting inlier points to 2-D
            ground_mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
            h, w = depth_image.shape

            # Map each valid pixel to its point-cloud index
            valid_indices = np.where(depth_image > 0)
            point_idx = 0
            pixel_to_point = {}
            for y, x in zip(valid_indices[0], valid_indices[1]):
                pixel_to_point[(x, y)] = point_idx
                point_idx += 1

            for point_idx in inliers:
                if point_idx < len(points):
                    point = points[point_idx]
                    u, v = self.pointcloud_to_pixel(point, self.camera_info)
                    if 0 <= u < w and 0 <= v < h:
                        ground_mask[v, u] = 255

            plants_mask = 255 - ground_mask
            plants_cloud = pcd.select_by_index(inliers, invert=True)

            # Publish plane coefficients for downstream volume estimation
            plane_msg = Float32MultiArray()
            plane_msg.data = [float(a), float(b), float(c), float(d)]
            self.pub_plane_coeffs.publish(plane_msg)

            ground_msg = self.bridge.cv2_to_imgmsg(ground_mask, encoding='mono8')
            ground_msg.header = msg.header
            self.pub_ground_mask.publish(ground_msg)

            plants_msg = self.bridge.cv2_to_imgmsg(plants_mask, encoding='mono8')
            plants_msg.header = msg.header
            self.pub_plants_mask.publish(plants_msg)

            if len(plants_cloud.points) > 0:
                cloud_msg = self.pointcloud_to_ros(plants_cloud, msg.header)
                self.pub_filtered_cloud.publish(cloud_msg)

            if self.frame_count % 100 == 0:
                self.get_logger().info(f"Using frame_id: {msg.header.frame_id}")

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing depth: {e}")

    def depth_to_pointcloud(self, depth_image, camera_info):
        """Convert a depth image to an Open3D PointCloud using camera intrinsics."""
        h, w = depth_image.shape
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)

        depth_m = depth_image.astype(np.float32) / 1000.0  # mm → m

        z = depth_m
        x_3d = (xx - cx) * z / fx
        y_3d = (yy - cy) * z / fy

        points = np.stack([x_3d.flatten(), y_3d.flatten(), z.flatten()], axis=1)
        valid_mask = (depth_m.flatten() > 0) & (depth_m.flatten() < 10)
        points = points[valid_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def pointcloud_to_pixel(self, point, camera_info):
        """Project a 3-D camera-frame point back to 2-D image coordinates."""
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        x, y, z = point
        if z <= 0:
            return -1, -1

        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        return u, v

    def pointcloud_to_ros(self, pcd, header):
        """Convert an Open3D PointCloud to a ROS 2 PointCloud2 message."""
        from sensor_msgs.msg import PointField
        points = np.asarray(pcd.points)

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12  # 3 × float32
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.data = points.astype(np.float32).tobytes()
        return cloud_msg


def main(args=None):
    rclpy.init(args=args)
    node = RANSACGroundRemoval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()