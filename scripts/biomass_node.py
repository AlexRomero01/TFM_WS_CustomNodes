#!/usr/bin/env python3
"""
biomass_node.py
================
ROS 2 node that estimates plant biomass (m³) from an RGB-D point cloud
using a type-specific volumetric heuristic.

For lettuce, the volume is derived from the downsampled surface point count
via a sphere-packing approximation.  For other crop types a simpler linear
scaling is used.  The voxel size controls the trade-off between accuracy
and computation time.

The node also publishes the configured crop type on /crop_type at 1 Hz so
that downstream nodes (MQTT publisher, CSV logger) can log it.

Subscriptions:
    /camera/camera/color/image_raw          sensor_msgs/Image
    /camera/camera/depth/image_rect_raw     sensor_msgs/Image
    /camera/camera/depth/camera_info        sensor_msgs/CameraInfo

Publications:
    /plant_biomass   std_msgs/Float32   (volumetric estimate in m³)
    /crop_type       std_msgs/String    (configured plant type, 1 Hz)
"""
import rclpy
import numpy as np
import open3d as o3d
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import os
import csv
from datetime import datetime

VOXEL_SIZE   = 0.01   # voxel side length in metres
PRINT_VOLUME = True   # log estimated volume to console
SAVE_DATA    = False  # save point clouds and CSV to disk
VISUALIZE   = False  # open Open3D viewer (blocks until closed)


class BiomassNode(Node):
    def __init__(self):
        super().__init__('biomass_node')

        # Plant type — change to 'other_crop' or any other string to use the generic heuristic
        self.plant_type = "lettuce"
        self.get_logger().info(f"Plant type: {self.plant_type}")

        self.count = 0
        self.name = "plant"
        self.save_dir = f"/tmp/biomass_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.save_dir, exist_ok=True)

        # Subscribers: color, depth, camera info
        self.sub_color = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.sub_depth = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/camera/depth/camera_info', self.info_callback, 10)

        # Publisher for biomass
        self.pub_biomass = self.create_publisher(Float32, '/plant_biomass', 10)

        # Publisher interno para /crop_type
        self.pub_crop_type = self.create_publisher(String, '/crop_type', 10)
        self.timer = self.create_timer(1.0, self.publish_crop_type)

        # publish once immediately and log so we can debug topic presence
        self.get_logger().info(f"Publishing initial crop_type='{self.plant_type}' on /crop_type")
        self.publish_crop_type()

        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.camera_info = None

    def publish_crop_type(self):
        msg = String()
        msg.data = self.plant_type
        self.pub_crop_type.publish(msg)
        self.get_logger().debug(f"Published /crop_type: {msg.data}")

    def color_callback(self, msg: Image):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_process()

    def depth_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.try_process()

    def info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def try_process(self):
        """Try processing when all data is available."""
        if self.color_image is None or self.depth_image is None or self.camera_info is None:
            return

        # Build point cloud from depth + color + intrinsics
        pcd = self.create_pointcloud(self.color_image, self.depth_image, self.camera_info)

        if len(pcd.points) == 0:
            self.get_logger().warn("Generated empty point cloud.")
            return

        if VISUALIZE:
            o3d.visualization.draw_geometries([pcd])

        # Estimate volume/biomass
        volume = self.estimate_volume(pcd, self.plant_type, VOXEL_SIZE)

        # Publish
        biomass_msg = Float32()
        biomass_msg.data = float(volume)
        self.pub_biomass.publish(biomass_msg)

        if PRINT_VOLUME:
            self.get_logger().info(
                f"\n##############################\n"
                f"Biomass: {volume:.6f} m³ ({volume * 1e6:.2f} cm³)\n"
                f"Plant type: {self.plant_type}\n"
                f"##############################"
            )

        # Optional: save
        if SAVE_DATA:
            filename = f"{self.save_dir}/plant_{self.count:04d}.pcd"
            o3d.io.write_point_cloud(filename, pcd)
            csv_file = f"{self.save_dir}/biomass.csv"
            self.update_csv(csv_file, self.count, self.plant_type, volume)

        self.count += 1

    def create_pointcloud(self, color_img, depth_img, cam_info):
        """Create Open3D PointCloud from RGB + Depth + CameraInfo."""
        fx = cam_info.k[0]
        fy = cam_info.k[4]
        cx = cam_info.k[2]
        cy = cam_info.k[5]

        # Depth image is in meters (RealSense outputs float32 or uint16)
        if depth_img.dtype != np.float32:
            depth_img = depth_img.astype(np.float32) / 1000.0  # if in mm → convert to meters

        rows, cols = depth_img.shape
        mask = depth_img > 0  # ignore invalid depths

        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        u = u[mask]
        v = v[mask]
        z = depth_img[mask]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Colors
        colors = color_img[v, u, :] / 255.0

        # Open3D cloud
        pts = np.vstack((x, y, z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def estimate_volume(self, pcd, plant_type, voxel_size):
        """Estimate plant volume depending on type."""
        down = pcd.voxel_down_sample(voxel_size=voxel_size)
        surface_points = len(down.points)

        if plant_type == "lettuce":
            # Heuristic formula for lettuce
            volume = (2 * surface_points) ** (3 / 2) / (6 * np.sqrt(np.pi))
        else:
            # Simpler formula for other crops
            volume = surface_points

        volume *= voxel_size ** 3
        return volume

    def update_csv(self, filename, count, plant_type, volume):
        row = {"index": count, "type": plant_type, "volume_m3": volume}
        exists = os.path.exists(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    node = BiomassNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
