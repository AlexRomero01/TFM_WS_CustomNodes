#!/usr/bin/env python3
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
        
        # Parámetros RANSAC
        self.ransac_distance_threshold = 0.05  # metros (5 cm)
        self.ransac_n = 3  # puntos para estimar el modelo
        self.num_iterations = 1000
        self.ground_angle_threshold = 15  # grados
        
        # Subscribers
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
        
        # Publishers
        self.pub_ground_mask = self.create_publisher(
            Image,
            '/ransac_ground_mask',
            10
        )
        
        self.pub_plants_mask = self.create_publisher(
            Image,
            '/ransac_plants_mask',
            10
        )
        
        self.pub_filtered_cloud = self.create_publisher(
            PointCloud2,
            '/ransac_filtered_cloud',
            10
        )

        self.pub_plane_coeffs = self.create_publisher(
            Float32MultiArray,
            '/ransac_plane',
            10
        )
        
        self.bridge = CvBridge()
        self.camera_info = None
        self.frame_count = 0
        
        self.get_logger().info(
            f"RANSACGroundRemoval node started\n"
            f"  Distance threshold: {self.ransac_distance_threshold}m\n"
            f"  Num iterations: {self.num_iterations}"
        )
    
    def info_callback(self, msg: CameraInfo):
        """Store camera intrinsics."""
        self.camera_info = msg
    
    def depth_callback(self, msg: Image):
        """Process depth image with RANSAC to remove ground."""
        try:
            # Convert to numpy
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            if self.camera_info is None:
                self.get_logger().warn("Camera info not available yet")
                return
            
            # Build point cloud from depth + intrinsics
            pcd = self.depth_to_pointcloud(depth_image, self.camera_info)
            
            if len(pcd.points) == 0:
                self.get_logger().warn("Empty point cloud generated")
                return
            
            # Apply RANSAC to detect ground plane
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.ransac_distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )
            
            # Extract plane coefficients (a, b, c, d where ax + by + cz + d = 0)
            [a, b, c, d] = plane_model
            
            # Check if plane is roughly horizontal (ground)
            # Normal vector is (a, b, c), z component should be dominant
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            z_angle = np.degrees(np.arcsin(abs(normal[2])))
            
            # Calculate ground depth from camera
            # Distance from origin (camera) to plane: |d| / ||normal||
            plane_normal_magnitude = np.sqrt(a**2 + b**2 + c**2)
            ground_depth = abs(d) / plane_normal_magnitude
            
            # Also calculate average z-coordinate of ground points for verification
            points = np.asarray(pcd.points)
            ground_points = points[inliers]
            avg_z_depth = np.mean(ground_points[:, 2]) if len(ground_points) > 0 else 0
            
            self.get_logger().info(
                f"Frame {self.frame_count}: Plane normal angle from vertical: {z_angle:.1f}°, "
                f"Ground depth (plane distance): {ground_depth:.3f}m, "
                f"Ground depth (avg z): {avg_z_depth:.3f}m, "
                f"Inliers: {len(inliers)}/{len(pcd.points)}"
            )
            
            # Create masks based on depth thresholding
            # Better approach: use the inliers directly to create masks
            ground_mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
            
            # Get points and build spatial index for faster lookup
            points = np.asarray(pcd.points)
            h, w = depth_image.shape
            
            # For each inlier point, mark its approximate location
            inliers_set = set(inliers)
            
            # Build a mapping of 2D pixels to point indices
            valid_indices = np.where(depth_image > 0)
            point_idx = 0
            pixel_to_point = {}
            
            for y, x in zip(valid_indices[0], valid_indices[1]):
                pixel_to_point[(x, y)] = point_idx
                point_idx += 1
            
            # Mark ground pixels (inliers) in the mask
            for point_idx in inliers:
                if point_idx < len(points):
                    point = points[point_idx]
                    u, v = self.pointcloud_to_pixel(point, self.camera_info)
                    if 0 <= u < w and 0 <= v < h:
                        ground_mask[v, u] = 255
            
            # Plants mask is inverse of ground
            plants_mask = 255 - ground_mask
            
            # Create filtered point cloud (without ground) - simply invert the selection
            plants_cloud = pcd.select_by_index(inliers, invert=True)
            
            # Publish plane coefficients [a, b, c, d] for downstream nodes
            plane_msg = Float32MultiArray()
            plane_msg.data = [float(a), float(b), float(c), float(d)]
            self.pub_plane_coeffs.publish(plane_msg)

            # Publish masks
            ground_msg = self.bridge.cv2_to_imgmsg(ground_mask, encoding='mono8')
            ground_msg.header = msg.header
            self.pub_ground_mask.publish(ground_msg)
            
            plants_msg = self.bridge.cv2_to_imgmsg(plants_mask, encoding='mono8')
            plants_msg.header = msg.header
            self.pub_plants_mask.publish(plants_msg)
            
            # Publish filtered point cloud
            if len(plants_cloud.points) > 0:
                # Use the depth frame from the camera, not 'map'
                header = msg.header
                cloud_msg = self.pointcloud_to_ros(plants_cloud, header)
                self.pub_filtered_cloud.publish(cloud_msg)
            
            if self.frame_count % 100 == 0:  # Log every 100 frames
                self.get_logger().info(f"Using frame_id: {msg.header.frame_id}")
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error processing depth: {e}")
    
    def depth_to_pointcloud(self, depth_image, camera_info):
        """Convert depth image to Open3D point cloud."""
        h, w = depth_image.shape
        
        # Get intrinsic parameters
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        
        # Create mesh grid
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        # Convert to meters (depth is usually in mm)
        depth_m = depth_image.astype(np.float32) / 1000.0
        
        # Calculate 3D coordinates
        z = depth_m
        x_3d = (xx - cx) * z / fx
        y_3d = (yy - cy) * z / fy
        
        # Flatten and stack
        points = np.stack([x_3d.flatten(), y_3d.flatten(), z.flatten()], axis=1)
        
        # Remove invalid points (z=0 or negative)
        valid_mask = (depth_m.flatten() > 0) & (depth_m.flatten() < 10)
        points = points[valid_mask]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
    
    def pointcloud_to_pixel(self, point, camera_info):
        """Project 3D point back to 2D image coordinates."""
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
        """Convert Open3D point cloud to ROS PointCloud2 message."""
        points = np.asarray(pcd.points)
        
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        
        # Create field descriptors for x, y, z
        from sensor_msgs.msg import PointField
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Pack point data
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