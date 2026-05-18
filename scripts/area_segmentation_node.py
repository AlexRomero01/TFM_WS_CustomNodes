import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque
import threading
from itertools import combinations

class AreaSegmentNode(Node):
    def __init__(self):
        super().__init__('area_segment_node')
        self.bridge = CvBridge()

        # --- Intrínsecos RealSense D457 ---
        self.fx = 391.92132568359375
        self.fy = 391.92132568359375
        self.cx = 323.88165283203125
        self.cy = 240.40322875976562
        self.z_ground = 130.0  # Altura fija al suelo en cm

        # --- Parámetros configurables ---
        # Percentil usado para la altura máxima (default 90 → descarta el 10% más ruidoso)
        self.declare_parameter('height_max_percentile', 90.0)
        self._height_pct = float(self.get_parameter('height_max_percentile').value)

        self.depth_buffer = deque(maxlen=1)
        self.depth_lock = threading.Lock()

        self.subscription = self.create_subscription(
            Image, '/Temperature_and_CSWI/rescaled_yolo_masks', self.mask_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)

        # --- Publicadores agregados (todos los plantas en conjunto) ---
        self.area_pub       = self.create_publisher(Float32, '/plant/projected_area_cm2',  10)
        self.diag_max_pub   = self.create_publisher(Float32, '/plant/diagonal_max_cm',     10)
        self.diag_min_pub   = self.create_publisher(Float32, '/plant/diagonal_min_cm',     10)
        self.diag_mean_pub  = self.create_publisher(Float32, '/plant/diagonal_mean_cm',    10)
        self.height_max_pub = self.create_publisher(Float32, '/plant/height_max_cm',       10)
        self.height_mean_pub= self.create_publisher(Float32, '/plant/height_mean_cm',      10)

        # --- Publicador por planta individual (JSON String) ---
        self.per_plant_pub  = self.create_publisher(String, '/plant/geometric_measurements', 10)

        self.get_logger().info(
            f"Nodo de Análisis 2D (Convex Hull) Iniciado\n"
            f"  height_max_percentile : {self._height_pct}  "
            f"(usa el p{self._height_pct:.0f} de la profundidad para altura máxima)"
        )

    # ------------------------------------------------------------------ depth
    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            with self.depth_lock:
                self.depth_buffer.append(depth_image)
        except Exception as e:
            self.get_logger().error(f"Error en depth_callback: {e}")

    # ------------------------------------------------------------------ mask
    def mask_callback(self, msg):
        try:
            mask_uint8 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

            with self.depth_lock:
                depth_raw = self.depth_buffer[0] if self.depth_buffer else None

            has_depth = depth_raw is not None

            # 1. Alineación máscara ↔ profundidad
            if has_depth and depth_raw.shape != mask_uint8.shape:
                depth_raw = cv2.resize(
                    depth_raw,
                    (mask_uint8.shape[1], mask_uint8.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 2. Contornos de la máscara YOLO
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return

            # ── Aggregate: all plants merged into one Convex Hull ──────────
            all_pts = np.concatenate(contours)
            hull_all = cv2.convexHull(all_pts)

            # Depth-dependent aggregate metrics
            depth_cm_all = np.array([])
            if has_depth:
                mask_pixels = mask_uint8 > 0
                depth_cm_all = depth_raw[mask_pixels].astype(np.float32) / 10.0
                depth_cm_all = depth_cm_all[(depth_cm_all >= 100.0) & (depth_cm_all <= 150.0)]

            if not has_depth or depth_cm_all.size == 0:
                if not has_depth:
                    self.get_logger().warn("No depth data available; publishing 2D-only measurements.", throttle_duration_sec=5.0)

                # Publish 2D-only aggregate measurements (pixel area in cm² is not meaningful without z, skip)
                real_area_hull = None
                diag_max = diag_min = diag_mean = None
                height_mean = height_max = None
                z_mean_all = None
            else:
                heights_all = self.z_ground - depth_cm_all
                z_mean_all  = float(np.mean(depth_cm_all))

                h_pos_all = heights_all[heights_all > 0]
                if h_pos_all.size > 0:
                    height_mean = float(np.mean(h_pos_all))
                    height_max  = float(np.percentile(h_pos_all, self._height_pct))
                else:
                    height_mean = 0.0
                    height_max  = 0.0

                hull_pts_cm_all = []
                for pt in hull_all:
                    u, v = pt[0]
                    hull_pts_cm_all.append([
                        (u - self.cx) * z_mean_all / self.fx,
                        (v - self.cy) * z_mean_all / self.fy,
                    ])
                hull_pts_cm_all = np.array(hull_pts_cm_all, dtype=np.float32)
                real_area_hull = cv2.contourArea(hull_pts_cm_all)

                n_verts = len(hull_pts_cm_all)
                if n_verts >= 2:
                    idx = np.array(list(combinations(range(n_verts), 2)))
                    diffs = hull_pts_cm_all[idx[:, 0]] - hull_pts_cm_all[idx[:, 1]]
                    diag_lengths = np.linalg.norm(diffs, axis=1)
                    diag_max  = float(diag_lengths.max())
                    diag_min  = float(diag_lengths.min())
                    diag_mean = float(diag_lengths.mean())
                else:
                    diag_max = diag_min = diag_mean = 0.0

                # Publicar resultados agregados
                self.area_pub.publish(       Float32(data=float(real_area_hull)))
                self.diag_max_pub.publish(   Float32(data=diag_max))
                self.diag_min_pub.publish(   Float32(data=diag_min))
                self.diag_mean_pub.publish(  Float32(data=diag_mean))
                self.height_max_pub.publish( Float32(data=height_max))
                self.height_mean_pub.publish(Float32(data=height_mean))

            # ── Per-plant: process each contour individually ───────────────
            per_plant_list = []
            # Sort contours by area (largest first) so IDs are stable frame-to-frame
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for plant_idx, contour in enumerate(sorted_contours):
                if len(contour) < 3:   # need at least 3 points for a hull
                    continue

                hull_p = cv2.convexHull(contour)

                # Depth pixels inside this specific plant contour
                plant_mask = np.zeros_like(mask_uint8)
                cv2.drawContours(plant_mask, [contour], -1, 255, thickness=cv2.FILLED)
                plant_pixels = plant_mask > 0

                depth_p = np.array([])
                if has_depth:
                    depth_p = depth_raw[plant_pixels].astype(np.float32) / 10.0
                    depth_p = depth_p[(depth_p >= 100.0) & (depth_p <= 150.0)]

                if depth_p.size > 0:
                    z_mean_p = float(np.mean(depth_p))
                    heights_p = self.z_ground - depth_p
                    h_pos_p   = heights_p[heights_p > 0]

                    if h_pos_p.size > 0:
                        p_height_mean = float(np.mean(h_pos_p))
                        p_height_max  = float(np.percentile(h_pos_p, self._height_pct))
                    else:
                        p_height_mean = 0.0
                        p_height_max  = 0.0

                    # Project hull vertices to real-world cm
                    hull_pts_p = []
                    for pt in hull_p:
                        u, v = pt[0]
                        hull_pts_p.append([
                            (u - self.cx) * z_mean_p / self.fx,
                            (v - self.cy) * z_mean_p / self.fy,
                        ])
                    hull_pts_p = np.array(hull_pts_p, dtype=np.float32)
                    p_area = round(float(cv2.contourArea(hull_pts_p)), 2)

                    n_p = len(hull_pts_p)
                    if n_p >= 2:
                        idx_p = np.array(list(combinations(range(n_p), 2)))
                        diffs_p = hull_pts_p[idx_p[:, 0]] - hull_pts_p[idx_p[:, 1]]
                        dl_p = np.linalg.norm(diffs_p, axis=1)
                        p_diag_max  = round(float(dl_p.max()), 2)
                        p_diag_min  = round(float(dl_p.min()), 2)
                    else:
                        p_diag_max = p_diag_min = 0.0

                    # Volume proxy (cm³): area_cm2 × height_max_cm
                    p_volume = round(p_area * p_height_max, 2)
                else:
                    # No valid depth data for this plant: use pixel area as fallback
                    p_area = round(float(cv2.contourArea(hull_p)), 2)  # pixel area (no depth scale)
                    p_height_mean = p_height_max = None
                    p_diag_max = p_diag_min = None
                    p_volume = None

                per_plant_list.append({
                    "id":             f"Plant_{plant_idx + 1}",
                    "area_cm2":       p_area,
                    "diag_max_cm":    p_diag_max,
                    "diag_min_cm":    p_diag_min,
                    "height_max_cm":  p_height_max,
                    "height_mean_cm": p_height_mean,
                    "volume_cm3":     p_volume,
                    "depth_available": depth_p.size > 0,
                })

            if per_plant_list:
                payload = json.dumps({
                    "msg_type":   "geometric_measurements",
                    "plant_count": len(per_plant_list),
                    "plants":     per_plant_list,
                })
                self.per_plant_pub.publish(String(data=payload))

            self.get_logger().info(
                f"[Aggregate] Area: {real_area_hull:.1f} cm²  |  "
                f"Diag max/min: {diag_max:.1f}/{diag_min:.1f} cm  |  "
                f"Height max(p{self._height_pct:.0f})/mean: {height_max:.1f}/{height_mean:.1f} cm  |  "
                f"Plants detected: {len(per_plant_list)}"
            )

        except Exception as e:
            self.get_logger().error(f"Error en mask_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = AreaSegmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()