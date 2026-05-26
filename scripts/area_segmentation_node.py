import json
import math
import threading
from collections import deque
from itertools import combinations
from typing import Optional

import cv2
import numpy as np
import rclpy
import utm
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Float32, String

class AreaSegmentNode(Node):
    def __init__(self):
        super().__init__('area_segment_node')
        self.bridge = CvBridge()

        # --- RealSense D457 Intrinsics ---
        self.fx = 391.92132568359375
        self.fy = 391.92132568359375
        self.cx = 323.88165283203125
        self.cy = 240.40322875976562
        # z_ground, depth_min, depth_max are now ROS2 parameters (see below)

        # --- Configurable parameters ---
        # Percentile used for max height (default 90 → discards the 10% noisiest)
        self.declare_parameter('height_max_percentile', 90.0)
        self._height_pct = float(self.get_parameter('height_max_percentile').value)

        # Valid depth range of the plant (cm from the camera).
        # IMPORTANT: z_ground must be >= depth_max_cm so that
        # z_ground - depth is never negative on valid pixels.
        self.declare_parameter('depth_min_cm',  100.0)   # minimum: closest plants
        self.declare_parameter('depth_max_cm',  150.0)   # maximum: furthest plants
        self.declare_parameter('z_ground_cm',   150.0)   # camera-to-ground distance (cm)
        #   ↑ Adjust z_ground_cm to the real height of your camera above the ground.
        #     Must be >= depth_max_cm to avoid negative heights.
        self._depth_min  = float(self.get_parameter('depth_min_cm').value)
        self._depth_max  = float(self.get_parameter('depth_max_cm').value)
        self.z_ground    = float(self.get_parameter('z_ground_cm').value)

        self.declare_parameter('spatial_calibration_factor', 0.9765)
        self.spatial_calib = float(self.get_parameter('spatial_calibration_factor').value)

        # --- GPS gate parameters ---
        self.declare_parameter('gps_topic',         '/gps/fix')
        self.declare_parameter('camera_fov_length', 1.55)   # metres — Camera FoV Length
        self.declare_parameter('gps_gate_enabled',  True)

        gps_topic              = str(self.get_parameter('gps_topic').value)
        self._fov_length       = float(self.get_parameter('camera_fov_length').value)
        self._gps_gate_enabled = bool(self.get_parameter('gps_gate_enabled').value)

        # --- GPS gate state ---
        # Rolling reference: updated every time a frame is accepted.
        # Works on second passes: the robot must travel FoV_Length from
        # the *last accepted segmentation*, regardless of the pass number.
        self._latest_gps:         Optional[NavSatFix] = None
        self._ref_easting:        Optional[float]     = None   # E₀ (m)
        self._ref_northing:       Optional[float]     = None   # N₀ (m)
        self._last_displacement_m: float              = 0.0
        self._gps_skipped:  int = 0
        self._gps_accepted: int = 0
        self._mask_frame_count: int = 0

        self.depth_buffer = deque(maxlen=1)
        self.depth_lock = threading.Lock()

        # --- Subscriptions ---
        if self._gps_gate_enabled:
            self.create_subscription(
                NavSatFix, gps_topic, self._gps_callback, 10)
        self.subscription = self.create_subscription(
            Image, '/Temperature_and_CSWI/rescaled_yolo_masks', self.mask_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)

        # --- Aggregated publishers (all plants combined) ---
        self.area_pub       = self.create_publisher(Float32, '/plant/projected_area_cm2',  10)
        self.diag_max_pub   = self.create_publisher(Float32, '/plant/diagonal_max_cm',     10)
        self.diag_min_pub   = self.create_publisher(Float32, '/plant/diagonal_min_cm',     10)
        self.diag_mean_pub  = self.create_publisher(Float32, '/plant/diagonal_mean_cm',    10)
        self.height_max_pub = self.create_publisher(Float32, '/plant/height_max_cm',       10)
        self.height_mean_pub= self.create_publisher(Float32, '/plant/height_mean_cm',      10)

        # --- Individual plant publisher (JSON String) ---
        self.per_plant_pub  = self.create_publisher(String, '/plant/geometric_measurements', 10)

        # --- Publisher: depth-filtered YOLO mask (100-150 cm) ---
        # Visualizable in RViz as Image (mono8). Only contains pixels whose
        # depth falls within the valid range of the plant.
        self.depth_filtered_mask_pub = self.create_publisher(
            Image, '/plant/yolo_depth_filtered_mask', 10)

        _gate_label = (
            f"ENABLED  (FoV={self._fov_length:.2f} m, topic={gps_topic})"
            if self._gps_gate_enabled else "DISABLED"
        )
        self.get_logger().info(
            f"2D Analysis Node (Convex Hull) Started\n"
            f"  height_max_percentile : {self._height_pct}  "
            f"(uses depth percentile p{self._height_pct:.0f} for max height)\n"
            f"  depth range           : [{self._depth_min:.0f}, {self._depth_max:.0f}] cm\n"
            f"  z_ground              : {self.z_ground:.0f} cm  "
            f"(must be >= depth_max_cm={self._depth_max:.0f} to avoid negative heights)\n"
            f"  GPS gate              : {_gate_label}"
        )
        if self.z_ground < self._depth_max:
            self.get_logger().warn(
                f"⚠️  z_ground_cm ({self.z_ground} cm) < depth_max_cm ({self._depth_max} cm)! "
                f"Plants at depth > {self.z_ground} cm will have height=0. "
                f"Adjust z_ground_cm or depth_max_cm in launch parameters."
            )


    # ------------------------------------------------------------------ depth
    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            with self.depth_lock:
                self.depth_buffer.append(depth_image)
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")

    # ------------------------------------------------------------------ GPS gate
    def _gps_callback(self, msg: NavSatFix) -> None:
        """Stores the last received GPS fix."""
        self._latest_gps = msg

    @staticmethod
    def _latlon_to_utm(lat: float, lon: float) -> tuple[float, float]:
        """Converts (lat, lon) degrees → UTM (easting, northing) in meters."""
        easting, northing, _zone_num, _zone_let = utm.from_latlon(lat, lon)
        return easting, northing

    def _should_process_frame(self) -> bool:
        """
        GPS gate decision.

        Returns True  → process frame AND update UTM reference.
        Returns False → skip frame (Δd_RTK < Camera_FoV_Length).

        Fail-safe (always returns True + warn):
          1. No NavSatFix received on /gps/fix.
          2. NavSatFix.status.status < 0 (STATUS_NO_FIX).
          3. UTM conversion fails for any reason.
        """
        gps = self._latest_gps

        # Fail-safe 1: no GPS message
        if gps is None:
            if self._mask_frame_count % 100 == 1:
                self.get_logger().warn(
                    "GPS gate: no message on /gps/fix — segmenting anyway (fail-safe). "
                    "Check: ros2 topic hz /gps/fix"
                )
            return True

        # Fail-safe 2: invalid fix
        if gps.status.status < 0:
            if self._mask_frame_count % 100 == 1:
                self.get_logger().warn(
                    f"GPS gate: NavSatFix status={gps.status.status} (NO_FIX) — "
                    "segmenting anyway (fail-safe)."
                )
            return True

        # UTM conversion
        try:
            cur_e, cur_n = self._latlon_to_utm(gps.latitude, gps.longitude)
        except Exception as exc:
            self.get_logger().error(
                f"GPS gate: UTM conversion failed ({exc}) — segmenting anyway (fail-safe)."
            )
            return True

        # First frame: save reference
        if self._ref_easting is None or self._ref_northing is None:
            self._ref_easting  = cur_e
            self._ref_northing = cur_n
            self._last_displacement_m = 0.0
            self.get_logger().info(
                f"GPS gate: initial reference → E={cur_e:.2f} m  N={cur_n:.2f} m "
                f"(lat={gps.latitude:.7f}, lon={gps.longitude:.7f})"
            )
            return True

        # Calculate Δd_RTK = sqrt((E_t-E₀)² + (N_t-N₀)²)
        dE = cur_e - self._ref_easting
        dN = cur_n - self._ref_northing
        delta_d = math.sqrt(dE * dE + dN * dN)
        self._last_displacement_m = delta_d

        if delta_d < self._fov_length:
            self.get_logger().debug(
                f"GPS gate: Δd={delta_d:.3f} m < FoV={self._fov_length:.2f} m — frame skipped."
            )
            return False

        # Sufficient displacement — update rolling reference
        self.get_logger().debug(
            f"GPS gate: Δd={delta_d:.3f} m ≥ FoV={self._fov_length:.2f} m — processing."
        )
        self._ref_easting  = cur_e
        self._ref_northing = cur_n
        return True

    # ------------------------------------------------------------------ mask
    def mask_callback(self, msg):
        self._mask_frame_count += 1

        # GPS gate — skip frame if the robot has not moved enough
        if self._gps_gate_enabled and not self._should_process_frame():
            self._gps_skipped += 1
            if self._mask_frame_count % 50 == 0:
                self.get_logger().info(
                    f"[GPS gate] Δd={self._last_displacement_m:.2f} m / {self._fov_length:.2f} m"
                    f" | accepted={self._gps_accepted}  skipped={self._gps_skipped}"
                )
            return
        self._gps_accepted += 1

        try:
            mask_uint8 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

            with self.depth_lock:
                depth_raw = self.depth_buffer[0] if self.depth_buffer else None

            has_depth = depth_raw is not None

            # 1. Mask ↔ depth alignment
            if has_depth and depth_raw.shape != mask_uint8.shape:
                depth_raw = cv2.resize(
                    depth_raw,
                    (mask_uint8.shape[1], mask_uint8.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 2. Depth filtering applied to the entire image
            #    ───────────────────────────────────────────────────────────────
            #    A full-image boolean mask is created with pixels that
            #    have valid depth, bitwise AND is performed with the YOLO mask,
            #    and published to /plant/yolo_depth_filtered_mask for RViz.
            if has_depth:
                depth_cm_full = depth_raw.astype(np.float32) / 10.0        # raw→cm
                depth_valid   = ((depth_cm_full >= self._depth_min) & (depth_cm_full <= self._depth_max))
                # Apply filter: zero-out YOLO pixels outside valid depth range
                filtered_mask = np.where(depth_valid, mask_uint8, 0).astype(np.uint8)
            else:
                # No depth available — pass the raw YOLO mask through unchanged
                filtered_mask = mask_uint8.copy()

            # Publish depth-filtered YOLO mask for RViz visualization
            try:
                filtered_msg = self.bridge.cv2_to_imgmsg(filtered_mask, encoding='mono8')
                filtered_msg.header = msg.header   # reuse original stamp + frame_id
                self.depth_filtered_mask_pub.publish(filtered_msg)
            except Exception as pub_err:
                self.get_logger().error(f"Error publishing depth_filtered_mask: {pub_err}")

            # 3. Contours — uses the depth-filtered mask
            raw_contours, _ = cv2.findContours(
                filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not raw_contours:
                return

            # ── Instance-level depth filter ────────────────────────────────
            # For each contour, compute the median depth of its interior pixels
            # and only keep instances whose median depth is within [depth_min_cm,
            # depth_max_cm].  This prevents partial / boundary contours that
            # survived the pixel-level mask from being analysed.
            if has_depth:
                contours = []
                discarded = 0
                for cnt in raw_contours:
                    # Build a single-contour mask
                    cnt_mask = np.zeros(depth_raw.shape[:2], dtype=np.uint8)
                    cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    cnt_pixels = depth_raw[cnt_mask > 0].astype(np.float32) / 10.0  # mm→cm
                    # Only consider non-zero (valid sensor) readings
                    valid_px = cnt_pixels[(cnt_pixels >= self._depth_min) & (cnt_pixels <= self._depth_max)]
                    if valid_px.size == 0:
                        discarded += 1
                        continue
                    median_depth = float(np.median(valid_px))
                    if self._depth_min <= median_depth <= self._depth_max:
                        contours.append(cnt)
                    else:
                        discarded += 1
                if discarded:
                    self.get_logger().debug(
                        f"[depth filter] kept={len(contours)}  "
                        f"discarded={discarded} contour(s) outside "
                        f"[{self._depth_min:.0f}, {self._depth_max:.0f}] cm"
                    )
                if not contours:
                    self.get_logger().debug(
                        "[depth filter] all contours discarded — no plants in depth range."
                    )
                    return
            else:
                contours = list(raw_contours)

            # ── Aggregate: all plants merged into one Convex Hull ──────────
            all_pts = np.concatenate(contours)
            hull_all = cv2.convexHull(all_pts)

            # Depth-dependent aggregate metrics
            depth_cm_all = np.array([])
            if has_depth:
                mask_pixels  = filtered_mask > 0          # only depth-valid plant pixels
                depth_cm_all = depth_raw[mask_pixels].astype(np.float32) / 10.0
                depth_cm_all = depth_cm_all[(depth_cm_all >= self._depth_min) & (depth_cm_all <= self._depth_max)]

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
                hull_pts_cm_all = np.array(hull_pts_cm_all, dtype=np.float32) * self.spatial_calib
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

                # Publish aggregated results
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
                    depth_p = depth_p[(depth_p >= self._depth_min) & (depth_p <= self._depth_max)]

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
                    hull_pts_p = np.array(hull_pts_p, dtype=np.float32) * self.spatial_calib
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
            self.get_logger().error(f"Error in mask_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = AreaSegmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()