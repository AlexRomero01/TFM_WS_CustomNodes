#!/usr/bin/env python3
"""
volume_estimation_node.py
==========================
ROS2 node for crop volume estimation via Depth-to-PointCloud back-projection,
RANSAC-based volumetric fill bounded by a height band, and voxelisation.

Pipeline
--------
  1. Back-project YOLO-masked depth pixels to 3-D camera-frame points.
  2. Voxelise + fill the volume between the RANSAC ground plane and the
     plant surface, keeping only voxels whose height above the ground is
     in (0, max_height_above_ground_m] (default 0.60 m).  This removes
     bare soil returns and anything taller than the limit (walls …).
  3. Publish the filled voxel cloud in GREEN in the raw camera optical frame
     (camera_depth_optical_frame, X-right Y-down Z-depth).  In RViz2 enable
     “Invert Z Axis” on the PointCloud2 display – same setting already used
     by the RANSAC cloud which works correctly.

Subscriptions (synchronized via ApproximateTimeSynchronizer):
    - Depth Image       : sensor_msgs/Image  (16-bit UC, depth in mm)
    - Segmented Mask    : sensor_msgs/Image  (YOLO binary output)
    - Camera Info       : sensor_msgs/CameraInfo
    - RANSAC Plane      : std_msgs/Float32MultiArray  [a, b, c, d] (ax+by+cz+d=0)

Publications:
    - /volume/marker_array   : visualization_msgs/MarkerArray  (CUBE markers for RViz2)
    - /volume/point_cloud    : sensor_msgs/PointCloud2          (XYZRGB, green)
    - /volume/estimate       : std_msgs/Float32                 (m³)

Parameters:
    depth_topic                 (str)   : depth image topic
    mask_topic                  (str)   : segmented mask topic
    camera_info_topic           (str)   : camera info topic
    ransac_topic                (str)   : RANSAC plane coefficients topic
    voxel_resolution            (float) : voxel edge length in metres  (default: 0.01)
    max_depth_m                 (float) : maximum valid depth in metres (default: 1.4)
    max_height_above_ground_m   (float) : height band cap above RANSAC plane (default: 0.60)
    max_fill_voxels             (int)   : per-column fill cap              (default: 500)
    sync_queue_size             (int)   : synchronizer queue size          (default: 10)
    sync_slop                   (float) : ApproximateTimeSynchronizer slop (default: 0.05)
    marker_frame_id             (str)   : TF frame for cloud and markers
    marker_lifetime_s           (float) : marker lifetime in seconds       (default: 0.5)

RViz2 setup
-----------
  Fixed Frame  = camera_depth_optical_frame
  PointCloud2 display on /volume/point_cloud:
    • Invert Z Axis : true       ← same as the RANSAC cloud display
    • Color Transformer : RGB8
    • Style : Flat Squares

Author: Àlex Romero Segués  –  custom_nodes package
"""
import struct
import math
import threading
from collections import deque
from typing import Optional

import rclpy
import numpy as np
import utm
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, NavSatFix
from std_msgs.msg import Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from builtin_interfaces.msg import Duration

# ---------------------------------------------------------------------------
# Helper: pack a (N,3) float32 array into a green XYZRGB PointCloud2 message
# ---------------------------------------------------------------------------
# Plant-green colour packed as a PCL-compatible float32 (0x00RRGGBB bit-cast).
_PLANT_GREEN_BYTES = struct.pack('<I', (0x00 << 24) | (0x32 << 16) | (0xC8 << 8) | 0x14)
# R=0x32=50  G=0xC8=200  B=0x14=20  → rich, saturated plant green
_PLANT_GREEN_F32   = struct.unpack('<f', _PLANT_GREEN_BYTES)[0]


def _points_to_cloud_msg(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Convert an (N,3) float32 array to a green XYZRGB PointCloud2 message."""
    n = len(points)
    # Build flat (N,4) float32 array: x, y, z, rgb_packed
    xyzrgb = np.empty((n, 4), dtype=np.float32)
    xyzrgb[:, :3] = points.astype(np.float32)
    xyzrgb[:, 3]  = _PLANT_GREEN_F32

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp    = stamp
    msg.height          = 1
    msg.width           = n
    msg.is_bigendian    = False
    msg.point_step      = 16   # 4 × float32
    msg.row_step        = msg.point_step * msg.width
    msg.is_dense        = True
    msg.fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = xyzrgb.tobytes()
    return msg


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
class VolumeEstimationNode(Node):
    """
    Back-projects YOLO-masked depth pixels to 3-D, fills the voxel volume
    between the crop surface and the RANSAC ground plane (height band 0–60 cm),
    and publishes the filled cloud in green plus a volume estimate.
    """

    # ------------------------------------------------------------------ init
    def __init__(self):
        super().__init__('volume_estimation_node')

        # ------------------------------------------------------------------
        # Declare ROS2 parameters
        # ------------------------------------------------------------------
        self.declare_parameter('depth_topic',        '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('mask_topic',         '/Temperature_and_CSWI/rescaled_yolo_masks')
        self.declare_parameter('gps_topic',          '/gps/fix')
        self.declare_parameter('camera_fov_length',  1.55)   # metres — Camera FoV Length
        self.declare_parameter('gps_gate_enabled',   True)
        self.declare_parameter('ransac_topic',       '/ransac_plane')
        self.declare_parameter('voxel_resolution',           0.03)   # metres  (3cm = 27x fewer voxels than 1cm)
        self.declare_parameter('min_depth_m',                1.0)    # metres  – near depth cut-off  (100 cm)
        self.declare_parameter('max_depth_m',                1.5)    # metres  – far  depth cut-off  (150 cm)
        self.declare_parameter('max_height_above_ground_m',  5.0)    # metres - set large to disable cap (debug)
        self.declare_parameter('max_fill_voxels',            100)    # voxels per column (safety cap)
        self.declare_parameter('marker_frame_id',            'volume_frame')
        self.declare_parameter('marker_lifetime_s',          0.5)    # seconds
        self.declare_parameter('process_every_nth_frame',    3)      # publish every 3rd sync (~3 Hz output)
        self.declare_parameter('max_cloud_points',           8000)   # cap points sent to RViz
        self.declare_parameter('publish_markers',            False)   # MarkerArray is slow; disable by default

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        depth_topic       = self.get_parameter('depth_topic').value
        mask_topic        = self.get_parameter('mask_topic').value
        gps_topic         = self.get_parameter('gps_topic').value
        self._fov_length  = float(self.get_parameter('camera_fov_length').value)
        self._gps_gate_enabled = bool(self.get_parameter('gps_gate_enabled').value)
        ransac_topic      = self.get_parameter('ransac_topic').value
        self.voxel_res        = float(self.get_parameter('voxel_resolution').value)
        self.min_depth        = float(self.get_parameter('min_depth_m').value)
        self.max_depth        = float(self.get_parameter('max_depth_m').value)
        self._max_height      = float(self.get_parameter('max_height_above_ground_m').value)
        self._max_fill_vox    = int(self.get_parameter('max_fill_voxels').value)
        self.marker_frame     = self.get_parameter('marker_frame_id').value
        marker_life           = float(self.get_parameter('marker_lifetime_s').value)
        self._nth_frame       = int(self.get_parameter('process_every_nth_frame').value)
        self._max_cloud_pts   = int(self.get_parameter('max_cloud_points').value)
        self._pub_markers_en  = bool(self.get_parameter('publish_markers').value)

        # Pre-compute marker lifetime as builtin_interfaces/Duration
        self._marker_lifetime = Duration()
        self._marker_lifetime.sec     = int(marker_life)
        self._marker_lifetime.nanosec = int((marker_life % 1.0) * 1e9)

        # ------------------------------------------------------------------
        # cv_bridge
        # ------------------------------------------------------------------
        self.bridge = CvBridge()

        # ------------------------------------------------------------------
        # RealSense D457 intrinsics (fx, fy, cx, cy)
        # ------------------------------------------------------------------
        self.fx = 391.92132568359375
        self.fy = 391.92132568359375
        self.cx = 323.88165283203125
        self.cy = 240.40322875976562

        # ------------------------------------------------------------------
        # GPS gate state
        # ------------------------------------------------------------------
        self._latest_gps:         Optional[NavSatFix] = None
        self._ref_easting:        Optional[float]     = None
        self._ref_northing:       Optional[float]     = None
        self._last_displacement_m: float              = 0.0
        self._gps_skipped:  int = 0
        self._gps_accepted: int = 0
        self._mask_frame_count: int = 0

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        if self._gps_gate_enabled:
            self.create_subscription(
                NavSatFix, gps_topic, self._gps_callback, 10)

        self._sub_mask = self.create_subscription(
            Image, mask_topic, self._mask_callback, 10)

        self.depth_buffer = deque(maxlen=1)
        self.depth_lock = threading.Lock()
        self._sub_depth = self.create_subscription(
            Image, depth_topic, self._depth_callback, 10)

        # RANSAC plane – separate subscription (updated whenever available)
        self._ransac_plane: np.ndarray | None = None  # [a, b, c, d]
        self._sub_ransac = self.create_subscription(
            Float32MultiArray,
            ransac_topic,
            self._ransac_callback,
            10,
        )

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self._pub_markers = self.create_publisher(
            MarkerArray, '/volume/marker_array', 10)
        self._pub_cloud = self.create_publisher(
            PointCloud2, '/volume/point_cloud', 10)
        self._pub_volume = self.create_publisher(
            Float32, '/volume/estimate', 10)

        # ------------------------------------------------------------------
        # Runtime statistics
        # ------------------------------------------------------------------
        self._frame_count        = 0
        self._ransac_plane_count = 0   # how many RANSAC plane msgs received

        # Heartbeat timer: logs every 10 s so it is always obvious whether the
        # synchronizer is firing or the node is silently idle.
        self._heartbeat = self.create_timer(10.0, self._heartbeat_cb)

        gate_label = (
            f"ENABLED  (FoV={self._fov_length:.2f} m, topic={gps_topic})"
            if self._gps_gate_enabled else "DISABLED"
        )
        self.get_logger().info(
            f"VolumeEstimationNode started\n"
            f"  depth_topic             : {depth_topic}\n"
            f"  mask_topic              : {mask_topic}\n"
            f"  GPS gate                : {gate_label}\n"
            f"  ransac_topic            : {ransac_topic}\n"
            f"  voxel_resolution        : {self.voxel_res} m\n"
            f"  max_height_above_ground : {self._max_height} m\n"
            f"  max_fill_voxels         : {self._max_fill_vox}\n"
            f"  depth range             : {self.min_depth} m – {self.max_depth} m\n"
            f"  process_every_nth_frame : {self._nth_frame}\n"
            f"  max_cloud_points        : {self._max_cloud_pts}\n"
            f"  publish_markers         : {self._pub_markers_en}\n"
            f"  frame_id                : {self.marker_frame}\n"
            f"  → Cloud published in {self.marker_frame} (raw camera frame)\n"
            f"    RViz2: Fixed Frame={self.marker_frame}, Invert Z Axis=true, Color=RGB8"
        )

    # --------------------------------------------------- Depth callback
    def _depth_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            with self.depth_lock:
                self.depth_buffer.append((depth_image, msg.header.stamp))
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")

    # --------------------------------------------------- GPS logic
    def _gps_callback(self, msg: NavSatFix) -> None:
        """Stores the latest GPS fix."""
        self._latest_gps = msg

    @staticmethod
    def _latlon_to_utm(lat: float, lon: float) -> tuple[float, float]:
        """Converts (lat, lon) degrees → UTM (easting, northing) in metres."""
        easting, northing, _zone_num, _zone_let = utm.from_latlon(lat, lon)
        return easting, northing

    def _should_process_frame(self) -> bool:
        """
        GPS gate decision.

        Returns True  → process the frame AND update the UTM reference.
        Returns False → skip the frame (Δd_RTK < Camera_FoV_Length).

        Fail-safes (always return True + warn):
          1. No NavSatFix received on /gps/fix.
          2. NavSatFix.status.status < 0 (STATUS_NO_FIX).
          3. UTM conversion fails.
        """
        gps = self._latest_gps

        if gps is None:
            if self._mask_frame_count % 100 == 1:
                self.get_logger().warn(
                    "GPS gate: no message on /gps/fix — segmenting anyway (fail-safe). "
                    "Check: ros2 topic hz /gps/fix"
                )
            return True

        if gps.status.status < 0:
            if self._mask_frame_count % 100 == 1:
                self.get_logger().warn(
                    f"GPS gate: NavSatFix status={gps.status.status} (NO_FIX) — "
                    "segmenting anyway (fail-safe)."
                )
            return True

        try:
            cur_e, cur_n = self._latlon_to_utm(gps.latitude, gps.longitude)
        except Exception as exc:
            self.get_logger().error(
                f"GPS gate: UTM conversion failed ({exc}) — segmenting anyway (fail-safe)."
            )
            return True

        if self._ref_easting is None or self._ref_northing is None:
            self._ref_easting  = cur_e
            self._ref_northing = cur_n
            self._last_displacement_m = 0.0
            self.get_logger().info(
                f"GPS gate: initial reference → E={cur_e:.2f} m  N={cur_n:.2f} m "
                f"(lat={gps.latitude:.7f}, lon={gps.longitude:.7f})"
            )
            return True

        dE = cur_e - self._ref_easting
        dN = cur_n - self._ref_northing
        delta_d = math.sqrt(dE * dE + dN * dN)
        self._last_displacement_m = delta_d

        if delta_d < self._fov_length:
            self.get_logger().debug(
                f"GPS gate: Δd={delta_d:.3f} m < FoV={self._fov_length:.2f} m — frame skipped."
            )
            return False

        self.get_logger().debug(
            f"GPS gate: Δd={delta_d:.3f} m >= FoV={self._fov_length:.2f} m — processing."
        )
        self._ref_easting  = cur_e
        self._ref_northing = cur_n
        return True

    # --------------------------------------------------- Mask callback
    def _mask_callback(self, msg: Image):
        """Processes depth and mask when a new mask arrives, subject to GPS gating."""
        self._mask_frame_count += 1
        
        # Frame-skip: only process every Nth frame to reduce CPU + RViz load
        if (self._mask_frame_count % self._nth_frame) != 0:
            return

        if self._gps_gate_enabled and not self._should_process_frame():
            self._gps_skipped += 1
            if self._mask_frame_count % 50 == 0:
                self.get_logger().info(
                    f"[GPS gate] Δd={self._last_displacement_m:.2f} m / {self._fov_length:.2f} m"
                    f" | accepted={self._gps_accepted}  skipped={self._gps_skipped}"
                )
            return
        
        self._gps_accepted += 1
        self._frame_count += 1

        try:
            mask_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            with self.depth_lock:
                depth_data = self.depth_buffer[0] if self.depth_buffer else None
                
            if depth_data is None:
                if self._frame_count % 30 == 0:
                    self.get_logger().warn("No depth data available yet - skipping frame.")
                return
                
            depth_raw, stamp = depth_data
            self._process(depth_raw, mask_raw, stamp)

        except Exception as exc:
            self.get_logger().error(f"Mask/Processing error (frame {self._frame_count}): {exc}")

    # --------------------------------------------------- Heartbeat timer
    def _heartbeat_cb(self):
        """Fires every 10 s so it is clear whether the node is working."""
        self.get_logger().info(
            f"[heartbeat] frames_processed={self._frame_count} | "
            f"ransac_msgs={self._ransac_plane_count} | "
            f"gps_accepted={self._gps_accepted} | gps_skipped={self._gps_skipped}"
        )
        if self._frame_count == 0:
            self.get_logger().warn(
                "  No frames processed yet!\n"
                "  Check if depth and mask topics are publishing."
            )

    # --------------------------------------------------- RANSAC callback
    def _ransac_callback(self, msg: Float32MultiArray):
        """Store the latest RANSAC plane coefficients [a, b, c, d]."""
        if len(msg.data) >= 4:
            self._ransac_plane = np.array(msg.data[:4], dtype=np.float64)
            self._ransac_plane_count += 1
            if self._ransac_plane_count == 1:
                a, b, c, d = self._ransac_plane
                self.get_logger().info(
                    f"\n★ First RANSAC plane received!\n"
                    f"  [a={a:.4f}  b={b:.4f}  c={c:.4f}  d={d:.4f}]\n"
                    f"  |c| = {abs(c):.4f}  (>0.2 → plane is reasonably horizontal)\n"
                    f"  z_ground at image centre ≈ {abs(d/c) if abs(c)>1e-4 else 999:.3f} m"
                )
        else:
            self.get_logger().warn(
                f"RANSAC message has {len(msg.data)} values; expected 4 [a,b,c,d]."
            )

    # Removed _synced_callback as we are no longer using ApproximateTimeSynchronizer.

    # --------------------------------------------------- Core processing
    def _process(
        self,
        depth_raw: np.ndarray,
        mask_img:  np.ndarray,
        stamp,
    ):
        """
        Volume estimation pipeline:
          1. Decode depth image.  Apply latest YOLO mask if available.
          2. Back-project masked pixels to 3-D camera-frame points.
          3. Voxelise + fill columns between the crop surface and the RANSAC
             ground plane, bounded to [0, max_height_above_ground_m].
          4. Publish estimated volume on /volume/estimate.
          5. Publish filled voxel cloud in GREEN on /volume/point_cloud.
        """
        # -- 1. Image decoding -----------------------------------------------
        depth_m = depth_raw.astype(np.float32) / 1000.0      # mm -> m

        if mask_img is not None:
            fg_mask  = (np.any(mask_img > 0, axis=2)
                        if mask_img.ndim == 3 else mask_img > 0)
            # Resize mask to depth dimensions if they differ
            if fg_mask.shape != depth_m.shape:
                import cv2 as _cv2
                fg_mask = _cv2.resize(
                    fg_mask.astype(np.uint8),
                    (depth_m.shape[1], depth_m.shape[0]),
                    interpolation=_cv2.INTER_NEAREST,
                ).astype(bool)
        else:
            # No mask yet - process the full depth image
            fg_mask = np.ones(depth_m.shape, dtype=bool)

        # ── 2. Camera intrinsics ─────────────────────────────────────────
        fx = self.fx; cx = self.cx
        fy = self.fy; cy = self.cy

        # ── 3. Back-project YOLO-masked pixels to 3D (vectorised) ────────
        # Only pixels inside the YOLO segmentation mask are used.
        surface_pts = self._depth_to_pointcloud(
            depth_m, fg_mask, fx, fy, cx, cy,
            min_depth=self.min_depth, max_depth=self.max_depth
        )
        if surface_pts.shape[0] == 0:
            self.get_logger().warn("No valid YOLO-masked depth points – skipping frame.")
            return

        # ── 4. Voxelise + fill to ground, bounded by height band ──────────
        n_surf_warn = (self._frame_count % 30 == 0)
        if self._ransac_plane is not None:
            voxel_centers = self._voxelize_and_fill(
                surface_pts,
                self._ransac_plane,
                self.voxel_res,
                self._max_fill_vox,
                self._max_height,
            )
        else:
            if n_surf_warn:
                self.get_logger().warn(
                    f"★ RANSAC plane NOT received yet (count={self._ransac_plane_count}).\n"
                    f"  → Showing surface voxels only (no volumetric fill).\n"
                    f"  Check: ros2 topic echo /ransac_plane --once"
                )
            voxel_centers = self._voxelize(surface_pts, self.voxel_res)

        voxel_count = voxel_centers.shape[0]

        if voxel_count == 0:
            self.get_logger().warn("Empty voxel set – skipping frame.")
            return

        # ── 5. Volume ─────────────────────────────────────────────────────
        volume_m3 = voxel_count * (self.voxel_res ** 3)

        if (self._frame_count % 30) == 0:
            self.get_logger().info(
                f"Frame {self._frame_count:05d} | "
                f"RANSAC msgs: {self._ransac_plane_count} | "
                f"voxels: {voxel_count:,} | "
                f"vol: {volume_m3*1e6:.1f} cm³"
            )

        vol_msg = Float32()
        vol_msg.data = float(volume_m3)
        self._pub_volume.publish(vol_msg)

        # -- 6. Publish green cloud (with point cap for RViz performance) ---
        # Randomly subsample if above max_cloud_points to keep RViz smooth.
        pub_pts = voxel_centers
        if len(pub_pts) > self._max_cloud_pts:
            idx = np.random.choice(len(pub_pts), self._max_cloud_pts, replace=False)
            pub_pts = pub_pts[idx]

        cloud_msg = _points_to_cloud_msg(
            pub_pts.astype(np.float32), self.marker_frame, stamp
        )
        self._pub_cloud.publish(cloud_msg)

        # -- 7. MarkerArray (optional, disabled by default - very slow) -----
        if self._pub_markers_en:
            marker_array = self._build_marker_array(voxel_centers, stamp, volume_m3)
            self._pub_markers.publish(marker_array)

    # ----------------------------------------- depth → point cloud helper
    @staticmethod
    def _depth_to_pointcloud(
        depth_m: np.ndarray,
        fg_mask: np.ndarray,
        fx: float, fy: float,
        cx: float, cy: float,
        min_depth: float = 0.05,
        max_depth: float = 10.0,
    ) -> np.ndarray:
        """
        Vectorized back-projection of masked pixels to 3D camera-frame points.

        Returns
        -------
        points : (N, 3) float32 array  [X, Y, Z] in metres
        """
        h, w = depth_m.shape

        # Pixel coordinate grids
        u_grid, v_grid = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )

        # Combined validity mask: foreground + valid depth range
        valid = fg_mask & (depth_m > min_depth) & (depth_m < max_depth)

        u = u_grid[valid]   # (N,)
        v = v_grid[valid]   # (N,)
        z = depth_m[valid]  # (N,)

        # Back-projection formulae
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return np.stack([x, y, z], axis=1).astype(np.float32)   # (N, 3)

    # ----------------------------------------- voxelisation (surface only)
    @staticmethod
    def _voxelize(points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Simple voxel-grid downsampling used as fallback (no RANSAC plane)."""
        vox = np.floor(points / voxel_size).astype(np.int32)
        uv  = np.unique(vox, axis=0)
        return (uv.astype(np.float32) + 0.5) * voxel_size

    # -------------------- voxelise + fill columns bounded by height band
    @staticmethod
    def _voxelize_and_fill(
        surface_pts: np.ndarray,
        plane: np.ndarray,
        voxel_size: float,
        max_fill_voxels: int = 500,
        max_height_above_ground: float = 0.60,
    ) -> np.ndarray:
        """
        Single-pass voxelise-and-fill that works entirely in voxel-index
        space (integers) – no dense float arrays, no memory explosion.

        Coordinate conventions (camera optical frame)
        ─────────────────────────────────────────────
          X = right,  Y = down,  Z = forward (larger Z = farther from camera)

        The RANSAC ground plane is at large Z (far).  Plant surfaces are at
        smaller Z (close).  "Height above ground" in camera-space is therefore
          h = z_gnd − z_point   (positive for points above the ground)

        Height band filter
        ───────────────────
        Only voxels in the range  0 < h ≤ max_height_above_ground  are kept:
          • voxels at h ≤ 0  are at or below the ground  → excluded
          • voxels at h > max_height_above_ground         → excluded
        This removes bare soil returns and objects taller than the limit
        (walls, machinery …), leaving only the low vegetation volume.

        Parameters
        ----------
        surface_pts             : (N, 3) float32  – plant surface [m, camera frame]
        plane                   : (4,)  float64   – RANSAC coefficients [a, b, c, d]
                                  where  a·x + b·y + c·z + d = 0
        voxel_size              : float  – voxel edge length [m]
        max_fill_voxels         : int    – per-column fill cap (safety guard)
        max_height_above_ground : float  – upper height band limit [m] (default 0.60)

        Returns
        -------
        centers : (M, 3) float32  – metric centres of every filled voxel [m]
        """
        a, b, c, d = plane

        # ── Plane sanity: need a reasonably horizontal plane ──────────────
        # For a forward/downward-facing camera the ground plane's c-component
        # (Z-component of the normal) must be significant.
        # Enforce c > 0 so that z_gnd = -(aX+bY+d)/c is a positive (forward) value.
        if abs(c) < 0.15:   # plane too vertical → fall back to surface voxels
            return VolumeEstimationNode._voxelize(surface_pts, voxel_size)
        if c < 0:
            a, b, c, d = -a, -b, -c, -d

        # ── Step 1 – voxelise surface ─────────────────────────────────────
        vox = np.floor(surface_pts / voxel_size).astype(np.int32)  # (N, 3)
        vx, vy, vz = vox[:, 0], vox[:, 1], vox[:, 2]

        # ── Step 2 – find minimum vz per (vx, vy) column ─────────────────
        # minimum vz = closest to camera = top of the plant surface
        OFFSET = 100_000
        col_key = (
            (vx.astype(np.int64) + OFFSET) * 200_000
            + (vy.astype(np.int64) + OFFSET)
        )
        sort_idx       = np.argsort(col_key)
        col_key_sorted = col_key[sort_idx]
        vz_sorted      = vz[sort_idx]

        unique_col_keys, first_in_sorted = np.unique(
            col_key_sorted, return_index=True
        )
        M = len(unique_col_keys)

        src_idx = sort_idx[first_in_sorted]
        vx_c    = vx[src_idx]   # (M,) representative vx per column
        vy_c    = vy[src_idx]   # (M,) representative vy per column
        # Surface = shallowest voxel (minimum vz = closest to camera = top of plant)
        vz_surf = np.minimum.reduceat(vz_sorted, first_in_sorted)  # (M,)

        # ── Step 3 – RANSAC ground voxel and height-band voxel per column ─
        x_m     = (vx_c.astype(np.float64) + 0.5) * voxel_size
        y_m     = (vy_c.astype(np.float64) + 0.5) * voxel_size

        # Ground depth for this (X, Y) ray – should be > z_surf (farther)
        z_gnd_m = -(a * x_m + b * y_m + d) / c

        # Upper limit: 60 cm above the ground (in camera Z, that means CLOSER)
        z_band_top_m = z_gnd_m - max_height_above_ground   # (M,) smaller Z

        # ── Step 4 – drop columns where ground is not reachably deeper ────
        z_surf_m = (vz_surf.astype(np.float64) + 0.5) * voxel_size
        bad_col  = z_gnd_m <= z_surf_m   # surface already at/below ground
        keep_col = ~bad_col
        if not np.any(keep_col):
            return np.zeros((0, 3), dtype=np.float32)

        # Restrict to valid columns
        vx_c    = vx_c[keep_col]
        vy_c    = vy_c[keep_col]
        vz_surf = vz_surf[keep_col]
        z_gnd_m      = z_gnd_m[keep_col]
        z_band_top_m = z_band_top_m[keep_col]

        # Convert ground and band-top to voxel indices
        vz_gnd      = np.floor(z_gnd_m      / voxel_size).astype(np.int32)
        vz_band_top = np.floor(z_band_top_m / voxel_size).astype(np.int32)

        # ── Step 5 – compute fill range [vz_lo .. vz_hi] ─────────────────
        # In camera Z: smaller index = closer to camera = higher above ground.
        # Fill range is [vz_surf .. vz_gnd-1] (exclude the ground voxel itself)
        # clipped to [vz_band_top .. vz_gnd-1] (height band cap).
        #
        #   vz_lo = max(vz_surf, vz_band_top)   ← farther from camera = lower
        #   vz_hi = vz_gnd - 1                  ← exclude bare ground voxel
        vz_lo = np.maximum(vz_surf,      vz_band_top)         # (M,)
        vz_hi = vz_gnd - 1                                    # (M,)

        # Drop columns where nothing survives the height band
        valid_range = vz_lo <= vz_hi
        if not np.any(valid_range):
            return np.zeros((0, 3), dtype=np.float32)

        vx_c    = vx_c[valid_range]
        vy_c    = vy_c[valid_range]
        vz_lo   = vz_lo[valid_range]
        vz_hi   = vz_hi[valid_range]

        fill_len = np.clip(
            vz_hi - vz_lo + 1, 1, max_fill_voxels
        ).astype(np.int32)
        vz_hi = vz_lo + fill_len - 1   # recompute after cap

        M_valid = len(vx_c)

        # ── Step 6 – vectorised column expansion (no Python loop) ─────────
        total      = int(fill_len.sum())
        ends       = np.cumsum(fill_len)
        seg_starts = np.concatenate([[0], ends[:-1]])

        vz_offsets = (
            np.arange(total, dtype=np.int32)
            - np.repeat(seg_starts.astype(np.int32), fill_len)
        )
        col_rep = np.repeat(np.arange(M_valid, dtype=np.int32), fill_len)

        vx_fill = vx_c[col_rep]
        vy_fill = vy_c[col_rep]
        vz_fill = vz_lo[col_rep] + vz_offsets

        # ── Step 7 – de-duplicate and convert to metric centres ───────────
        all_vox = np.stack([vx_fill, vy_fill, vz_fill], axis=1)   # (total, 3)
        all_vox = np.unique(all_vox, axis=0)
        return (all_vox.astype(np.float32) + 0.5) * voxel_size

    # ---------------------------------------- RViz2 MarkerArray builder
    def _build_marker_array(
        self,
        voxel_centers: np.ndarray,
        stamp,
        volume_m3: float,
    ) -> MarkerArray:
        """
        Build a MarkerArray with one DeleteAll marker followed by one CUBE
        marker per voxel.  Voxels are colour-coded by depth (Z value):
        green (near) → yellow → red (far).

        A final TEXT_VIEW_FACING marker displays the computed volume.
        """
        marker_array = MarkerArray()

        # ── DeleteAll: clear previous markers ────────────────────────────
        delete_all            = Marker()
        delete_all.header.frame_id = self.marker_frame
        delete_all.header.stamp    = stamp
        delete_all.ns              = 'volume'
        delete_all.id              = 0
        delete_all.action          = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        if voxel_centers.shape[0] == 0:
            return marker_array

        # Depth range for color mapping
        z_vals  = voxel_centers[:, 2]
        z_min   = float(z_vals.min())
        z_max   = float(z_vals.max())
        z_range = max(z_max - z_min, 1e-6)

        voxel_size = self.voxel_res

        for idx, (x, y, z) in enumerate(voxel_centers.tolist()):
            m = Marker()
            m.header.frame_id = self.marker_frame
            m.header.stamp    = stamp
            m.ns              = 'volume'
            m.id              = idx + 1   # +1 because id=0 is the DeleteAll
            m.type            = Marker.CUBE
            m.action          = Marker.ADD
            m.lifetime        = self._marker_lifetime

            # Position
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = float(z)
            m.pose.orientation.w = 1.0

            # Scale = voxel edge length
            m.scale.x = voxel_size
            m.scale.y = voxel_size
            m.scale.z = voxel_size

            # Colour: solid plant green (matches the PointCloud2 colour)
            m.color.r   = 50.0  / 255.0
            m.color.g   = 200.0 / 255.0
            m.color.b   = 20.0  / 255.0
            m.color.a   = 0.7

            marker_array.markers.append(m)

        # ── Volume text label ─────────────────────────────────────────────
        if voxel_centers.shape[0] > 0:
            centroid = voxel_centers.mean(axis=0)
            txt           = Marker()
            txt.header.frame_id = self.marker_frame
            txt.header.stamp    = stamp
            txt.ns              = 'volume_label'
            txt.id              = 0
            txt.type            = Marker.TEXT_VIEW_FACING
            txt.action          = Marker.ADD
            txt.lifetime        = self._marker_lifetime
            txt.pose.position.x = float(centroid[0])
            txt.pose.position.y = float(centroid[1]) - 0.05
            txt.pose.position.z = float(centroid[2])
            txt.pose.orientation.w = 1.0
            txt.scale.z         = 0.03   # text height in metres
            txt.color.r         = 1.0
            txt.color.g         = 1.0
            txt.color.b         = 1.0
            txt.color.a         = 1.0
            txt.text = f"{volume_m3 * 1e6:.1f} cm³"
            marker_array.markers.append(txt)

        return marker_array


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = VolumeEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
