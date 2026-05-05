#!/usr/bin/env python3
"""
classical_rgb_segmentation_node.py
====================================
Classical vegetation segmentation pipeline using only RGB data.
Serves as a baseline comparison for the YOLO-based segmentation model.

Two independent methods run in parallel (configurable via 'method' parameter):

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  METHOD 1 — ExG (Excess Green Index)                                    │
  │    ExG = 2·G − R − B  →  Otsu binarization  →  morphological opening   │
  │    Strength : fast, simple.                                              │
  │    Weakness : sensitive to bright soil / lighting changes.               │
  │                                                                          │
  │  METHOD 2 — HSV Green Range                                             │
  │    BGR → HSV  →  inRange(H:[35,85], S:[sat_min,255], V:[0,255])        │
  │    Hue selects green, Saturation rejects grey/soil, Value is ignored.   │
  │    Strength : robust to shadow-induced brightness variation inside       │
  │               broccoli leaves (Value channel discarded).                 │
  │    Weakness : needs hue/saturation range tuning per environment.         │
  └─────────────────────────────────────────────────────────────────────────┘

Pre-processing (shared, applied before both methods):
  A light GaussianBlur is applied to the decoded BGR frame before feeding
  it to either pipeline.  This suppresses per-pixel sensor noise that would
  otherwise cause:
    • ExG : spurious Otsu splits on noisy flat surfaces
    • HSV : flickering hue values at leaf/shadow boundaries
  The kernel is configurable (blur_kernel_size, default 5×5).  Set to 1 to
  disable blur entirely (no-op passthrough).

Subscriptions:
    - /camera/camera/color/image_raw  (sensor_msgs/Image)

Publications (ExG method):
    - /perception/exg/mask            sensor_msgs/Image  (mono8)
    - /perception/exg/overlay         sensor_msgs/Image  (bgr8)
    - /perception/exg/latency_ms      std_msgs/Float32
    - /perception/exg/debug           sensor_msgs/Image  (mono8, optional)

Publications (HSV method):
    - /perception/hsv/mask            sensor_msgs/Image  (mono8)
    - /perception/hsv/overlay         sensor_msgs/Image  (bgr8)
    - /perception/hsv/latency_ms      std_msgs/Float32

Parameters
----------
  rgb_topic           str    Input RGB topic   (/camera/camera/color/image_raw)
  method              str    'exg' | 'hsv' | 'both'                   ('both')
  blur_kernel_size    int    GaussianBlur kernel (odd, 1 = disabled)       (5)
  morph_kernel_size   int    Morphological kernel side length (odd)        (5)
  overlay_alpha       float  Green-tint blend weight [0,1]             (0.45)
  publish_debug       bool   Publish raw ExG grayscale image          (False)
  queue_size          int    Sub/pub queue depth                          (10)

  --- ExG-specific ---
  (none beyond the shared parameters above)

  --- HSV-specific ---
  hsv_hue_low         int    Lower Hue bound in OpenCV units [0,179]    (35)
  hsv_hue_high        int    Upper Hue bound in OpenCV units [0,179]    (85)
  hsv_sat_min         int    Minimum Saturation to reject grey/soil     (40)

Author: Àlex Romero Segués  –  custom_nodes package
"""

import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class ClassicalRgbSegmentationNode(Node):
    """
    Classical RGB vegetation segmentation — two methods in one node.

    ExG  : Excess Green Index + Otsu auto-threshold
    HSV  : Hue-Saturation green range mask (Value channel discarded)

    Both run per frame when method='both', allowing direct latency/accuracy
    comparison by subscribing to their respective overlay topics in rqt.
    """

    # Overlay tint colours (BGR)
    _EXG_COLOUR = np.array([20, 220, 60],  dtype=np.uint8)   # vivid green
    _HSV_COLOUR = np.array([255, 100, 0],  dtype=np.uint8)   # cyan-blue

    # ------------------------------------------------------------------ init
    def __init__(self):
        super().__init__('classical_rgb_segmentation_node')

        # ── Declare parameters ────────────────────────────────────────────
        self.declare_parameter('rgb_topic',         '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',       '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('method',            'both')   # 'exg' | 'hsv' | 'both'
        # Pre-processing
        self.declare_parameter('blur_kernel_size',  5)        # GaussianBlur kernel (1 = off)
        # Morphological post-processing
        self.declare_parameter('morph_kernel_size', 5)
        self.declare_parameter('overlay_alpha',     0.45)
        self.declare_parameter('publish_debug',     False)
        self.declare_parameter('queue_size',        10)
        self.declare_parameter('sync_queue_size',   20)
        self.declare_parameter('sync_slop',         0.05)   # seconds
        # Depth ROI — only pixels in [depth_min_m, depth_max_m] are segmented
        self.declare_parameter('depth_min_m',       1.0)    # 100 cm
        self.declare_parameter('depth_max_m',       1.5)    # 150 cm
        # HSV-specific
        self.declare_parameter('hsv_hue_low',       35)
        self.declare_parameter('hsv_hue_high',      85)
        self.declare_parameter('hsv_sat_min',       40)

        # ── Read parameters ───────────────────────────────────────────────
        rgb_topic         = self.get_parameter('rgb_topic').value
        depth_topic       = self.get_parameter('depth_topic').value
        self._method      = str(self.get_parameter('method').value).lower()
        blur_k            = int(self.get_parameter('blur_kernel_size').value)
        self._kernel_size = int(self.get_parameter('morph_kernel_size').value)
        self._alpha       = float(np.clip(self.get_parameter('overlay_alpha').value, 0.0, 1.0))
        self._debug       = bool(self.get_parameter('publish_debug').value)
        queue_size        = int(self.get_parameter('queue_size').value)
        sync_queue_size   = int(self.get_parameter('sync_queue_size').value)
        sync_slop         = float(self.get_parameter('sync_slop').value)
        self._depth_min   = float(self.get_parameter('depth_min_m').value)  # metres
        self._depth_max   = float(self.get_parameter('depth_max_m').value)  # metres
        self._hue_low     = int(self.get_parameter('hsv_hue_low').value)
        self._hue_high    = int(self.get_parameter('hsv_hue_high').value)
        self._sat_min     = int(self.get_parameter('hsv_sat_min').value)

        if self._method not in ('exg', 'hsv', 'both'):
            self.get_logger().warn(
                f"Unknown method='{self._method}'. Falling back to 'both'."
            )
            self._method = 'both'

        # ── Blur kernel — must be positive and odd (1 = disabled) ────────
        blur_k = max(1, blur_k)
        if blur_k % 2 == 0:
            self.get_logger().warn(
                f"blur_kernel_size is even — incrementing to {blur_k + 1}."
            )
            blur_k += 1
        # Store as tuple for cv2.GaussianBlur; (1,1) is a guaranteed no-op
        self._blur_ksize: tuple[int, int] = (blur_k, blur_k)
        self._blur_enabled: bool = blur_k > 1

        # ── Morphological kernel — must be positive and odd ───────────────
        self._kernel_size = max(1, self._kernel_size)
        if self._kernel_size % 2 == 0:
            self.get_logger().warn(
                f"morph_kernel_size is even — incrementing to {self._kernel_size + 1}."
            )
            self._kernel_size += 1

        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self._kernel_size, self._kernel_size))

        # Pre-compute HSV bounds (constant across frames)
        self._hsv_lower = np.array([self._hue_low,  self._sat_min, 0],   dtype=np.uint8)
        self._hsv_upper = np.array([self._hue_high, 255,           255],  dtype=np.uint8)

        # ── cv_bridge ────────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── Subscribers (synchronized RGB + Depth) ───────────────────────
        # Store the latest valid depth mask so RGB frames never block waiting.
        self._latest_depth_mask: Optional[np.ndarray] = None

        _sub_rgb   = Subscriber(self, Image, rgb_topic)
        _sub_depth = Subscriber(self, Image, depth_topic)

        self._sync = ApproximateTimeSynchronizer(
            [_sub_rgb, _sub_depth],
            queue_size=sync_queue_size,
            slop=sync_slop,
        )
        self._sync.registerCallback(self._synced_callback)

        # ── Publishers — ExG ─────────────────────────────────────────────
        self._pub_exg_mask    = self.create_publisher(Image,   '/perception/exg/mask',       queue_size)
        self._pub_exg_overlay = self.create_publisher(Image,   '/perception/exg/overlay',    queue_size)
        self._pub_exg_latency = self.create_publisher(Float32, '/perception/exg/latency_ms', queue_size)
        self._pub_exg_debug: Optional[rclpy.publisher.Publisher] = None
        if self._debug:
            self._pub_exg_debug = self.create_publisher(Image, '/perception/exg/debug', queue_size)

        # ── Publishers — HSV ─────────────────────────────────────────────
        self._pub_hsv_mask    = self.create_publisher(Image,   '/perception/hsv/mask',       queue_size)
        self._pub_hsv_overlay = self.create_publisher(Image,   '/perception/hsv/overlay',    queue_size)
        self._pub_hsv_latency = self.create_publisher(Float32, '/perception/hsv/latency_ms', queue_size)

        # ── Runtime stats ────────────────────────────────────────────────
        self._frame_count       = 0
        _BUF                    = 100
        self._exg_latency_buf: list[float] = []
        self._hsv_latency_buf: list[float] = []
        self._BUF_SIZE          = _BUF

        self._heartbeat = self.create_timer(10.0, self._heartbeat_cb)

        _blur_label = (
            f"GaussianBlur {self._blur_ksize[0]}×{self._blur_ksize[1]}"
            if self._blur_enabled else "disabled (kernel=1)"
        )
        self.get_logger().info(
            "ClassicalRgbSegmentationNode started\n"
            f"  rgb_topic         : {rgb_topic}\n"
            f"  depth_topic       : {depth_topic}\n"
            f"  depth ROI         : [{self._depth_min:.2f} m, {self._depth_max:.2f} m]  "
            f"({self._depth_min*100:.0f}–{self._depth_max*100:.0f} cm)\n"
            f"  method            : {self._method}\n"
            f"  pre-blur          : {_blur_label}\n"
            f"  morph_kernel      : {self._kernel_size}×{self._kernel_size}\n"
            f"  overlay_alpha     : {self._alpha:.2f}\n"
            f"  HSV range Hue     : [{self._hue_low}, {self._hue_high}]\n"
            f"  HSV sat_min       : {self._sat_min}\n"
            f"  sync_slop         : {sync_slop} s\n"
            "  Topics (ExG) : /perception/exg/mask  /perception/exg/overlay  "
            "/perception/exg/latency_ms\n"
            "  Topics (HSV) : /perception/hsv/mask  /perception/hsv/overlay  "
            "/perception/hsv/latency_ms"
        )

    # ──────────────────────────────────────────── helpers ──────────────────

    @staticmethod
    def _latency_stats(buf: list[float]) -> str:
        if not buf:
            return "no data"
        arr = np.array(buf, dtype=np.float64)
        return (f"mean={arr.mean():.2f}  p50={np.median(arr):.2f}  "
                f"p95={np.percentile(arr, 95):.2f}  "
                f"min={arr.min():.2f}  max={arr.max():.2f}")

    def _heartbeat_cb(self):
        """Every 10 s — print alive status + rolling latency stats for both methods."""
        if self._frame_count == 0:
            self.get_logger().warn(
                "No frames processed yet! Is the RGB topic publishing?\n"
                f"  Check: ros2 topic hz {self.get_parameter('rgb_topic').value}"
            )
            return

        lines = [f"[heartbeat] frames={self._frame_count}"]
        if self._method in ('exg', 'both'):
            lines.append(f"  ExG latency (ms) : {self._latency_stats(self._exg_latency_buf)}")
        if self._method in ('hsv', 'both'):
            lines.append(f"  HSV latency (ms) : {self._latency_stats(self._hsv_latency_buf)}")
        lines.append("  → rqt_plot /perception/exg/latency_ms/data /perception/hsv/latency_ms/data")
        self.get_logger().info("\n".join(lines))

    # ──────────────────────────────────────────── main callback ────────────

    def _synced_callback(self, rgb_msg: Image, depth_msg: Image):
        """Fires when RGB and depth are synchronized. Decode both, run pipelines."""
        self._frame_count += 1

        # ── Decode RGB ───────────────────────────────────────────────────
        try:
            bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f"RGB decode error (frame {self._frame_count}): {exc}")
            return

        # ── Pre-processing : GaussianBlur ────────────────────────────────
        # Applied once here, before both ExG and HSV pipelines, so the cost
        # is paid only once per frame regardless of method='both'.
        # Suppresses per-pixel sensor noise that would otherwise cause:
        #   • ExG : spurious Otsu splits on noisy flat surfaces
        #   • HSV : flickering hue values at leaf / shadow boundaries
        if self._blur_enabled:
            bgr = cv2.GaussianBlur(bgr, self._blur_ksize, 0)

        # ── Decode Depth and build ROI mask ──────────────────────────────
        # RealSense depth is uint16 in millimetres. Convert to metres,
        # then keep only pixels inside [depth_min_m, depth_max_m].
        try:
            depth_raw = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth_m   = depth_raw.astype(np.float32) / 1000.0
            depth_mask = (
                (depth_m >= self._depth_min) &
                (depth_m <= self._depth_max) &
                (depth_raw > 0)
            ).astype(np.uint8) * 255  # uint8 binary, same convention as color masks

            # If depth resolution differs from RGB, resize to match
            if depth_mask.shape[:2] != bgr.shape[:2]:
                depth_mask = cv2.resize(
                    depth_mask,
                    (bgr.shape[1], bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            self._latest_depth_mask = depth_mask
        except Exception as exc:
            self.get_logger().warn(
                f"Depth decode error (frame {self._frame_count}): {exc} — using previous mask."
            )

        # Fall back to an all-pass mask if depth has never arrived
        depth_mask = self._latest_depth_mask
        if depth_mask is None:
            depth_mask = np.full(bgr.shape[:2], 255, dtype=np.uint8)

        stamp = rgb_msg.header.stamp
        frame = rgb_msg.header.frame_id

        if self._method in ('exg', 'both'):
            try:
                self._process_exg(bgr, depth_mask, stamp, frame)
            except Exception as exc:
                self.get_logger().error(f"ExG error (frame {self._frame_count}): {exc}")

        if self._method in ('hsv', 'both'):
            try:
                self._process_hsv(bgr, depth_mask, stamp, frame)
            except Exception as exc:
                self.get_logger().error(f"HSV error (frame {self._frame_count}): {exc}")

    # ──────────────────────────────────────── METHOD 1 : ExG ───────────────

    def _process_exg(self, bgr: np.ndarray, depth_mask: np.ndarray, stamp, frame: str):
        """
        Excess Green Index pipeline.

          ExG = 2G − R − B  →  Otsu binarization  →  morphological opening
          AND depth_mask  (keeps only pixels at 100–150 cm)
        """
        t0 = time.perf_counter()

        # ── 1. ExG ──────────────────────────────────────────────────────
        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        exg_uint8 = (np.clip(2 * g - r - b, 0, 510).astype(np.uint16) >> 1).astype(np.uint8)

        # ── 2. Otsu binarization ─────────────────────────────────────────
        _, binary = cv2.threshold(
            exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── 3. Morphological opening (noise removal) ──────────────────────
        color_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel)

        # ── 4. Apply depth ROI mask ───────────────────────────────────────
        mask = cv2.bitwise_and(color_mask, depth_mask)

        latency_ms = (time.perf_counter() - t0) * 1e3

        # ── Publish ───────────────────────────────────────────────────────
        self._publish_mask(mask, stamp, frame, self._pub_exg_mask)
        self._publish_overlay(
            bgr, mask, stamp, frame,
            self._pub_exg_overlay,
            tint_colour=self._EXG_COLOUR,
            method_label="ExG+Otsu",
            latency_ms=latency_ms,
        )
        self._publish_latency(latency_ms, self._pub_exg_latency, self._exg_latency_buf)

        if self._debug and self._pub_exg_debug is not None:
            dbg_msg = self._bridge.cv2_to_imgmsg(exg_uint8, encoding='mono8')
            dbg_msg.header.stamp    = stamp
            dbg_msg.header.frame_id = frame
            self._pub_exg_debug.publish(dbg_msg)

        self.get_logger().debug(
            f"[ExG] frame={self._frame_count:05d}  "
            f"crop={100.0 * mask.astype(bool).sum() / mask.size:.1f}%  "
            f"latency={latency_ms:.2f} ms"
        )

    # ──────────────────────────────────────── METHOD 2 : HSV ───────────────

    def _process_hsv(self, bgr: np.ndarray, depth_mask: np.ndarray, stamp, frame: str):
        """
        HSV green-range pipeline.

          BGR → HSV  →  cv2.inRange(H, S, V=any)  →  morphological opening
          AND depth_mask  (keeps only pixels at 100–150 cm)

          Value channel deliberately ignored → robust to leaf shadows.
        """
        t0 = time.perf_counter()

        # ── 1. BGR → HSV ──────────────────────────────────────────────────
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # ── 2. Green range mask ───────────────────────────────────────────
        raw_mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # ── 3. Morphological opening (noise removal) ──────────────────────
        color_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self._kernel)

        # ── 4. Apply depth ROI mask ───────────────────────────────────────
        mask = cv2.bitwise_and(color_mask, depth_mask)

        latency_ms = (time.perf_counter() - t0) * 1e3

        # ── Publish ───────────────────────────────────────────────────────
        self._publish_mask(mask, stamp, frame, self._pub_hsv_mask)
        self._publish_overlay(
            bgr, mask, stamp, frame,
            self._pub_hsv_overlay,
            tint_colour=self._HSV_COLOUR,
            method_label="HSV-green",
            latency_ms=latency_ms,
        )
        self._publish_latency(latency_ms, self._pub_hsv_latency, self._hsv_latency_buf)

        self.get_logger().debug(
            f"[HSV] frame={self._frame_count:05d}  "
            f"crop={100.0 * mask.astype(bool).sum() / mask.size:.1f}%  "
            f"latency={latency_ms:.2f} ms"
        )

    # ──────────────────────────────── shared publishing helpers ────────────

    def _publish_mask(self, mask: np.ndarray, stamp, frame: str, pub) -> None:
        """Publish a mono8 binary mask (0 / 255)."""
        msg = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
        msg.header.stamp    = stamp
        msg.header.frame_id = frame
        pub.publish(msg)

    def _publish_overlay(
        self,
        bgr: np.ndarray,
        mask: np.ndarray,
        stamp,
        frame: str,
        pub,
        tint_colour: np.ndarray,
        method_label: str,
        latency_ms: float,
    ) -> None:
        """
        Blend a solid tint over the foreground pixels, draw contours,
        and burn a stats label into the bottom-left corner.
        """
        fg      = mask > 0
        overlay = bgr.copy()

        # Colour blend: alpha * tint + (1-alpha) * original
        overlay[fg] = (
            self._alpha       * tint_colour.astype(np.float32)
            + (1.0 - self._alpha) * bgr[fg].astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        # Thin white contour outline around each detected region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

        # Burned-in label: method name, latency, crop coverage %
        fg_px    = int(fg.sum())
        total_px = mask.size
        pct      = 100.0 * fg_px / total_px if total_px > 0 else 0.0
        label    = f"{method_label}  {latency_ms:.1f} ms  |  crop={pct:.1f}%  ({fg_px} px)"
        cv2.putText(
            overlay, label,
            (8, overlay.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        msg = self._bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        msg.header.stamp    = stamp
        msg.header.frame_id = frame
        pub.publish(msg)

    def _publish_latency(
        self,
        latency_ms: float,
        pub,
        buf: list[float],
    ) -> None:
        """Publish a Float32 latency message and update the rolling buffer."""
        msg = Float32()
        msg.data = float(latency_ms)
        pub.publish(msg)
        buf.append(latency_ms)
        if len(buf) > self._BUF_SIZE:
            buf.pop(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ClassicalRgbSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # already shut down by rclpy internals on Ctrl-C


if __name__ == '__main__':
    main()
