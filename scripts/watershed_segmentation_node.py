#!/usr/bin/env python3
"""
watershed_segmentation_node.py
=================================
Post-processing node that applies the Watershed algorithm on top of the
binary vegetation masks produced by the ClassicalRgbSegmentationNode.

For each method (ExG and HSV) the pipeline is:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  WATERSHED PIPELINE (per method)                                        │
  │                                                                         │
  │  binary mask (mono8, 0/255)                                             │
  │    │                                                                    │
  │    ├─ morphological opening  (remove noise)                             │
  │    ├─ sure background  = dilate(mask)                                   │
  │    ├─ distance transform  →  local maxima  →  sure foreground           │
  │    ├─ unknown region  = sure_bg − sure_fg                               │
  │    ├─ connected-component labels on sure_fg  →  marker image            │
  │    ├─ cv2.watershed on original RGB image                               │
  │    └─ coloured instance map published as bgr8 image                    │
  └─────────────────────────────────────────────────────────────────────────┘

Subscriptions:
    /perception/exg/mask      sensor_msgs/Image  (mono8)  — from ClassicalRgbSegmentation
    /perception/hsv/mask      sensor_msgs/Image  (mono8)  — from ClassicalRgbSegmentation
    /camera/camera/color/image_raw  sensor_msgs/Image  (bgr8)  — original RGB (for watershed)

Publications:
    /perception/exg/watershed          sensor_msgs/Image  (bgr8)  — colour-labelled plant instances
    /perception/hsv/watershed          sensor_msgs/Image  (bgr8)  — colour-labelled plant instances
    /perception/exg/watershed_overlay  sensor_msgs/Image  (bgr8)  — watershed blended on RGB
    /perception/hsv/watershed_overlay  sensor_msgs/Image  (bgr8)  — watershed blended on RGB

Parameters
----------
  rgb_topic          str   Input RGB topic                 (/camera/camera/color/image_raw)
  exg_mask_topic     str   ExG mask topic                  (/perception/exg/mask)
  hsv_mask_topic     str   HSV mask topic                  (/perception/hsv/mask)
  queue_size         int   Sub/pub queue depth                              (10)
  sync_queue_size    int   ApproximateTimeSynchronizer queue depth          (30)
  sync_slop          float Sync slop in seconds                            (0.1)
  dist_thresh_ratio  float Fraction of dist-transform max used as fg thresh (0.5)
  morph_kernel_size  int   Morphological kernel for pre/post processing       (3)
  min_area_px        int   Minimum connected-component area to keep          (50)
  overlay_alpha      float Blend weight of coloured labels over RGB [0,1]  (0.50)

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
# Colour palette for labelled instances (BGR, cyclic)
# ---------------------------------------------------------------------------
_PALETTE = np.array([
    [  0, 255,   0],   # vivid green
    [  0, 128, 255],   # orange
    [255,   0, 128],   # magenta-rose
    [  0, 200, 255],   # yellow
    [255,  50,  50],   # blue
    [128, 255,   0],   # lime
    [ 50,  50, 255],   # red
    [255, 255,   0],   # cyan
    [200,   0, 200],   # purple
    [  0, 180, 180],   # olive
], dtype=np.uint8)


def _apply_watershed(
    bgr: np.ndarray,
    mask: np.ndarray,
    dist_thresh_ratio: float,
    morph_kernel: np.ndarray,
    min_area_px: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Run the full watershed pipeline on a binary mask.

    Parameters
    ----------
    bgr              : Original BGR image (H, W, 3) — required by cv2.watershed.
    mask             : Binary mask (H, W) uint8, values 0/255.
    dist_thresh_ratio: Fraction of the distance transform maximum used as the
                       sure-foreground threshold (typically 0.4–0.6).
    morph_kernel     : Pre-built structuring element for morphological ops.
    min_area_px      : Connected components smaller than this (px) are ignored.

    Returns
    -------
    coloured    : BGR image with each plant labelled in a distinct colour
                  (black background — useful as a pure instance map).
    label_mask  : int32 array of per-pixel instance labels (0 = background,
                  -1 = border); used externally to build the RGB overlay.
    n_plants    : Number of detected plant instances.
    latency_ms  : Wall-clock time of this function in milliseconds.
    """
    t0 = time.perf_counter()

    h, w = mask.shape[:2]

    # ── 0. Ensure mask is binary uint8 ──────────────────────────────────────
    binary = (mask > 0).astype(np.uint8) * 255

    # ── 1. Morphological opening — remove pepper noise ──────────────────────
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  morph_kernel, iterations=2)

    # ── 2. Sure background — dilate slightly beyond the blobs ───────────────
    sure_bg = cv2.dilate(opening, morph_kernel, iterations=3)

    # ── 3. Distance transform → sure foreground ─────────────────────────────
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist < 1.0:
        # No foreground content; return plain black image + empty label mask
        black      = np.zeros((h, w, 3), dtype=np.uint8)
        label_mask = np.zeros((h, w),    dtype=np.int32)
        return black, label_mask, 0, (time.perf_counter() - t0) * 1e3

    thresh_val = dist_thresh_ratio * max_dist
    _, sure_fg = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # ── 4. Unknown region (border zone between sure_fg and sure_bg) ─────────
    unknown = cv2.subtract(sure_bg, sure_fg)

    # ── 5. Connected-component markers ──────────────────────────────────────
    n_labels, markers = cv2.connectedComponents(sure_fg)

    # Filter out tiny components before watershed
    for lbl in range(1, n_labels):
        if np.count_nonzero(markers == lbl) < min_area_px:
            markers[markers == lbl] = 0
    # Re-label contiguously after filtering
    _, markers = cv2.connectedComponents((markers > 0).astype(np.uint8))
    n_labels = markers.max() + 1  # includes background (0)

    # Shift markers by 1 so background is 1, unknown is 0
    markers = markers + 1
    markers[unknown > 0] = 0

    # ── 6. Watershed ─────────────────────────────────────────────────────────
    markers_ws = markers.copy()
    cv2.watershed(bgr, markers_ws)
    # watershed marks borders as -1

    # ── 7. Colour each instance ──────────────────────────────────────────────
    coloured = np.zeros((h, w, 3), dtype=np.uint8)
    n_plants = 0

    for lbl in range(2, markers_ws.max() + 1):   # label 1 = background, skip
        region = markers_ws == lbl
        if region.sum() < min_area_px:
            continue
        colour = _PALETTE[(lbl - 2) % len(_PALETTE)]
        coloured[region] = colour
        n_plants += 1

    # ── 8. Draw watershed borders in white ───────────────────────────────────
    border = markers_ws == -1
    coloured[border] = [255, 255, 255]

    latency_ms = (time.perf_counter() - t0) * 1e3
    return coloured, markers_ws, n_plants, latency_ms


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class WatershedSegmentationNode(Node):
    """
    Subscribes to ExG and HSV binary masks plus the original RGB image,
    then applies the Watershed algorithm to separate individual plant instances.
    Publishes a coloured instance map for each method.
    """

    def __init__(self):
        super().__init__('watershed_segmentation_node')

        # ── Declare parameters ────────────────────────────────────────────
        self.declare_parameter('rgb_topic',        '/camera/camera/color/image_raw')
        self.declare_parameter('exg_mask_topic',   '/perception/exg/mask')
        self.declare_parameter('hsv_mask_topic',   '/perception/hsv/mask')
        self.declare_parameter('queue_size',       10)
        self.declare_parameter('sync_queue_size',  30)
        self.declare_parameter('sync_slop',        0.1)
        self.declare_parameter('dist_thresh_ratio', 0.5)
        self.declare_parameter('morph_kernel_size', 3)
        self.declare_parameter('min_area_px',       50)
        self.declare_parameter('overlay_alpha',     0.50)

        # ── Read parameters ───────────────────────────────────────────────
        rgb_topic         = self.get_parameter('rgb_topic').value
        exg_mask_topic    = self.get_parameter('exg_mask_topic').value
        hsv_mask_topic    = self.get_parameter('hsv_mask_topic').value
        queue_size        = int(self.get_parameter('queue_size').value)
        sync_queue_size   = int(self.get_parameter('sync_queue_size').value)
        sync_slop         = float(self.get_parameter('sync_slop').value)
        self._dist_thresh = float(self.get_parameter('dist_thresh_ratio').value)
        k                 = int(self.get_parameter('morph_kernel_size').value)
        self._min_area    = int(self.get_parameter('min_area_px').value)
        self._alpha       = float(
            np.clip(self.get_parameter('overlay_alpha').value, 0.0, 1.0)
        )

        # Kernel must be a positive odd integer
        k = max(1, k)
        if k % 2 == 0:
            self.get_logger().warn(
                f"morph_kernel_size is even — incrementing to {k + 1}."
            )
            k += 1
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        # ── cv_bridge ────────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── Runtime stats ────────────────────────────────────────────────
        self._frame_count          = 0
        self._exg_latency_buf: list[float] = []
        self._hsv_latency_buf: list[float] = []
        _BUF = 100
        self._BUF_SIZE             = _BUF

        # ── Subscribers (synchronized RGB + ExG mask + HSV mask) ─────────
        _sub_rgb = Subscriber(self, Image, rgb_topic)
        _sub_exg = Subscriber(self, Image, exg_mask_topic)
        _sub_hsv = Subscriber(self, Image, hsv_mask_topic)

        self._sync = ApproximateTimeSynchronizer(
            [_sub_rgb, _sub_exg, _sub_hsv],
            queue_size=sync_queue_size,
            slop=sync_slop,
        )
        self._sync.registerCallback(self._synced_callback)

        # ── Publishers ───────────────────────────────────────────────────
        # Pure instance-colour maps (black background)
        self._pub_exg_ws = self.create_publisher(
            Image, '/perception/exg/watershed', queue_size)
        self._pub_hsv_ws = self.create_publisher(
            Image, '/perception/hsv/watershed', queue_size)
        # Overlay: watershed colours blended on original RGB
        self._pub_exg_overlay = self.create_publisher(
            Image, '/perception/exg/watershed_overlay', queue_size)
        self._pub_hsv_overlay = self.create_publisher(
            Image, '/perception/hsv/watershed_overlay', queue_size)
        # Latency diagnostics
        self._pub_exg_latency = self.create_publisher(
            Float32, '/perception/exg/watershed_latency_ms', queue_size)
        self._pub_hsv_latency = self.create_publisher(
            Float32, '/perception/hsv/watershed_latency_ms', queue_size)

        # ── Heartbeat ────────────────────────────────────────────────────
        self._heartbeat = self.create_timer(10.0, self._heartbeat_cb)

        self.get_logger().info(
            "WatershedSegmentationNode started\n"
            f"  rgb_topic         : {rgb_topic}\n"
            f"  exg_mask_topic    : {exg_mask_topic}\n"
            f"  hsv_mask_topic    : {hsv_mask_topic}\n"
            f"  sync_slop         : {sync_slop} s\n"
            f"  dist_thresh_ratio : {self._dist_thresh}\n"
            f"  morph_kernel      : {k}×{k}\n"
            f"  min_area_px       : {self._min_area}\n"
            f"  overlay_alpha     : {self._alpha:.2f}\n"
            "  Out (ExG) : /perception/exg/watershed"
            "  /perception/exg/watershed_overlay"
            "  /perception/exg/watershed_latency_ms\n"
            "  Out (HSV) : /perception/hsv/watershed"
            "  /perception/hsv/watershed_overlay"
            "  /perception/hsv/watershed_latency_ms"
        )

    # ──────────────────────────────────────── helpers ──────────────────────

    @staticmethod
    def _latency_stats(buf: list[float]) -> str:
        if not buf:
            return "no data"
        arr = np.array(buf, dtype=np.float64)
        return (
            f"mean={arr.mean():.2f}  p50={np.median(arr):.2f}  "
            f"p95={np.percentile(arr, 95):.2f}  "
            f"min={arr.min():.2f}  max={arr.max():.2f}"
        )

    def _heartbeat_cb(self):
        if self._frame_count == 0:
            self.get_logger().warn(
                "No frames processed yet! "
                "Check that /perception/exg/mask and /perception/hsv/mask are publishing."
            )
            return
        self.get_logger().info(
            f"[heartbeat] frames={self._frame_count}\n"
            f"  ExG watershed latency (ms) : {self._latency_stats(self._exg_latency_buf)}\n"
            f"  HSV watershed latency (ms) : {self._latency_stats(self._hsv_latency_buf)}"
        )

    def _make_overlay(
        self,
        bgr: np.ndarray,
        coloured: np.ndarray,
        markers_ws: np.ndarray,
        n_plants: int,
        latency_ms: float,
        label: str,
    ) -> np.ndarray:
        """
        Blend the coloured watershed map over the original RGB image.

        Pixels that belong to a plant instance (marker >= 2) are alpha-blended;
        border pixels (marker == -1) are drawn in white;
        background pixels (marker == 1) show the unmodified RGB.
        """
        overlay = bgr.copy()

        # Alpha-blend labelled pixels only (skip background label 1)
        plant_fg = (markers_ws >= 2)  # boolean mask of all plant pixels
        if plant_fg.any():
            overlay[plant_fg] = (
                self._alpha       * coloured[plant_fg].astype(np.float32)
                + (1.0 - self._alpha) * bgr[plant_fg].astype(np.float32)
            ).clip(0, 255).astype(np.uint8)

        # Draw watershed border lines in white
        border = markers_ws == -1
        overlay[border] = [255, 255, 255]

        # Draw contour outlines per instance for extra clarity
        for lbl in range(2, markers_ws.max() + 1):
            region_mask = ((markers_ws == lbl).astype(np.uint8) * 255)
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            colour = _PALETTE[(lbl - 2) % len(_PALETTE)].tolist()
            cv2.drawContours(overlay, contours, -1, colour, 2)

        # Burn-in label
        info = f"{label}  {n_plants} plant(s)  {latency_ms:.1f} ms"
        cv2.putText(
            overlay, info,
            (8, overlay.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        return overlay

    def _publish_watershed(
        self,
        bgr: np.ndarray,
        coloured: np.ndarray,
        markers_ws: np.ndarray,
        n_plants: int,
        latency_ms: float,
        stamp,
        frame: str,
        pub_image,
        pub_overlay,
        pub_latency,
        buf: list[float],
        label: str,
    ) -> None:
        """Publish the pure instance map, the RGB overlay, and the latency."""
        # ── 1. Pure instance-colour map (black background) ────────────────
        info = f"{label}  {n_plants} plant(s)  {latency_ms:.1f} ms"
        coloured_out = coloured.copy()
        cv2.putText(
            coloured_out, info,
            (8, coloured_out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        img_msg = self._bridge.cv2_to_imgmsg(coloured_out, encoding='bgr8')
        img_msg.header.stamp    = stamp
        img_msg.header.frame_id = frame
        pub_image.publish(img_msg)

        # ── 2. RGB overlay ────────────────────────────────────────────────
        overlay = self._make_overlay(
            bgr, coloured, markers_ws, n_plants, latency_ms, label
        )
        ov_msg = self._bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        ov_msg.header.stamp    = stamp
        ov_msg.header.frame_id = frame
        pub_overlay.publish(ov_msg)

        # ── 3. Latency ────────────────────────────────────────────────────
        lat_msg = Float32()
        lat_msg.data = float(latency_ms)
        pub_latency.publish(lat_msg)

        buf.append(latency_ms)
        if len(buf) > self._BUF_SIZE:
            buf.pop(0)

    # ──────────────────────────────────────── main callback ────────────────

    def _synced_callback(
        self,
        rgb_msg: Image,
        exg_msg: Image,
        hsv_msg: Image,
    ) -> None:
        """Fires when RGB, ExG mask and HSV mask are temporally aligned."""
        self._frame_count += 1

        stamp = rgb_msg.header.stamp
        frame = rgb_msg.header.frame_id

        # ── Decode original RGB ──────────────────────────────────────────
        try:
            bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(
                f"RGB decode error (frame {self._frame_count}): {exc}"
            )
            return

        # ── Decode ExG mask ──────────────────────────────────────────────
        try:
            exg_mask = self._bridge.imgmsg_to_cv2(exg_msg, desired_encoding='mono8')
        except Exception as exc:
            self.get_logger().error(
                f"ExG mask decode error (frame {self._frame_count}): {exc}"
            )
            exg_mask = None

        # ── Decode HSV mask ──────────────────────────────────────────────
        try:
            hsv_mask = self._bridge.imgmsg_to_cv2(hsv_msg, desired_encoding='mono8')
        except Exception as exc:
            self.get_logger().error(
                f"HSV mask decode error (frame {self._frame_count}): {exc}"
            )
            hsv_mask = None

        # ── Ensure mask dimensions match RGB (safety resize) ─────────────
        if exg_mask is not None and exg_mask.shape[:2] != bgr.shape[:2]:
            exg_mask = cv2.resize(
                exg_mask,
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if hsv_mask is not None and hsv_mask.shape[:2] != bgr.shape[:2]:
            hsv_mask = cv2.resize(
                hsv_mask,
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # ── ExG Watershed ────────────────────────────────────────────────
        if exg_mask is not None:
            try:
                coloured_exg, markers_exg, n_exg, lat_exg = _apply_watershed(
                    bgr, exg_mask,
                    self._dist_thresh,
                    self._kernel,
                    self._min_area,
                )
                self._publish_watershed(
                    bgr, coloured_exg, markers_exg, n_exg, lat_exg,
                    stamp, frame,
                    self._pub_exg_ws,
                    self._pub_exg_overlay,
                    self._pub_exg_latency,
                    self._exg_latency_buf,
                    "ExG+Watershed",
                )
                self.get_logger().debug(
                    f"[ExG] frame={self._frame_count:05d}  "
                    f"plants={n_exg}  latency={lat_exg:.2f} ms"
                )
            except Exception as exc:
                self.get_logger().error(
                    f"ExG watershed error (frame {self._frame_count}): {exc}"
                )

        # ── HSV Watershed ────────────────────────────────────────────────
        if hsv_mask is not None:
            try:
                coloured_hsv, markers_hsv, n_hsv, lat_hsv = _apply_watershed(
                    bgr, hsv_mask,
                    self._dist_thresh,
                    self._kernel,
                    self._min_area,
                )
                self._publish_watershed(
                    bgr, coloured_hsv, markers_hsv, n_hsv, lat_hsv,
                    stamp, frame,
                    self._pub_hsv_ws,
                    self._pub_hsv_overlay,
                    self._pub_hsv_latency,
                    self._hsv_latency_buf,
                    "HSV+Watershed",
                )
                self.get_logger().debug(
                    f"[HSV] frame={self._frame_count:05d}  "
                    f"plants={n_hsv}  latency={lat_hsv:.2f} ms"
                )
            except Exception as exc:
                self.get_logger().error(
                    f"HSV watershed error (frame {self._frame_count}): {exc}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = WatershedSegmentationNode()
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
