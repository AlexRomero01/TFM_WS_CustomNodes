#!/usr/bin/env python3
"""
yolo_measure_tool.py
=====================
Interactive OpenCV measurement tool that renders on top of the live
/yolo_image topic.

Features
--------
  • YOLO image displayed as background (all detections already drawn)
  • LEFT-CLICK + DRAG  → ruler line with real-world length in cm
  • RIGHT-CLICK        → add a polygon vertex (snapped to cursor)
  • Press  'c'         → close / finish polygon (shows area cm² + perimeter cm)
  • Press  'r'         → clear all ruler lines
  • Press  'p'         → clear polygon
  • Press  'q' / ESC  → quit

Subscriptions
-------------
  /yolo_image                           sensor_msgs/Image  (BGR8) — YOLO viz
  /camera/camera/depth/image_rect_raw   sensor_msgs/Image  (16UC1, mm)

Measurements use depth back-projection with RealSense D457 intrinsics.

Author: Àlex Romero Segués — custom_nodes package
"""

import threading
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

# ──────────────────────────────────────────────────────────────────────────────
# Appearance constants
# ──────────────────────────────────────────────────────────────────────────────
RULER_COLOR      = (0,   220, 255)   # vivid cyan
RULER_DONE_COLOR = (255, 160,  0)    # amber  (finished rulers)
POLY_VERTEX_COLOR= (50,  255, 100)   # bright green
POLY_EDGE_COLOR  = (50,  255, 100)
POLY_FILL_COLOR  = (50,  255, 100)
POLY_FILL_ALPHA  = 0.25
LABEL_BG_ALPHA   = 0.55
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.55
FONT_THICK       = 1
HUD_COLOR        = (200, 200, 200)   # light grey for HUD text

# ──────────────────────────────────────────────────────────────────────────────
# Depth helper
# ──────────────────────────────────────────────────────────────────────────────
_DEPTH_WIN = 5   # pixel neighbourhood radius for median depth sampling


def _sample_depth_cm(depth_raw: np.ndarray, u: int, v: int) -> float:
    """Return median depth in cm from a small window around (u, v).
    Returns 0.0 if no valid pixel is found (depth = 0 means invalid for D4xx)."""
    h, w = depth_raw.shape
    u0, u1 = max(0, u - _DEPTH_WIN), min(w, u + _DEPTH_WIN + 1)
    v0, v1 = max(0, v - _DEPTH_WIN), min(h, v + _DEPTH_WIN + 1)
    patch = depth_raw[v0:v1, u0:u1].astype(np.float32)
    valid = patch[(patch > 100) & (patch < 30000)]   # 10 cm – 30 m range (mm)
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) / 10.0   # mm → cm


SPATIAL_CALIB_FACTOR = 0.9765

def _project_pixel(u: int, v: int, z_cm: float,
                   fx: float, fy: float, cx: float, cy: float
                   ) -> np.ndarray:
    """Back-project pixel (u,v) + depth z_cm → 3-D camera-frame point in cm."""
    x = ((u - cx) * z_cm / fx) * SPATIAL_CALIB_FACTOR
    y = ((v - cy) * z_cm / fy) * SPATIAL_CALIB_FACTOR
    return np.array([x, y, z_cm], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────
def _draw_label(img: np.ndarray, text: str, pos, color=(255, 255, 255),
                bg=(30, 30, 30)):
    """Draw a text label with a semi-transparent background."""
    x, y = int(pos[0]), int(pos[1])
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    pad = 4
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  bg, -1)
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, img, 1 - LABEL_BG_ALPHA, 0, img)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, color, FONT_THICK,
                cv2.LINE_AA)


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 Node
# ──────────────────────────────────────────────────────────────────────────────
class YoloMeasureNode(Node):
    """Subscribes to /yolo_image and depth; hosts interactive measurement UI."""

    # ── RealSense D457 intrinsics (from camera/camera/color/camera_info) ──────
    FX = 391.92132568359375
    FY = 391.92132568359375
    CX = 323.88165283203125
    CY = 240.40322875976562

    def __init__(self):
        super().__init__('yolo_measure_tool')
        self.bridge = CvBridge()
        self._lock  = threading.Lock()

        # ── Latest frames ────────────────────────────────────────────────────
        self._yolo_img:  np.ndarray | None = None   # BGR8
        self._depth_raw: np.ndarray | None = None   # 16UC1 (mm)

        # ── Ruler state ──────────────────────────────────────────────────────
        self._rulers: list[tuple] = []   # list of (pt1, pt2, length_cm)
        self._ruler_drawing = False
        self._ruler_pt1 = (0, 0)
        self._ruler_pt2 = (0, 0)

        # ── Polygon state ────────────────────────────────────────────────────
        self._poly_pts: list[tuple[int, int]] = []  # pixel vertices
        self._poly_closed = False
        self._poly_area_cm2: float = 0.0
        self._poly_perim_cm: float = 0.0
        self._cursor = (0, 0)            # for live preview line

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._yolo_cb, 10)
        self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self._depth_cb, 10)

        self.get_logger().info(
            "YoloMeasureTool ready.\n"
            "  LEFT-CLICK+DRAG  → draw ruler (length in cm)\n"
            "  RIGHT-CLICK      → add polygon vertex\n"
            "  'c'              → close polygon\n"
            "  'r'              → clear rulers\n"
            "  'p'              → clear polygon\n"
            "  'q' / ESC        → quit"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _yolo_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._lock:
                self._yolo_img = img
        except Exception as e:
            self.get_logger().error(f"yolo_image decode: {e}")

    def _depth_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self._lock:
                self._depth_raw = img
        except Exception as e:
            self.get_logger().error(f"depth decode: {e}")

    # ── Mouse callback ────────────────────────────────────────────────────────
    def mouse_cb(self, event, x, y, flags, param):
        self._cursor = (x, y)

        # ── LEFT button: ruler ───────────────────────────────────────────────
        if event == cv2.EVENT_LBUTTONDOWN:
            self._ruler_drawing = True
            self._ruler_pt1 = (x, y)
            self._ruler_pt2 = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._ruler_drawing:
            self._ruler_pt2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self._ruler_drawing:
            self._ruler_drawing = False
            self._ruler_pt2 = (x, y)
            length = self._measure_line(self._ruler_pt1, self._ruler_pt2)
            if length > 0:
                self._rulers.append((self._ruler_pt1, self._ruler_pt2, length))
            self._ruler_pt1 = self._ruler_pt2   # reset preview

        # ── RIGHT button: polygon vertex ─────────────────────────────────────
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._poly_closed:
                return  # locked until user presses 'p'
            self._poly_pts.append((x, y))

    # ── Metric computation ───────────────────────────────────────────────────
    def _get_depth_at(self, u: int, v: int) -> float:
        """Thread-safe depth sample (cm). Returns 0 if unavailable."""
        with self._lock:
            if self._depth_raw is None:
                return 0.0
            depth = self._depth_raw
        # Remap pixel if depth and yolo_image resolutions differ
        with self._lock:
            yimg = self._yolo_img
        if yimg is not None and depth.shape[:2] != yimg.shape[:2]:
            dh, dw = depth.shape[:2]
            yh, yw = yimg.shape[:2]
            u2 = int(u * dw / yw)
            v2 = int(v * dh / yh)
        else:
            u2, v2 = u, v
        return _sample_depth_cm(depth, u2, v2)

    def _measure_line(self, pt1, pt2) -> float:
        """Return Euclidean 3-D distance (cm) between two image pixels."""
        z1 = self._get_depth_at(*pt1)
        z2 = self._get_depth_at(*pt2)
        if z1 <= 0 or z2 <= 0:
            # Fall back to pixel distance if no depth
            return 0.0
        p1 = _project_pixel(*pt1, z1, self.FX, self.FY, self.CX, self.CY)
        p2 = _project_pixel(*pt2, z2, self.FX, self.FY, self.CX, self.CY)
        return float(np.linalg.norm(p1 - p2))

    def close_polygon(self):
        """Project polygon vertices and compute area + perimeter."""
        pts = self._poly_pts
        if len(pts) < 3:
            self.get_logger().warn("Need at least 3 vertices to close a polygon.")
            return

        pts_3d = []
        for u, v in pts:
            z = self._get_depth_at(u, v)
            if z <= 0:
                self.get_logger().warn(f"No depth at vertex ({u},{v}), using last valid depth.")
                z = pts_3d[-1][2] if pts_3d else 100.0
            pts_3d.append(_project_pixel(u, v, z, self.FX, self.FY, self.CX, self.CY))

        pts_3d = np.array(pts_3d, dtype=np.float32)   # (N, 3)
        pts_xy  = pts_3d[:, :2]                         # use XY plane for area

        self._poly_area_cm2 = float(cv2.contourArea(pts_xy))

        # Perimeter = sum of 3D edge lengths (closed)
        perim = 0.0
        n = len(pts_3d)
        for i in range(n):
            perim += float(np.linalg.norm(pts_3d[i] - pts_3d[(i + 1) % n]))
        self._poly_perim_cm = perim
        self._poly_closed   = True

    # ── Rendering ─────────────────────────────────────────────────────────────
    def _render_rulers(self, canvas: np.ndarray):
        """Draw all finished rulers and the active one being dragged."""
        # Finished rulers
        for pt1, pt2, length in self._rulers:
            cv2.line(canvas, pt1, pt2, RULER_DONE_COLOR, 2, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 4, RULER_DONE_COLOR, -1)
            cv2.circle(canvas, pt2, 4, RULER_DONE_COLOR, -1)
            mid = _midpoint(pt1, pt2)
            _draw_label(canvas, f"{length:.1f} cm", (mid[0]+6, mid[1]-6),
                        color=RULER_DONE_COLOR)

        # Active (being dragged)
        if self._ruler_drawing:
            cv2.line(canvas, self._ruler_pt1, self._ruler_pt2,
                     RULER_COLOR, 2, cv2.LINE_AA)
            cv2.circle(canvas, self._ruler_pt1, 4, RULER_COLOR, -1)
            cv2.circle(canvas, self._ruler_pt2, 4, RULER_COLOR, -1)
            # Live length estimate
            z1 = self._get_depth_at(*self._ruler_pt1)
            z2 = self._get_depth_at(*self._ruler_pt2)
            if z1 > 0 and z2 > 0:
                p1 = _project_pixel(*self._ruler_pt1, z1, self.FX, self.FY,
                                    self.CX, self.CY)
                p2 = _project_pixel(*self._ruler_pt2, z2, self.FX, self.FY,
                                    self.CX, self.CY)
                live_len = float(np.linalg.norm(p1 - p2))
                mid = _midpoint(self._ruler_pt1, self._ruler_pt2)
                _draw_label(canvas, f"{live_len:.1f} cm", (mid[0]+6, mid[1]-6),
                            color=RULER_COLOR)

    def _render_polygon(self, canvas: np.ndarray):
        """Draw in-progress or closed polygon with metrics."""
        pts = self._poly_pts
        if not pts:
            return

        # Vertices
        for p in pts:
            cv2.circle(canvas, p, 5, POLY_VERTEX_COLOR, -1, cv2.LINE_AA)

        # Edges between consecutive vertices
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], POLY_EDGE_COLOR, 2, cv2.LINE_AA)

        if self._poly_closed:
            # Closing edge
            cv2.line(canvas, pts[-1], pts[0], POLY_EDGE_COLOR, 2, cv2.LINE_AA)

            # Semi-transparent fill
            overlay = canvas.copy()
            arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [arr], POLY_FILL_COLOR)
            cv2.addWeighted(overlay, POLY_FILL_ALPHA, canvas,
                            1 - POLY_FILL_ALPHA, 0, canvas)

            # Metric label at centroid
            cx_poly = int(np.mean([p[0] for p in pts]))
            cy_poly = int(np.mean([p[1] for p in pts]))
            _draw_label(canvas,
                        f"Area: {self._poly_area_cm2:.1f} cm\xb2",
                        (cx_poly - 60, cy_poly - 14),
                        color=POLY_FILL_COLOR)
            _draw_label(canvas,
                        f"Perim: {self._poly_perim_cm:.1f} cm",
                        (cx_poly - 60, cy_poly + 12),
                        color=POLY_FILL_COLOR)
        else:
            # Live preview: line from last vertex to cursor
            cv2.line(canvas, pts[-1], self._cursor,
                     POLY_EDGE_COLOR, 1, cv2.LINE_AA)

    def _render_hud(self, canvas: np.ndarray):
        """Top-left help overlay."""
        lines = [
            "L-drag: ruler  |  R-click: poly vertex  |  c: close poly",
            "r: clear rulers  |  p: clear polygon  |  q/ESC: quit",
        ]
        y = 18
        for line in lines:
            _draw_label(canvas, line, (8, y), color=HUD_COLOR, bg=(10, 10, 10))
            y += 22

    def render_frame(self) -> np.ndarray | None:
        """Compose the full annotated frame. Returns None if no image yet."""
        with self._lock:
            if self._yolo_img is None:
                return None
            canvas = self._yolo_img.copy()

        self._render_rulers(canvas)
        self._render_polygon(canvas)
        self._render_hud(canvas)
        return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = YoloMeasureNode()

    # Spin ROS in a background thread so OpenCV runs on the main thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    WIN = "YOLO Measurement Tool"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, node.mouse_cb)

    try:
        while rclpy.ok():
            frame = node.render_frame()
            if frame is None:
                # Show a waiting splash until the first frame arrives
                splash = np.zeros((480, 640, 3), dtype=np.uint8)
                _draw_label(splash, "Waiting for /yolo_image ...",
                            (60, 240), color=(180, 180, 180))
                cv2.imshow(WIN, splash)
            else:
                cv2.imshow(WIN, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):        # q or ESC
                break
            elif key == ord('r'):
                node._rulers.clear()
                node.get_logger().info("Rulers cleared.")
            elif key == ord('p'):
                node._poly_pts.clear()
                node._poly_closed = False
                node._poly_area_cm2 = 0.0
                node._poly_perim_cm = 0.0
                node.get_logger().info("Polygon cleared.")
            elif key == ord('c'):
                node.close_polygon()

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
