import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading
from collections import deque

class InteractiveMeasureNode(Node):
    def __init__(self):
        super().__init__('interactive_measure_node')
        self.bridge = CvBridge()

        # --- RealSense D457 Intrinsics (Updated from your topic echo) ---
        self.fx = 385.0450439453125
        self.fy = 384.5845642089844
        self.cx = 325.48455810546875
        self.cy = 241.35191345214844

        # Thread-safe buffers
        self.latest_depth = None
        self.latest_color = None
        self.data_lock = threading.Lock()
        self.depth_buffer = deque(maxlen=5)

        # Mouse state
        self.drawing = False
        self.pt1 = (0, 0)
        self.pt2 = (0, 0)
        self.box_drawn = False

        # Topics
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)

        self.window_name = "Point-Projection Measurement Tool"
        self.get_logger().info("Node Started. Click and drag to measure using 3D Projection.")

    def color_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.data_lock: self.latest_color = img
        except Exception as e: self.get_logger().error(f"Color error: {e}")

    def depth_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            with self.data_lock:
                self.depth_buffer.append(img)
                self.latest_depth = np.mean(np.array(self.depth_buffer), axis=0)
        except Exception as e: self.get_logger().error(f"Depth error: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing, self.pt1, self.pt2, self.box_drawn = True, (x, y), (x, y), False
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.pt2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing, self.pt2, self.box_drawn = False, (x, y), True

    def calculate_metrics_projection(self, depth_raw, x1, y1, x2, y2):
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        
        if left == right or top == bottom: return 0.0, 0.0, 0.0

        # 1. Depth Analysis
        depth_roi = depth_raw[top:bottom, left:right]
        # RealSense 16UC1 is mm. We want cm for the calculation.
        depth_cm = depth_roi.astype(np.float32) / 10.0
        
        # Filter background/noise (standard range for your project)
        valid_depth = depth_cm[(depth_cm > 10.0) & (depth_cm < 300.0)]
        if valid_depth.size == 0: return 0.0, 0.0, 0.0
        
        z_mean = np.mean(valid_depth)

        # 2. Point Projection Methodology
        # We project the 4 corners of your selection into real 3D space (cm)
        corners_px = [(left, top), (right, top), (right, bottom), (left, bottom)]
        corners_cm = []
        
        for (u, v) in corners_px:
            x_cm = (u - self.cx) * z_mean / self.fx
            y_cm = (v - self.cy) * z_mean / self.fy
            corners_cm.append([x_cm, y_cm])
        
        corners_cm = np.array(corners_cm, dtype=np.float32)

        # 3. Geometric Metrics
        # Area of the projected rectangle in real space
        area_cm2 = cv2.contourArea(corners_cm)
        
        # Diagonal: Distance from top-left to bottom-right in real space
        diag_cm = np.linalg.norm(corners_cm[0] - corners_cm[2])

        return area_cm2, diag_cm, z_mean

def main(args=None):
    rclpy.init(args=args)
    node = InteractiveMeasureNode()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cv2.namedWindow(node.window_name)
    cv2.setMouseCallback(node.window_name, node.mouse_callback)

    try:
        while rclpy.ok():
            with node.data_lock:
                if node.latest_color is None or node.latest_depth is None: continue
                color_img = node.latest_color.copy()
                depth_img = node.latest_depth.copy()

            if node.drawing or node.box_drawn:
                cv2.rectangle(color_img, node.pt1, node.pt2, (0, 255, 0), 2)
                if node.box_drawn:
                    area, diag, z = node.calculate_metrics_projection(
                        depth_img, node.pt1[0], node.pt1[1], node.pt2[0], node.pt2[1])

                    # UI Labeling
                    cv2.putText(color_img, f"Area: {area:.1f} cm2", (node.pt1[0], node.pt1[1]-35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(color_img, f"Diag: {diag:.1f} cm", (node.pt1[0], node.pt1[1]-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(node.window_name, color_img)
            if cv2.waitKey(30) & 0xFF == ord('q'): break
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()