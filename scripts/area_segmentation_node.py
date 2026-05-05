import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
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

        # --- Publicadores ---
        self.area_pub       = self.create_publisher(Float32, '/plant/projected_area_cm2',  10)
        self.diag_max_pub   = self.create_publisher(Float32, '/plant/diagonal_max_cm',     10)
        self.diag_min_pub   = self.create_publisher(Float32, '/plant/diagonal_min_cm',     10)
        self.diag_mean_pub  = self.create_publisher(Float32, '/plant/diagonal_mean_cm',    10)
        self.height_max_pub = self.create_publisher(Float32, '/plant/height_max_cm',       10)
        self.height_mean_pub= self.create_publisher(Float32, '/plant/height_mean_cm',      10)

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
                if not self.depth_buffer:
                    return
                depth_raw = self.depth_buffer[0]

            # 1. Alineación máscara ↔ profundidad
            if depth_raw.shape != mask_uint8.shape:
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

            all_pts = np.concatenate(contours)

            # 3. Convex Hull en píxeles
            hull = cv2.convexHull(all_pts)

            # 4. Profundidades de los píxeles de la planta (cm)
            mask_pixels = mask_uint8 > 0
            depth_cm = depth_raw[mask_pixels].astype(np.float32) / 10.0

            # Filtrado por rango (100 cm – 150 cm, igual que el nodo de volumen)
            depth_cm = depth_cm[(depth_cm >= 100.0) & (depth_cm <= 150.0)]
            if depth_cm.size == 0:
                return

            # 5. Alturas (en cámara, "altura" = z_ground – z_pixel)
            #    Un valor positivo indica que el punto está POR ENCIMA del suelo.
            heights_cm = self.z_ground - depth_cm   # (N,)

            z_mean = float(np.mean(depth_cm))         # profundidad media para escala

            # Altura media: media de alturas positivas (descarta suelo/fondo)
            heights_positive = heights_cm[heights_cm > 0]
            if heights_positive.size > 0:
                height_mean = float(np.mean(heights_positive))
                # Altura máxima robusta: percentil configurado de las alturas positivas
                height_max = float(np.percentile(heights_positive, self._height_pct))
            else:
                height_mean = 0.0
                height_max  = 0.0

            # 6. Proyección del Hull a coordenadas reales (cm)
            hull_pts_cm = []
            for pt in hull:
                u, v = pt[0]
                x_cm = (u - self.cx) * z_mean / self.fx
                y_cm = (v - self.cy) * z_mean / self.fy
                hull_pts_cm.append([x_cm, y_cm])

            hull_pts_cm = np.array(hull_pts_cm, dtype=np.float32)

            # 7. Área del Convex Hull (cm²)
            real_area_hull = cv2.contourArea(hull_pts_cm)

            # 8. Diagonales: todas las distancias entre pares de vértices del hull
            #    (diagonales reales del polígono, no del bounding box)
            n_verts = len(hull_pts_cm)
            if n_verts >= 2:
                idx = np.array(list(combinations(range(n_verts), 2)))
                diffs = hull_pts_cm[idx[:, 0]] - hull_pts_cm[idx[:, 1]]
                diag_lengths = np.linalg.norm(diffs, axis=1)
                diag_max  = float(diag_lengths.max())
                diag_min  = float(diag_lengths.min())
                diag_mean = float(diag_lengths.mean())
            else:
                diag_max = diag_min = diag_mean = 0.0

            # 9. Publicar resultados
            self.area_pub.publish(       Float32(data=float(real_area_hull)))
            self.diag_max_pub.publish(   Float32(data=diag_max))
            self.diag_min_pub.publish(   Float32(data=diag_min))
            self.diag_mean_pub.publish(  Float32(data=diag_mean))
            self.height_max_pub.publish( Float32(data=height_max))
            self.height_mean_pub.publish(Float32(data=height_mean))

            self.get_logger().info(
                f"Area: {real_area_hull:.1f} cm²  |  "
                f"Diag max/min/mean: {diag_max:.1f}/{diag_min:.1f}/{diag_mean:.1f} cm  |  "
                f"Height max(p{self._height_pct:.0f})/mean: {height_max:.1f}/{height_mean:.1f} cm  |  "
                f"Z_mean: {z_mean:.1f} cm"
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