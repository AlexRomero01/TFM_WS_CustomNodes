#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthInspector(Node):
    def __init__(self):
        super().__init__('depth_inspector')
        self.bridge = CvBridge()
        
        # Ajusta el topic a tu configuración
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw', 
            self.depth_callback,
            10
        )
        
        self.current_depth_image = None
        self.mouse_x, self.mouse_y = 0, 0
        self.ground_height_cm = 130.0

        # Crear ventana de OpenCV
        cv2.namedWindow("Inspeccion de Profundidad TFM")
        cv2.setMouseCallback("Inspeccion de Profundidad TFM", self.on_mouse)
        
        self.get_logger().info("Visualizador iniciado. Mueve el raton sobre la imagen.")

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y

    def depth_callback(self, msg):
        try:
            # Convertir a milímetros (16UC1)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.current_depth_image = depth_image

            # Crear una visualización en color (Jet colormap)
            # Normalizamos de 500mm a 2000mm para que el contraste sea útil en tu rango (130cm)
            depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_mapped = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # Obtener valor del píxel actual
            h, w = depth_image.shape
            if 0 <= self.mouse_y < h and 0 <= self.mouse_x < w:
                depth_mm = depth_image[self.mouse_y, self.mouse_x]
                depth_cm = depth_mm / 10.0
                
                # Calcular altura respecto al suelo
                plant_height = self.ground_height_cm - depth_cm if depth_mm > 0 else 0.0

                # Dibujar info en pantalla
                info_text = f"X:{self.mouse_x} Y:{self.mouse_y} | Z:{depth_cm:.1f}cm | H:{plant_height:.1f}cm"
                cv2.putText(color_mapped, info_text, (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Dibujar punto de mira
                cv2.drawMarker(color_mapped, (self.mouse_x, self.mouse_y), (255, 255, 255), 
                               cv2.MARKER_CROSS, 20, 2)

            cv2.imshow("Inspeccion de Profundidad TFM", color_mapped)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Evita el error de doble shutdown
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()