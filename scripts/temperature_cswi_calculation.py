import rclpy
import os
import argparse
import cv2
import numpy as np
try:
    from ultralytics_ros.msg import YoloResult
except Exception:
    YoloResult = None
    # will log later and skip YOLO subscription
try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

class Calculator(Node):
    def __init__(self):
        super().__init__('Temperature_CSWI_Calculator')

        # Logging for debugging initialization
        self.get_logger().info("Initializing ImageRescaler node")

        # Crear archivo de log
        self.log_file_path = os.path.expanduser("~/tfm_ws/src/custom_nodes/processing_times.txt")
        self.log_file = open(self.log_file_path, "w")
        self.log_file.write("Process Time Logs:\n")

        # Suscriptores para los tópicos
        self.bridge = CvBridge() if CvBridge is not None else None
        if self.bridge is None:
            self.get_logger().error("cv_bridge not available. Install cv_bridge or fix NumPy ABI. Node will run but image processing is disabled.")

        self.rgb_subscriber = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, 10)
        self.thermal_subscriber = self.create_subscription(Image, '/thermal_image_view', self.thermal_callback, 10)
        self.temperature_sub = self.create_subscription(Image, '/thermal_image', self.temperature_callback, 10)
        if YoloResult is not None:
            self.yolo_subscriber = self.create_subscription(YoloResult, '/yolo_result', self.yolo_callback, 10)
        else:
            self.get_logger().warning("ultralytics_ros.msg.YoloResult not available — YOLO processing disabled.")

        # Publicadores bajo el tópico general 'Temperature_and_CSWI'
        base_topic = 'Temperature_and_CSWI'
        self.rescaled_image_publisher = self.create_publisher(Image, f'/{base_topic}/rescaled_rgb', 10)
        self.text_publisher = self.create_publisher(String, f'/{base_topic}/text', 10)
        self.rescaled_masks_publisher = self.create_publisher(Image, f'/{base_topic}/rescaled_yolo_masks', 10)
        self.masked_image_with_temperature_publisher = self.create_publisher(Image, f'/{base_topic}/masked_image_with_temperature', 10)

        # Fallback: subscribe to combined rescaled masks and split into connected components
        # This lets us compute per-object area even if ultralytics_ros/YoloResult is not available.
        self.combined_mask_subscriber = self.create_subscription(
            Image,
            f'/{base_topic}/rescaled_yolo_masks',
            self.combined_mask_callback,
            10
        )

        # Utilidad para convertir entre ROS Image y OpenCV Image
        # self.bridge = CvBridge()  # bridge already set above

        self.H = None  # Homography matrix

        # Additional parameters
        #self.name_class = name_class
        #self.number_class = number_class

        # Variables para almacenar las imágenes
        self.rgb_image = None
        self.thermal_image = None
        self.thermal_temperature_image = None
        self.rgb_rescaled = None
        self.person_colors = {}

        self.Twet = 22.0  # Ajusta este valor según sea necesario
        self.Tdry = 27.0  # Ajusta este valor según sea necesario


    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Obtener la resolución de la imagen
        height, width, _ = self.rgb_image.shape

        if self.H is None:
            # Cargar la homografía adecuada según la resolución
            if width == 1280:
                if self.H is None or self.H.shape[1] != 1280:
                    self.get_logger().info("Loading 1280 homography matrix.")
                    try:
                        self.H = np.loadtxt(os.path.expanduser("~/tfm_ws/src/custom_nodes/Homography/average_homography2.txt"))
                    except Exception as e:
                        self.get_logger().error(f"Failed to load 1280 homography matrix: {e}")
            elif width == 640:
                if self.H is None or self.H.shape[1] != 640:
                    self.get_logger().info("Loading 640 homography matrix.")
                    try:
                        self.H = np.loadtxt(os.path.expanduser("~/tfm_ws/src/custom_nodes/Homography/average_homography_640.txt"))
                    except Exception as e:
                        self.get_logger().error(f"Failed to load 640 homography matrix: {e}")
            else:
                self.get_logger().error(f"Unsupported resolution: {width}x{height}. No homography applied.")
                self.H = None

        # Procesar las imágenes
        self.process_images()

    def thermal_callback(self, msg):
        self.thermal_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.process_images()

    def temperature_callback(self, msg):
        self.thermal_temperature_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')


    def yolo_callback(self, msg):
        if YoloResult is None:
            self.get_logger().warn("Received YOLO message but ultralytics_ros not available; skipping.")
            return
        if self.bridge is None:
            self.get_logger().warn("cv_bridge not available; cannot process YOLO masks.")
            return
        self.process_and_publish_yolo_masks(msg)
    
    def process_images(self):
        if self.rgb_image is not None and self.thermal_image is not None:
            height, width = self.thermal_image.shape
            if self.H is not None:
                self.rgb_rescaled = cv2.warpPerspective(self.rgb_image, self.H, (width, height))
                rescaled_image_msg = self.bridge.cv2_to_imgmsg(self.rgb_rescaled, encoding="bgr8")
                self.rescaled_image_publisher.publish(rescaled_image_msg)
            else:
                self.get_logger().warn("Homography matrix not loaded, skipping image rescaling")

    def process_and_publish_yolo_masks(self, yolo_result_msg):
        if self.rgb_rescaled is None:
            self.get_logger().warn("RGB rescaled image is not available yet.")
            return
    
        if hasattr(yolo_result_msg, 'masks') and len(yolo_result_msg.masks) > 0:
            #self.get_logger().info("Processing YOLO masks")
            person_temperatures = []
            combined_mask = None  # Para almacenar la máscara combinada
            for i, detection in enumerate(yolo_result_msg.detections.detections):
                class_id = detection.results[0].hypothesis.class_id

                # if class_id == self.name_class or class_id == self.number_class:
                #self.get_logger().info(f"Detected class matching target: {class_id}")
                mask_msg = yolo_result_msg.masks[i]
                if not mask_msg:
                        self.get_logger().warn("Empty mask list. make sure to use a segmentation model and name it with a -seg suffix. For example: your_model-seg.pt")
                        return
                yolo_mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')

                if self.thermal_image is not None:
                    height, width = self.thermal_image.shape
                    rescaled_mask = cv2.warpPerspective(yolo_mask, self.H, (width, height))

                    # Calcular el área de la máscara reescalada
                    # count non-zero pixels (works whether mask uses 255 or other nonzero)
                    mask_area = np.count_nonzero(rescaled_mask > 0)
                    image_area = float(self.rgb_rescaled.shape[0] * self.rgb_rescaled.shape[1])
                    percentage = (mask_area / image_area) * 100.0 

                    # Filter out small detections (area < 1%)
                    if percentage < 1.0:
                        continue

                    if self.is_mask_within_bounds(rescaled_mask, width, height):
                        temperature = self.calculate_mask_temperature(rescaled_mask)
                        if temperature != 0.0:
                            cwsi = (temperature - self.Twet) / (self.Tdry - self.Twet)
                            color = (128, 0, 128)  # Morado para todas las máscaras
                            # store percentage with temperature and cwsi
                            person_temperatures.append((rescaled_mask, temperature, cwsi, color, percentage))
                            
                            # Si no hay una máscara combinada, inicializarla con la primera
                            if combined_mask is None:
                                combined_mask = rescaled_mask
                            else:
                                # Combinar máscaras utilizando operación bitwise OR
                                combined_mask = cv2.bitwise_or(combined_mask, rescaled_mask)

            # Publicar la máscara combinada en 'rescaled_yolo_masks'
            if combined_mask is not None:
                #self.get_logger().info("Publishing combined YOLO mask")
                rescaled_mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding="mono8")
                self.rescaled_masks_publisher.publish(rescaled_mask_msg)  # Publicación de la máscara combinada

            if person_temperatures:
                image_with_temperatures = self.rgb_rescaled.copy()
                for mask, temp, cwsi, color, percentage in person_temperatures:
                    self.add_temperature_and_cwsi_to_image(image_with_temperatures, mask, temp, cwsi, color)

                #self.get_logger().info("Publishing image with temperatures and CSWI")
                masked_image_msg = self.bridge.cv2_to_imgmsg(image_with_temperatures, encoding="bgr8")
                self.masked_image_with_temperature_publisher.publish(masked_image_msg)

                # publish text including Area numeric value (no percent sign)
                for idx, (mask, temp, cwsi, color, percentage) in enumerate(person_temperatures):
                    self._publish_object_text(idx, temp, cwsi, mask=mask, area=percentage)


    def add_temperature_and_cwsi_to_image(self, image, mask, temperature, cwsi, color):
        # Convert the mask to binary and adjust size if needed
        resized_mask = cv2.resize(mask, (self.rgb_rescaled.shape[1], self.rgb_rescaled.shape[0]))
        resized_mask = (resized_mask > 0).astype(np.uint8)

        # Apply the fixed purple mask over the rescaled RGB image
        image[resized_mask > 0] = image[resized_mask > 0] * 0.5 + np.array(color) * 0.5

        # Text color is always white for visibility
        text_color = (255, 255, 255)

        # Find the center of the mask to place the text
        y_coords, x_coords = np.where(resized_mask > 0)
        if len(x_coords) > 0 and len(y_coords) > 0:
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            cv2.putText(image, f'{temperature:.2f} C', (center_x - 20, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            cv2.putText(image, f'CSWI: {cwsi:.2f}', (center_x - 20, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    def is_mask_within_bounds(self, mask, width, height):
        valid_pixels = np.argwhere(mask > 0)
        for pixel in valid_pixels:
            y, x = pixel[0], pixel[1]
            if not (0 <= x < width and 0 <= y < height):
                return False
        return True

    def calculate_mask_temperature(self, mask):
        if self.thermal_temperature_image is None:
            return 0.0

        valid_pixels = np.argwhere(mask > 0)
        mask_temperature_values = []

        for pixel in valid_pixels:
            y, x = pixel[0], pixel[1]

            if 0 <= x < self.thermal_temperature_image.shape[1] and 0 <= y < self.thermal_temperature_image.shape[0]:
                raw_temperature = self.thermal_temperature_image[y, x]
                temperature_celsius = (raw_temperature - 1000) / 10.0

                if -20 <= temperature_celsius <= 100:
                    mask_temperature_values.append(temperature_celsius)

        if mask_temperature_values:
            return np.mean(mask_temperature_values)
        else:
            return 0.0
        
    def destroy_node(self):
        # Cerrar archivo de log
        self.log_file.close
        super().destroy_node()

    def combined_mask_callback(self, msg):
        """Fallback: split combined rescaled mask into connected components and publish per-object text (with Area)."""
        if self.rgb_rescaled is None:
            self.get_logger().warn("RGB rescaled image is not available yet (combined_mask_callback).")
            return
        if self.thermal_image is None or self.thermal_temperature_image is None:
            self.get_logger().warn("Thermal image/temperature not available yet (combined_mask_callback).")
            return

        # Convert mask image to OpenCV mono8
        try:
            combined_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert combined mask msg: {e}")
            return

        # Binary threshold (any non-zero pixel belongs to an object)
        binary = (combined_mask > 0).astype('uint8') * 255

        # Connected components to separate objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            # no objects found
            return

        image_area = float(self.rgb_rescaled.shape[0] * self.rgb_rescaled.shape[1])
        person_infos = []
        for label in range(1, num_labels):
            comp_mask = (labels == label).astype('uint8')
            mask_area = int(np.count_nonzero(comp_mask))
            if mask_area == 0:
                continue
            percentage = (mask_area / image_area) * 100.0

            # Filter out small detections (area < 1%)
            if percentage < 1.0:
                continue

            # convert to same format as other functions expect (255-based mask)
            mask_255 = (comp_mask * 255).astype('uint8')

            # Compute temperature for this component
            temperature = self.calculate_mask_temperature(mask_255)
            if temperature == 0.0:
                # skip if no valid temperature pixels
                continue
            cwsi = (temperature - self.Twet) / (self.Tdry - self.Twet)
            color = (128, 0, 128)
            person_infos.append((mask_255, temperature, cwsi, color, percentage))

        if not person_infos:
            return

        # Build and publish masked image (visual)
        image_with_temperatures = self.rgb_rescaled.copy()
        for mask, temp, cwsi, color, percentage in person_infos:
            self.add_temperature_and_cwsi_to_image(image_with_temperatures, mask, temp, cwsi, color)
        try:
            masked_image_msg = self.bridge.cv2_to_imgmsg(image_with_temperatures, encoding="bgr8")
            self.masked_image_with_temperature_publisher.publish(masked_image_msg)
        except Exception:
            pass

        # Publish per-object text including Area
        for idx, (mask, temp, cwsi, color, percentage) in enumerate(person_infos):
            self._publish_object_text(idx, temp, cwsi, mask=mask, area=percentage)


    def _publish_object_text(self, idx, temperature, cwsi, mask=None, area=None):
        """Publish one text message for an object; compute area from mask if needed."""
        if area is None and mask is not None and self.rgb_rescaled is not None:
            try:
                mask_area = int(np.count_nonzero(mask > 0))
                image_area = float(self.rgb_rescaled.shape[0] * self.rgb_rescaled.shape[1])
                area = (mask_area / image_area) * 100.0 if image_area > 0 else 0.0
            except Exception:
                area = 0.0

        if area is None:
            area = 0.0

        msg_text = String()
        msg_text.data = f" Objeto {idx + 1}: Temperatura = {temperature:.2f} °C, CSWI = {cwsi:.2f}, Area = {area:.2f}"
        self.text_publisher.publish(msg_text)
        self.get_logger().info(f"Objeto {idx + 1}: Temperatura = {temperature:.2f} °C, CSWI = {cwsi:.2f}, Area = {area:.2f}")


def main(args=None):
    # Parsear los argumentos de la línea de comandos
    parser = argparse.ArgumentParser(description='Image Rescaler Node')
    #parser.add_argument('--homography_file', type=int, required=True, help='1280 o 640')
    #parser.add_argument('--name_class', type=str, required=True, help='Name of the class in YOLO')
    #parser.add_argument('--number_class', type=int, required=True, help='Index ID of the class in YOLO')
    #args = parser.parse_args()

    rclpy.init()  # Inicializar sin pasar 'args' aquí

    image_rescaler = Calculator()
    rclpy.spin(image_rescaler)
    image_rescaler.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
