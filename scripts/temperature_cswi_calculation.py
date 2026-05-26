#!/usr/bin/env python3
"""
temperature_cswi_calculation.py
================================
ROS 2 node that overlays YOLO segmentation masks onto the thermal camera
frame using a pre-computed homography, then calculates the Crop Water
Stress Index (CWSI) for each detected plant.

The node operates in two modes:
  1. YOLO mode   (primary): consumes ultralytics_ros/YoloResult masks.
  2. Fallback mode         : splits the combined rescaled mask into connected
                             components when YoloResult is unavailable.

Subscriptions:
    /camera/camera/color/image_raw          sensor_msgs/Image  (bgr8)
    /thermal_image_view                     sensor_msgs/Image  (mono8)
    /thermal_image                          sensor_msgs/Image  (mono16)
    /yolo_result                            ultralytics_ros/YoloResult  (optional)
    /Temperature_and_CSWI/rescaled_yolo_masks  sensor_msgs/Image  (mono8, fallback)

Publications:
    /Temperature_and_CSWI/rescaled_rgb                    sensor_msgs/Image  (bgr8)
    /Temperature_and_CSWI/text                            std_msgs/String
    /Temperature_and_CSWI/rescaled_yolo_masks             sensor_msgs/Image  (mono8)
    /Temperature_and_CSWI/masked_image_with_temperature   sensor_msgs/Image  (bgr8)

Note: published text strings keep the "Objeto / Temperatura / CSWI / Area"
format because they are parsed by ros2_mqtt_publisher.py and csv_safecopy.py.
"""
import rclpy
import os
import cv2
import numpy as np

try:
    from ultralytics_ros.msg import YoloResult
except Exception:
    YoloResult = None  # YOLO mode disabled if the message type is not installed

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

        self.get_logger().info("Initializing Temperature_CSWI_Calculator node")

        # Log file for processing time diagnostics
        self.log_file_path = os.path.expanduser(
            "~/tfm_ws/src/custom_nodes/processing_times.txt")
        self.log_file = open(self.log_file_path, "w")
        self.log_file.write("Process Time Logs:\n")

        # cv_bridge setup
        self.bridge = CvBridge() if CvBridge is not None else None
        if self.bridge is None:
            self.get_logger().error(
                "cv_bridge not available. Image processing is disabled.")

        # Subscriptions
        self.rgb_subscriber = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.rgb_callback, 10)
        self.thermal_subscriber = self.create_subscription(
            Image, '/thermal_image_view', self.thermal_callback, 10)
        self.temperature_sub = self.create_subscription(
            Image, '/thermal_image', self.temperature_callback, 10)

        if YoloResult is not None:
            self.yolo_subscriber = self.create_subscription(
                YoloResult, '/yolo_result', self.yolo_callback, 10)
        else:
            self.get_logger().warning(
                "ultralytics_ros.msg.YoloResult not available — YOLO mode disabled.")

        # Publishers
        base_topic = 'Temperature_and_CSWI'
        self.rescaled_image_publisher = self.create_publisher(
            Image, f'/{base_topic}/rescaled_rgb', 10)
        self.text_publisher = self.create_publisher(
            String, f'/{base_topic}/text', 10)
        self.rescaled_masks_publisher = self.create_publisher(
            Image, f'/{base_topic}/rescaled_yolo_masks', 10)
        self.masked_image_with_temperature_publisher = self.create_publisher(
            Image, f'/{base_topic}/masked_image_with_temperature', 10)

        # Fallback: subscribe to the combined rescaled mask and split into
        # connected components to compute per-object area when YoloResult
        # is not available.
        self.combined_mask_subscriber = self.create_subscription(
            Image,
            f'/{base_topic}/rescaled_yolo_masks',
            self.combined_mask_callback,
            10
        )

        self.H = None           # Homography matrix (RGB → thermal frame)

        # Image buffers
        self.rgb_image = None
        self.thermal_image = None
        self.thermal_temperature_image = None
        self.rgb_rescaled = None
        self.person_colors = {}

        # CWSI reference temperatures (°C) — adjust to field conditions
        self.Twet = 22.0   # wet-bulb reference (fully irrigated canopy)
        self.Tdry = 27.0   # dry-bulb reference (non-transpiring surface)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        height, width, _ = self.rgb_image.shape

        # Lazy-load the homography matrix matching the current resolution
        if self.H is None:
            if width == 1280:
                self.get_logger().info("Loading 1280 homography matrix.")
                try:
                    self.H = np.loadtxt(os.path.expanduser(
                        "~/tfm_ws/src/custom_nodes/Homography/average_homography2.txt"))
                except Exception as e:
                    self.get_logger().error(
                        f"Failed to load 1280 homography matrix: {e}")
            elif width == 640:
                self.get_logger().info("Loading 640 homography matrix.")
                try:
                    self.H = np.loadtxt(os.path.expanduser(
                        "~/tfm_ws/src/custom_nodes/Homography/average_homography_640.txt"))
                except Exception as e:
                    self.get_logger().error(
                        f"Failed to load 640 homography matrix: {e}")
            else:
                self.get_logger().error(
                    f"Unsupported resolution: {width}x{height}. No homography applied.")
                self.H = None

        self.process_images()

    def thermal_callback(self, msg):
        self.thermal_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.process_images()

    def temperature_callback(self, msg):
        self.thermal_temperature_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='mono16')

    def yolo_callback(self, msg):
        if YoloResult is None:
            self.get_logger().warn(
                "Received YOLO message but ultralytics_ros not available; skipping.")
            return
        if self.bridge is None:
            self.get_logger().warn(
                "cv_bridge not available; cannot process YOLO masks.")
            return
        self.process_and_publish_yolo_masks(msg)

    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------

    def process_images(self):
        """Warp the RGB image into the thermal camera frame using the homography."""
        if self.rgb_image is not None and self.thermal_image is not None:
            height, width = self.thermal_image.shape
            if self.H is not None:
                self.rgb_rescaled = cv2.warpPerspective(
                    self.rgb_image, self.H, (width, height))
                rescaled_image_msg = self.bridge.cv2_to_imgmsg(
                    self.rgb_rescaled, encoding="bgr8")
                self.rescaled_image_publisher.publish(rescaled_image_msg)
            else:
                self.get_logger().warn(
                    "Homography matrix not loaded; skipping image rescaling.")

    def process_and_publish_yolo_masks(self, yolo_result_msg):
        """Warp each YOLO mask into the thermal frame and compute temperature/CWSI."""
        if self.rgb_rescaled is None:
            self.get_logger().warn("RGB rescaled image not available yet.")
            return

        if not (hasattr(yolo_result_msg, 'masks') and len(yolo_result_msg.masks) > 0):
            return

        person_temperatures = []
        combined_mask = None

        for i, detection in enumerate(yolo_result_msg.detections.detections):
            mask_msg = yolo_result_msg.masks[i]
            if not mask_msg:
                self.get_logger().warn(
                    "Empty mask — ensure a segmentation model is used (-seg suffix).")
                return

            yolo_mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')

            if self.thermal_image is not None:
                height, width = self.thermal_image.shape
                rescaled_mask = cv2.warpPerspective(yolo_mask, self.H, (width, height))

                mask_area = np.count_nonzero(rescaled_mask > 0)
                image_area = float(self.rgb_rescaled.shape[0] * self.rgb_rescaled.shape[1])
                percentage = (mask_area / image_area) * 100.0

                # Discard detections smaller than 1 % of the frame (noise/partial views)
                if percentage < 1.0:
                    continue

                if self.is_mask_within_bounds(rescaled_mask, width, height):
                    temperature = self.calculate_mask_temperature(rescaled_mask)
                    if temperature != 0.0:
                        cwsi = (temperature - self.Twet) / (self.Tdry - self.Twet)
                        color = (128, 0, 128)  # purple overlay for all instances
                        person_temperatures.append(
                            (rescaled_mask, temperature, cwsi, color, percentage))

                        combined_mask = (
                            rescaled_mask if combined_mask is None
                            else cv2.bitwise_or(combined_mask, rescaled_mask)
                        )

        # Publish the union of all valid masks
        if combined_mask is not None:
            rescaled_mask_msg = self.bridge.cv2_to_imgmsg(
                combined_mask, encoding="mono8")
            self.rescaled_masks_publisher.publish(rescaled_mask_msg)

        if person_temperatures:
            image_with_temperatures = self.rgb_rescaled.copy()
            for mask, temp, cwsi, color, percentage in person_temperatures:
                self.add_temperature_and_cwsi_to_image(
                    image_with_temperatures, mask, temp, cwsi, color)

            masked_image_msg = self.bridge.cv2_to_imgmsg(
                image_with_temperatures, encoding="bgr8")
            self.masked_image_with_temperature_publisher.publish(masked_image_msg)

            for idx, (mask, temp, cwsi, color, percentage) in enumerate(person_temperatures):
                self._publish_object_text(idx, temp, cwsi, mask=mask, area=percentage)

    def add_temperature_and_cwsi_to_image(self, image, mask, temperature, cwsi, color):
        """Overlay a coloured mask and burn temperature / CWSI text into the image."""
        resized_mask = cv2.resize(
            mask, (self.rgb_rescaled.shape[1], self.rgb_rescaled.shape[0]))
        resized_mask = (resized_mask > 0).astype(np.uint8)

        image[resized_mask > 0] = (
            image[resized_mask > 0] * 0.5 + np.array(color) * 0.5)

        y_coords, x_coords = np.where(resized_mask > 0)
        if len(x_coords) > 0 and len(y_coords) > 0:
            cx = int(np.mean(x_coords))
            cy = int(np.mean(y_coords))
            cv2.putText(image, f'{temperature:.2f} C',
                        (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, f'CSWI: {cwsi:.2f}',
                        (cx - 20, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def is_mask_within_bounds(self, mask, width, height):
        """Return True if all foreground pixels lie within the image bounds."""
        valid_pixels = np.argwhere(mask > 0)
        for pixel in valid_pixels:
            y, x = pixel[0], pixel[1]
            if not (0 <= x < width and 0 <= y < height):
                return False
        return True

    def calculate_mask_temperature(self, mask):
        """Return the mean temperature (°C) of pixels inside the mask."""
        if self.thermal_temperature_image is None:
            return 0.0

        valid_pixels = np.argwhere(mask > 0)
        mask_temperature_values = []

        for pixel in valid_pixels:
            y, x = pixel[0], pixel[1]
            if (0 <= x < self.thermal_temperature_image.shape[1]
                    and 0 <= y < self.thermal_temperature_image.shape[0]):
                raw_temperature = self.thermal_temperature_image[y, x]
                # Optris encoding: raw = (T_celsius + 100) * 10
                temperature_celsius = (raw_temperature - 1000) / 10.0
                if -20 <= temperature_celsius <= 100:
                    mask_temperature_values.append(temperature_celsius)

        return np.mean(mask_temperature_values) if mask_temperature_values else 0.0

    def destroy_node(self):
        self.log_file.close()
        super().destroy_node()

    # -------------------------------------------------------------------------
    # Fallback: split combined mask into connected components
    # -------------------------------------------------------------------------

    def combined_mask_callback(self, msg):
        """
        Fallback path used when ultralytics_ros is unavailable.
        Splits the combined rescaled mask into individual connected components
        and publishes per-object temperature / CWSI text.
        """
        if self.rgb_rescaled is None:
            self.get_logger().warn(
                "RGB rescaled image not available yet (combined_mask_callback).")
            return
        if self.thermal_image is None or self.thermal_temperature_image is None:
            self.get_logger().warn(
                "Thermal image/temperature not available yet (combined_mask_callback).")
            return

        try:
            combined_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert combined mask msg: {e}")
            return

        binary = (combined_mask > 0).astype('uint8') * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        if num_labels <= 1:
            return  # no foreground objects found

        image_area = float(self.rgb_rescaled.shape[0] * self.rgb_rescaled.shape[1])
        person_infos = []

        for label in range(1, num_labels):
            comp_mask = (labels == label).astype('uint8')
            mask_area = int(np.count_nonzero(comp_mask))
            if mask_area == 0:
                continue

            percentage = (mask_area / image_area) * 100.0
            if percentage < 1.0:
                continue  # discard tiny components

            mask_255 = (comp_mask * 255).astype('uint8')
            temperature = self.calculate_mask_temperature(mask_255)
            if temperature == 0.0:
                continue

            cwsi = (temperature - self.Twet) / (self.Tdry - self.Twet)
            person_infos.append((mask_255, temperature, cwsi, (128, 0, 128), percentage))

        if not person_infos:
            return

        image_with_temperatures = self.rgb_rescaled.copy()
        for mask, temp, cwsi, color, percentage in person_infos:
            self.add_temperature_and_cwsi_to_image(
                image_with_temperatures, mask, temp, cwsi, color)
        try:
            masked_image_msg = self.bridge.cv2_to_imgmsg(
                image_with_temperatures, encoding="bgr8")
            self.masked_image_with_temperature_publisher.publish(masked_image_msg)
        except Exception:
            pass

        for idx, (mask, temp, cwsi, color, percentage) in enumerate(person_infos):
            self._publish_object_text(idx, temp, cwsi, mask=mask, area=percentage)

    def _publish_object_text(self, idx, temperature, cwsi, mask=None, area=None):
        """
        Publish a text message for one detected object.

        The format "Objeto N: Temperatura = X °C, CSWI = Y, Area = Z" is kept
        intentionally because it is parsed by ros2_mqtt_publisher.py and
        csv_safecopy.py using a fixed regex.
        """
        if area is None and mask is not None and self.rgb_rescaled is not None:
            try:
                mask_area = int(np.count_nonzero(mask > 0))
                image_area = float(
                    self.rgb_rescaled.shape[0] * self.rgb_rescaled.shape[1])
                area = (mask_area / image_area) * 100.0 if image_area > 0 else 0.0
            except Exception:
                area = 0.0

        if area is None:
            area = 0.0

        msg_text = String()
        msg_text.data = (
            f" Objeto {idx + 1}: Temperatura = {temperature:.2f} °C, "
            f"CSWI = {cwsi:.2f}, Area = {area:.2f}"
        )
        self.text_publisher.publish(msg_text)
        self.get_logger().info(
            f"Objeto {idx + 1}: Temperatura = {temperature:.2f} °C, "
            f"CSWI = {cwsi:.2f}, Area = {area:.2f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init()
    image_rescaler = Calculator()
    rclpy.spin(image_rescaler)
    image_rescaler.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
