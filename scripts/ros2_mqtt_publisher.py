#!/usr/bin/env python3
"""
ROS2 MQTT Publisher Node
========================
Publishes sensor data to MQTT broker in real-time at 2 messages/second.
CSV logging is handled separately by csv_safecopy.py
"""

import re
import json
import math
import os
import csv
from datetime import datetime
import paho.mqtt.client as mqtt
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Temperature, RelativeHumidity
from geometry_msgs.msg import PointStamped

# Configuraciones
MQTT_DEFAULT_HOST = "localhost"  
MQTT_DEFAULT_PORT = 1883         
MQTT_DEFAULT_TIMEOUT = 120       
PUBLISH_RATE_HZ = 2.0 

NO_DATA_TIMEOUT = 10.0
RECONNECT_ATTEMPT_INTERVAL = 5.0

# Topics ROS2
ROS2_TOPIC_GPS = "gps/fix"
ROS2_TOPIC_TEMPERATURE = "/Temperature_and_CSWI/text"
ROS2_TOPIC_NDVI = "/NDVI"
ROS2_TOPIC_HEADING = '/ublox_rover/navheading'
ROS2_TOPIC_AREA = '/segmentation_area_info'
ROS2_TOPIC_LOCATION = '/navigation/information'
ROS2_TOPIC_BIOMASS = '/plant_biomass'          
ROS2_TOPIC_CROP_LIGHT_STATE = '/crop_light_state'
ROS2_TOPIC_CROP_TYPE = '/crop_type'
ROS2_TOPIC_AMBIENT_TEMPERATURE = 'pce_p18/temperature'
ROS2_TOPIC_RELATIVE_HUMIDITY = 'pce_p18/rel_humidity' 
ROS2_TOPIC_ABSOLUTE_HUMIDITY = 'pce_p18/abs_humidity'
ROS2_TOPIC_DEW_POINT = 'pce_p18/dew_point'
ROS2_TOPIC_UTM_BASELINK = '/tf_utm_baselink'
ROS2_TOPIC_GEOMETRIC_MEASUREMENTS = '/plant/geometric_measurements'

MQTT_GLOBAL_TOPIC = "mqtt/global"

class Ros2MqttPublisher(Node):
    def __init__(self, host=MQTT_DEFAULT_HOST, port=MQTT_DEFAULT_PORT):
        super().__init__('ros2_mqtt_publisher')

        # --- MQTT Setup ---
        self.mqtt_host = host
        self.mqtt_port = port
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.on_connect = self.on_server_connection
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        try:
            self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, MQTT_DEFAULT_TIMEOUT)
            self.mqtt_client.loop_start() 
        except Exception as e:
            self.get_logger().error(f"Failed to connect to MQTT broker: {e}")

        self.is_connected = False
        self._last_reconnect_attempt = 0.0
        self.last_data_activity = 0  # Initialize to 0 so the protection logic works
        self._publish_log_counter = 0

        # --- Data Buffers ---
        # Inicializamos a None
        self.latest_data = {
            "gps": None, "temperature": None, "ndvi": None, "heading": None,
            "area": None, "location": None, "biomass": None, "light_state": None,
            "crop_type": None, "ambient_temperature": None, "relative_humidity": None,
            "absolute_humidity": None, "dew_point": None, "tf_position": None,
            "geometric_measurements": None,
        }

        self.detected_plants = {} 

        # --- CSV Setup ---
        home = os.path.expanduser("~/tfm_ws/data_collection")
        os.makedirs(home, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.join(home, f"Data_collection_{timestamp}.csv")
        self.csv_fields = [
            "ts",
            "gps_lat","gps_lon","gps_alt", "gps_service", "gps_status",
            "temperature_canopy","temperature_cwsi",
            "ndvi", "ndvi3d", "ndvi_ir", "ndvi_visible","heading_deg","area","location",
            "biomass","crop_light_state","crop_type",
            "ambient_temperature","relative_humidity","absolute_humidity", "dew_point",
            "tf_utm_baselink_X","tf_utm_baselink_Y"
        ]
        
        with open(self.csv_file, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writeheader()

        self.subscribe_to_topics()
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self.publish_cycle)

    def on_server_connection(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self.get_logger().info("Connected to MQTT broker.")
            self.is_connected = True

    def on_disconnect(self, client, userdata, disconnect_flags, reason_code=None, properties=None):
        self.is_connected = False

    def subscribe_to_topics(self):
        qos = QoSProfile(depth=10)
        self.create_subscription(NavSatFix, ROS2_TOPIC_GPS, self.gps_callback, qos)
        self.create_subscription(String, ROS2_TOPIC_TEMPERATURE, self.temperature_callback, qos)
        self.create_subscription(String, ROS2_TOPIC_NDVI, self.ndvi_callback, qos)
        self.create_subscription(Imu, ROS2_TOPIC_HEADING, self.heading_callback, 10)
        self.create_subscription(Float32, ROS2_TOPIC_AREA, self.area_callback, 10)
        self.create_subscription(String, ROS2_TOPIC_LOCATION, self.location_callback, 10)
        self.create_subscription(Float32, ROS2_TOPIC_BIOMASS, self.biomass_callback, 10)
        self.create_subscription(String, ROS2_TOPIC_CROP_LIGHT_STATE, self.crop_light_state_callback, 10)
        self.create_subscription(String, ROS2_TOPIC_CROP_TYPE, self.crop_type_callback, 10)
        self.create_subscription(Temperature, ROS2_TOPIC_AMBIENT_TEMPERATURE, self.ambient_temperature_callback, 10)
        self.create_subscription(RelativeHumidity, ROS2_TOPIC_RELATIVE_HUMIDITY, self.relative_humidity_callback, 10)
        self.create_subscription(Temperature, ROS2_TOPIC_ABSOLUTE_HUMIDITY, self.absolute_humidity_callback, 10)
        self.create_subscription(Temperature, ROS2_TOPIC_DEW_POINT, self.dew_point_callback, 10)
        self.create_subscription(PointStamped, ROS2_TOPIC_UTM_BASELINK, self.utm_baselink_callback, 10)
        self.create_subscription(String, ROS2_TOPIC_GEOMETRIC_MEASUREMENTS, self.geometric_measurements_callback, 10)

    def ensure_mqtt_connected(self):
        self.last_data_activity = self.get_clock().now().to_msg().sec
        # Lógica de reconexión simplificada para brevedad
        if not self.is_connected and (self.last_data_activity - self._last_reconnect_attempt > RECONNECT_ATTEMPT_INTERVAL):
            try:
                self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, MQTT_DEFAULT_TIMEOUT)
                self._last_reconnect_attempt = self.last_data_activity
            except: pass

    # ### HELPER: Extraer timestamp del mensaje si existe, sino usar el actual
    def get_msg_time(self, msg):
        if hasattr(msg, 'header'):
            # Convertir stamp (sec, nanosec) a float seconds
            return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        else:
            # Para mensajes sin header (String, Float32), usamos el tiempo de recepción
            return self.get_clock().now().to_msg().sec

    # ================= Callbacks =================
    
    def gps_callback(self, msg: NavSatFix):
        self.latest_data["gps"] = {
            "msg_type": "gps",
            "ts": self.get_msg_time(msg), 
            "latitude": msg.latitude, "longitude": msg.longitude, "altitude": msg.altitude,
            "status": msg.status.status, "service": msg.status.service
        }
        self.ensure_mqtt_connected()

    def temperature_callback(self, msg: String):
            try:
                # 1. Parse the message
                match = re.search(r'Objeto\s*(\d+).*?Temperatura\s*=\s*(-?[\d.]+).*?CSWI\s*=\s*(-?[\d.]+)(?:.*?Area\s*=\s*(-?[\d.]+))?', msg.data)
                if match:
                    obj_id, temp, cwsi, area = match.groups()
                    current_ts = self.get_msg_time(msg)
                    
                    # --- NEW LOGIC: Clear old detections if this is a new frame ---
                    # If the timestamp changes (more than a tiny jitter), it's a new camera frame.
                    # We check against the stored timestamp in latest_data.
                    last_temp_data = self.latest_data.get("temperature")
                    if last_temp_data and abs(current_ts - last_temp_data["ts"]) > 0.01:
                        self.detected_plants = {} 
                    # --------------------------------------------------------------

                    entry = {"id": f"Objeto_{obj_id.strip()}", "canopy_temperature": float(temp), "cwsi": float(cwsi)}
                    if area: entry["area"] = float(area)
                    
                    self.detected_plants[entry["id"]] = entry
                    all_objects = list(self.detected_plants.values())
                    
                    self.latest_data["temperature"] = {
                        "msg_type": "temperature",
                        "ts": current_ts, 
                        "entity_count": len(all_objects),
                        "plants": all_objects,
                        "avg_temp": sum(o["canopy_temperature"] for o in all_objects) / len(all_objects),
                        "avg_cwsi": sum(o["cwsi"] for o in all_objects) / len(all_objects)
                    }
                    self.ensure_mqtt_connected()
            except Exception as e:
                self.get_logger().error(f"Error in temperature_callback: {e}")

    def ndvi_callback(self, msg: String):
        try:
            v = [float(x.strip()) for x in msg.data.split(',')]
            if len(v) >= 4:
                self.latest_data["ndvi"] = {
                    "msg_type": "ndvi", 
                    "ts": self.get_msg_time(msg),
                    "ndvi": v[0], "ndvi3d": v[1], "ndvi_ir": v[2], "ndvi_visible": v[3]
                }
                self.ensure_mqtt_connected()
        except: pass

    def area_callback(self, msg: Float32):
        self.latest_data["area"] = {"msg_type": "area", "ts": self.get_msg_time(msg), "area": msg.data}
        self.ensure_mqtt_connected()

    def heading_callback(self, msg: Imu):
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        self.latest_data["heading"] = {"msg_type": "heading", "ts": self.get_msg_time(msg), "heading_deg": yaw}
        self.ensure_mqtt_connected()

    def location_callback(self, msg: String):
        self.latest_data["location"] = {"msg_type": "location", "ts": self.get_msg_time(msg), "location": msg.data}
        self.ensure_mqtt_connected()
    
    def biomass_callback(self, msg: Float32):
        self.latest_data["biomass"] = {"msg_type": "biomass", "ts": self.get_msg_time(msg), "biomass": msg.data}
        self.ensure_mqtt_connected()

    def crop_light_state_callback(self, msg: String):
        self.latest_data["light_state"] = {"msg_type": "light_state", "ts": self.get_msg_time(msg), "crop_light_state": msg.data}
        self.ensure_mqtt_connected()

    def crop_type_callback(self, msg: String):
        self.latest_data["crop_type"] = {"msg_type": "crop_type", "ts": self.get_msg_time(msg), "crop_type": msg.data}
        self.ensure_mqtt_connected()

    def ambient_temperature_callback(self, msg: Temperature):
        self.latest_data["ambient_temperature"] = {"msg_type": "ambient_temperature", "ts": self.get_msg_time(msg), "ambient_temperature": msg.temperature}
        self.ensure_mqtt_connected()

    def relative_humidity_callback(self, msg: RelativeHumidity):
        self.latest_data["relative_humidity"] = {"msg_type": "relative_humidity", "ts": self.get_msg_time(msg), "relative_humidity": msg.relative_humidity}
        self.ensure_mqtt_connected()

    def absolute_humidity_callback(self, msg: Temperature):
        self.latest_data["absolute_humidity"] = {"msg_type": "absolute_humidity", "ts": self.get_msg_time(msg), "absolute_humidity": msg.temperature}
        self.ensure_mqtt_connected()

    def dew_point_callback(self, msg: Temperature):
        self.latest_data["dew_point"] = {"msg_type": "dew_point", "ts": self.get_msg_time(msg), "dew_point": msg.temperature}
        self.ensure_mqtt_connected()

    def utm_baselink_callback(self, msg: PointStamped):
        self.latest_data["tf_position"] = {"msg_type": "tf_position", "ts": self.get_msg_time(msg), "x": msg.point.x, "y": msg.point.y, "z": msg.point.z}
        self.ensure_mqtt_connected()

    def geometric_measurements_callback(self, msg: String):
        """Receive per-plant JSON from area_segmentation_node and store for MQTT publication."""
        try:
            data = json.loads(msg.data)
            # Enrich with timestamp and ensure msg_type is present
            data["msg_type"] = "geometric_measurements"
            data["ts"] = self.get_msg_time(msg)
            self.latest_data["geometric_measurements"] = data
            self.ensure_mqtt_connected()
        except Exception as e:
            self.get_logger().error(f"Error in geometric_measurements_callback: {e}")

    def publish_cycle(self):
            # 1. Get current time for the CSV row
            now = self.get_clock().now().to_msg()
            current_wall_time = now.sec + (now.nanosec * 1e-9)
            
            # --- CSV LOGIC (Independent of MQTT) ---
            # Initialize row with last known data from our buffer
            csv_row = {k: "" for k in self.csv_fields}
            csv_row["ts"] = current_wall_time

            # Map current buffer state to CSV
            # GPS
            gps = self.latest_data.get("gps")
            if gps:
                csv_row.update({
                    "gps_lat": gps.get("latitude"), "gps_lon": gps.get("longitude"),
                    "gps_alt": gps.get("altitude"), "gps_status": gps.get("status"),
                    "gps_service": gps.get("service")
                })

            # Temperature (Canopy)
            temp = self.latest_data.get("temperature")
            if temp:
                csv_row["temperature_canopy"] = round(temp.get("avg_temp", 0), 2)
                csv_row["temperature_cwsi"] = round(temp.get("avg_cwsi", 0), 2)

            # NDVI
            ndvi = self.latest_data.get("ndvi")
            if ndvi:
                csv_row.update({
                    "ndvi": ndvi.get("ndvi"), "ndvi3d": ndvi.get("ndvi3d"),
                    "ndvi_ir": ndvi.get("ndvi_ir"), "ndvi_visible": ndvi.get("ndvi_visible")
                })

            # Simple Fields
            if self.latest_data.get("heading"): csv_row["heading_deg"] = self.latest_data["heading"].get("heading_deg")
            if self.latest_data.get("area"): csv_row["area"] = self.latest_data["area"].get("area")
            if self.latest_data.get("location"): csv_row["location"] = self.latest_data["location"].get("location")
            if self.latest_data.get("biomass"): csv_row["biomass"] = self.latest_data["biomass"].get("biomass")
            if self.latest_data.get("light_state"): csv_row["crop_light_state"] = self.latest_data["light_state"].get("crop_light_state")
            if self.latest_data.get("crop_type"): csv_row["crop_type"] = self.latest_data["crop_type"].get("crop_type")
            if self.latest_data.get("ambient_temperature"): csv_row["ambient_temperature"] = self.latest_data["ambient_temperature"].get("ambient_temperature")
            if self.latest_data.get("relative_humidity"): csv_row["relative_humidity"] = self.latest_data["relative_humidity"].get("relative_humidity")
            if self.latest_data.get("absolute_humidity"): csv_row["absolute_humidity"] = self.latest_data["absolute_humidity"].get("absolute_humidity")
            if self.latest_data.get("dew_point"): csv_row["dew_point"] = self.latest_data["dew_point"].get("dew_point")
            
            # TF Position
            tf = self.latest_data.get("tf_position")
            if tf:
                csv_row["tf_utm_baselink_X"] = tf.get("x")
                csv_row["tf_utm_baselink_Y"] = tf.get("y")

            # ALWAYS write to CSV (2 times per second)
            try:
                with open(self.csv_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                    writer.writerow(csv_row)
            except Exception as e:
                self.get_logger().error(f"CSV Write Error: {e}")

            # --- MQTT LOGIC (Kept separate/Untouched behavior) ---
            # We only publish via MQTT if there is actually data in the buffers
            published_any = False
            if self.is_connected:
                for key, data in self.latest_data.items():
                    if data is not None:
                        payload = json.dumps(data)
                        self.mqtt_client.publish(MQTT_GLOBAL_TOPIC, payload, qos=0)
                        published_any = True
                        
            if published_any:
                self._publish_log_counter += 1
                if self._publish_log_counter % 10 == 0:
                    self.get_logger().info("Actively publishing data to MQTT broker...")
                
            # IMPORTANT: We no longer set latest_data[key] = None
            # This keeps the CSV full even if a topic doesn't send a message in a specific 0.5s window.

def main(args=None):
    rclpy.init(args=args)
    node = Ros2MqttPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mqtt_client.disconnect()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()