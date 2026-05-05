#!/usr/bin/env python3
"""
CSV Safe Copy Node
==================
Subscribes to all sensor topics and writes data to a local CSV file.
Runs independently from MQTT publishing for decoupled operation.
"""

import re
import csv
import os
import threading
import queue
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from datetime import datetime
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Temperature, RelativeHumidity
from geometry_msgs.msg import PointStamped

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


class CsvSafeCopy(Node):
    def __init__(self):
        super().__init__('csv_safecopy')

        # --- Data Buffers ---
        self.latest_data = {
            "gps": None, "temperature": None, "ndvi": None, "heading": None,
            "area": None, "location": None, "biomass": None, "light_state": None,
            "crop_type": None, "ambient_temperature": None, "relative_humidity": None,
            "absolute_humidity": None, "dew_point": None, "tf_position": None
        }

        self.detected_plants = {} 

        # --- CSV writer queue & background worker ---
        self.csv_queue = queue.Queue()
        self._csv_thread = threading.Thread(target=self._csv_worker, daemon=True)
        self._csv_thread.start()

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

        self.get_logger().info(f"CSV file created: {self.csv_file}")

        self.subscribe_to_topics()
        self.create_timer(1.0, self.publish_cycle)  # 1Hz CSV write timer

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

    def _csv_worker(self):
        """Background worker that writes queued CSV rows to disk."""
        while True:
            try:
                row = self.csv_queue.get()
                if row is None:
                    break
                with open(self.csv_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                    writer.writerow(row)
            except Exception as e:
                try:
                    self.get_logger().error(f"CSV worker error: {e}")
                except Exception:
                    pass

    def get_msg_time(self, msg):
        """Extract timestamp from message header or use current time."""
        if hasattr(msg, 'header'):
            return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        else:
            return self.get_clock().now().to_msg().sec

    # ================= Callbacks =================
    
    def gps_callback(self, msg: NavSatFix):
        self.latest_data["gps"] = {
            "ts": self.get_msg_time(msg), 
            "latitude": msg.latitude, "longitude": msg.longitude, "altitude": msg.altitude,
            "status": msg.status.status, "service": msg.status.service
        }

    def temperature_callback(self, msg: String):
        try:
            match = re.search(r'Objeto\s*(\d+).*?Temperatura\s*=\s*(-?[\d.]+).*?CSWI\s*=\s*(-?[\d.]+)(?:.*?Area\s*=\s*(-?[\d.]+))?', msg.data)
            if match:
                obj_id, temp, cwsi, area = match.groups()
                current_ts = self.get_msg_time(msg)
                
                # Clear old detections if this is a new frame
                last_temp_data = self.latest_data.get("temperature")
                if last_temp_data and abs(current_ts - last_temp_data["ts"]) > 0.01:
                    self.detected_plants = {} 

                entry = {"id": f"Objeto_{obj_id.strip()}", "canopy_temperature": float(temp), "cwsi": float(cwsi)}
                if area: entry["area"] = float(area)
                
                self.detected_plants[entry["id"]] = entry
                all_objects = list(self.detected_plants.values())
                
                self.latest_data["temperature"] = {
                    "ts": current_ts, 
                    "entity_count": len(all_objects),
                    "plants": all_objects,
                    "avg_temp": sum(o["canopy_temperature"] for o in all_objects) / len(all_objects),
                    "avg_cwsi": sum(o["cwsi"] for o in all_objects) / len(all_objects)
                }
        except Exception as e:
            self.get_logger().error(f"Error in temperature_callback: {e}")

    def ndvi_callback(self, msg: String):
        try:
            v = [float(x.strip()) for x in msg.data.split(',')]
            if len(v) >= 4:
                self.latest_data["ndvi"] = {
                    "ts": self.get_msg_time(msg),
                    "ndvi": v[0], "ndvi3d": v[1], "ndvi_ir": v[2], "ndvi_visible": v[3]
                }
        except: pass

    def area_callback(self, msg: Float32):
        self.latest_data["area"] = {"ts": self.get_msg_time(msg), "area": msg.data}

    def heading_callback(self, msg: Imu):
        import math
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        self.latest_data["heading"] = {"ts": self.get_msg_time(msg), "heading_deg": yaw}

    def location_callback(self, msg: String):
        self.latest_data["location"] = {"ts": self.get_msg_time(msg), "location": msg.data}
    
    def biomass_callback(self, msg: Float32):
        self.latest_data["biomass"] = {"ts": self.get_msg_time(msg), "biomass": msg.data}

    def crop_light_state_callback(self, msg: String):
        self.latest_data["light_state"] = {"ts": self.get_msg_time(msg), "crop_light_state": msg.data}

    def crop_type_callback(self, msg: String):
        self.latest_data["crop_type"] = {"ts": self.get_msg_time(msg), "crop_type": msg.data}

    def ambient_temperature_callback(self, msg: Temperature):
        self.latest_data["ambient_temperature"] = {"ts": self.get_msg_time(msg), "ambient_temperature": msg.temperature}

    def relative_humidity_callback(self, msg: RelativeHumidity):
        self.latest_data["relative_humidity"] = {"ts": self.get_msg_time(msg), "relative_humidity": msg.relative_humidity}

    def absolute_humidity_callback(self, msg: Temperature):
        self.latest_data["absolute_humidity"] = {"ts": self.get_msg_time(msg), "absolute_humidity": msg.temperature}

    def dew_point_callback(self, msg: Temperature):
        self.latest_data["dew_point"] = {"ts": self.get_msg_time(msg), "dew_point": msg.temperature}

    def utm_baselink_callback(self, msg: PointStamped):
        self.latest_data["tf_position"] = {"ts": self.get_msg_time(msg), "x": msg.point.x, "y": msg.point.y, "z": msg.point.z}

    def publish_cycle(self):
        """Periodic CSV write cycle."""
        csv_row = {k: "" for k in self.csv_fields}
        data_processed = False

        for key, data in self.latest_data.items():
            if data is None:
                continue

            ts_to_use = data["ts"]
            csv_row["ts"] = ts_to_use 

            # Fill CSV fields
            if key == "gps":
                csv_row["gps_lat"] = data.get("latitude")
                csv_row["gps_lon"] = data.get("longitude")
                csv_row["gps_alt"] = data.get("altitude")
                csv_row["gps_status"] = data.get("status")
                csv_row["gps_service"] = data.get("service")
            
            elif key == "temperature":
                csv_row["temperature_canopy"] = round(data.get("avg_temp", 0), 2)
                csv_row["temperature_cwsi"] = round(data.get("avg_cwsi", 0), 2)
                self.detected_plants = {}

            elif key == "ndvi":
                csv_row["ndvi"] = data.get("ndvi")
                csv_row["ndvi3d"] = data.get("ndvi3d")
                csv_row["ndvi_ir"] = data.get("ndvi_ir")
                csv_row["ndvi_visible"] = data.get("ndvi_visible")
            elif key == "heading":
                csv_row["heading_deg"] = data.get("heading_deg")
            elif key == "area":
                csv_row["area"] = data.get("area")
            elif key == "location":
                csv_row["location"] = data.get("location")
            elif key == "biomass":
                csv_row["biomass"] = data.get("biomass")
            elif key == "light_state":
                csv_row["crop_light_state"] = data.get("crop_light_state")
            elif key == "crop_type":
                csv_row["crop_type"] = data.get("crop_type")
            elif key == "ambient_temperature":
                csv_row["ambient_temperature"] = data.get("ambient_temperature")
            elif key == "relative_humidity":
                csv_row["relative_humidity"] = data.get("relative_humidity")
            elif key == "absolute_humidity":
                csv_row["absolute_humidity"] = data.get("absolute_humidity")
            elif key == "dew_point":
                csv_row["dew_point"] = data.get("dew_point")
            elif key == "tf_position":
                csv_row["tf_utm_baselink_X"] = data.get("x")
                csv_row["tf_utm_baselink_Y"] = data.get("y")

            data_processed = True
            self.latest_data[key] = None

        if data_processed:
            try:
                self.csv_queue.put(csv_row)
            except Exception as e:
                self.get_logger().error(f"Error queueing CSV row: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CsvSafeCopy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if hasattr(node, 'csv_queue'):
                node.csv_queue.put(None)
            if hasattr(node, '_csv_thread'):
                node._csv_thread.join(timeout=1.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
