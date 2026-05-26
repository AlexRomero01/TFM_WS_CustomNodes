#!/usr/bin/env python3
"""
GPS_coordinates_node.py
========================
ROS 2 node that maps incoming GPS fixes to a pre-surveyed field layout
(sections and row corridors) and publishes the robot's current position
within the field as a human-readable string.

The field is divided into three sections (solar opaque, solar semi, open air),
each with numbered row corridors defined by North/South GPS endpoints.
The robot's direction of travel is inferred from consecutive GPS fixes
to determine which side of the corridor is being worked on.

Subscriptions:
    /gps/fix   sensor_msgs/NavSatFix

Publications:
    /navigation/information   std_msgs/String
    Format: "section: <name>, row: <rowN>, position_from_N: <m>, direction: <dir>"
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
from math import radians, cos, sin, sqrt, atan2

# ---------------------------------------------------------------------------
# Field layout — GPS endpoints for each row corridor
# ---------------------------------------------------------------------------
# Each corridor is defined by a North (N) and South (S) endpoint.
# Coordinates were surveyed on-site at the agrivoltaic test field.
rows = {
    "section1 (solar opaque)": {
        "row1-2":  {"N": {"lat": 41.28795045,    "lon": 2.04391535},  "S": {"lat": 41.287786,      "lon": 2.0439733}},
        "row2-3":  {"N": {"lat": 41.2879391,     "lon": 2.04390545},  "S": {"lat": 41.28777615,    "lon": 2.04396275}},
        "row3-4":  {"N": {"lat": 41.287934345,   "lon": 2.04387395},  "S": {"lat": 41.28777085,    "lon": 2.0439341}},
        "row4-5":  {"N": {"lat": 41.287933395,   "lon": 2.04384965},  "S": {"lat": 41.28776495,    "lon": 2.0439096}},
        "row5-6":  {"N": {"lat": 41.2879266,     "lon": 2.04383785},  "S": {"lat": 41.2877586,     "lon": 2.04389625}},
        "row6-7":  {"N": {"lat": 41.2879228,     "lon": 2.0438055},   "S": {"lat": 41.28775595,    "lon": 2.0438658}},
        "row7-8":  {"N": {"lat": 41.2879194,     "lon": 2.04378085},  "S": {"lat": 41.287752,      "lon": 2.0438422}},
        "row8-9":  {"N": {"lat": 41.2879124,     "lon": 2.04376925},  "S": {"lat": 41.28774585,    "lon": 2.04382865}},
        "row9-10": {"N": {"lat": 41.28790795,    "lon": 2.04374125},  "S": {"lat": 41.28774515,    "lon": 2.0437953}},
        "row10-11":{"N": {"lat": 41.2879038,     "lon": 2.0437127},   "S": {"lat": 41.2877464,     "lon": 2.04376645}},
        "row11-12":{"N": {"lat": 41.28789725,    "lon": 2.04369785},  "S": {"lat": 41.28774005,    "lon": 2.04375415}},
    },
    "section2 (solar semi)": {
        "row1-2":  {"N": {"lat": 41.28799645,    "lon": 2.04417645},  "S": {"lat": 41.2878341,     "lon": 2.04423515}},
        "row2-3":  {"N": {"lat": 41.2879938,     "lon": 2.0441546},   "S": {"lat": 41.28783065,    "lon": 2.0442118}},
        "row3-4":  {"N": {"lat": 41.28799155,    "lon": 2.04413305},  "S": {"lat": 41.28782815,    "lon": 2.0441896}},
        "row4-5":  {"N": {"lat": 41.2879811,     "lon": 2.0441115},   "S": {"lat": 41.2878246,     "lon": 2.0441662}},
        "row5-6":  {"N": {"lat": 41.2879662,     "lon": 2.04408135},  "S": {"lat": 41.2878162,     "lon": 2.04414025}},
        "row6-7":  {"N": {"lat": 41.28796375,    "lon": 2.04404955},  "S": {"lat": 41.2878106,     "lon": 2.0441147}},
        "row7-8":  {"N": {"lat": 41.2879671,     "lon": 2.04402915},  "S": {"lat": 41.28781125,    "lon": 2.04408985}},
    },
    "open air": {
        "row1-2":  {"N": {"lat": 41.288049995,   "lon": 2.04442805},  "S": {"lat": 41.28788735,    "lon": 2.0444869}},
        "row2-3":  {"N": {"lat": 41.288044795,   "lon": 2.0444113},   "S": {"lat": 41.28788565,    "lon": 2.04446855}},
        "row3-4":  {"N": {"lat": 41.288041595,   "lon": 2.0443915},   "S": {"lat": 41.2878844,     "lon": 2.04444585}},
        "row4-5":  {"N": {"lat": 41.2880357,     "lon": 2.04437205},  "S": {"lat": 41.2878797,     "lon": 2.0444266}},
        "row5-6":  {"N": {"lat": 41.288030145,   "lon": 2.04435095},  "S": {"lat": 41.2878699,     "lon": 2.04440825}},
        "row6-7":  {"N": {"lat": 41.28802909,    "lon": 2.044331},    "S": {"lat": 41.28786505,    "lon": 2.04438705}},
        "row7-8":  {"N": {"lat": 41.288024745,   "lon": 2.04430955},  "S": {"lat": 41.28785875,    "lon": 2.0443678}},
    }
}

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in metres between two GPS coordinates."""
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def gps_to_local(lat_ref, lon_ref, lat, lon):
    """Convert GPS coords to a local 2-D metric frame centred at (lat_ref, lon_ref)."""
    x = haversine(lat_ref, lon_ref, lat_ref, lon) * (1 if lon > lon_ref else -1)
    y = haversine(lat_ref, lon_ref, lat, lon_ref) * (1 if lat > lat_ref else -1)
    return x, y


def project_point_on_row(lat, lon, N, S):
    """
    Project (lat, lon) onto the N→S row segment.

    Returns:
        dist_from_N  : distance along the row from the North endpoint [m]
        dist_perp    : perpendicular distance from the row centre line [m]
        row_length   : total row length [m]
    """
    Nx, Ny = 0.0, 0.0
    Sx, Sy = gps_to_local(N["lat"], N["lon"], S["lat"], S["lon"])
    Px, Py = gps_to_local(N["lat"], N["lon"], lat, lon)

    dx, dy = Sx - Nx, Sy - Ny
    if dx == 0 and dy == 0:
        return 0.0, sqrt(Px ** 2 + Py ** 2), 0.0

    # Scalar projection onto the segment (clamped to [0, 1])
    t = ((Px - Nx) * dx + (Py - Ny) * dy) / (dx * dx + dy * dy)
    t_clamped = max(0, min(1, t))

    proj_x = Nx + t_clamped * dx
    proj_y = Ny + t_clamped * dy

    dist_from_N = sqrt(proj_x ** 2 + proj_y ** 2)
    dist_perp = sqrt((Px - proj_x) ** 2 + (Py - proj_y) ** 2)
    return dist_from_N, dist_perp, sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class GPSRowDetector(Node):
    def __init__(self):
        super().__init__('gps_row_detector')
        self.sub_gps = self.create_subscription(
            NavSatFix, '/gps/fix', self.listener_callback, 10)
        self.pub_info = self.create_publisher(String, '/navigation/information', 10)
        self.last_dist_from_N = None
        self.last_row = None
        self.get_logger().info("GPSRowDetector started.")

    def listener_callback(self, msg: NavSatFix):
        lat, lon = msg.latitude, msg.longitude

        # Find the closest row corridor by minimising perpendicular distance
        best_section = best_row = None
        best_dist_from_N = None
        min_perp_dist = float("inf")

        for section, section_rows in rows.items():
            for row_name, coords in section_rows.items():
                dist_from_N, dist_perp, _ = project_point_on_row(
                    lat, lon, coords["N"], coords["S"])
                if dist_perp < min_perp_dist:
                    min_perp_dist = dist_perp
                    best_section, best_row = section, row_name
                    best_dist_from_N = dist_from_N

        # Infer direction of travel from consecutive positions within the same corridor
        direction = "unknown"
        final_row_str = best_row

        if (self.last_row == (best_section, best_row)
                and self.last_dist_from_N is not None
                and abs(best_dist_from_N - self.last_dist_from_N) > 0.01):
            try:
                # Corridor name format: 'rowA-B' — pick the working row from direction
                row_parts = best_row.replace('row', '').split('-')
                if len(row_parts) == 2:
                    row_num1 = int(row_parts[0])
                    row_num2 = int(row_parts[1])

                    if best_dist_from_N > self.last_dist_from_N:
                        direction = "North → South"
                        final_row_str = f"row{row_num1}"
                    elif best_dist_from_N < self.last_dist_from_N:
                        direction = "South → North"
                        final_row_str = f"row{row_num2}"
            except (ValueError, IndexError):
                pass  # keep the corridor name if parsing fails

        self.last_dist_from_N = best_dist_from_N
        self.last_row = (best_section, best_row)

        # Publish location string (format consumed by ros_data.py parser)
        info_str = (
            f"section: {best_section}, "
            f"row: {final_row_str}, "
            f"position_from_N: {best_dist_from_N:.2f} m, "
            f"direction: {direction}"
        )
        self.get_logger().info(info_str)
        self.pub_info.publish(String(data=info_str))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = GPSRowDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
