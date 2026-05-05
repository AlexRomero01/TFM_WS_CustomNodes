from nodos.NodeOptions import NodeOptions
import tkinter as tk
from tkinter import ttk
import threading
import rclpy
from std_msgs.msg import String


class GPSCoordinatesOptions(NodeOptions):
    def __init__(self):
        super().__init__(
            package='custom_nodes',
            executable='~/tfm_ws/src/custom_nodes/scripts/GPS_coordinates_node.py',
            node_name='GPS_coordinates_node'
        )
        self.internal_node = True
        self.thread = None
        self.node = None
        self.running = False

        self.status_indicator = None  # LED-style indicator
        self.status_label = None      # Label with last received value

    def create_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="GPS Coordinates Node")
        frame.pack(fill='x', padx=10, pady=10)

        # LED-style indicator
        self.status_indicator = tk.Canvas(
            frame, width=20, height=20, bg='black',
            highlightthickness=1, highlightbackground="gray"
        )
        self.status_indicator.pack(side='left', padx=10, pady=5)

        # Status label
        self.status_label = ttk.Label(frame, text="No data received")
        self.status_label.pack(side='left', padx=10)

    def start(self):
        if self.running:
            print("[GPSCoordinatesOptions] Node is already running")
            return
        self.running = True
        self.thread = threading.Thread(target=self.listen_to_topic, daemon=True)
        self.thread.start()
        print("[GPSCoordinatesOptions] Node started")
        self._update_indicator("green")  # Green = running

    def stop(self):
        if not self.running:
            print("[GPSCoordinatesOptions] Node is not running")
            return
        self.running = False
        if self.node:
            self.node.destroy_node()
            self.node = None
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"[GPSCoordinatesOptions] Error while shutting down rclpy: {e}")
        print("[GPSCoordinatesOptions] Node stopped")
        self._update_indicator("black")  # Black = stopped
        self._update_status("No data received")

    def listen_to_topic(self):
        try:
            rclpy.init(args=None)
            self.node = rclpy.create_node('gps_coordinates_gui_listener')

            def callback(msg):
                info = msg.data
                self._update_status(info)
                print(f"[GUI] Received: {info}")  # Debug log

            self.node.create_subscription(
                String, '/navigation/information', callback, 10
            )

            while self.running and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)

            if self.node:
                self.node.destroy_node()
                self.node = None
            rclpy.shutdown()
        except Exception as e:
            print(f"[GPSCoordinatesOptions] Exception in listen_to_topic: {e}")
            self.running = False

    def _update_indicator(self, color):
        if self.status_indicator:
            self.status_indicator.after(
                0, lambda: self.status_indicator.config(bg=color)
            )

    def _update_status(self, text):
        if self.status_label:
            self.status_label.after(
                0, lambda: self.status_label.config(text=text)
            )

