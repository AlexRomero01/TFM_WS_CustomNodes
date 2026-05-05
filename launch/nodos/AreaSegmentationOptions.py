from nodos.NodeOptions import NodeOptions
import tkinter as tk
from tkinter import ttk
import threading
import rclpy
from std_msgs.msg import Float32

class AreaSegmentationOptions(NodeOptions):
    def __init__(self):
        super().__init__(
            package='custom_nodes',
            executable='~/tfm_ws/src/custom_nodes/scripts/area_segmentation_node.py',
            node_name='area_segment_node'
        )
        self.internal_node = True
        self.thread = None
        self.node = None
        self.running = False

        self.status_indicator = None  # Visual indicator (LED-style square)
        self.status_label = None      # Label to show last received value

    def create_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Area Segmentation Node")
        frame.pack(fill='x', padx=10, pady=10)

        # LED-style status indicator (black = inactive, green = running)
        self.status_indicator = tk.Canvas(frame, width=20, height=20, bg='black', highlightthickness=1, highlightbackground="gray")
        self.status_indicator.pack(side='left', padx=10, pady=5)

        # Label to show the last received value from topic
        self.status_label = ttk.Label(frame, text="No data received")
        self.status_label.pack(side='left', padx=10)

    def start(self):
        if self.running:
            print("[AreaSegmentationOptions] Node is already running")
            return
        self.running = True
        self.thread = threading.Thread(target=self.listen_to_topic, daemon=True)
        self.thread.start()
        print("[AreaSegmentationOptions] Node started")
        self._update_indicator("green")  # Green = node running

    def stop(self):
        if not self.running:
            print("[AreaSegmentationOptions] Node is not running")
            return
        self.running = False
        if self.node:
            self.node.destroy_node()
            self.node = None
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"[AreaSegmentationOptions] Error while shutting down rclpy: {e}")
        print("[AreaSegmentationOptions] Node stopped")
        self._update_indicator("black")  # Black = node stopped
        self._update_status("No data received")

    def listen_to_topic(self):
        try:
            rclpy.init(args=None)
            self.node = rclpy.create_node('area_segmentation_gui_listener')

            def callback(msg):
                percentage = msg.data
                self._update_status(f"{percentage:.2f}%")
                print(f"[GUI] Received: {percentage:.2f}%")  # Debug

            self.node.create_subscription(Float32, '/segmentation_area_info', callback, 10)

            while self.running and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)

            if self.node:
                self.node.destroy_node()
                self.node = None
            rclpy.shutdown()
        except Exception as e:
            print(f"[AreaSegmentationOptions] Exception in listen_to_topic: {e}")
            self.running = False

    def _update_indicator(self, color):
        if self.status_indicator:
            self.status_indicator.after(0, lambda: self.status_indicator.config(bg=color))

    def _update_status(self, text):
        if self.status_label:
            self.status_label.after(0, lambda: self.status_label.config(text=text))

