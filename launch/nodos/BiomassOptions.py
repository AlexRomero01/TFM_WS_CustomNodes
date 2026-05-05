from nodos.NodeOptions import NodeOptions
import tkinter as tk
from tkinter import ttk
import threading
import rclpy
from std_msgs.msg import Float32, String

class BiomassOptions(NodeOptions):
    def __init__(self):
        super().__init__(
            package='custom_nodes',
            executable='~/tfm_ws/src/custom_nodes/scripts/biomass_node.py',
            node_name='biomass_node'
        )
        self.internal_node = True
        self.thread = None
        self.node = None
        self.running = False

        self.status_indicator = None
        self.status_label = None
        self.combo_box = None
        self.publisher = None

    def create_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Biomass Estimation Node")
        frame.pack(fill='x', padx=10, pady=10)

        # LED-style indicator
        self.status_indicator = tk.Canvas(frame, width=20, height=20, bg='black', highlightthickness=1, highlightbackground="gray")
        self.status_indicator.pack(side='left', padx=10, pady=5)

        # Dropdown for selecting plant type
        self.combo_box = ttk.Combobox(frame, values=["lettuce", "other"], state="readonly")
        self.combo_box.current(0)
        self.combo_box.pack(side='left', padx=10)
        self.combo_box.bind("<<ComboboxSelected>>", self._publish_selection)

        # Label to show last received biomass value
        self.status_label = ttk.Label(frame, text="No data received")
        self.status_label.pack(side='left', padx=10)

    def start(self):
        if self.running:
            print("[BiomassOptions] Node is already running")
            return
        self.running = True
        self.thread = threading.Thread(target=self.listen_to_topic, daemon=True)
        self.thread.start()
        print("[BiomassOptions] Node started")
        self._update_indicator("green")

    def stop(self):
        if not self.running:
            print("[BiomassOptions] Node is not running")
            return
        self.running = False
        if self.node:
            self.node.destroy_node()
            self.node = None
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"[BiomassOptions] Error while shutting down rclpy: {e}")
        print("[BiomassOptions] Node stopped")
        self._update_indicator("black")
        self._update_status("No data received")

    def listen_to_topic(self):
        try:
            rclpy.init(args=None)
            self.node = rclpy.create_node('biomass_gui_listener')

            self.publisher = self.node.create_publisher(String, '/biomass_type', 10)

            # Publish the default selection once
            self._publish_selection()

            def callback(msg):
                biomass = msg.data
                self._update_status(f"{biomass:.2f} m³")
                print(f"[GUI] Received biomass: {biomass:.2f}")  # Debug

            self.node.create_subscription(Float32, '/plant_biomass', callback, 10)

            while self.running and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)

            if self.node:
                self.node.destroy_node()
                self.node = None
            rclpy.shutdown()
        except Exception as e:
            print(f"[BiomassOptions] Exception in listen_to_topic: {e}")
            self.running = False

    def _publish_selection(self, event=None):
        if self.publisher and self.combo_box:
            selected = self.combo_box.get()
            msg = String()
            msg.data = selected
            self.publisher.publish(msg)
            print(f"[BiomassOptions] Published plant type: {selected}")

    def _update_indicator(self, color):
        if self.status_indicator:
            self.status_indicator.after(0, lambda: self.status_indicator.config(bg=color))

    def _update_status(self, text):
        if self.status_label:
            self.status_label.after(0, lambda: self.status_label.config(text=text))
