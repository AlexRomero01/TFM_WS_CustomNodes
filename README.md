# Custom Nodes for ROS2

This workspace contains custom ROS2 nodes for various functionalities, including sensor data processing, computer vision, and communication.


## Dependencies

This package depends on the following ROS2 packages:
- `rclcpp`
- `sensor_msgs`
- `cv_bridge`
- `image_transport`
- `realsense2_camera`
- `ultralytics_ros`
- `optris_drivers2`
- `launch`
- `launch_ros`

## Usage

To launch the main application, run the following command:

```bash
ros2 launch custom_nodes main_launch.py
```

This will launch all the necessary nodes, including the camera drivers, YOLO object detection, and the custom processing nodes.

The main launch file (`main_launch.py`) accepts several arguments to customize the execution:

- `align_depth`: set to `true` to align the depth and color streams. Default: `false`.
- `pointcloud_enable`: set to `true` to enable point cloud generation. Default: `false`.
- `debug`: set to `true` to launch in debug mode. Default: `false`.
- `focus`: sets the focus of the Optris camera. Default: `70`.
- `yolo_model`: the YOLO model to be used for object detection. Default: `TFM_NoAugmentation-seg.pt`.
- `rgbd_resolution`: the resolution of the RGBD camera. Default: `640,480,30`.
- `homography_file`: the homography file to be used. Default: `1280`.
- `name_class`: the name of the class to be detected by YOLO. Default: `plant`.
- `number_class`: the number of the class to be detected by YOLO. Default: `0`.

For example, to launch with depth alignment and point cloud enabled, you can run:

```bash
ros2 launch custom_nodes main_launch.py align_depth:=true pointcloud_enable:=true
```

## Scripts

The `scripts` directory contains several Python scripts that implement the core functionalities of this package.

- `ros2_mqtt_publisher.py`: This script subscribes to various ROS2 topics, collects data, and publishes it to an MQTT broker. It also logs the data to a CSV file.
- `area_segmentation_node.py`: This script performs area segmentation on images.
- `biomass_node.py`: This script calculates the biomass of plants.
- `crop_light_state_node.py`: This script determines the light state of the crops (sun or shade).
- `GPS_coordinates_node.py`: This script reads GPS coordinates and publishes them.
- `higrometer_node.py`: This script reads data from a hygrometer.
- `modify_xml.py`: This script modifies the configuration file for the Optris camera.
- `record_temperature.py`: This script records temperature data.
- `sun_shadow_calibration.py`: This script performs sun/shadow calibration.
- `temperature_cswi_calculation.py`: This script calculates the Crop Water Stress Index (CWSI).
- `utm_base_link_xy.py`: This script publishes the TF transform between the UTM and base_link frames.
