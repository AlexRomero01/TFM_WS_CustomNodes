"""Launch file for the agrivoltaic crop-monitoring pipeline.

Supports two execution modes controlled by the `use_sim_time` argument:
  - Live mode  (use_sim_time=false): starts all hardware drivers
    (RealSense, Optris thermal camera, higrometer, NDVI sensor).
  - Rosbag mode (use_sim_time=true): skips hardware drivers; all
    processing nodes still run against the replayed topics.

Launch arguments
----------------
use_sim_time      : bool  — true for rosbag playback, false for live sensors
align_depth       : bool  — enable RealSense depth alignment
pointcloud_enable : bool  — enable RealSense point cloud
debug             : bool  — enable YOLO debug output
focus             : int   — Optris camera focus value
yolo_model        : str   — YOLO model filename (.pt)
rgbd_resolution   : str   — RealSense RGB+D profile  (WxHxFPS, default 640,480,15)
homography_file   : str   — homography key ('1280' or '640')
name_class        : str   — YOLO class name filter
number_class      : int   — YOLO class index filter
publish_mqtt      : bool  — enable the MQTT publisher node
"""
import os
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
    OpaqueFunction
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
    AnyLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():

    # -------------------------------------------------------------------------
    # Launch arguments
    # -------------------------------------------------------------------------
    align_depth_launch_arg        = DeclareLaunchArgument("align_depth",        default_value='false')
    pointcloud_enable_launch_arg  = DeclareLaunchArgument("pointcloud_enable",  default_value='false')
    debug_launch_arg              = DeclareLaunchArgument("debug",               default_value='false')
    focus_launch_arg              = DeclareLaunchArgument("focus",               default_value='70')
    yolo_model_launch_arg         = DeclareLaunchArgument("yolo_model",          default_value='TFM_YOLO26_augmentation-seg.pt')
    rgbd_resolution_launch_arg    = DeclareLaunchArgument("rgbd_resolution",     default_value='640,480,15')
    homography_file_launch_arg    = DeclareLaunchArgument("homography_file",     default_value='1280')
    name_class_id_launch_arg      = DeclareLaunchArgument("name_class",          default_value='plant')
    number_class_id_launch_arg    = DeclareLaunchArgument("number_class",        default_value='0')
    publish_mqtt_launch_arg       = DeclareLaunchArgument("publish_mqtt",        default_value='true')

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value='false',
        description="True for rosbag playback (disables hardware drivers), False for live sensors"
    )

    # =========================================================================
    # HARDWARE DRIVERS — only when use_sim_time=false
    # =========================================================================

    # RealSense D457
    launch_include_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'align_depth.enable':          LaunchConfiguration('align_depth'),
            'pointcloud.enable':           LaunchConfiguration('pointcloud_enable'),
            'rgb_camera.color_profile':    LaunchConfiguration('rgbd_resolution'),
            'depth_module.depth_profile':  LaunchConfiguration('rgbd_resolution'),
        }.items(),
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_before_launches = TimerAction(period=0.5, actions=[launch_include_1])

    # Optris thermal camera — update config.xml focus value before launch
    focus_value = LaunchConfiguration('focus')
    config_file_path    = os.path.expanduser('~/tfm_ws/src/custom_nodes/Homography/config.xml')
    modify_xml_file_path = os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/modify_xml.py')

    modify_xml_process = ExecuteProcess(
        cmd=['python3', modify_xml_file_path, config_file_path, focus_value],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_config = TimerAction(period=0.5, actions=[modify_xml_process])

    optris_imager_node = ExecuteProcess(
        cmd=['xterm', '-hold', '-e', 'ros2', 'run', 'optris_drivers2',
             'optris_imager_node', config_file_path],
        shell=True,
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_node_1 = TimerAction(period=1.5, actions=[optris_imager_node])

    optris_colorconvert_node = ExecuteProcess(
        cmd=['xterm', '-hold', '-e', 'ros2', 'run', 'optris_drivers2',
             'optris_colorconvert_node'],
        shell=True,
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_node_2 = TimerAction(period=2.5, actions=[optris_colorconvert_node])

    # Higrometer (serial sensor — hardware only)
    higrometer_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/higrometer_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True,
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_higrometer_node = TimerAction(period=6.0, actions=[higrometer_node])

    # NDVI sensor (hardware only)
    ndvi_node_process = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/ndvi_sensor/scripts/ndvi_sensor_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True,
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_ndvi_node = TimerAction(period=6.0, actions=[ndvi_node_process])

    # GPS row-detector node (always enabled — works with live GPS or replayed /gps/fix)
    GPS_coordinates_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/GPS_coordinates_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_gps_node = TimerAction(period=6.0, actions=[GPS_coordinates_node])

    # =========================================================================
    # PROCESSING & VISUALISATION NODES — always run (live and rosbag modes)
    # =========================================================================

    # YOLO tracker
    launch_include_2 = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ultralytics_ros'), 'launch', 'tracker.launch.xml'
            ])
        ]),
        launch_arguments={
            'debug':        LaunchConfiguration('debug'),
            'yolo_model':   LaunchConfiguration('yolo_model'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items()
    )
    delay_between_launches = TimerAction(period=3.0, actions=[launch_include_2])

    # Temperature + CWSI calculation node
    script_path = os.path.expanduser(
        '~/tfm_ws/src/custom_nodes/scripts/temperature_cswi_calculation.py')
    mean_processor2_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', script_path,
            '--homography_file', LaunchConfiguration('homography_file'),
            '--name_class',      LaunchConfiguration('name_class'),
            '--number_class',    LaunchConfiguration('number_class'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_py_node_2 = TimerAction(period=6.0, actions=[mean_processor2_node])

    # RViz2
    rviz_config_file = os.path.expanduser(
        '~/tfm_ws/src/custom_nodes/launch/rviz2_tfm_config.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )
    delay_rviz = TimerAction(period=9.0, actions=[rviz_node])

    # CSV safe-copy node
    csv_safecopy_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/csv_safecopy.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_csv_safecopy_node = TimerAction(period=6.0, actions=[csv_safecopy_node])

    # 2-D area segmentation node
    area_segment_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/area_segmentation_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_area_node = TimerAction(period=6.0, actions=[area_segment_node])

    # Crop light state classifier
    crop_light_state_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/crop_light_state_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_crop_light_state_node = TimerAction(period=6.0, actions=[crop_light_state_node])

    # RANSAC ground removal node
    ransac_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/RANSAC.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_ransac_node = TimerAction(period=6.0, actions=[ransac_node])

    # Volume estimation node (depends on RANSAC plane — delayed 1 s more)
    volume_estim_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/volume_estimation_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_volume_estim_node = TimerAction(period=7.0, actions=[volume_estim_node])

    # MQTT publisher node (conditional — skipped when publish_mqtt=false)
    mqtt_node_process = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/ros2_mqtt_publisher.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_mqtt_node = TimerAction(period=6.0, actions=[mqtt_node_process])

    def launch_mqtt_if_enabled(context, *args, **kwargs):
        if LaunchConfiguration('publish_mqtt').perform(context).lower() == 'true':
            return [delay_mqtt_node]
        return []

    conditional_mqtt_launch = OpaqueFunction(function=launch_mqtt_if_enabled)

    # UTM → base_link TF publisher
    utm_base_link_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3',
            os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/utm_base_link_xy.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_utm_node = TimerAction(period=6.0, actions=[utm_base_link_node])

    # =========================================================================
    # Assemble LaunchDescription
    # =========================================================================
    return LaunchDescription([
        use_sim_time_arg,
        align_depth_launch_arg,
        pointcloud_enable_launch_arg,
        debug_launch_arg,
        focus_launch_arg,
        yolo_model_launch_arg,
        rgbd_resolution_launch_arg,
        homography_file_launch_arg,
        name_class_id_launch_arg,
        number_class_id_launch_arg,
        publish_mqtt_launch_arg,

        # Hardware drivers (skipped in rosbag/sim mode)
        delay_before_launches,
        delay_config,
        delay_node_1,
        delay_node_2,
        delay_higrometer_node,
        delay_ndvi_node,
        delay_gps_node,

        # Processing & visualisation (always run)
        delay_between_launches,
        delay_py_node_2,
        delay_rviz,
        delay_csv_safecopy_node,
        delay_area_node,
        delay_crop_light_state_node,
        delay_ransac_node,
        delay_volume_estim_node,
        conditional_mqtt_launch,
        delay_utm_node,
    ])