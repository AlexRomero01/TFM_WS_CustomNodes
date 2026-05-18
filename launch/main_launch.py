"""Launch adjusted for Rosbag Playback vs Live Mode"""
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
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition # <--- IMPORTANTE

def generate_launch_description():

    # --- Arguments ---
    align_depth_launch_arg = DeclareLaunchArgument("align_depth", default_value='false')
    pointcloud_enable_launch_arg = DeclareLaunchArgument("pointcloud_enable", default_value='false')
    debug_launch_arg = DeclareLaunchArgument("debug", default_value='false')
    focus_launch_arg = DeclareLaunchArgument("focus", default_value='70')
    yolo_model_launch_arg = DeclareLaunchArgument("yolo_model", default_value='TFM_YOLO26_augmentation-seg.pt')
    rgbd_resolution_launch_arg = DeclareLaunchArgument("rgbd_resolution", default_value='640,480,15')
    homography_file_launch_arg = DeclareLaunchArgument("homography_file", default_value='1280')
    name_class_id_launch_arg = DeclareLaunchArgument("name_class", default_value='plant')
    number_class_id_launch_arg = DeclareLaunchArgument("number_class", default_value='0')
    publish_mqtt_launch_arg = DeclareLaunchArgument("publish_mqtt", default_value='true')
    
    # Este argumento controla si arrancamos hardware o no
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", 
        default_value='false', 
        description="True for rosbags (disables hardware drivers), False for live sensors"
    )

    # =========================================
    # HARDWARE DRIVERS (Solo si use_sim_time es FALSE)
    # =========================================

    # --- Realsense Camera (HARDWARE) ---
    launch_include_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'align_depth.enable': LaunchConfiguration('align_depth'),
            'pointcloud.enable': LaunchConfiguration('pointcloud_enable'),
            'rgb_camera.color_profile': LaunchConfiguration('rgbd_resolution'),
            'depth_module.depth_profile': LaunchConfiguration('rgbd_resolution')
        }.items(),
        condition=UnlessCondition(LaunchConfiguration('use_sim_time')) # <--- Solo si NO es simulación
    )
    # Sin delay si es hardware, pero mantenemos la estructura
    delay_before_launches = TimerAction(period=0.5, actions=[launch_include_1])

    # --- Optris Camera Config (HARDWARE) ---
    focus_value = LaunchConfiguration('focus')
    config_file_path = os.path.expanduser('~/tfm_ws/src/custom_nodes/Homography/config.xml')
    modify_xml_file_path = os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/modify_xml.py')

    modify_xml_process = ExecuteProcess(
        cmd=['python3', modify_xml_file_path, config_file_path, focus_value],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_config = TimerAction(period=0.5, actions=[modify_xml_process])

    optris_imager_node = ExecuteProcess(
        cmd=['xterm', '-hold', '-e', 'ros2', 'run', 'optris_drivers2', 'optris_imager_node', config_file_path],
        shell=True,
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_node_1 = TimerAction(period=1.5, actions=[optris_imager_node])

    optris_colorconvert_node = ExecuteProcess(
        cmd=['xterm', '-hold', '-e', 'ros2', 'run', 'optris_drivers2', 'optris_colorconvert_node'],
        shell=True,
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_node_2 = TimerAction(period=2.5, actions=[optris_colorconvert_node])

    # --- Higrometer Node (HARDWARE) ---
    # Asumo que este lee serial. Si es un nodo de procesado, quita la condición.
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

    # --- NDVI Node (HARDWARE) ---
    # Este es el que daba el error en la foto. No lanzarlo con Rosbag.
    ndvi_node_process = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', os.path.expanduser('~/tfm_ws/src/ndvi_sensor/scripts/ndvi_sensor_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True,
        condition=UnlessCondition(LaunchConfiguration('use_sim_time'))
    )
    delay_ndvi_node = TimerAction(period=6.0, actions=[ndvi_node_process])
    
    # --- GPS Node (Always enabled) ---
    GPS_coordinates_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/GPS_coordinates_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_gps_node = TimerAction(period=6.0, actions=[GPS_coordinates_node])


    # =========================================
    # SOFTWARE / PROCESSING NODES (Siempre se ejecutan)
    # =========================================

    # --- YOLO Tracker ---
    launch_include_2 = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ultralytics_ros'),
                'launch',
                'tracker.launch.xml'
            ])
        ]),
        launch_arguments={
            'debug': LaunchConfiguration('debug'),
            'yolo_model': LaunchConfiguration('yolo_model'),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    delay_between_launches = TimerAction(period=3.0, actions=[launch_include_2])

    # --- Temperature + CWSI ---
    script_path = os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/temperature_cswi_calculation.py')
    mean_processor2_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', script_path,
            '--homography_file', LaunchConfiguration('homography_file'),
            '--name_class', LaunchConfiguration('name_class'),
            '--number_class', LaunchConfiguration('number_class'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')] 
        ],
        output='screen',
        shell=True
    )
    delay_py_node_2 = TimerAction(period=6.0, actions=[mean_processor2_node])

    # --- RViz ---
    rviz_config_file = os.path.expanduser('~/tfm_ws/src/custom_nodes/launch/rviz2_tfm_config.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )
    delay_rviz = TimerAction(period=9.0, actions=[rviz_node])

    # --- CSV SafeCopy Node (Always enabled) ---
    csv_safecopy_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/csv_safecopy.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_csv_safecopy_node = TimerAction(period=6.0, actions=[csv_safecopy_node])

    # --- Area Segmentation Node ---
    area_segment_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/area_segmentation_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_area_node = TimerAction(period=6.0, actions=[area_segment_node])

    # --- Crop Light State Node ---
    crop_light_state_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/crop_light_state_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_crop_light_state_node = TimerAction(period=6.0, actions=[crop_light_state_node])
    
    # --- RANSAC Ground Removal Node ---
    ransac_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/RANSAC.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_ransac_node = TimerAction(period=6.0, actions=[ransac_node])

    # --- Volume Estimation Node ---
    volume_estim_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/volume_estimation_node.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_volume_estim_node = TimerAction(period=7.0, actions=[volume_estim_node])

    # --- MQTT Publisher Node ---
    mqtt_node_process = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/ros2_mqtt_publisher.py'),
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

    # --- UTM → base_link TF Publisher ---
    utm_base_link_node = ExecuteProcess(
        cmd=[
            'xterm', '-hold', '-e', 'python3', os.path.expanduser('~/tfm_ws/src/custom_nodes/scripts/utm_base_link_xy.py'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]
        ],
        output='screen',
        shell=True
    )
    delay_utm_node = TimerAction(period=6.0, actions=[utm_base_link_node])


    # --- Return Launch Description ---
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
        
        # Drivers (con condicion Unless use_sim_time)
        delay_before_launches,
        delay_config,
        delay_node_1,
        delay_node_2,
        delay_higrometer_node,
        delay_ndvi_node,
        delay_gps_node,

        # Processing & Viz (Siempre corren)
        delay_between_launches,
        delay_py_node_2,
        delay_rviz,
        delay_csv_safecopy_node,
        delay_area_node,
        delay_crop_light_state_node,
        delay_ransac_node,
        delay_volume_estim_node,
        conditional_mqtt_launch,
        delay_utm_node
    ])