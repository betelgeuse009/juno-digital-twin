from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import SetRemap
from launch_ros.actions import Node as LaunchNode

def generate_launch_description() -> LaunchDescription:

    rviz_cfg = PathJoinSubstitution(
        [
            FindPackageShare("carla_ros_bridge"),
            "launch",
            "configs",
            "rtabmap.rviz",
        ]
    )
    # Identity transform: map and odom are the same in simulation (ground truth)
    static_odom_tf = LaunchNode(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )
    rtabmap_launch = GroupAction([
        SetRemap("goal", "/rtabmap/goal_internal"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("rtabmap_launch"),
                        "launch",
                        "rtabmap.launch.py",
                    ]
                )
            ),
            launch_arguments={
                "namespace":          "rtabmap",
                "rviz_cfg":           rviz_cfg,
                "args":               "--delete_db_on_start",
                "use_sim_time": "true",
                # CARLA frame and odometry
                "frame_id":           "hero",
                "odom_topic":         "/carla/hero/odometry",
                "visual_odometry":    "false",
                "publish_tf":         "false",
                "odom_frame_id": "odom",
                "approx_sync_max_interval": "0.05",
                # CARLA camera topics
                "rgb_topic":          "/carla/hero/rgb_front/image",
                "depth_topic":        "/carla/hero/depth_front/image",
                "camera_info_topic":  "/carla/hero/rgb_front/camera_info",

                # No IMU needed — CARLA odom is already complete
                "wait_imu_to_init":   "false",

                # CARLA topics aren't hardware-synced
                "approx_sync":        "true",
                "rgbd_sync":          "true",
                "approx_rgbd_sync":   "true",
                "topic_queue_size":   "10",
                "wait_for_transform": "1.0",
                "tf_tolerance": "0.5",
                "rviz":               "true",
                "rtabmap_viz":        "false",
            }.items(),
        ),
    ])

    return LaunchDescription([
        rtabmap_launch,
    ])