from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import SetRemap
from launch_ros.actions import Node as LaunchNode

def generate_launch_description() -> LaunchDescription:

    # Odometry Relay 
    # Converts CARLA's absolute odometry (frame_id='map', world origin)
    # into relative odometry (frame_id='odom', starting at vehicle spawn).
    # Also publishes odom -> hero TF.
    odom_relay = LaunchNode(
        package="carla_ros_bridge",             
        executable="carla_odom_relay",
        name="carla_odom_relay",
        parameters=[{
            "input_odom_topic": "/carla/hero/odometry",
            "output_odom_topic": "/odom",
            "odom_frame_id": "odom",
            "child_frame_id": "hero",
            "use_sim_time": True,
        }],
        output="screen",
    )

    # RTABMAP
    rtabmap_launch = GroupAction([
        SetRemap("goal", "/rtabmap/goal_internal"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("rtabmap_launch"),
                    "launch",
                    "rtabmap.launch.py",
                ])
            ),
            launch_arguments={
                "namespace":            "rtabmap",
                "rviz_cfg":             "/home/ubuntu/Workspace/ros-bridge/src/carla_ros_bridge/launch/configs/rtabmap.rviz",
                "args": " ".join([
                    "--delete_db_on_start",
                    "--Grid/RangeMax 20.0",
                    "--Grid/RangeMin 0.3",
                    "--Grid/MaxGroundAngle 25",
                    "--Grid/MaxObstacleHeight 3.0",
                    "--Grid/MinGroundHeight -0.5",
                    "--Grid/MaxGroundHeight 0.3",
                    "--Grid/NormalK 20",
                    "--Grid/CellSize 0.1",
                    "--Grid/RayTracing true",
                    #"--Grid/3D false",

                ]),               
                "use_sim_time":         "true",

                # Frame configuration 
                "frame_id":             "hero",
                "visual_odometry":      "false",

                # RTABMAP publishes map -> odom (loop closure corrections)
                "publish_tf_map":       "true",
                # Don't publish odom -> hero (our relay node does that)
                "publish_tf_odom":      "false",

                # Odom frame is now 'odom' (created by our relay node),
                # NOT 'map' — this lets RTABMAP compute map->odom corrections
                "odom_frame_id":        "odom",

                # Odometry topic (output of our relay node)
                "odom_topic":           "/odom",

                # CARLA camera topics 
                "rgb_topic":            "/carla/hero/rgb_front/image",
                "depth_topic":          "/carla/hero/depth_front/image",
                "depth_camera_info_topic":          "/carla/hero/depth_front/camera_info",
                "camera_info_topic":    "/carla/hero/rgb_front/camera_info",

                # The rgbd_sync nodelet bundles RGB + depth into a
                # single rgbd_image message before sending to rtabmap.
                "rgbd_sync":            "true",
                # CARLA renders RGB and depth on the same tick with
                # identical timestamps, so exact sync is fine.
                "approx_rgbd_sync":     "false",
                # rtabmap subscribes to the bundled rgbd_image, NOT
                # to raw depth separately.
                "subscribe_rgbd":       "true",
                "subscribe_depth":      "false",

                # Odom may publish at different rate than cameras
                "approx_sync":          "true",
                "approx_sync_max_interval": "0.05",

                # No IMU needed — CARLA odom is already complete
                "wait_imu_to_init":     "false",

                # Simulation burst-publish tolerance
                "topic_queue_size":     "30",
                "wait_for_transform":   "1.0",
                "tf_tolerance":         "0.5",
                "qos": "1", 

                # Visualization
                "rviz":                 "true",
                "rtabmap_viz":          "false",
            }.items(),
        ),
    ])

    return LaunchDescription([
        odom_relay,
        rtabmap_launch,
    ])