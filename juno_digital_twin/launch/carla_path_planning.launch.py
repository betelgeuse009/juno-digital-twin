from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


USE_SIM = True
NAV2_YAML = "/home/ubuntu/Workspace/ros-bridge/src/carla_ros_bridge/launch/configs/nav2_rpp_controller.yaml"


def generate_launch_description() -> LaunchDescription:
    force_color = SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1')
    odom_relay = Node(
        package="carla_ros_bridge", executable="carla_odom_relay",
        name="carla_odom_relay", output="screen",
        parameters=[{"use_sim_time": USE_SIM}],
    )

    # Nav2 with our RPP controller YAML
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py",
            ])
        ),
        launch_arguments={
            "params_file": NAV2_YAML,
            "use_sim_time": str(USE_SIM).lower(),
        }.items(),
    )

    # carla_ros_bridge pipeline
    seg_node = Node(
        package="carla_ros_bridge", executable="carla_segnode",
        name="carla_segnode", output="screen",
        parameters=[{"use_sim_time": USE_SIM}],
    )
    path_planning_node = Node(
        package="carla_ros_bridge", executable="carla_path_planning",
        name="carla_path_planning", output="screen",
        parameters=[{"use_sim_time": USE_SIM}],
        arguments=['--ros-args', '--log-level', 'info']
    )
    steering_control_node = Node(
        package="carla_ros_bridge", executable="carla_steering_throttle_control",
        name="carla_steering_throttle_control", output="screen",
        parameters=[{"use_sim_time": USE_SIM}],
    )
    throttle_node = Node(
        package="carla_ros_bridge", executable="carla_throttle_node_v2",
        name="carla_throttle_node_v2", output="screen",
        parameters=[{"use_sim_time": USE_SIM}],
    )

    return LaunchDescription([
        force_color,
        odom_relay,
        nav2,
        seg_node,
        path_planning_node,
        steering_control_node,
        throttle_node,
    ])