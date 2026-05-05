import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    ld = launch.LaunchDescription()

    rtabmap_include = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('carla_ros_bridge'), 'rtabmap_depth.launch.py')
        )
    )

    path_planning_node = Node(
        package='carla_ros_bridge',
        executable='carla_path_planning_plus3',
        name='carla_path_planning_plus3',
        output='screen',
    )

    seg_node = Node(
        package='carla_ros_bridge',
        executable='carla_segnode',
        name='carla_segnode',
        output='screen',
    )

    throttle_node = Node(
        package='carla_ros_bridge',
        executable='carla_throttle_node',
        name='carla_throttle_node',
        output='screen',
    )

    steering_control_node = Node(
        package='carla_ros_bridge',
        executable='carla_steering_throttle_control',
        name='carla_steering_throttle_control',
        output='screen',
    )

    ld.add_action(rtabmap_include)
    ld.add_action(path_planning_node)
    ld.add_action(seg_node)
    ld.add_action(throttle_node)
    ld.add_action(steering_control_node)

    return ld

if __name__ == '__main__':
    generate_launch_description()