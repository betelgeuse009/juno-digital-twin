import os
from glob import glob

from setuptools import setup

package_name = 'juno_digital_twin'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'launch', 'configs'), glob('launch/configs/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Erdeniz Esmeli',
    maintainer_email='erdenizesmeli@proton.me',
    description='CARLA digital twin of the H2politO Juno autonomous driving stack',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'carla_path_planning_plus1 = juno_digital_twin.carla_path_planning_plus1:main',
            'carla_path_planning_plus2 = juno_digital_twin.carla_path_planning_plus2:main',
            'carla_path_planning_plus3 = juno_digital_twin.carla_path_planning_plus3:main',
            'carla_path_planning_plus5 = juno_digital_twin.carla_path_planning_plus5:main',
            'carla_path_planning = juno_digital_twin.carla_path_planning:main',
            'carla_obstacle_avoidance = juno_digital_twin.carla_obstacle_avoidance_v2:main',
            'carla_old_obstacle_avoidance = juno_digital_twin.carla_obstacle_avoidance:main',
            'carla_throttle_node = juno_digital_twin.carla_throttle_node:main',
            'carla_throttle_node_v2 = juno_digital_twin.carla_throttle_node_v2:main',
            'carla_segnode = juno_digital_twin.carla_segnode:main',
            'carla_odom_relay = juno_digital_twin.carla_odom_relay:main',
            'carla_steering_throttle_control = juno_digital_twin.carla_steering_throttle_control:main',
            'carla_stop_node = juno_digital_twin.carla_stop_node:main',
            'waypoints_eloborate = juno_digital_twin.waypoints_eloborate:main',
            'waypoints = juno_digital_twin.waypoints:main',
        ],
    },
)
