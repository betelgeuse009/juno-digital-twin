#!/usr/bin/env python3
import sys
import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from shared_objects.ROS_utils import Topics, SHOW
from shared_objects.utils_stop_v1 import analysis

# CARLA sim version of stop_node.py
# Differences from real vehicle:
#   - Camera topic: /carla/hero/rgb_front/image  (not ZED)
#   - No Stepper/CAN hardware — CARLA brake is driven by publishing stop=True,
#     which carla_steering_throttle_control.py translates to brake=1.0


class StopSignDetector(Node):
    def __init__(self):
        super().__init__('stop_node', parameter_overrides=[])

        self.declare_parameter('brake_wait_duration', 10.0)
        self.brake_wait_duration = self.get_parameter('brake_wait_duration').value

        self.enable = True
        self.count_stop = 0
        self.count_img = 0
        self.threshold_stop = 1
        self.threshold_img = 45
        self.bridge = CvBridge()
        self.is_waiting = False

        topics = Topics()
        self.topic_names = topics.topic_names

        self.stop_pub = self.create_publisher(Bool, self.topic_names["stop"], 1)
        self.original_sub = self.create_subscription(
            Image,
            '/carla/hero/rgb_front/image',
            self.original_img_callback,
            1
        )

        self.timer = None

        self.get_logger().info("StopSignDetector node has been started (CARLA mode)")

    def original_img_callback(self, msg):
        if self.enable and not self.is_waiting:
            self.count_img += 1
            if self.count_img % self.threshold_img != 0:
                return

            img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            container = analysis(img)
            result, img = container[0], container[1]

            if result:
                if self.count_stop < self.threshold_stop:
                    self.count_stop += 1
                else:
                    self.get_logger().info("Stop sign detected. Activating brake.")
                    # In CARLA, brake=1.0 is applied by carla_steering_throttle_control
                    # when it receives stop=True (mirrors real vehicle Stepper.brake())
                    self.get_logger().info("Brake activated. Waiting...")

                    self.is_waiting = True
                    self.enable = False
                    self.count_stop = 0

                    stop_msg = Bool(data=True)
                    self.stop_pub.publish(stop_msg)

                    self.timer = self.create_timer(self.brake_wait_duration, self.end_waiting)
            else:
                self.count_stop = 0

            if SHOW:
                cv2.imshow('Cropped Image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def end_waiting(self):
        stop_msg = Bool(data=False)
        self.stop_pub.publish(stop_msg)

        self.is_waiting = False
        self.enable = True
        self.count_stop = 0

        self.get_logger().info(
            f"{self.brake_wait_duration:.0f}-second wait complete. Resuming stop detection."
        )

        if self.timer:
            self.timer.cancel()
            self.timer = None


def main():
    rclpy.init(args=sys.argv)
    node = StopSignDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly')
    except Exception as e:
        node.get_logger().error('Error: %r' % (e))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
