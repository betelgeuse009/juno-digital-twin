import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from rclpy.parameter import Parameter
def main(args=None):
    rclpy.init(args=args)
    
    # 1. Create a node to replace 'self'
    node = Node(
    'send_goal_pose_node',
    parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
    )

    # 2. Pass the node instead of self
    client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
    client.wait_for_server()

    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = 'hero'
    goal.pose.header.stamp = node.get_clock().now().to_msg()
    goal.pose.pose.position.x = 15.0
    goal.pose.pose.position.y = 0.0
    goal.pose.pose.orientation.w = 1.0

    # 3. Define the missing callback
    def on_feedback(feedback_msg):
        pass 

    future = client.send_goal_async(goal, feedback_callback=on_feedback)

    # 4. Process network traffic until the goal is accepted/rejected
    rclpy.spin_until_future_complete(node, future)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
