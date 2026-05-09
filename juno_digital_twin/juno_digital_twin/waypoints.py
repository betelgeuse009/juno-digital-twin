import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import csv
import math

class WaypointRecorder(Node):
    def __init__(self):
        super().__init__('waypoint_recorder')
        
        # Subscribe to GPS fixes and process each message in gps_callback.
        self.subscription = self.create_subscription(
            NavSatFix, '/carla/hero/gnss', self.gps_callback, 20)
        
        self.waypoints = []
        # Minimum spacing (meters) between saved waypoints to avoid dense duplicates.
        self.min_dist = 7.5 
        self.get_logger().info("Waypoint Recorder Started. Waiting for RTK Fix...")

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Great-circle distance on Earth in meters between two lat/lon coordinates.
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def gps_callback(self, msg):
        lat, lon, alt = msg.latitude, msg.longitude, msg.altitude
        # First accepted point becomes the route start.
        if not self.waypoints:
            self.waypoints.append([lat, lon, alt])
            self.get_logger().info(f"Start Point: {lat:.6f}, {lon:.6f}")
            return
        # Compare current position with last stored waypoint.
        last_lat, last_lon, _ = self.waypoints[-1]
        dist = self.haversine_distance(last_lat, last_lon, lat, lon)
        # Save only if movement exceeds threshold.
        if dist >= self.min_dist:
            self.waypoints.append([lat, lon, alt])
            self.get_logger().info(f"Recorded: {lat:.6f}, {lon:.6f} | Dist: {dist:.2f}m")

    def save_waypoints(self):
        # Persist waypoints to a CSV file with header row.
        if self.waypoints:
            filename = 'track_waypoints.csv'
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['latitude', 'longitude', 'altitude'])
                for lat, lon, alt in self.waypoints:
                    writer.writerow([f"{lat:.8f}", f"{lon:.8f}", f"{alt:.3f}"])
            self.get_logger().info(f"Saved {len(self.waypoints)} RTK waypoints to {filename}.")
        else:
            self.get_logger().warn("No RTK waypoints were recorded.")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Always save what has been collected before shutting down.
        node.save_waypoints()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()