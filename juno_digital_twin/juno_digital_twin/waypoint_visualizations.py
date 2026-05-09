import numpy as np
import matplotlib.pyplot as plt

def plot_track(filename='track_waypoints.npy'):
    try:
        # Load the waypoints array (Shape: [N, 3] -> Lat, Lon, Alt)
        waypoints = np.load(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Make sure you are in the correct directory.")
        return

    if len(waypoints) == 0:
        print("The waypoint array is empty.")
        return

    # Extract Latitude (y-axis) and Longitude (x-axis)
    lats = waypoints[:, 0]
    lons = waypoints[:, 1]

    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the full track
    plt.plot(lons, lats, marker='o', linestyle='-', color='dodgerblue', markersize=4, label='Recorded Path')
    
    # Highlight Start and End points
    plt.plot(lons[0], lats[0], marker='s', color='limegreen', markersize=8, label='Start')
    plt.plot(lons[-1], lats[-1], marker='X', color='crimson', markersize=8, label='End')

    # Formatting the graph
    plt.title('Recorded RTK GPS Track', fontsize=14, fontweight='bold')
    plt.xlabel('Longitude (Degrees)', fontsize=12)
    plt.ylabel('Latitude (Degrees)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Keep the aspect ratio equal so the spatial path doesn't look stretched
    plt.gca().set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_track()