import carla
import math
import time

client = carla.Client('10.97.94.1', 2000)
world = client.get_world()

print("Waiting for ROS bridge to spawn actors...")

# 1. Loop until the camera is found in the world state
hero_cam = None
while hero_cam is None:
    # Re-fetch actors on every iteration to catch the latest synchronous tick
    for actor in world.get_actors().filter('sensor.camera.rgb'):
        if actor.attributes.get('role_name') == 'rgb_front':
            hero_cam = actor
            print(f"Found hero camera! (ID: {actor.id})")
            break
    
    if hero_cam is None:
        print("Camera not found yet. Retrying in 1 second...")
        time.sleep(1.0)

# 2. Locate the static cylinder in the environment
cylinder_loc = None
env_objects = world.get_environment_objects(carla.CityObjectLabel.Any)
for obj in env_objects:
    if 'Cylinder8' in obj.name: 
        print('Found cylinder!')
        cylinder_loc = obj.transform.location
        break

# 3. Calculate distance
if hero_cam and cylinder_loc:
    cam_loc = hero_cam.get_transform().location
    
    distance = math.sqrt(
        (cylinder_loc.x - cam_loc.x)**2 + 
        (cylinder_loc.y - cam_loc.y)**2 + 
        (cylinder_loc.z - cam_loc.z)**2
    )
    print(f"Distance to cylinder: {distance:.2f} meters")
else:
    print("Error: Missing cylinder. Ensure it is named 'Cylinder8' in the Unreal Editor.")
