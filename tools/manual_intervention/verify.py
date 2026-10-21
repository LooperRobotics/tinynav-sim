"""Verify manual command ownership and deadman timeout on an idle simulator."""
import asyncio
import json
import math
import threading
import time
import urllib.request
import rclpy
import websockets
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage
from rclpy.qos import qos_profile_sensor_data

BASE = 'http://127.0.0.1:8000'
def request(path, post=False):
    req = urllib.request.Request(BASE + path, method='POST' if post else 'GET')
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)

async def verify():
    status = request('/device/status')
    if status['navActive']:
        raise RuntimeError('Cancel the test task before running this idle simulator check')
    rclpy.init()
    node = rclpy.create_node('verify_manual_intervention')
    commands, positions = [], []
    node.create_subscription(Twist, '/cmd_vel', lambda m: commands.append(
        (time.monotonic(), m.linear.x, m.linear.y, m.angular.z)), 10)
    def pose(msg):
        for tf in msg.transforms:
            if tf.child_frame_id == 'lekiwi':
                p = tf.transform.translation
                positions.append((time.monotonic(), p.x, p.y))
    node.create_subscription(TFMessage, '/sim/world_pose', pose, qos_profile_sensor_data)
    running = True
    def spin():
        while running: rclpy.spin_once(node, timeout_sec=.05)
    thread = threading.Thread(target=spin)
    thread.start()
    try:
        request('/nav/manual/enter', True)
        await asyncio.sleep(1)
        assert [p.node_name for p in node.get_publishers_info_by_topic('/cmd_vel')] == ['ros2_node_manager']
        async with websockets.connect('ws://127.0.0.1:8000/ws/teleop') as ws:
            sent = time.monotonic()
            await ws.send(json.dumps({'linear_x': .2, 'linear_y': 0, 'angular_z': 0}))
            await asyncio.sleep(1.2)
            assert any(t >= sent and x > .1 for t,x,y,z in commands), 'No manual command received'
            late = [(x,y,z) for t,x,y,z in commands if t > sent + .45]
            assert late and all(v == (0,0,0) for v in late), 'Deadman did not stop command stream'
            settled = [(x,y) for t,x,y in positions if t > sent + .65]
            assert len(settled) > 5, 'No Gazebo poses received'
            drift = math.dist(settled[0], settled[-1])
            assert drift < .01, f'Simulator kept moving after timeout: {drift}'
        print(json.dumps({'manual_publishers': 1, 'deadman_timeout_s': .3, 'settled_drift_m': drift, 'result': 'PASS'}))
    finally:
        request('/nav/manual/exit', True)
        running = False
        thread.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': asyncio.run(verify())
