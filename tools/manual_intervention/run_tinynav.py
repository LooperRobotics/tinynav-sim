"""Run TinyNav against the Go2 Gazebo sensor adapter without a physical chassis."""
import os
import sys
import uvicorn

sys.path.insert(0, '/tinynav')
from app.backend import node_manager


def init_looper(self):
    self._sensor_mode = 'looper'
    self._launch_sensor_procs(os.environ.copy())
    for topic in node_manager._IMAGE_TOPICS_LOOPER:
        self._last_frame[topic] = b''
        self._last_frame_time[topic] = 0.0
        self.preview_callbacks[topic] = []


def skip_hardware(self):
    self.get_logger().info('Go2 Gazebo: physical chassis disabled')


if __name__ == '__main__':
    node_manager.BackendNode._detect_and_init_sensor = init_looper
    node_manager.BackendNode._start_unitree_if_configured = skip_hardware
    uvicorn.run('app.backend.main:app', host='0.0.0.0', port=8000,
                timeout_graceful_shutdown=3)
