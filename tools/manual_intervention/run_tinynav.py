"""Run the Python TinyNav backend with Gazebo-only runtime adapters."""
import os
from pathlib import Path
import runpy
import sys


def configure_robot():
    os.environ['ROBOT_TYPE'] = 'go2'
    from tinynav.core import robot_specs
    robot_specs.ROBOT_CONFIG = robot_specs.RobotConfig(
        name='lekiwi', shape='circle', radius=0.14,
        camera_x=0.09, camera_y=0.0255, max_linear_vel=0.3, min_linear_vel=0.05,
        obstacle=robot_specs.ObstacleConfig(robot_z_bottom=-0.08, robot_z_top=0.3))


def wrap_launch(original):
    def launch(self, name, cmd, *args, **kwargs):
        if cmd[:3] == ['uv', 'run', 'python']:
            cmd = [sys.executable, str(Path(__file__).resolve()), 'node', *cmd[3:]]
        return original(self, name, cmd, *args, **kwargs)
    return launch


def start_backend():
    import uvicorn
    from app.backend import node_manager

    def init_looper(self):
        self._sensor_mode = 'looper'
        self._launch_sensor_procs(os.environ.copy())
        for topic in node_manager._IMAGE_TOPICS_LOOPER:
            self._last_frame[topic] = b''
            self._last_frame_time[topic] = 0.0
            self.preview_callbacks[topic] = []

    def skip_hardware(self):
        self.get_logger().info('Gazebo fixture: hardware chassis disabled')

    cls = node_manager.BackendNode
    cls._detect_and_init_sensor = init_looper
    cls._start_unitree_if_configured = skip_hardware
    cls._launch_proc = wrap_launch(cls._launch_proc)
    cls._launch_proc_tee = wrap_launch(cls._launch_proc_tee)
    uvicorn.run('app.backend.main:app', host='0.0.0.0', port=8000,
                timeout_graceful_shutdown=3)


if __name__ == '__main__':
    sys.path.insert(0, '/tinynav')
    configure_robot()
    if len(sys.argv) > 1 and sys.argv[1] == 'node':
        sys.argv = sys.argv[2:]
        runpy.run_path(sys.argv[0], run_name='__main__')
    else:
        start_backend()
