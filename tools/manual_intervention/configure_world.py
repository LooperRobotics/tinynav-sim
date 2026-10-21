"""Enable planar velocity control in an expanded Gazebo Fortress world."""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('world')
parser.add_argument('--output', required=True)
args = parser.parse_args()
tree = ET.parse(args.world)
model = tree.getroot().find(".//model[@name='lekiwi']")
if model is None:
    raise SystemExit('Expand the lekiwi model into the world first')
for plugin in list(model.findall('plugin')):
    if 'DiffDrive' in plugin.get('name', '') or 'VelocityControl' in plugin.get('name', ''):
        model.remove(plugin)
plugin = ET.SubElement(model, 'plugin', filename='ignition-gazebo-velocity-control-system',
                       name='ignition::gazebo::systems::VelocityControl')
ET.SubElement(plugin, 'topic').text = '/cmd_vel'
tree.write(args.output, encoding='unicode')
print(f'Planar command model written to {Path(args.output)}')
