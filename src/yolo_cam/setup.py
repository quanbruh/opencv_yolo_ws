from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'yolo_cam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='quannotreal',
    maintainer_email='quannotreal@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolocam = yolo_cam.camera:main',
            'camera_node = yolo_cam.camera_node:main',
            'yolo_node = yolo_cam.yolo_node:main',
            'pub_error = yolo_cam.pub_error:main'

        ],
    },
)
