from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    camera = Node(
        package='yolo_cam',
        executable='camera_node',
        output='screen',
        
    )
    
    yolo = Node(
        package='yolo_cam',
        executable='yolo_node',
        output='screen',
        
    )
    
    

    viewer = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        output='screen',
        arguments=['/camera/image_raw'],
    )
    
    start_viewer = RegisterEventHandler(
        OnProcessStart(
            target_action=yolo,
            on_start=[viewer],
        )
    )
    
    
    pub_lech = Node(
        package='yolo_cam',
        executable='pub_error',
        output='screen',
        
    )
    
    

    

    return LaunchDescription([
        yolo,
        start_viewer,
        pub_lech,
        camera,
    ])