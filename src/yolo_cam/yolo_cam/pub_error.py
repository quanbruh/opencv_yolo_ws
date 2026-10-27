import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection2DArray
import cv2
import numpy as np
from geometry_msgs.msg import Twist


class TargetOffsetNode(Node):
    """So tâm box YOLO với tâm khung hình -> LEFT / RIGHT / CENTER."""

    def __init__(self):
        super().__init__("target_offset_node")

        dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("image_width", 0, dyn)       # 0 = tự lấy từ topic ảnh
        self.declare_parameter("deadband", 0.10, dyn)       # |offset| < 10% nửa khung = CENTER
        self.declare_parameter("target_class", "")          # rỗng = mọi class
        self.declare_parameter("topic_detections", "/detections")
        self.declare_parameter("topic_raw", "/image_raw/compressed")
        self.declare_parameter("topic_offset", "/target/offset")
        self.declare_parameter("topic_direction", "/target/direction")

        p = lambda n: self.get_parameter(n).value
        self.width = int(p("image_width"))
        self.deadband = float(p("deadband"))
        self.target_class = str(p("target_class"))

        self.pub_off = self.create_publisher(Float32, str(p("topic_offset")), 10)
        self.pub_dir = self.create_publisher(String, str(p("topic_direction")), 10)
        self.create_subscription(
            Detection2DArray, str(p("topic_detections")), self._on_det, 10)

        # chưa biết bề rộng ảnh -> đọc 1 frame đầu để lấy
        if self.width <= 0:
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=ReliabilityPolicy.BEST_EFFORT)
            self.sub_img = self.create_subscription(
                CompressedImage, str(p("topic_raw")), self._on_img, qos)
            self.get_logger().info("Đang chờ frame đầu để lấy bề rộng ảnh...")

    def _on_img(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        self.width = img.shape[1]
        self.get_logger().info(f"Bề rộng ảnh: {self.width}px")
        self.destroy_subscription(self.sub_img)

    def _on_det(self, msg):
        if self.width <= 0:
            return

        # lọc theo class, chọn box to nhất (thường là vật gần nhất)
        best, best_area = None, 0.0
        for d in msg.detections:
            if not d.results:
                continue
            if self.target_class and d.results[0].hypothesis.class_id != self.target_class:
                continue
            area = d.bbox.size_x * d.bbox.size_y
            if area > best_area:
                best, best_area = d, area

        if best is None:
            self._publish(0.0, "NONE")
            self.pub_center()
            return

        half = self.width / 2.0
        offset = (best.bbox.center.position.x - half) / half   # -1 (mép trái) .. +1 (mép phải)

        if abs(offset) < self.deadband:
            direction = "CENTER"
            self.pub_center()
        elif offset < 0:
            direction = "LEFT"
            self.pub_left()
            
        else:
            direction = "RIGHT"
            self.pub_right()
            
        self._publish(offset, direction)

    def _publish(self, offset, direction):
        self.pub_off.publish(Float32(data=float(offset)))
        self.pub_dir.publish(String(data=direction))
        self.get_logger().info(f"{direction}  offset={offset:+.2f}",
                               throttle_duration_sec=0.5)
        
    def pub_left(self):
        self.pub_angular = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
        self.timer = self.create_timer(0.05, self.gui_giatri)  # 10 Hz
        self.angular = 0.5
    
    def pub_center(self):
            self.pub_angular = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
            self.timer = self.create_timer(0.05, self.gui_giatri)  # 10 Hz
            self.angular = 0.0
    
    def pub_right(self):
            self.pub_angular = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
            self.timer = self.create_timer(0.05, self.gui_giatri)  # 10 Hz
            self.angular = -0.5
                
    def gui_giatri(self):
        giatri = Twist()
        giatri.angular.z = self.angular
        self.pub_angular.publish(giatri)
        # self.get_logger().info(f"gia tri angular tao set la {giatri.angular.z}")

def main():
    rclpy.init()
    node = TargetOffsetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()