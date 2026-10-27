import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CompressedImage


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("source", "0", dyn)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("pub_fps", 15.0)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("topic", "/image_raw/compressed")

        p = lambda n: self.get_parameter(n).value
        self.width, self.height = int(p("width")), int(p("height"))
        self.pub_period = 1.0 / float(p("pub_fps"))
        self.jpeg_q = [int(cv2.IMWRITE_JPEG_QUALITY), int(p("jpeg_quality"))]

        src = p("source")
        self.src = int(src) if str(src).isdigit() else str(src)
        self.cap = None
        if not self._open():
            raise RuntimeError(f"Không mở được camera: {self.src}")

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(CompressedImage, str(p("topic")), qos)

        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.src)
        if isinstance(self.src, int):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    def _loop(self):
        fails, last_pub = 0, 0.0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    self.get_logger().warn("Mất stream, đang reconnect...")
                    self._open()
                    fails = 0
                time.sleep(0.05)
                continue
            fails = 0

            now = time.monotonic()
            if now - last_pub < self.pub_period:
                continue
            last_pub = now

            if self.pub.get_subscription_count() == 0:
                continue
            ok, buf = cv2.imencode(".jpg", frame, self.jpeg_q)
            if not ok:
                continue
            m = CompressedImage()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "camera"
            m.format = "jpeg"
            m.data = buf.tobytes()
            self.pub.publish(m)

    def destroy_node(self):
        self.running = False
        self.th.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()