import os

import cv2
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

DEFAULT_MODEL = "/home/quannotreal/yolo_ws/src/yolo_cam/models/masteryi.pt"


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")
        dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("models", DEFAULT_MODEL)
        self.declare_parameter("imgsz", 320)
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("device", "cpu", dyn)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("input_topic", "/image_raw/compressed")
        self.declare_parameter("output_topic", "/yolo/image_annotated/compressed")

        p = lambda n: self.get_parameter(n).value
        self.imgsz = int(p("imgsz"))
        self.conf = float(p("conf"))
        self.jpeg_q = [int(cv2.IMWRITE_JPEG_QUALITY), int(p("jpeg_quality"))]

        self.device = str(p("device"))
        if self.device != "cpu" and not torch.cuda.is_available():
            self.get_logger().warn("Không có CUDA -> dùng CPU")
            self.device = "cpu"

        model_path = str(p("models"))
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            try:
                model_path = os.path.join(
                    get_package_share_directory("yolo_cam"), "models", model_path)
            except Exception:
                pass
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không thấy model: {model_path}")

        self.model = YOLO(model_path)
        self.get_logger().info(f"Classes: {self.model.names}")

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(
            CompressedImage, str(p("input_topic")), self.cb, qos)
        self.pub_img = self.create_publisher(CompressedImage, str(p("output_topic")), qos)
        self.pub_det = self.create_publisher(Detection2DArray, "/detections", 10)

    def cb(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]

        out = Detection2DArray()
        out.header = msg.header
        for b in r.boxes:
            cx, cy, w, h = b.xywh[0].tolist()
            label = self.model.names[int(b.cls)]
            conf = float(b.conf)

            d = Detection2D()
            d.header = msg.header
            d.bbox.center.position.x = cx
            d.bbox.center.position.y = cy
            d.bbox.size_x = w
            d.bbox.size_y = h
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = conf
            d.results.append(hyp)
            out.detections.append(d)
        self.pub_det.publish(out)

        if self.pub_img.get_subscription_count() > 0:
            ok, buf = cv2.imencode(".jpg", r.plot(), self.jpeg_q)   # r.plot() vẽ bbox
            if ok:
                m = CompressedImage()
                m.header = msg.header
                m.format = "jpeg"
                m.data = buf.tobytes()
                self.pub_img.publish(m)


def main():
    rclpy.init()
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()