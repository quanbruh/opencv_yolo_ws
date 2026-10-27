import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class YoloCameraNode(Node):
    def __init__(self):
        super().__init__("yolo_camera_node")

        self.declare_parameter("model", "yolo11_model.pt")
        self.declare_parameter("source", "1")       # "0" = webcam, hoặc URL ESP32-CAM
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("device", "0")     # "0" nếu có GPU
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("show", False)       # True = mở cửa sổ OpenCV

        self.model = YOLO(self.get_parameter("model").value)
        self.get_logger().info(f"Classes: {self.model.names}")
        self.bridge = CvBridge()

        src = self.get_parameter("source").value
        src = int(src) if str(src).isdigit() else src
        self.cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được camera: {src}")

        self.pub_raw = self.create_publisher(Image, "/image_raw", 1)
        self.pub_img = self.create_publisher(Image, "/yolo/image_annotated", 1)
        self.pub_det = self.create_publisher(Detection2DArray, "/detections", 10)

        period = 1.0 / self.get_parameter("fps").value
        self.timer = self.create_timer(period, self.loop)

    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Mất frame từ camera")
            return

        stamp = self.get_clock().now().to_msg()

        raw = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        raw.header.stamp = stamp
        raw.header.frame_id = "camera"
        self.pub_raw.publish(raw)

        r = self.model.predict(
            frame,
            imgsz=self.get_parameter("imgsz").value,
            conf=self.get_parameter("conf").value,
            device=self.get_parameter("device").value,
            verbose=False,
        )[0]

        out = Detection2DArray()
        out.header = raw.header
        for b in r.boxes:
            cx, cy, w, h = b.xywh[0].tolist()
            d = Detection2D()
            d.header = raw.header
            d.bbox.center.position.x = cx
            d.bbox.center.position.y = cy
            d.bbox.size_x = w
            d.bbox.size_y = h
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.model.names[int(b.cls)]
            hyp.hypothesis.score = float(b.conf)
            d.results.append(hyp)
            out.detections.append(d)
        self.pub_det.publish(out)

        annotated = r.plot()
        img = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        img.header = raw.header
        self.pub_img.publish(img)

        if self.get_parameter("show").value:
            cv2.imshow("YOLO ROS2", annotated)
            cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = YoloCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
