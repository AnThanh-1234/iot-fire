#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot chữa cháy tự động - Raspberry Pi + YOLOv5 (bản tối ưu)
-------------------------------------------------------------
- Webcam phát hiện lửa bằng YOLOv5 (model lửa tự tải về)
- 2 servo Pan (GPIO 23) / Tilt (GPIO 18) xoay để đưa ngọn lửa vào tâm khung hình
- (Tùy chọn --pump) Khi khóa mục tiêu -> bật relay (GPIO 14) kích máy bơm 365

Tối ưu cho Raspberry Pi:
- Lần chạy đầu tự chuyển model .pt -> .onnx, các lần sau chạy bằng onnxruntime
  (nhanh hơn PyTorch trên CPU ARM, không cần nạp torch -> khởi động nhanh, ít RAM)
- Camera đọc ở luồng riêng, luôn xử lý khung hình MỚI NHẤT (không bị trễ)
- Camera chụp 320x240 MJPG -> giảm tải giải mã/vẽ

Chuyển động servo:
- Không thấy lửa: servo ngang quét RADAR qua lại 0°..180°, servo dọc về 90°
- Thấy lửa: dừng radar, bám lửa. Servo do luồng riêng 50Hz điều khiển, trượt mượt
  tới góc đích; có vùng chồng lấn (hysteresis) + lọc mượt tâm lửa để chống giật

Chạy:  python fire_tracker.py              (chỉ servo theo dõi lửa)
       python fire_tracker.py --pump       (bật thêm máy bơm khi khóa mục tiêu)
       python fire_tracker.py --imgsz 256  (nhanh hơn nữa, kém chính xác hơn chút)
       python fire_tracker.py --torch      (bỏ qua ONNX, dùng PyTorch như cũ)
Thoát: nhấn 'q' trên cửa sổ camera (hoặc Ctrl+C trong terminal)
"""

import argparse
import ast
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

# ===================== CẤU HÌNH PHẦN CỨNG =====================
PAN_PIN = 23          # Servo trục ngang
TILT_PIN = 24         # Servo trục dọc
RELAY_PIN = 14        # IN của relay máy bơm
RELAY_ACTIVE_HIGH = False   # Đa số module relay 5V kích mức THẤP -> False. Relay kích mức CAO -> True

# Thang góc servo 0°..180° (90° = chính giữa)
PAN_MIN, PAN_MAX = 0, 180       # Giới hạn góc servo ngang (độ)
TILT_MIN, TILT_MAX = 45, 150    # Giới hạn góc servo dọc (tránh gập quá, va chạm cơ khí)
PAN_HOME, TILT_HOME = 90, 90    # Góc ban đầu: khởi động/thoát chương trình đều quay về đây
HOME_SETTLE = 0.6               # Thời gian chờ (giây) cho servo quay tới góc ban đầu

PAN_DIR = 1           # Đảo chiều nếu servo xoay ngược hướng lửa (1 hoặc -1)
TILT_DIR = -1

# ===================== CẤU HÌNH CAMERA =====================
FRAME_W, FRAME_H = 320, 240   # Độ phân giải camera (nhỏ = nhanh)
NUM_THREADS = 4               # Số nhân CPU dùng cho model (Pi 4/5 có 4 nhân)
CAM_HFOV, CAM_VFOV = 60.0, 45.0   # Góc nhìn ngang/dọc của webcam (độ) - đổi lệch pixel ra độ

# ===================== RADAR (KHÔNG THẤY LỬA) =====================
RADAR_MIN, RADAR_MAX = 0, 180  # Vùng quét của servo ngang (độ)
RADAR_SPEED = 30.0             # Tốc độ quét (độ/giây). Nhanh quá model sẽ bỏ sót lửa
RADAR_START_DELAY = 1.5        # Mất lửa bao lâu (giây) thì bắt đầu quét radar

# ===================== BÁM LỬA CHỐNG GIẬT =====================
# Vùng chồng lấn (hysteresis), tính theo TỈ LỆ khung hình:
#   lửa lệch RA NGOÀI ZONE_OUTER mới bắt đầu xoay,
#   đã xoay thì xoay tới khi lệch VÀO TRONG ZONE_INNER mới dừng.
#   -> lửa dao động nhẹ quanh tâm sẽ KHÔNG làm servo giật qua lại.
ZONE_OUTER = 0.15
ZONE_INNER = 0.05
TRACK_GAIN = 0.6      # Mỗi lần nhận diện chỉ sửa 60% độ lệch -> không vọt lố
CENTER_SMOOTH = 0.5   # Lọc làm mượt tâm lửa (0..1): nhỏ = mượt hơn nhưng phản ứng chậm hơn
LOCK_TOLERANCE = 0.10 # Lệch tâm < 10% khung hình coi là "đã nhắm"
LOCK_FRAMES = 3       # Số lần liên tiếp nhắm trúng mới coi là khóa mục tiêu

# Luồng điều khiển servo chạy riêng, servo trượt mượt tới góc đích
SERVO_RATE = 50          # Số lần cập nhật servo mỗi giây
SERVO_MAX_SPEED = 90.0   # Tốc độ xoay tối đa khi bám lửa (độ/giây)
SERVO_EASE = 5.0         # Độ "mềm": càng nhỏ servo càng chậm rãi khi gần tới đích
SERVO_MIN_STEP = 0.3     # Chỉ gửi lệnh khi góc đổi > 0.3° -> bớt rung
IDLE_DETACH = 0.8        # Đứng yên quá 0.8s thì ngắt xung PWM -> hết rung (0 = tắt tính năng)

# ===================== MÁY BƠM =====================
LOST_TIMEOUT = 1.5    # Mất dấu lửa quá bao lâu (giây) thì tắt bơm
MAX_PUMP_TIME = 15.0  # Bảo vệ: bơm liên tục tối đa (giây), sau đó nghỉ
PUMP_COOLDOWN = 3.0   # Thời gian nghỉ sau khi bơm quá lâu

WINDOW_NAME = "Raspberry Pi - Fire Tracker"

# Model YOLOv5s đã huấn luyện nhận diện lửa (giấy phép MIT, ~14 MB)
# Nguồn: https://github.com/spacewalk01/yolov5-fire-detection
MODEL_URL = ("https://github.com/spacewalk01/yolov5-fire-detection/"
             "raw/main/model/yolov5s_best.pt")
YOLOV5_HUB_DIR = os.path.join(
    os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")),
    "hub", "ultralytics_yolov5_master")


# ===================== PHẦN CỨNG (GPIO) =====================
class Hardware:
    """Servo + relay. Servo được điều khiển bởi một luồng riêng (SERVO_RATE Hz):
    chương trình chính chỉ đặt GÓC ĐÍCH, luồng servo đưa servo trượt mượt tới đó.

    Chế độ (mode):
      "radar" - servo ngang quét qua lại RADAR_MIN..RADAR_MAX, servo dọc về TILT_HOME
      "track" - bám theo lửa (góc đích do chương trình chính đặt)
      "hold"  - đứng yên tại chỗ (vừa mất dấu lửa, chờ trước khi quét radar)
    Không chạy trên Pi thì tự chuyển sang chế độ giả lập (vẫn tính góc, in log)."""

    def __init__(self, use_pump=False, log=True):
        self.sim = False
        self.relay = None
        self.log = log
        self.step = 0
        self.servos = {}
        try:
            from gpiozero import AngularServo, OutputDevice
            factory = None
            try:
                # pigpio cho xung PWM chính xác hơn -> servo không rung (Pi 5 không hỗ trợ)
                # Cần: sudo apt install pigpio && sudo systemctl enable --now pigpiod
                from gpiozero.pins.pigpio import PiGPIOFactory
                factory = PiGPIOFactory()
                print("[HW] Dùng pigpio (PWM chính xác, servo mượt).")
            except Exception:
                print("[HW] Không có pigpiod -> dùng PWM phần mềm "
                      "(đã bật ngắt xung khi đứng yên để chống rung).")

            kw = dict(min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000)
            if factory:
                kw["pin_factory"] = factory
            # initial_angle = góc HOME -> vừa cấp xung là servo quay thẳng về 90°
            self.servos["pan"] = AngularServo(PAN_PIN, min_angle=0, max_angle=180,
                                              initial_angle=PAN_HOME, **kw)
            self.servos["tilt"] = AngularServo(TILT_PIN, min_angle=0, max_angle=180,
                                               initial_angle=TILT_HOME, **kw)

            if use_pump:
                rkw = dict(active_high=RELAY_ACTIVE_HIGH, initial_value=False)
                if factory:
                    rkw["pin_factory"] = factory
                self.relay = OutputDevice(RELAY_PIN, **rkw)
            else:
                print("[HW] Máy bơm TẮT (chỉ servo theo dõi lửa). Dùng --pump để bật.")
        except Exception as e:
            print(f"[HW] Không khởi tạo được GPIO ({e}) -> CHẾ ĐỘ GIẢ LẬP.")
            self.sim = True

        self.pump_on = False
        self.lock = threading.Lock()
        self.mode = "hold"
        self.radar_dir = 1
        self.angle = {"pan": float(PAN_HOME), "tilt": float(TILT_HOME)}    # góc hiện tại
        self.target = {"pan": float(PAN_HOME), "tilt": float(TILT_HOME)}   # góc đích
        self.sent = dict(self.angle)                  # góc đã gửi xuống servo lần cuối
        self.last_move = {"pan": time.time(), "tilt": time.time()}
        self.attached = {"pan": True, "tilt": True}
        self.go_home("Khởi động")

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---------- dùng từ chương trình chính ----------
    @property
    def pan_angle(self):
        return self.angle["pan"]

    @property
    def tilt_angle(self):
        return self.angle["tilt"]

    def get_angles(self):
        with self.lock:
            return self.angle["pan"], self.angle["tilt"]

    def start_radar(self):
        """Bắt đầu quét radar từ vị trí hiện tại, hướng về phía đầu xa hơn."""
        with self.lock:
            if self.mode == "radar":
                return
            self.mode = "radar"
            self.target["pan"] = self.angle["pan"]
            mid = (RADAR_MIN + RADAR_MAX) / 2
            self.radar_dir = 1 if self.angle["pan"] < mid else -1
        print(f"[SERVO] Không thấy lửa -> RADAR quét {RADAR_MIN}°↔{RADAR_MAX}° "
              f"({RADAR_SPEED:.0f}°/s), TILT về {TILT_HOME}°")

    def hold(self):
        """Dừng tại chỗ."""
        with self.lock:
            self.mode = "hold"
            self.target = dict(self.angle)

    def track(self, pan=None, tilt=None, reason="BÁM LỬA"):
        """Đặt góc đích khi bám lửa. None = giữ nguyên góc đích trục đó."""
        with self.lock:
            was_radar = self.mode == "radar"
            self.mode = "track"
            old = dict(self.target)
            parts, clipped = [], False
            for axis, val, lo, hi, sign, pos, neg in (
                    ("pan", pan, PAN_MIN, PAN_MAX, PAN_DIR, "sang PHẢI", "sang TRÁI"),
                    ("tilt", tilt, TILT_MIN, TILT_MAX, TILT_DIR, "XUỐNG", "LÊN")):
                if val is None:
                    continue
                c = float(np.clip(val, lo, hi))
                clipped |= c != val
                self.target[axis] = c
                d = c - old[axis]
                if abs(d) >= SERVO_MIN_STEP:
                    parts.append(f"{axis.upper()} {old[axis]:6.1f}° -> {c:6.1f}° "
                                 f"({d:+5.1f}° {pos if d * sign > 0 else neg})")
        if was_radar:
            print(f"[SERVO] Dừng RADAR tại PAN {self.angle['pan']:.1f}° -> chuyển sang bám lửa")
        if self.log and parts:
            self.step += 1
            if clipped:
                parts.append("[CHẠM GIỚI HẠN GÓC]")
            print(f"[SERVO #{self.step:04d}] {reason:<9} | " + " | ".join(parts))

    # ---------- luồng servo ----------
    def _loop(self):
        dt = 1.0 / SERVO_RATE
        while self.running:
            t0 = time.time()
            with self.lock:
                if self.mode == "radar":
                    # Góc đích chạy đều qua lại, servo bám theo mượt (đổi chiều êm ở 2 đầu)
                    p = self.target["pan"] + self.radar_dir * RADAR_SPEED * dt
                    if p >= RADAR_MAX:
                        p, self.radar_dir = RADAR_MAX, -1
                        if self.log:
                            print(f"[RADAR] Chạm {RADAR_MAX}° -> quét ngược về {RADAR_MIN}°")
                    elif p <= RADAR_MIN:
                        p, self.radar_dir = RADAR_MIN, 1
                        if self.log:
                            print(f"[RADAR] Chạm {RADAR_MIN}° -> quét ngược về {RADAR_MAX}°")
                    self.target["pan"] = p
                    self.target["tilt"] = float(TILT_HOME)
                for axis in ("pan", "tilt"):
                    self.angle[axis] = self._approach(self.angle[axis], self.target[axis], dt)
                angles = dict(self.angle)
            for axis in ("pan", "tilt"):
                self._write(axis, angles[axis], t0)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    @staticmethod
    def _approach(cur, tgt, dt):
        """Tiến về đích: xa thì nhanh (tối đa SERVO_MAX_SPEED), gần thì chậm dần -> không giật."""
        diff = tgt - cur
        if abs(diff) < 0.05:
            return tgt
        v = float(np.clip(SERVO_EASE * diff, -SERVO_MAX_SPEED, SERVO_MAX_SPEED))
        step = v * dt
        return tgt if abs(step) >= abs(diff) else cur + step

    def _write(self, axis, angle, now):
        if abs(angle - self.sent[axis]) >= SERVO_MIN_STEP:
            if not self.sim:
                self.servos[axis].angle = angle
            self.sent[axis] = angle
            self.last_move[axis] = now
            self.attached[axis] = True
        elif (IDLE_DETACH and self.attached[axis]
              and now - self.last_move[axis] > IDLE_DETACH):
            # Đứng yên đủ lâu -> ngắt xung PWM: servo giữ vị trí bằng ma sát bánh răng, hết rung
            if not self.sim:
                self.servos[axis].detach()
            self.attached[axis] = False

    def go_home(self, why):
        """Đưa cả 2 servo về góc ban đầu (90°) ngay lập tức và chờ servo quay tới nơi."""
        print(f"[SERVO] {why}: quay về góc ban đầu PAN {PAN_HOME}° | TILT {TILT_HOME}°")
        if not self.sim:
            self.servos["pan"].angle = PAN_HOME
            self.servos["tilt"].angle = TILT_HOME
            time.sleep(HOME_SETTLE)
        home = {"pan": float(PAN_HOME), "tilt": float(TILT_HOME)}
        self.angle, self.target, self.sent = dict(home), dict(home), dict(home)

    def pump(self, on):
        if on == self.pump_on:
            return
        self.pump_on = on
        print(f"[PUMP] {'BẬT' if on else 'TẮT'}")
        if self.relay is not None:
            self.relay.on() if on else self.relay.off()

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.pump(False)
        if self.relay is not None:
            self.relay.close()
        self.go_home("Thoát chương trình")
        if not self.sim:
            for s in self.servos.values():
                s.detach()
                s.close()


# ===================== CAMERA (LUỒNG RIÊNG) =====================
class Camera:
    """Đọc camera liên tục ở luồng nền, chỉ giữ khung hình mới nhất."""

    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Không mở được webcam!")
        self.frame, self.frame_id = None, 0
        self.cond = threading.Condition()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.cond:
                self.frame, self.frame_id = frame, self.frame_id + 1
                self.cond.notify_all()

    def read(self, last_id, timeout=2.0):
        """Chờ khung hình mới hơn last_id. Trả về (id, frame) hoặc (last_id, None)."""
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id != last_id, timeout)
            if self.frame_id == last_id:
                return last_id, None
            return self.frame_id, self.frame

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


# ===================== MODEL =====================
def download_model(path):
    """Tải model lửa nếu chưa có file trên máy."""
    import urllib.request
    print(f"[MODEL] Chưa có '{path}', đang tải từ:\n        {MODEL_URL}")
    tmp = path + ".part"

    def progress(blocks, bsize, total):
        if total > 0:
            print(f"\r        {min(100, blocks * bsize * 100 // total):3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, tmp, progress)
        os.replace(tmp, path)
        print(f"\n[MODEL] Đã lưu model vào '{path}'")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit(f"[LỖI] Tải model thất bại ({e}). Kiểm tra Internet, "
                         f"hoặc tải thủ công từ link trên và đặt tên '{path}'.")


def export_onnx(pt_path, onnx_path, imgsz):
    """Chuyển .pt -> .onnx bằng export.py của YOLOv5 (chạy 1 lần, ở tiến trình con)."""
    print(f"[ONNX] Lần đầu: chuyển '{pt_path}' sang ONNX (imgsz={imgsz}), mất ~1-3 phút...")
    try:
        if not os.path.isdir(YOLOV5_HUB_DIR):
            # Tải mã nguồn YOLOv5 về cache của torch.hub
            subprocess.run([sys.executable, "-c",
                            "import torch; torch.hub.load('ultralytics/yolov5', 'custom', "
                            f"path={pt_path!r}, trust_repo=True)"], check=True)
        subprocess.run([sys.executable, os.path.join(YOLOV5_HUB_DIR, "export.py"),
                        "--weights", pt_path, "--include", "onnx",
                        "--imgsz", str(imgsz), "--device", "cpu"], check=True)
        produced = os.path.splitext(pt_path)[0] + ".onnx"
        os.replace(produced, onnx_path)
        print(f"[ONNX] Xong -> '{onnx_path}'")
        return True
    except Exception as e:
        print(f"[ONNX] Chuyển đổi thất bại ({e}) -> dùng PyTorch.")
        return False


def letterbox(img, size):
    """Thu ảnh về size x size, giữ tỉ lệ, viền xám (giống YOLOv5)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, px, py


class OnnxDetector:
    """Chạy model YOLOv5 .onnx bằng onnxruntime (nhanh), hoặc OpenCV DNN nếu thiếu."""

    def __init__(self, onnx_path, conf, iou=0.45):
        self.conf, self.iou = conf, iou
        self.class_names = ["fire"]
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = NUM_THREADS
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
            inp = self.sess.get_inputs()[0]
            self.in_name, self.size = inp.name, int(inp.shape[2])
            meta = self.sess.get_modelmeta().custom_metadata_map
            if "names" in meta:
                names = ast.literal_eval(meta["names"])
                self.class_names = list(names.values()) if isinstance(names, dict) else list(names)
            self.backend = "onnxruntime"
        except ImportError:
            print("[MODEL] Chưa có onnxruntime (pip install onnxruntime) -> dùng OpenCV DNN.")
            self.sess = None
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            cv2.setNumThreads(NUM_THREADS)
            self.size = int(os.path.splitext(onnx_path)[0].rsplit("_", 1)[-1])
            self.backend = "opencv-dnn"

    def detect(self, frame):
        img, r, px, py = letterbox(frame, self.size)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)   # NCHW, RGB, float32
        if self.sess is not None:
            pred = self.sess.run(None, {self.in_name: blob})[0][0]
        else:
            self.net.setInput(blob)
            pred = self.net.forward()[0]

        # pred: (N, 5 + số lớp) = cx, cy, w, h, obj, cls...
        pred = pred[pred[:, 4] > self.conf]
        if not len(pred):
            return []
        cls_ids = pred[:, 5:].argmax(1)
        scores = pred[:, 4] * pred[np.arange(len(pred)), 5 + cls_ids]
        keep = scores > self.conf
        pred, cls_ids, scores = pred[keep], cls_ids[keep], scores[keep]
        if not len(pred):
            return []

        # Đổi tọa độ về khung hình gốc
        x = (pred[:, 0] - pred[:, 2] / 2 - px) / r
        y = (pred[:, 1] - pred[:, 3] / 2 - py) / r
        w, h = pred[:, 2] / r, pred[:, 3] / r
        boxes = np.stack([x, y, w, h], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf, self.iou)
        H, W = frame.shape[:2]
        dets = []
        for i in np.array(idx).flatten():
            x1, y1 = max(0, int(x[i])), max(0, int(y[i]))
            x2, y2 = min(W - 1, int(x[i] + w[i])), min(H - 1, int(y[i] + h[i]))
            dets.append((x1, y1, x2, y2, float(scores[i]), str(self.class_names[cls_ids[i]])))
        return dets


class TorchDetector:
    """Dự phòng: chạy file .pt bằng PyTorch (torch.hub YOLOv5) - chậm hơn ONNX."""

    def __init__(self, pt_path, conf, imgsz):
        try:
            import torch
        except ModuleNotFoundError:
            raise SystemExit("[LỖI] Chưa cài PyTorch: pip install torch torchvision "
                             "gitpython seaborn thop pandas")
        except ImportError as e:
            raise SystemExit(
                f"[LỖI] PyTorch đã cài nhưng không chạy được:\n  {e}\n"
                "Nếu có 'undefined symbol': pip install --force-reinstall --no-cache-dir "
                "--index-url https://pypi.org/simple torch torchvision && "
                "pip install --ignore-installed numpy")
        torch.set_num_threads(NUM_THREADS)
        self.torch = torch
        self.imgsz = imgsz
        self.model = torch.hub.load("ultralytics/yolov5", "custom",
                                    path=pt_path, trust_repo=True)
        self.model.conf = conf
        names = self.model.names
        self.class_names = list(names.values()) if isinstance(names, dict) else list(names)
        self.backend = "pytorch"

    def detect(self, frame):
        with self.torch.inference_mode():
            res = self.model(frame[:, :, ::-1], size=self.imgsz)   # BGR -> RGB
        return [(int(a), int(b), int(c), int(d), float(s), str(self.class_names[int(k)]))
                for a, b, c, d, s, k in res.xyxy[0].tolist()]


def load_detector(pt_path, conf, imgsz, force_torch=False):
    if not os.path.exists(pt_path):
        download_model(pt_path)
    if not force_torch:
        onnx_path = f"{os.path.splitext(pt_path)[0]}_{imgsz}.onnx"
        if os.path.exists(onnx_path) or export_onnx(pt_path, onnx_path, imgsz):
            try:
                return OnnxDetector(onnx_path, conf)
            except Exception as e:
                print(f"[MODEL] Không chạy được ONNX ({e}) -> dùng PyTorch.")
    return TorchDetector(pt_path, conf, imgsz)


def pick_target(dets, fire_classes):
    """Chọn ngọn lửa lớn nhất (thường là gần nhất / nguy hiểm nhất)."""
    fires = [d for d in dets if not fire_classes or d[5].lower() in fire_classes]
    if not fires:
        return None
    return max(fires, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))


# ===================== CHƯƠNG TRÌNH CHÍNH =====================
def main():
    ap = argparse.ArgumentParser(description="Raspberry Pi Fire Tracker")
    ap.add_argument("--model", default="fire_yolov5s.pt", help="File trọng số YOLOv5 nhận diện lửa")
    ap.add_argument("--cam", type=int, default=0, help="Chỉ số webcam")
    ap.add_argument("--conf", type=float, default=0.45, help="Ngưỡng tin cậy")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="Kích thước ảnh vào model: 320 (mặc định), 256 nhanh hơn, 416 chính xác hơn")
    ap.add_argument("--classes", default="fire",
                    help="Tên lớp coi là lửa, cách nhau dấu phẩy. Rỗng '' = mọi lớp")
    ap.add_argument("--pump", action="store_true",
                    help="Bật máy bơm khi khóa mục tiêu (mặc định chỉ servo theo dõi)")
    ap.add_argument("--torch", action="store_true", help="Dùng PyTorch thay vì ONNX")
    ap.add_argument("--no-servo-log", action="store_true", help="Không in log từng bước xoay servo")
    ap.add_argument("--no-display", action="store_true", help="Không hiển thị cửa sổ")
    args = ap.parse_args()

    fire_classes = {c.strip().lower() for c in args.classes.split(",") if c.strip()}

    # Chạy qua SSH không có màn hình -> không mở cửa sổ được
    if (not args.no_display and os.name == "posix"
            and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")):
        print("[DISPLAY] Không có màn hình (đang chạy qua SSH) -> tắt cửa sổ hiển thị.\n"
              "          Muốn xem trên VNC/màn hình Pi: chạy 'export DISPLAY=:0' rồi chạy lại.")
        args.no_display = True

    cv2.setNumThreads(NUM_THREADS)
    detector = load_detector(args.model, args.conf, args.imgsz, args.torch)
    names = [str(n).lower() for n in detector.class_names]
    print(f"[MODEL] Backend: {detector.backend} | các lớp: {names}")
    if fire_classes and not fire_classes & set(names):
        print(f"[MODEL] Không có lớp {sorted(fire_classes)} trong model -> coi mọi vật là lửa.")
        fire_classes = set()

    hw = Hardware(use_pump=args.pump, log=not args.no_servo_log)
    try:
        cam = Camera(args.cam, FRAME_W, FRAME_H)
    except RuntimeError as e:
        hw.close()
        raise SystemExit(f"[LỖI] {e}")

    lock_count = 0
    last_seen = 0.0
    pump_start = cooldown_until = 0.0
    fps, t_prev, last_log = 0.0, time.time(), 0.0
    frame_id = 0
    prev_target = prev_locked = False
    ex_f = ey_f = 0.0                    # tâm lửa đã làm mượt
    active_x = active_y = False          # trục đang trong pha "đang xoay" (hysteresis)

    print("[RUN] Bắt đầu theo dõi lửa. Nhấn 'q' (hoặc Ctrl+C) để thoát.")
    try:
        while True:
            frame_id, frame = cam.read(frame_id)
            if frame is None:
                print("[CAM] Không nhận được khung hình, đang chờ...")
                continue
            # Góc servo ĐÚNG LÚC chụp khung hình (servo vẫn chạy trong lúc model xử lý)
            snap_pan, snap_tilt = hw.get_angles()

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            now = time.time()

            target = pick_target(detector.detect(frame), fire_classes)
            locked = False

            if target:
                x1, y1, x2, y2, conf, name = target
                tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
                ex, ey = (tx - cx) / w, (ty - cy) / h      # lệch tâm chuẩn hóa [-0.5, 0.5]
                last_seen = now
                first = not prev_target

                # --- Làm mượt tâm lửa (khung lửa nhảy nhẹ mỗi lần nhận diện) ---
                if first:
                    ex_f, ey_f = ex, ey
                    active_x = active_y = False
                    print(f"[SERVO] Phát hiện lửa tại ({tx},{ty}) -> bắt đầu bám theo")
                else:
                    ex_f = CENTER_SMOOTH * ex + (1 - CENTER_SMOOTH) * ex_f
                    ey_f = CENTER_SMOOTH * ey + (1 - CENTER_SMOOTH) * ey_f

                # --- Vùng chồng lấn (hysteresis) ---
                if abs(ex_f) > ZONE_OUTER:
                    active_x = True
                elif abs(ex_f) < ZONE_INNER:
                    active_x = False
                if abs(ey_f) > ZONE_OUTER:
                    active_y = True
                elif abs(ey_f) < ZONE_INNER:
                    active_y = False

                # --- Góc đích = góc lúc chụp + độ lệch đổi ra độ (chỉ sửa TRACK_GAIN phần) ---
                new_pan = new_tilt = None
                if active_x:
                    new_pan = snap_pan + PAN_DIR * ex_f * CAM_HFOV * TRACK_GAIN
                elif first:
                    new_pan = snap_pan           # vừa thấy lửa: dừng radar ngay tại chỗ
                if active_y:
                    new_tilt = snap_tilt + TILT_DIR * ey_f * CAM_VFOV * TRACK_GAIN
                elif first:
                    new_tilt = snap_tilt
                if new_pan is not None or new_tilt is not None or first:
                    hw.track(new_pan, new_tilt)

                # --- Kiểm tra khóa mục tiêu ---
                if abs(ex_f) < LOCK_TOLERANCE and abs(ey_f) < LOCK_TOLERANCE:
                    lock_count = min(lock_count + 1, LOCK_FRAMES)
                else:
                    lock_count = max(lock_count - 1, 0)
                locked = lock_count >= LOCK_FRAMES
            else:
                lock_count = 0
                if prev_target:
                    print(f"[SERVO] Mất dấu lửa -> đứng yên chờ {RADAR_START_DELAY:.1f}s "
                          "rồi quét radar")
                    hw.hold()
                if now - last_seen > RADAR_START_DELAY:
                    hw.start_radar()

            if locked != prev_locked:
                print(f"[SERVO] {'ĐÃ KHÓA MỤC TIÊU (lửa ở giữa khung hình)' if locked else 'Mất khóa mục tiêu'}"
                      f" | PAN {hw.pan_angle:.1f}° TILT {hw.tilt_angle:.1f}°")
            prev_target, prev_locked = bool(target), locked

            # --- Điều khiển máy bơm (chỉ khi có --pump) ---
            if args.pump:
                fire_recent = last_seen and (now - last_seen) < LOST_TIMEOUT
                want = (locked or (hw.pump_on and fire_recent)) and now >= cooldown_until
                if want and not hw.pump_on:
                    pump_start = now
                if want and now - pump_start > MAX_PUMP_TIME:
                    print("[PUMP] Bơm quá lâu -> nghỉ bảo vệ máy bơm.")
                    want = False
                    cooldown_until = now + PUMP_COOLDOWN
                hw.pump(want)

            # --- FPS ---
            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)

            state = ("TARGET LOCKED! PUMPING..." if hw.pump_on else
                     "TARGET LOCKED!" if locked else
                     "TRACKING FIRE..." if target else
                     "RADAR SCANNING..." if hw.mode == "radar" else "SEARCHING...")

            if args.no_display:
                if now - last_log >= 1.0:
                    last_log = now
                    extra = (f" | {target[5]} {target[4]:.2f} lệch ({ex_f:+.2f},{ey_f:+.2f})"
                             if target else "")
                    print(f"[{state}] FPS {fps:.1f} | Pan {hw.pan_angle:.0f} | "
                          f"Tilt {hw.tilt_angle:.0f}{extra}")
                continue

            # --- Vẽ & hiển thị (vẽ thẳng lên frame, không copy) ---
            cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            # Vùng vàng: lửa trong vùng này servo không xoay (vùng chồng lấn ngoài)
            # Vùng xanh dương: vùng khóa mục tiêu
            for frac, col in ((ZONE_OUTER, (0, 255, 255)), (LOCK_TOLERANCE, (255, 128, 0))):
                zx, zy = int(frac * w), int(frac * h)
                cv2.rectangle(frame, (cx - zx, cy - zy), (cx + zx, cy + zy), col, 1)
            if target:
                color = (0, 0, 255) if locked else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.line(frame, (cx, cy), (tx, ty), color, 1)
            color = (0, 0, 255) if locked else (0, 255, 0) if target else (200, 200, 200)
            cv2.putText(frame, state, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            info = f"FPS {fps:.1f} Pan {hw.pan_angle:.0f} Tilt {hw.tilt_angle:.0f}"
            if hw.sim:
                info += " SIM"
            cv2.putText(frame, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[RUN] Dừng bởi người dùng.")
    finally:
        hw.close()
        cam.release()
        cv2.destroyAllWindows()
        print("[RUN] Đã dừng và giải phóng tài nguyên.")


if __name__ == "__main__":
    main()
