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
  tới góc đích; có vùng chồng lấn (hysteresis) + lọc mượt hướng lửa để chống giật
- AI (YOLO ~4 FPS) chạy ở luồng riêng chỉ để phát hiện/xác nhận lửa; giữa các lần AI,
  vị trí lửa được bám bằng lọc màu ở tốc độ camera (~25-30 FPS) -> servo không giật

Chạy:  python fire_tracker.py              (chỉ servo theo dõi lửa)
       python fire_tracker.py --pump       (bật thêm máy bơm khi khóa mục tiêu)
       python fire_tracker.py --imgsz 256  (nhanh hơn nữa, kém chính xác hơn chút)
       python fire_tracker.py --torch      (bỏ qua ONNX, dùng PyTorch như cũ)
       python fire_tracker.py --servo hwpwm (servo dùng PWM PHẦN CỨNG - mượt nhất)
       python fire_tracker.py --servo-trace (ghi thêm từng lệnh servo để tìm nguyên nhân giật)
Log: mọi thông báo được lưu vào logs/fire_tracker_<ngày>_<giờ>.txt (tắt: --no-log-file)

Bật PWM phần cứng (làm 1 lần) để servo hết giật:
  1) pip install rpi-hardware-pwm
  2) Thêm vào cuối /boot/firmware/config.txt:
       dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4
  3) sudo reboot. Cắm dây tín hiệu servo PAN -> GPIO 12 (chân 32), TILT -> GPIO 13 (chân 33)
  4) python fire_tracker.py --servo hwpwm
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
TILT_MIN, TILT_MAX = 0, 10      # Giới hạn góc servo dọc: chỉ trong 0°..10°
PAN_HOME, TILT_HOME = 90, 5     # Góc ban đầu: khởi động/thoát chương trình đều quay về đây
                                # (TILT_HOME phải nằm trong TILT_MIN..TILT_MAX)
TILT_HOME = min(max(TILT_HOME, TILT_MIN), TILT_MAX)   # an toàn: ép HOME vào trong giới hạn
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

# ===================== BỘ NHỚ LỬA =====================
# Mất dấu lửa (hoặc radar lướt qua lửa): KHÔNG quét lại từ đầu mà
#   1) QUAY LẠI chỗ lửa cuối cùng (có dự đoán theo hướng lửa đang di chuyển), đứng chờ AI
#   2) QUÉT HẸP quanh chỗ đó
#   3) vẫn không thấy mới QUÉT RADAR toàn bộ; radar đi chậm lại ở những chỗ từng có lửa
RECALL_DWELL = 1.2     # Đứng chờ AI tại chỗ lửa cuối cùng bao lâu (giây)
PREDICT_TIME = 0.5     # Dự đoán lửa đi thêm bao lâu theo vận tốc lúc mất dấu (giây)
LOCAL_RANGE = 25.0     # Quét hẹp ± 25° quanh chỗ lửa cuối cùng
LOCAL_SPEED = 20.0     # Tốc độ quét hẹp (độ/giây)
LOCAL_TURNS = 3        # Số lần chạm biên khi quét hẹp (3 = qua lại đủ 1.5 lượt)
HOTSPOT_TTL = 120.0    # Nhớ các chỗ từng có lửa trong 120 giây
HOTSPOT_RANGE = 15.0   # Radar đi qua gần chỗ từng có lửa (± 15°)...
HOTSPOT_SLOW = 0.4     # ...thì chậm lại còn 40% tốc độ để AI kịp nhận diện

# ===================== TỰ PHÁT HIỆN SERVO QUAY NGƯỢC =====================
# Khi servo xoay mà lửa trong ảnh dịch NGƯỢC với dự đoán -> PAN_DIR/TILT_DIR bị sai,
# chương trình tự đảo chiều và in ra giá trị đúng để sửa vĩnh viễn trong file.

# ===================== PHÁT HIỆN SERVO BỊ CẤN (KẸT CƠ KHÍ) =====================
# Servo thường không báo vị trí thật -> dùng CAMERA làm cảm biến: ra lệnh xoay mà khung cảnh
# trong ảnh KHÔNG dịch chuyển => servo đang bị cấn. Khi đó lùi ra, thu hẹp giới hạn góc
# và LƯU vào servo_limits.json (lần chạy sau vẫn nhớ). Xóa: chạy với --reset-limits
STALL_DETECT = False      # THỬ NGHIỆM, đang TẮT: còn báo nhầm khi khung cảnh trơn. Bật: True
STALL_MIN_CMD = 8.0       # Phải ra lệnh xoay tổng ≥ 8° liên tục mới xét (tránh nhầm do nhiễu)
STALL_RATIO = 0.25        # Ảnh dịch < 25% so với lệnh => coi là bị cấn
STALL_GAP = 5.0           # Lúc đổi chiều: góc lệnh đi quá vị trí thật ≥ 5° => cấn (gần biên)
STALL_MARGIN = 3.0        # Giới hạn mới = chỗ cấn lùi vào 3°
STALL_MIN_RESPONSE = 0.05 # Độ tin cậy đo dịch ảnh tối thiểu
STALL_MIN_TEXTURE = 8.0   # Khung cảnh phải đủ chi tiết (độ lệch chuẩn độ sáng) - tường trắng/tối thì bỏ qua
STALL_MIN_SPAN = 0.3      # Không thu hẹp giới hạn nhỏ hơn 30% khoảng ban đầu
LIMITS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servo_limits.json")

AUTO_DIR = False        # Mặc định TẮT: chỉnh PAN_DIR/TILT_DIR bằng tay 1 lần (cách chuẩn)
AUTO_DIR_MIN_MOVE = 3.0   # Servo phải xoay tổng cộng ≥ 3° trong lúc thấy lửa mới kết luận

# ===================== BÁM LỬA CHỐNG GIẬT =====================
# AI (YOLO) chậm (~4 FPS trên Pi) nên chạy ở LUỒNG RIÊNG, chỉ để PHÁT HIỆN / XÁC NHẬN lửa.
# Giữa 2 lần AI chạy, vị trí lửa được bám bằng LỌC MÀU LỬA (rất nhẹ) ở tốc độ camera
# (~25-30 FPS) -> servo nhận vị trí lửa liên tục, không phải chờ AI -> chuyển động mượt.
FAST_TRACK = True         # False = chỉ dùng AI (giật hơn)
ROI_EXPAND = 1.8          # Vùng tìm màu lửa = khung lửa của AI phóng to 1.8 lần
FIRE_V_MIN = 190          # Độ sáng tối thiểu của điểm ảnh lửa (0-255). Phòng tối có thể giảm 160
FIRE_MIN_PIXELS = 12      # Ít hơn số điểm ảnh này coi như lọc màu không thấy lửa
AI_CONFIRM_TIMEOUT = 1.0  # AI không thấy lửa quá 1s thì bỏ bám (chống bám nhầm đèn/nắng)

# Vùng chồng lấn (hysteresis), tính theo TỈ LỆ khung hình:
#   lửa lệch RA NGOÀI ZONE_OUTER mới bắt đầu xoay,
#   đã xoay thì xoay tới khi lệch VÀO TRONG ZONE_INNER mới dừng.
#   -> lửa dao động nhẹ quanh tâm sẽ KHÔNG làm servo giật qua lại.
ZONE_OUTER = 0.12
ZONE_INNER = 0.04
WORLD_SMOOTH = 0.25   # Lọc mượt hướng lửa (dùng cho bộ nhớ lửa / dự đoán khi mất dấu)

# Bộ điều khiển PID cho từng trục - cách chuẩn trong các dự án pan-tilt tracking.
# Đầu vào: độ lệch tâm e (tỉ lệ khung hình, -0.5..0.5). Đầu ra: góc cần xoay thêm (× góc nhìn).
#   KP: lệch càng xa xoay càng mạnh (chính). Tăng nếu bám chậm, giảm nếu dao động.
#   KI: bù lệch nhỏ còn sót lại lâu (lửa dừng hơi lệch tâm mãi). Để nhỏ.
#   KD: hãm khi đang tiến nhanh về tâm -> chống vọt lố. Tăng nếu bị vọt qua tâm.
PID_KP = 0.6
PID_KI = 0.15
PID_KD = 0.05
PID_I_LIMIT = 0.3     # Giới hạn phần tích phân (chống "tích lũy quá đà" - anti-windup)
ERR_SMOOTH = 0.5      # Lọc mượt độ lệch tâm trước khi đưa vào PID (0..1)
LOCK_TOLERANCE = 0.10 # Lệch tâm < 10% khung hình coi là "đã nhắm"
LOCK_FRAMES = 8       # Số khung hình liên tiếp nhắm trúng mới coi là khóa mục tiêu

# Luồng điều khiển servo chạy riêng, servo trượt mượt tới góc đích
SERVO_RATE = 50          # Số lần cập nhật servo mỗi giây
SERVO_MAX_SPEED = 90.0   # Tốc độ xoay tối đa khi bám lửa (độ/giây)
SERVO_EASE = 4.0         # Độ "mềm": càng nhỏ servo càng chậm rãi khi gần tới đích
SERVO_LOG_INTERVAL = 0.4 # Khi bám lửa, log servo tối đa mỗi 0.4s (tránh tràn màn hình)
SERVO_MIN_STEP = 0.3     # Chỉ gửi lệnh khi góc đổi > 0.3° -> bớt rung
# Đứng yên quá N giây thì ngắt xung PWM (0 = TẮT). Mặc định TẮT vì khi servo quay lại,
# xung đầu tiên sau lúc ngắt làm servo GIẬT 1 cái ở đầu mỗi lần quay.
IDLE_DETACH = 0

# Cách tạo xung servo (chọn bằng --servo):
#   "hwpwm"   : PWM PHẦN CỨNG - xung chuẩn tuyệt đối, MƯỢT NHẤT. Servo phải cắm vào
#               GPIO 12 (PAN) và GPIO 13 (TILT), cần cài đặt 1 lần (xem hướng dẫn).
#   "pigpio"  : PWM qua DMA (Pi 1-4, cần pigpiod). Mượt.
#   "gpiozero": PWM PHẦN MỀM - dễ rung/giật do CPU bận chạy AI.
#   "auto"    : thử pigpio, không được thì gpiozero.
HW_PWM_PAN_CHANNEL = 0    # Kênh PWM0 = GPIO 12 (chân vật lý 32)
HW_PWM_TILT_CHANNEL = 1   # Kênh PWM1 = GPIO 13 (chân vật lý 33)
HW_PWM_CHIP = None        # None = tự chọn (Pi 5: chip 2, Pi 1-4: chip 0)

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
class HwPwmServo:
    """Servo chạy bằng PWM PHẦN CỨNG của Pi (thư viện rpi-hardware-pwm).
    Xung do chip tạo ra, không phụ thuộc CPU -> không rung/giật kể cả khi AI chạy nặng.
    Dùng giống AngularServo của gpiozero: .angle = 0..180, .detach(), .close()"""

    def __init__(self, channel, angle):
        from rpi_hardware_pwm import HardwarePWM
        chip = HW_PWM_CHIP
        if chip is None:
            try:
                with open("/proc/device-tree/model") as f:
                    chip = 2 if "Raspberry Pi 5" in f.read() else 0
            except OSError:
                chip = 0
        self.chip = chip
        self.pwm = HardwarePWM(pwm_channel=channel, hz=50, chip=chip)
        self.pwm.start(self._duty(angle))
        self.running = True

    @staticmethod
    def _duty(angle):
        pulse_ms = 0.5 + float(np.clip(angle, 0, 180)) / 180.0 * 2.0   # 0.5ms..2.5ms
        return pulse_ms / 20.0 * 100.0                                  # % của chu kỳ 20ms

    @property
    def angle(self):
        return None

    @angle.setter
    def angle(self, value):
        if self.running:
            self.pwm.change_duty_cycle(self._duty(value))
        else:
            self.pwm.start(self._duty(value))
            self.running = True

    def detach(self):
        if self.running:
            self.pwm.stop()
            self.running = False

    def close(self):
        self.detach()


class Hardware:
    """Servo + relay. Servo được điều khiển bởi một luồng riêng (SERVO_RATE Hz):
    chương trình chính chỉ đặt GÓC ĐÍCH, luồng servo đưa servo trượt mượt tới đó.

    Chế độ (mode):
      "radar" - servo ngang quét qua lại RADAR_MIN..RADAR_MAX, servo dọc về TILT_HOME
      "track" - bám theo lửa (góc đích do chương trình chính đặt)
      "hold"  - đứng yên tại chỗ (vừa mất dấu lửa, chờ trước khi quét radar)
    Không chạy trên Pi thì tự chuyển sang chế độ giả lập (vẫn tính góc, in log)."""

    def __init__(self, use_pump=False, log=True, backend="auto", trace=None):
        self.sim = False
        self.relay = None
        self.log = log
        self.step = 0
        self.servos = {}
        self.trace = trace            # file ghi từng lệnh gửi servo (--servo-trace)
        self.t0 = time.time()
        self.backend = "sim"
        try:
            from gpiozero import AngularServo, OutputDevice
            factory = None
            if backend in ("auto", "pigpio"):
                try:
                    # pigpio: xung PWM qua DMA, chính xác (Pi 1-4; Pi 5 không hỗ trợ)
                    # Cần: sudo apt install pigpio && sudo systemctl enable --now pigpiod
                    from gpiozero.pins.pigpio import PiGPIOFactory
                    factory = PiGPIOFactory()
                except Exception as e:
                    if backend == "pigpio":
                        raise RuntimeError(f"không kết nối được pigpiod ({e})")

            if backend == "hwpwm":
                self.servos["pan"] = HwPwmServo(HW_PWM_PAN_CHANNEL, PAN_HOME)
                self.servos["tilt"] = HwPwmServo(HW_PWM_TILT_CHANNEL, TILT_HOME)
                self.backend = "hwpwm"
                print(f"[HW] Servo dùng PWM PHẦN CỨNG (chip {self.servos['pan'].chip}): "
                      "PAN = GPIO 12, TILT = GPIO 13 -> mượt nhất.")
            else:
                kw = dict(min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000)
                if factory:
                    kw["pin_factory"] = factory
                # initial_angle = góc HOME -> vừa cấp xung là servo quay thẳng về 90°
                self.servos["pan"] = AngularServo(PAN_PIN, min_angle=0, max_angle=180,
                                                  initial_angle=PAN_HOME, **kw)
                self.servos["tilt"] = AngularServo(TILT_PIN, min_angle=0, max_angle=180,
                                                   initial_angle=TILT_HOME, **kw)
                if factory:
                    self.backend = "pigpio"
                    print("[HW] Servo dùng pigpio (PWM qua DMA, mượt).")
                else:
                    self.backend = "gpiozero"
                    print("[HW] CẢNH BÁO: servo dùng PWM PHẦN MỀM -> dễ rung/giật. "
                          "Nên dùng --servo hwpwm (xem hướng dẫn trong file).")

            if use_pump:
                rkw = dict(active_high=RELAY_ACTIVE_HIGH, initial_value=False)
                if factory:
                    rkw["pin_factory"] = factory
                self.relay = OutputDevice(RELAY_PIN, **rkw)
            else:
                print("[HW] Máy bơm TẮT (chỉ servo theo dõi lửa). Dùng --pump để bật.")
        except Exception as e:
            if backend == "hwpwm":
                raise SystemExit(
                    f"[LỖI] Không bật được PWM phần cứng: {e}\n"
                    "  1) pip install rpi-hardware-pwm\n"
                    "  2) Thêm vào cuối /boot/firmware/config.txt dòng:\n"
                    "       dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4\n"
                    "  3) sudo reboot, rồi cắm servo PAN vào GPIO 12, TILT vào GPIO 13")
            if backend == "pigpio":
                raise SystemExit(f"[LỖI] Không dùng được pigpio: {e}\n"
                                 "  sudo apt install -y pigpio && sudo systemctl enable --now pigpiod")
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
        self._log_t, self._log_ref = 0.0, dict(self.target)   # chống spam log
        self.hotspots = []          # các góc PAN từng có lửa (radar đi chậm lại ở đó)
        self.sweep, self.search_done = None, False
        # Giới hạn góc ĐANG DÙNG (có thể bị thu hẹp khi phát hiện servo bị cấn)
        self.limits = {"pan": [float(PAN_MIN), float(PAN_MAX)],
                       "tilt": [float(TILT_MIN), float(TILT_MAX)]}
        self._load_limits()
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

    # ---------- giới hạn góc (tự học khi bị cấn) ----------
    def radar_range(self):
        lo, hi = self.limits["pan"]
        return max(RADAR_MIN, lo), min(RADAR_MAX, hi)

    def _load_limits(self):
        try:
            import json
            with open(LIMITS_FILE, encoding="utf-8") as f:
                saved = json.load(f)
        except (OSError, ValueError):
            return
        for axis, (lo0, hi0) in (("pan", (PAN_MIN, PAN_MAX)), ("tilt", (TILT_MIN, TILT_MAX))):
            if axis in saved:
                lo, hi = saved[axis]
                self.limits[axis] = [max(lo0, float(lo)), min(hi0, float(hi))]
        print(f"[CẤN] Nạp giới hạn đã học từ {os.path.basename(LIMITS_FILE)}: "
              f"PAN {self.limits['pan'][0]:.0f}°..{self.limits['pan'][1]:.0f}° | "
              f"TILT {self.limits['tilt'][0]:.0f}°..{self.limits['tilt'][1]:.0f}°")

    def _save_limits(self):
        try:
            import json
            with open(LIMITS_FILE, "w", encoding="utf-8") as f:
                json.dump({**self.limits, "cap_nhat": time.strftime("%Y-%m-%d %H:%M:%S")},
                          f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[CẤN] Không lưu được {LIMITS_FILE}: {e}")

    def report_stall(self, axis, side, angle):
        """Servo `axis` bị cấn khi xoay về phía `side` (+1 tăng góc / -1 giảm góc) tại `angle`.
        Thu hẹp giới hạn, lùi servo ra khỏi chỗ cấn, lưu lại."""
        lo0, hi0 = (PAN_MIN, PAN_MAX) if axis == "pan" else (TILT_MIN, TILT_MAX)
        min_span = STALL_MIN_SPAN * (hi0 - lo0)
        with self.lock:
            lo, hi = self.limits[axis]
            if side > 0:
                new = max(lo + min_span, angle - STALL_MARGIN)
                if new >= hi - 0.5:
                    return
                self.limits[axis][1] = new
            else:
                new = min(hi - min_span, angle + STALL_MARGIN)
                if new <= lo + 0.5:
                    return
                self.limits[axis][0] = new
            # Đồng bộ góc lệnh = vị trí THẬT của servo (lệnh đã chạy quá chỗ cấn),
            # rồi lùi ra khỏi chỗ cấn ngay (servo kẹt kéo dòng lớn, nóng, có thể cháy)
            lo, hi = self.limits[axis]
            self.angle[axis] = float(angle)
            self.target[axis] = float(np.clip(self.target[axis], lo, hi))
            if axis == "pan" and self.mode in ("radar", "search"):
                self.radar_dir = -side
                self.target["pan"] = new
                if self.mode == "search":
                    self.sweep["lo"] = max(self.sweep["lo"], lo)
                    self.sweep["hi"] = min(self.sweep["hi"], hi)
            old = (lo0, hi0)
        print(f"[CẤN] Servo {axis.upper()} bị CẤN tại ~{angle:.1f}° khi xoay "
              f"{'tăng' if side > 0 else 'giảm'} góc -> giới hạn mới {axis.upper()} "
              f"{lo:.0f}°..{hi:.0f}° (ban đầu {old[0]}°..{old[1]}°), lùi servo ra")
        self._save_limits()

    def start_radar(self):
        """Bắt đầu quét radar từ vị trí hiện tại, hướng về phía đầu xa hơn."""
        r_lo, r_hi = self.radar_range()
        with self.lock:
            if self.mode == "radar":
                return
            self.mode = "radar"
            self.target["pan"] = self.angle["pan"]
            mid = (r_lo + r_hi) / 2
            self.radar_dir = 1 if self.angle["pan"] < mid else -1
        print(f"[SERVO] Không thấy lửa -> RADAR quét {r_lo:.0f}°↔{r_hi:.0f}° "
              f"({RADAR_SPEED:.0f}°/s), TILT về {TILT_HOME}°")

    def hold(self):
        """Dừng tại chỗ."""
        with self.lock:
            self.mode = "hold"
            self.target = dict(self.angle)

    def start_search(self, center, tilt, rng, speed, turns):
        """Quét hẹp quanh `center` (± rng độ), chạm biên `turns` lần thì xong (search_done)."""
        lo, hi = max(self.limits["pan"][0], center - rng), min(self.limits["pan"][1], center + rng)
        with self.lock:
            self.mode = "search"
            self.sweep = dict(lo=lo, hi=hi, speed=speed, turns=turns,
                              tilt=float(np.clip(tilt, *self.limits["tilt"])))
            self.target["pan"] = self.angle["pan"]
            self.radar_dir = 1 if self.angle["pan"] < (lo + hi) / 2 else -1
            self.search_done = False
        print(f"[NHỚ] Quét hẹp quanh PAN {center:.0f}° ({lo:.0f}°↔{hi:.0f}°, {speed:.0f}°/s)")

    def track(self, pan=None, tilt=None, reason="BÁM LỬA"):
        """Đặt góc đích khi bám lửa. None = giữ nguyên góc đích trục đó.
        Được gọi ~30 lần/giây nên log chỉ in mỗi SERVO_LOG_INTERVAL giây."""
        clipped = False
        with self.lock:
            was_radar = self.mode in ("radar", "search")
            self.mode = "track"
            for axis, val, lo, hi in (("pan", pan, *self.limits["pan"]),
                                      ("tilt", tilt, *self.limits["tilt"])):
                if val is None:
                    continue
                c = float(np.clip(val, lo, hi))
                clipped |= c != val
                self.target[axis] = c
            new = dict(self.target)
            if was_radar:
                self._log_ref = dict(self.angle)
        if was_radar:
            print(f"[SERVO] Dừng quét tại PAN {self.angle['pan']:.1f}° -> chuyển sang bám lửa")
        now = time.time()
        if not self.log or now - self._log_t < SERVO_LOG_INTERVAL:
            return
        parts = []
        for axis, sign, pos, neg in (("pan", PAN_DIR, "sang PHẢI", "sang TRÁI"),
                                     ("tilt", TILT_DIR, "XUỐNG", "LÊN")):
            d = new[axis] - self._log_ref[axis]
            if abs(d) >= SERVO_MIN_STEP:
                parts.append(f"{axis.upper()} {self._log_ref[axis]:6.1f}° -> {new[axis]:6.1f}° "
                             f"({d:+5.1f}° {pos if d * sign > 0 else neg})")
        if parts:
            self.step += 1
            if clipped:
                parts.append("[CHẠM GIỚI HẠN GÓC]")
            print(f"[SERVO #{self.step:04d}] {reason:<9} | " + " | ".join(parts))
            self._log_t, self._log_ref = now, new

    # ---------- luồng servo ----------
    def _loop(self):
        dt = 1.0 / SERVO_RATE
        while self.running:
            t0 = time.time()
            with self.lock:
                if self.mode in ("radar", "search"):
                    # Góc đích chạy đều qua lại, servo bám theo mượt (đổi chiều êm ở 2 đầu)
                    if self.mode == "radar":
                        lo, hi = max(RADAR_MIN, self.limits["pan"][0]), min(RADAR_MAX, self.limits["pan"][1])
                        speed = RADAR_SPEED
                        tilt = float(np.clip(TILT_HOME, *self.limits["tilt"]))
                        # Gần chỗ từng có lửa -> đi chậm lại cho AI kịp nhận diện
                        if any(abs(self.angle["pan"] - hp) < HOTSPOT_RANGE for hp in self.hotspots):
                            speed *= HOTSPOT_SLOW
                    else:
                        sw = self.sweep
                        lo, hi, speed, tilt = sw["lo"], sw["hi"], sw["speed"], sw["tilt"]
                    p = self.target["pan"] + self.radar_dir * speed * dt
                    edge = None
                    if p >= hi:
                        p, self.radar_dir, edge = hi, -1, hi
                    elif p <= lo:
                        p, self.radar_dir, edge = lo, 1, lo
                    if edge is not None:
                        tag = "RADAR" if self.mode == "radar" else "QUÉT HẸP"
                        if self.log:
                            print(f"[{tag}] Chạm {edge:.0f}° -> quét ngược lại")
                        if self.mode == "search":
                            sw["turns"] -= 1
                            if sw["turns"] <= 0:
                                self.mode, self.search_done = "hold", True
                    self.target["pan"] = p
                    self.target["tilt"] = float(tilt)
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
            if self.trace:
                self.trace.write(f"{now - self.t0:8.3f}\t{axis}\t{angle:.2f}\t"
                                 f"{'' if self.attached[axis] else 'REATTACH'}\t{self.mode}\n")
            self.sent[axis] = angle
            self.last_move[axis] = now
            self.attached[axis] = True
        elif (IDLE_DETACH and self.attached[axis]
              and now - self.last_move[axis] > IDLE_DETACH):
            # Đứng yên đủ lâu -> ngắt xung PWM: servo giữ vị trí bằng ma sát bánh răng, hết rung
            if not self.sim:
                self.servos[axis].detach()
            if self.trace:
                self.trace.write(f"{now - self.t0:8.3f}\t{axis}\t{angle:.2f}\tDETACH\t{self.mode}\n")
            self.attached[axis] = False

    def go_home(self, why):
        """Đưa cả 2 servo về góc ban đầu (90°) ngay lập tức và chờ servo quay tới nơi."""
        # Góc HOME luôn nằm trong giới hạn đang dùng (kể cả giới hạn đã học khi bị cấn)
        home = {"pan": float(np.clip(PAN_HOME, *self.limits["pan"])),
                "tilt": float(np.clip(TILT_HOME, *self.limits["tilt"]))}
        print(f"[SERVO] {why}: quay về góc ban đầu PAN {home['pan']:.0f}° | TILT {home['tilt']:.0f}°")
        if not self.sim:
            self.servos["pan"].angle = home["pan"]
            self.servos["tilt"].angle = home["tilt"]
            time.sleep(HOME_SETTLE)
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


class StallDetector:
    """Phát hiện servo bị CẤN bằng camera.
    Mỗi khung hình đo khung cảnh dịch bao nhiêu (tương quan pha - cv2.phaseCorrelate),
    đổi ra độ rồi so với góc đã RA LỆNH cho servo. Lệnh xoay ≥ STALL_MIN_CMD độ mà ảnh dịch
    < STALL_RATIO lần lệnh => servo bị kẹt cơ khí ở đó."""

    SIZE = (160, 120)   # ảnh thu nhỏ để đo cho nhanh (~1ms)

    def __init__(self):
        self.prev = None
        self.win = cv2.createHanningWindow(self.SIZE, cv2.CV_32F)
        self.acc = {a: None for a in ("pan", "tilt")}
        self.carry = {a: None for a in ("pan", "tilt")}   # vị trí thật nối sang cửa sổ sau
        self.pause_until = 0.0

    def update(self, frame, pan, tilt, now):
        """Trả về danh sách (trục, chiều, góc_cấn) nếu phát hiện cấn."""
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.SIZE).astype(np.float32)
        events = []
        if self.prev is not None:
            pg, ppan, ptilt = self.prev
            (dx, dy), resp = cv2.phaseCorrelate(pg, g, self.win)
            if min(float(g.std()), float(pg.std())) < STALL_MIN_TEXTURE:
                resp = 0.0     # khung cảnh trơn (tường trắng, tối): không đo được dịch -> bỏ qua
            W, H = self.SIZE
            for axis, d_px, size, fov, sign, ang, pang in (
                    ("pan", dx, W, CAM_HFOV, PAN_DIR, pan, ppan),
                    ("tilt", dy, H, CAM_VFOV, TILT_DIR, tilt, ptilt)):
                d_cmd = ang - pang
                # Ảnh dịch đổi ra "độ servo thật sự đã quay" (≈ d_cmd nếu servo quay bình thường)
                d_real = -sign * d_px / size * fov
                a = self.acc[axis]
                if abs(d_cmd) < 0.1 or resp < STALL_MIN_RESPONSE:
                    if a and now - a["t"] > 0.4:
                        self.acc[axis] = None        # đứng yên / ảnh không tin cậy -> bỏ cửa sổ
                    continue
                if a is None or np.sign(d_cmd) != np.sign(a["cmd"]):
                    # ĐỔI CHIỀU (vd radar chạm biên): nếu góc lệnh đã đi quá vị trí thật
                    # ≥ STALL_GAP độ thì servo bị cấn ở gần biên (cửa sổ chưa kịp đủ STALL_MIN_CMD)
                    c = self.carry[axis]
                    est, sd = None, 0
                    if a is not None:
                        est, sd = a["start"] + a["real"], np.sign(a["cmd"])
                    elif c and now - c[2] < 0.5:
                        est, sd = c[0], c[1]
                    if est is not None and sd != np.sign(d_cmd) and (pang - est) * sd >= STALL_GAP \
                            and now >= self.pause_until:
                        events.append((axis, int(sd), est))
                        self.acc[axis] = self.carry[axis] = None
                        continue
                    # Cửa sổ mới bắt đầu từ VỊ TRÍ THẬT ước lượng của cửa sổ trước (nếu vừa
                    # xét xong cùng chiều), không phải góc lệnh - servo có thể đã kẹt một nửa
                    start = c[0] if c and c[1] == np.sign(d_cmd) and now - c[2] < 0.5 else pang
                    a = self.acc[axis] = {"cmd": 0.0, "real": 0.0, "start": start, "moving": None}
                a["cmd"] += d_cmd
                a["real"] += d_real
                a["t"] = now
                if abs(d_real) >= 0.5 * abs(d_cmd):
                    a["moving"] = ang                 # khung hình này servo vẫn quay thật
                if abs(a["cmd"]) >= STALL_MIN_CMD:
                    if abs(a["real"]) < STALL_RATIO * abs(a["cmd"]) and now >= self.pause_until:
                        # Vị trí THẬT của servo = góc lúc bắt đầu xét + quãng ảnh thực sự dịch
                        # (không dùng góc ra lệnh: góc lệnh chạy trước servo thật)
                        stuck_at = a["start"] + a["real"]
                        events.append((axis, 1 if a["cmd"] > 0 else -1, stuck_at))
                        self.carry[axis] = None
                    elif abs(a["real"]) > 0.9 * abs(a["cmd"]):
                        self.carry[axis] = None        # servo bám lệnh tốt -> không cần nối
                    else:
                        self.carry[axis] = (a["start"] + a["real"], np.sign(a["cmd"]), now)
                    self.acc[axis] = None
        self.prev = (g, pan, tilt)
        return events

    def pause(self, now, seconds):
        """Tạm ngừng xét (vừa lùi servo ra khỏi chỗ cấn: góc lệnh và góc thật đang lệch nhau)."""
        self.pause_until = now + seconds
        self.acc = {a: None for a in ("pan", "tilt")}
        self.carry = {a: None for a in ("pan", "tilt")}


class PID:
    """Bộ điều khiển PID cho 1 trục servo (giống các dự án pan-tilt phổ biến)."""

    def __init__(self, kp, ki, kd, i_limit):
        self.kp, self.ki, self.kd, self.i_limit = kp, ki, kd, i_limit
        self.reset()

    def reset(self):
        self.i, self.d, self.prev_e, self.prev_t = 0.0, 0.0, None, None

    def update(self, e, now):
        if self.prev_t is not None and now > self.prev_t:
            dt = now - self.prev_t
            self.i = float(np.clip(self.i + e * dt, -self.i_limit, self.i_limit))
            # Đạo hàm có lọc (đo lửa bị nhiễu -> D thô sẽ làm servo rung)
            self.d += 0.3 * ((e - self.prev_e) / dt - self.d)
        self.prev_e, self.prev_t = e, now
        return self.kp * e + self.ki * self.i + self.kd * self.d


class AsyncDetector:
    """Chạy YOLO ở luồng nền trên khung hình mới nhất -> luồng chính không phải chờ AI."""

    def __init__(self, detector, cam, hw, fire_classes):
        self.detector, self.cam, self.hw, self.fire_classes = detector, cam, hw, fire_classes
        self.lock = threading.Lock()
        self.result = None       # (seq, target, (pan, tilt) lúc chụp)
        self.seq = 0
        self.fps = 0.0
        self.error = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        fid, t_prev = 0, time.time()
        try:
            while self.running:
                fid, frame = self.cam.read(fid)
                if frame is None:
                    continue
                snap = self.hw.get_angles()          # góc servo đúng lúc chụp khung hình
                target = pick_target(self.detector.detect(frame), self.fire_classes)
                now = time.time()
                dt, t_prev = now - t_prev, now
                with self.lock:
                    self.seq += 1
                    self.result = (self.seq, target, snap)
                    self.fps = 0.8 * self.fps + 0.2 * (1.0 / dt if dt > 0 else 0)
        except Exception as e:
            self.error = e

    def latest(self):
        with self.lock:
            return self.result

    def stop(self):
        self.running = False


def color_track(frame, box):
    """Tìm tâm vùng màu lửa (sáng, đỏ-cam-vàng hoặc lõi trắng) quanh khung lửa cũ.
    Trả về (x, y, khung_mới) hoặc None. Chỉ xử lý vùng nhỏ nên rất nhanh."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 8), max(y2 - y1, 8)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = bw * ROI_EXPAND / 2, bh * ROI_EXPAND / 2
    ax1, ay1 = max(0, int(cx - rx)), max(0, int(cy - ry))
    ax2, ay2 = min(W, int(cx + rx)), min(H, int(cy + ry))
    if ax2 - ax1 < 4 or ay2 - ay1 < 4:
        return None
    hsv = cv2.cvtColor(frame[ay1:ay2, ax1:ax2], cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((v >= FIRE_V_MIN) & ((h <= 35) | (h >= 165) | (s <= 60))).astype(np.uint8)
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] < FIRE_MIN_PIXELS:
        return None
    fx, fy = ax1 + m["m10"] / m["m00"], ay1 + m["m01"] / m["m00"]
    return fx, fy, (int(fx - bw / 2), int(fy - bh / 2), int(fx + bw / 2), int(fy + bh / 2))


class TeeLog:
    """Mọi thứ in ra màn hình đều được ghi thêm vào file txt, mỗi dòng kèm thời gian."""

    def __init__(self, stream, f, lock):
        self.stream, self.f, self.lock = stream, f, lock
        self.line_start = True

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass
        with self.lock:
            out = []
            for part in s.splitlines(keepends=True):
                if self.line_start:
                    t = time.time()
                    out.append(time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d} ")
                out.append(part)
                self.line_start = part.endswith("\n")
            self.f.write("".join(out))
        return len(s)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        self.f.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def setup_file_log(log_dir, args):
    """Bắt đầu ghi log ra logs/fire_tracker_<ngày>_<giờ>.txt. Trả về đường dẫn file."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, time.strftime("fire_tracker_%Y%m%d_%H%M%S.txt"))
    f = open(path, "w", encoding="utf-8", buffering=1)     # ghi từng dòng -> Ctrl+C không mất log
    f.write(f"===== FIRE TRACKER LOG - bắt đầu {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    f.write(f"Lệnh chạy : {' '.join(sys.argv)}\n")
    f.write(f"Cấu hình  : PAN_PIN={PAN_PIN} TILT_PIN={TILT_PIN} PAN_DIR={PAN_DIR} "
            f"TILT_DIR={TILT_DIR} servo={args.servo} imgsz={args.imgsz} conf={args.conf}\n")
    f.write("=" * 60 + "\n")
    lock = threading.Lock()
    sys.stdout = TeeLog(sys.stdout, f, lock)
    sys.stderr = TeeLog(sys.stderr, f, lock)               # ghi cả lỗi/cảnh báo
    return path


def remember_hotspot(hotspots, pan, tilt, now):
    """Bộ nhớ các chỗ từng có lửa: gộp điểm gần nhau (< 10°), giữ 5 chỗ mới nhất."""
    for hs in hotspots:
        if abs(hs[0] - pan) < 10:
            hs[0] += 0.3 * (pan - hs[0])
            hs[1] += 0.3 * (tilt - hs[1])
            hs[2] = now
            break
    else:
        hotspots.append([pan, tilt, now])
    hotspots[:] = sorted((hs for hs in hotspots if now - hs[2] < HOTSPOT_TTL),
                         key=lambda hs: -hs[2])[:5]


# ===================== CHƯƠNG TRÌNH CHÍNH =====================
def main():
    global PAN_DIR, TILT_DIR          # có thể tự đảo khi phát hiện servo quay ngược (AUTO_DIR)
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
    ap.add_argument("--servo", default="auto", choices=["auto", "hwpwm", "pigpio", "gpiozero"],
                    help="Cách tạo xung servo. hwpwm = PWM phần cứng (mượt nhất, servo ở GPIO 12/13)")
    ap.add_argument("--log-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
                    help="Thư mục lưu file log .txt (mặc định: logs/ cạnh script)")
    ap.add_argument("--no-log-file", action="store_true", help="Không ghi log ra file")
    ap.add_argument("--reset-limits", action="store_true",
                    help="Xóa giới hạn góc đã tự học khi servo bị cấn (servo_limits.json)")
    ap.add_argument("--no-stall", action="store_true", help="Tắt phát hiện servo bị cấn")
    ap.add_argument("--servo-trace", action="store_true",
                    help="Ghi THÊM từng lệnh gửi servo ra file servo_trace_*.txt (để tìm nguyên nhân giật)")
    args = ap.parse_args()

    if not args.no_log_file:
        log_path = setup_file_log(args.log_dir, args)
        print(f"[LOG] Đang ghi log vào: {log_path}")
    trace = None
    if args.servo_trace:
        os.makedirs(args.log_dir, exist_ok=True)
        tpath = os.path.join(args.log_dir, time.strftime("servo_trace_%Y%m%d_%H%M%S.txt"))
        trace = open(tpath, "w", encoding="utf-8", buffering=1)
        trace.write("giay\ttruc\tgoc\tsu_kien\tche_do\n")
        print(f"[LOG] Đang ghi từng lệnh servo vào: {tpath}")

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

    if args.reset_limits and os.path.exists(LIMITS_FILE):
        os.remove(LIMITS_FILE)
        print(f"[CẤN] Đã xóa giới hạn đã học ({os.path.basename(LIMITS_FILE)}) -> dùng lại giới hạn gốc")
    hw = Hardware(use_pump=args.pump, log=not args.no_servo_log, backend=args.servo, trace=trace)
    stall = StallDetector() if STALL_DETECT and not args.no_stall else None
    try:
        cam = Camera(args.cam, FRAME_W, FRAME_H)
    except RuntimeError as e:
        hw.close()
        raise SystemExit(f"[LỖI] {e}")

    ai = AsyncDetector(detector, cam, hw, fire_classes)

    lock_count = 0
    last_seen = ai_seen = 0.0
    pump_start = cooldown_until = 0.0
    fps, t_prev, last_log = 0.0, time.time(), 0.0
    frame_id, last_seq = 0, 0
    box, conf, name = None, 0.0, "fire"  # khung lửa đang bám (tọa độ khung hình hiện tại)
    meas = None                          # tâm lửa đo được ở khung hình này
    need_init = False
    world_pan = world_tilt = 0.0         # hướng của ngọn lửa (độ, đã lọc mượt)
    active_x = active_y = False          # trục đang trong pha "đang xoay" (hysteresis)
    pid_x = PID(PID_KP, PID_KI, PID_KD, PID_I_LIMIT)
    pid_y = PID(PID_KP, PID_KI, PID_KD, PID_I_LIMIT)
    ex_f = ey_f = 0.0                    # độ lệch tâm đã lọc mượt
    prev_present = prev_locked = False
    # Bộ nhớ lửa
    hotspots = []                        # [[pan, tilt, thời điểm]] các chỗ từng có lửa
    vel_pan = vel_tilt = 0.0             # vận tốc hướng lửa (độ/giây)
    prev_world = None
    phase, phase_t, arrive_t, mem = "radar", 0.0, None, None
    # Tự phát hiện chiều ngược: tổng (Δlệch·Δgóc), tổng (Δgóc²), tổng |Δgóc|
    cal = {"pan": [0.0, 0.0, 0.0], "tilt": [0.0, 0.0, 0.0]}
    dir_ok = {"pan": False, "tilt": False}
    prev_m = ai_raw = None

    print("[RUN] Bắt đầu theo dõi lửa. Nhấn 'q' (hoặc Ctrl+C) để thoát.")
    try:
        while True:
            if ai.error:
                raise RuntimeError(f"Luồng AI bị lỗi: {ai.error}")
            frame_id, frame = cam.read(frame_id)
            if frame is None:
                print("[CAM] Không nhận được khung hình, đang chờ...")
                continue
            snap_pan, snap_tilt = hw.get_angles()   # góc servo lúc có khung hình này
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            now = time.time()

            # --- 0. Servo có bị cấn không? (ra lệnh xoay mà khung cảnh không dịch) ---
            if stall:
                for axis, side, at in stall.update(frame, snap_pan, snap_tilt, now):
                    hw.report_stall(axis, side, at)
                    stall.pause(now, 1.0)

            # --- 1. Có kết quả AI mới? ---
            res = ai.latest()
            ai_new = res is not None and res[0] != last_seq
            ai_hit = False
            if ai_new:
                last_seq, ai_target, (a_pan, a_tilt) = res
                if ai_target:
                    ai_hit, ai_seen = True, now
                    x1, y1, x2, y2, conf, name = ai_target
                    # Khung của AI tính trên khung hình CŨ -> dịch theo góc servo đã xoay từ đó
                    dx = -PAN_DIR * (snap_pan - a_pan) / CAM_HFOV * w
                    dy = -TILT_DIR * (snap_tilt - a_tilt) / CAM_VFOV * h
                    # Hướng lửa lúc AI chụp (đúng cả khi radar đã lướt qua) -> ghi vào bộ nhớ
                    hp = a_pan + PAN_DIR * ((x1 + x2) / 2 - cx) / w * CAM_HFOV
                    ht = a_tilt + TILT_DIR * ((y1 + y2) / 2 - cy) / h * CAM_VFOV
                    # Mẫu hiệu chỉnh chiều: vị trí GỐC trong ảnh AI + góc servo lúc AI chụp
                    ai_raw = (((x1 + x2) / 2 - cx) / w, ((y1 + y2) / 2 - cy) / h, a_pan, a_tilt)
                    remember_hotspot(hotspots, hp, ht, now)
                    hw.hotspots = [hs[0] for hs in hotspots]
                    if box is None:
                        print(f"[SERVO] AI phát hiện lửa ({name} {conf:.2f}) tại hướng "
                              f"PAN {hp:.1f}° TILT {ht:.1f}° -> quay thẳng về đó")
                        hw.track(hp, ht)       # radar có lướt quá cũng quay lại đúng chỗ
                        need_init, active_x, active_y = True, False, False
                        prev_m, prev_world = None, None
                    box = (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy))
            if box is not None and now - ai_seen > AI_CONFIRM_TIMEOUT:
                box = None                                  # AI lâu rồi không xác nhận -> bỏ

            # --- 2. Đo tâm lửa ở khung hình hiện tại ---
            meas = samp = None
            if box is not None:
                if FAST_TRACK:
                    r = color_track(frame, box)
                    if r:
                        fx, fy, box = r
                        meas = (fx, fy)
                        samp = ((fx - cx) / w, (fy - cy) / h, snap_pan, snap_tilt)
                if meas is None and ai_hit:
                    meas = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    samp = ai_raw
            present = box is not None
            locked = False

            # --- 3. Điều khiển servo ---
            if meas:
                last_seen = now
                ex, ey = (meas[0] - cx) / w, (meas[1] - cy) / h

                # --- Tự phát hiện servo quay ngược ---
                # Lửa đứng yên: servo xoay Δgóc thì lửa trong ảnh phải dịch
                # Δlệch ≈ -DIR·Δgóc/FOV. Nếu thực tế dịch cùng chiều -> DIR bị sai.
                # (Dùng vị trí đo THẬT trong ảnh + góc lúc chụp ảnh đó, không dùng khung đã dịch bù)
                if AUTO_DIR and prev_m is not None:
                    for key, e, pe, a, pa, fov in (
                            ("pan", samp[0], prev_m[0], samp[2], prev_m[2], CAM_HFOV),
                            ("tilt", samp[1], prev_m[1], samp[3], prev_m[3], CAM_VFOV)):
                        da = a - pa
                        if not 0.2 < abs(da) < 15:
                            continue
                        c = cal[key]
                        c[0] += (e - pe) * fov * da
                        c[1] += da * da
                        c[2] += abs(da)
                        if c[2] < AUTO_DIR_MIN_MOVE:
                            continue
                        k = c[0] / c[1]                     # ≈ -DIR nếu đúng chiều
                        d = PAN_DIR if key == "pan" else TILT_DIR
                        if k * d > 0.3:
                            d = -d
                            if key == "pan":
                                PAN_DIR = d
                            else:
                                TILT_DIR = d
                            print(f"[CALIB] Servo {key.upper()} đang quay NGƯỢC hướng lửa -> tự đảo "
                                  f"chiều. Sửa vĩnh viễn trong file: {key.upper()}_DIR = {d}")
                            need_init, active_x, active_y = True, False, False
                            prev_world = None
                        elif k * d < -0.3 and not dir_ok[key]:
                            dir_ok[key] = True
                            print(f"[CALIB] Chiều servo {key.upper()} đúng ({key.upper()}_DIR = {d})")
                        c[:] = [0.0, 0.0, 0.0]
                prev_m = samp

                # Hướng thật của ngọn lửa = góc servo + độ lệch đổi ra độ
                wp = snap_pan + PAN_DIR * ex * CAM_HFOV
                wt = snap_tilt + TILT_DIR * ey * CAM_VFOV
                if need_init:                            # lửa mới: khởi tạo bộ lọc + PID
                    world_pan, world_tilt, need_init = wp, wt, False
                    vel_pan = vel_tilt = 0.0
                    ex_f, ey_f = ex, ey
                    pid_x.reset()
                    pid_y.reset()
                else:
                    world_pan += WORLD_SMOOTH * (wp - world_pan)
                    world_tilt += WORLD_SMOOTH * (wt - world_tilt)
                # Vận tốc hướng lửa (để đoán lửa đi đâu khi mất dấu)
                if prev_world is not None and now > prev_world[2]:
                    dtw = now - prev_world[2]
                    vel_pan += 0.2 * ((world_pan - prev_world[0]) / dtw - vel_pan)
                    vel_tilt += 0.2 * ((world_tilt - prev_world[1]) / dtw - vel_tilt)
                prev_world = (world_pan, world_tilt, now)

                # --- PID + vùng chết (deadzone có hysteresis) cho từng trục ---
                ex_f += ERR_SMOOTH * (ex - ex_f)
                ey_f += ERR_SMOOTH * (ey - ey_f)
                err_x, err_y = ex_f, ey_f
                if abs(err_x) > ZONE_OUTER:
                    active_x = True
                elif abs(err_x) < ZONE_INNER:
                    active_x = False
                if abs(err_y) > ZONE_OUTER:
                    active_y = True
                elif abs(err_y) < ZONE_INNER:
                    active_y = False
                new_pan = new_tilt = None
                if active_x:
                    new_pan = snap_pan + PAN_DIR * pid_x.update(err_x, now) * CAM_HFOV
                else:
                    pid_x.reset()                        # vào vùng tâm -> xóa tích phân
                if active_y:
                    new_tilt = snap_tilt + TILT_DIR * pid_y.update(err_y, now) * CAM_VFOV
                else:
                    pid_y.reset()
                if new_pan is not None or new_tilt is not None:
                    hw.track(new_pan, new_tilt)

                if abs(err_x) < LOCK_TOLERANCE and abs(err_y) < LOCK_TOLERANCE:
                    lock_count = min(lock_count + 1, LOCK_FRAMES)
                else:
                    lock_count = max(lock_count - 1, 0)
            if present:
                locked = lock_count >= LOCK_FRAMES
                phase = "track"
            else:
                lock_count = 0
                prev_m = None
                if prev_present:
                    # --- Bộ nhớ: quay lại chỗ lửa cuối cùng (+ dự đoán theo vận tốc) ---
                    if not need_init:
                        vp = float(np.clip(vel_pan, -30, 30))
                        vt = float(np.clip(vel_tilt, -30, 30))
                        mem = (world_pan + vp * PREDICT_TIME, world_tilt + vt * PREDICT_TIME)
                    elif hotspots:                 # AI mới thấy thoáng qua -> dùng bộ nhớ AI
                        mem = (hotspots[0][0], hotspots[0][1])
                    else:
                        mem = None
                    if mem:
                        print(f"[NHỚ] Mất dấu lửa -> quay lại chỗ lửa cuối cùng "
                              f"PAN {mem[0]:.1f}° TILT {mem[1]:.1f}° (chờ {RECALL_DWELL:.1f}s)")
                        hw.track(mem[0], mem[1], reason="QUAY LẠI")
                        phase, phase_t, arrive_t = "recall", now, None
                    else:
                        phase = "radar"
                elif phase == "recall":
                    tp, tt = hw.target["pan"], hw.target["tilt"]
                    if arrive_t is None and abs(hw.pan_angle - tp) < 2 and abs(hw.tilt_angle - tt) < 2:
                        arrive_t = now
                    if (arrive_t and now - arrive_t > RECALL_DWELL) or now - phase_t > 4.0:
                        hw.start_search(tp, tt, LOCAL_RANGE, LOCAL_SPEED, LOCAL_TURNS)
                        phase = "search"
                elif phase == "search":
                    if hw.search_done:
                        print("[NHỚ] Quét hẹp không thấy lửa -> quét RADAR toàn bộ")
                        phase = "radar"
                else:
                    phase = "radar"
                if phase == "radar":
                    hw.start_radar()

            if locked != prev_locked:
                print(f"[SERVO] {'ĐÃ KHÓA MỤC TIÊU (lửa ở giữa khung hình)' if locked else 'Mất khóa mục tiêu'}"
                      f" | PAN {hw.pan_angle:.1f}° TILT {hw.tilt_angle:.1f}°")
            prev_present, prev_locked = present, locked

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
                     "TRACKING FIRE..." if present else
                     "RECALL: BACK TO LAST FIRE..." if phase == "recall" else
                     "LOCAL SEARCH..." if phase == "search" else
                     "RADAR SCANNING..." if hw.mode == "radar" else "SEARCHING...")

            if args.no_display:
                if now - last_log >= 1.0:
                    last_log = now
                    extra = (f" | {name} {conf:.2f} hướng lửa ({world_pan:.0f}°,{world_tilt:.0f}°)"
                             if present else "")
                    print(f"[{state}] CAM {fps:.0f} FPS | AI {ai.fps:.1f} FPS | "
                          f"Pan {hw.pan_angle:.0f} | Tilt {hw.tilt_angle:.0f}{extra}")
                continue

            # --- Vẽ & hiển thị (vẽ lên bản sao: luồng AI có thể đang đọc frame gốc) ---
            view = frame.copy()
            cv2.drawMarker(view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            # Vùng vàng: lửa trong vùng này servo không xoay | xanh dương: vùng khóa mục tiêu
            for frac, col in ((ZONE_OUTER, (0, 255, 255)), (LOCK_TOLERANCE, (255, 128, 0))):
                zx, zy = int(frac * w), int(frac * h)
                cv2.rectangle(view, (cx - zx, cy - zy), (cx + zx, cy + zy), col, 1)
            if present:
                color = (0, 0, 255) if locked else (0, 255, 0)
                cv2.rectangle(view, box[:2], box[2:], color, 2)
                cv2.putText(view, f"{name} {conf:.2f}", (box[0], max(box[1] - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                if meas:
                    cv2.circle(view, (int(meas[0]), int(meas[1])), 4, color, -1)
                    cv2.line(view, (cx, cy), (int(meas[0]), int(meas[1])), color, 1)
            color = (0, 0, 255) if locked else (0, 255, 0) if present else (200, 200, 200)
            cv2.putText(view, state, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            info = (f"CAM {fps:.0f} AI {ai.fps:.1f} | Pan {hw.pan_angle:.0f} "
                    f"Tilt {hw.tilt_angle:.0f}{' SIM' if hw.sim else ''}")
            cv2.putText(view, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[RUN] Dừng bởi người dùng.")
    finally:
        ai.stop()
        hw.close()
        cam.release()
        cv2.destroyAllWindows()
        print("[RUN] Đã dừng và giải phóng tài nguyên.")
        if trace:
            trace.close()


if __name__ == "__main__":
    main()
