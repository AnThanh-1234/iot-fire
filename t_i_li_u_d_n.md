# 🚒 Hệ Thống Robot Chữa Cháy Tự Động (Raspberry Pi & YOLOv5)

Dự án này sử dụng Raspberry Pi kết hợp với mô hình trí tuệ nhân tạo YOLOv5 để tự động phát hiện ngọn lửa qua Webcam. Khi phát hiện lửa, hệ thống sẽ điều khiển cụm động cơ Pan/Tilt (xoay ngang/gập lên xuống) để khóa mục tiêu và tự động kích hoạt máy bơm nước 365 để dập tắt lửa.

---

## 📡 Thông tin kết nối Raspberry Pi
*   **Địa chỉ IP (LAN):** `192.168.11.22`
*   **Kết nối SSH:** Mở terminal trên máy tính và gõ `ssh pi@192.168.11.22` (thay `pi` bằng username của bạn).
*   **Kết nối VNC:** Mở VNC Viewer, nhập IP `192.168.11.22`. Giao diện màn hình Raspberry Pi (kèm cửa sổ camera nhận diện) sẽ hiển thị tại đây.

---

## 🛠️ Danh sách Linh kiện (Hardware)
1.  **Bộ điều khiển trung tâm:** Raspberry Pi (Khuyên dùng Pi 4).
2.  **Camera:** Webcam USB (hoặc Pi Camera).
3.  **Động cơ nhắm mục tiêu:** 2 x Động cơ Servo (Cho trục Pan và Tilt).
4.  **Máy bơm:** Máy Bơm Nước 365 (3-12V DC).
5.  **Mạch kích dòng:** Module Relay 1 kênh 5V.
6.  **Nguồn điện:** Pin/Nguồn ngoài 5V-12V (Dùng để cấp điện cho Servo và Máy bơm, **không dùng chung nguồn tải với mạch Pi**).

---

## 🔌 Sơ đồ chân tín hiệu (Pinout)
*   **Servo Trục Ngang (Pan):** Kết nối chân tín hiệu vào **GPIO 23** trên Raspberry Pi.
*   **Servo Trục Dọc (Tilt):** Kết nối chân tín hiệu vào **GPIO 18** trên Raspberry Pi.
*   **Điều khiển Bơm (Relay):** Kết nối chân IN của Relay vào **GPIO 14** trên Raspberry Pi.
*   *Lưu ý quan trọng:* Phải nối chung chân GND của nguồn ngoài (cấp cho Servo/Bơm) với chân GND của Raspberry Pi.

---

## 💻 Cài đặt & Sử dụng (Software)

**Bước 1: Cài đặt thư viện**
Mở terminal trên Raspberry Pi (qua SSH hoặc VNC) và chạy lệnh sau:
```bash
pip install opencv-python gpiozero numpy torch torchvision ultralytics
```

**Bước 2: Cấp quyền phần cứng (Nếu cần)**
Đảm bảo bạn đã bật giao tiếp I2C/GPIO và Camera trong cài đặt của Pi:
```bash
sudo raspi-config
```
*(Vào Interface Options -> Enable Camera / I2C)*

**Bước 3: Chạy chương trình**
Khi đã cấp nguồn và cắm đủ dây, chạy file script Python:
```bash
python fire_tracker.py
```

**Bước 4: Quan sát**
*   Mở VNC Viewer lên, bạn sẽ thấy cửa sổ "Raspberry Pi - Fire Tracker".
*   Khi có ngọn lửa xuất hiện, khung xanh/đỏ sẽ bao quanh ngọn lửa, Servo tự động xoay để đưa lửa vào tâm khung hình.
*   Khi chữ `TARGET LOCKED! PUMPING...` hiện lên, Máy bơm 365 sẽ được kích hoạt!