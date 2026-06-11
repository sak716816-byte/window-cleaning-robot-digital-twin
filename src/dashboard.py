"""
Based on Data-driven and Physical Evolution High-Altitude Curtain Wall Robot Digital Twin Control Console.
Interactive Dashboard Module (dashboard.py)

Implements:
1. Multi-threaded asynchronous simulation (QThread).
2. PySide6 GUI with dark industrial sci-fi style.
3. Real-time pyqtgraph scrolling charts for Pressure vs Redline and Wind Load.
4. Custom 2D Digital Twin view with position, orientation, and cup health.
5. Simulated AI Cleanliness Camera View with real-time cleanliness percentage.
6. Interactive sliders to inject wind gusts, joint gaps, and climb controls.
"""

import os
import sys
import time
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QMutex, QPointF
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QSlider, QPushButton,
                             QGroupBox, QProgressBar, QFrame, QSizePolicy)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QImage

import pyqtgraph as pg

# Import Physics Engine & Control System
from physics_engine import PhysicsEngine, RobotParameters, PhysicsState, G_ACCEL, P_ATM
from control_system import SafetyControlHub, ControlState, ALARM_NONE

# Thread-safe environment variables shared between UI and Simulation Thread
class SharedEnvironment:
    def __init__(self):
        self.mutex = QMutex()
        self.wind_speed = 0.0
        self.wall_roughness = 1.0e-6
        self.spray_flow = 0.0
        self.cleaning_mode = False
        self.target_climb_speed = 0.0
        self.cross_gap_trigger = False
        self.reset_trigger = False
        
    def get_dict(self) -> dict:
        self.mutex.lock()
        data = {
            "wind_speed": self.wind_speed,
            "wall_roughness": self.wall_roughness,
            "spray_flow": self.spray_flow,
            "cleaning_mode": self.cleaning_mode,
            "target_climb_speed": self.target_climb_speed,
            "cross_gap_trigger": self.cross_gap_trigger,
            "reset_trigger": self.reset_trigger
        }
        # Reset one-shot triggers
        self.cross_gap_trigger = False
        self.reset_trigger = False
        self.mutex.unlock()
        return data

    def set_wind_speed(self, val: float):
        self.mutex.lock()
        self.wind_speed = val
        self.mutex.unlock()

    def set_wall_roughness(self, val: float):
        self.mutex.lock()
        self.wall_roughness = val
        self.mutex.unlock()

    def set_cleaning_mode(self, val: bool):
        self.mutex.lock()
        self.cleaning_mode = val
        if val:
            self.spray_flow = 2.0  # Turn on spray automatically
        else:
            self.spray_flow = 0.0
        self.mutex.unlock()

    def set_climb_speed(self, val: float):
        self.mutex.lock()
        self.target_climb_speed = val
        self.mutex.unlock()

    def trigger_gap(self):
        self.mutex.lock()
        self.cross_gap_trigger = True
        self.mutex.unlock()

    def trigger_reset(self):
        self.mutex.lock()
        self.reset_trigger = True
        self.mutex.unlock()

class SimulationThread(QThread):
    """
    Asynchronous worker thread running the Physics Engine and Control Hub.
    """
    state_signal = Signal(dict)

    def __init__(self, shared_env: SharedEnvironment):
        super().__init__()
        self.shared = shared_env
        self.running = True
        
        # Configure CUBEBOX Parameters
        self.params = RobotParameters(
            mass=9.0,
            num_cups=4,
            cup_diameter=0.05,
            cup_volume=2.0e-5,
            cog_height=0.08,
            length=0.35,
            width=0.30,
            preload_force=40.0
        )
        self.engine = PhysicsEngine(self.params)
        self.hub = SafetyControlHub(
            num_cups=self.params.num_cups,
            max_fan_rpm=self.params.max_fan_rpm,
            max_wheel_rpm=self.params.max_wheel_rpm
        )
        
        # Static Wall Joints/Gaps (representing glass panel boundaries)
        # Vertical joint at x = 0.5m, Horizontal joint at y = 1.0m
        # Gap width = 8mm, depth = 10mm
        self.wall_gaps = [(0.5, 999.0, 0.008, 0.010), (999.0, 1.0, 0.008, 0.010)]
        self.active_gaps_in_simulation = []

    def run(self):
        dt = 0.01
        sim_accum_t = 0.0
        
        while self.running:
            start_t = time.perf_counter()
            
            # Read shared parameters
            env_inputs = self.shared.get_dict()
            
            # Check Reset
            if env_inputs["reset_trigger"]:
                self.engine = PhysicsEngine(self.params)
                self.hub = SafetyControlHub(
                    num_cups=self.params.num_cups,
                    max_fan_rpm=self.params.max_fan_rpm,
                    max_wheel_rpm=self.params.max_wheel_rpm
                )
                self.active_gaps_in_simulation = []
            
            # Handle one-shot gap injection
            if env_inputs["cross_gap_trigger"]:
                # Inject a joint gap directly under the robot's current y position
                current_y = self.engine.state.pos_y
                # Horizontal gap crossing at current_y + 0.05m
                self.active_gaps_in_simulation = [(999.0, current_y + 0.05, 0.008, 0.010)]
                
            # Simulate sensor noise: Add Gaussian noise to raw physical states
            raw_state = self.engine.state
            noise_p = np.random.normal(0, 1200.0, self.params.num_cups) # 1.2 kPa noise
            noise_imu = np.random.normal(0, 0.15, 2)
            
            noisy_pressures = raw_state.cup_pressures + noise_p
            # Noisy IMU acceleration: [a_x, a_y]
            noisy_imu = np.array([raw_state.acc_x, raw_state.acc_y]) + noise_imu
            
            # Package telemetry
            telemetry = {
                "pressures": noisy_pressures,
                "imu_accel": noisy_imu,
                "is_attached": raw_state.is_attached,
                "slip_safety_factor": raw_state.slip_safety_factor,
                "overturn_safety_factor": raw_state.overturn_safety_factor
            }
            
            # Run Safety Control Hub Step
            user_cmds = {
                "cleaning_mode": env_inputs["cleaning_mode"],
                "target_climb_speed": env_inputs["target_climb_speed"],
                "spray_flow": env_inputs["spray_flow"]
            }
            
            control_actions = self.hub.step(telemetry, user_cmds, dt)
            
            # Feed control actions back to Physics Engine
            physics_env = {
                "dt": dt,
                "wind_speed": env_inputs["wind_speed"],
                "wind_angle": 0.0, # Wind directly hitting body
                "wall_inclination": np.pi / 2.0, # Vertical wall
                "wall_roughness": env_inputs["wall_roughness"],
                "wall_gaps": self.wall_gaps + self.active_gaps_in_simulation,
                "spray_flow": control_actions["spray_flow"],
                "cleaning_active": control_actions["cleaning_active"],
                "target_fan_rpm": control_actions["target_fan_rpm"],
                "target_wheel_rpm_l": control_actions["target_wheel_rpm_l"],
                "target_wheel_rpm_r": control_actions["target_wheel_rpm_r"],
                "mu": 0.6 # PDMS on glass friction
            }
            
            updated_physics = self.engine.simulate_tick(physics_env)
            
            # Emit telemetry state dictionary to UI thread
            state_dict = {
                "pos_x": updated_physics.pos_x,
                "pos_y": updated_physics.pos_y,
                "vel_y": updated_physics.vel_y,
                "pressures": updated_physics.cup_pressures.tolist(),
                "filtered_pressures": [self.hub.filters_p[i].x for i in range(self.params.num_cups)],
                "fan_rpm": updated_physics.fan_rpm,
                "slip_sf": updated_physics.slip_safety_factor,
                "overturn_sf": updated_physics.overturn_safety_factor,
                "is_attached": updated_physics.is_attached,
                "is_slipping": updated_physics.is_slipping,
                "is_overturned": updated_physics.is_overturned,
                "control_state": control_actions["control_state"],
                "alarm_code": control_actions["alarm_code"],
                "wind_speed": env_inputs["wind_speed"],
                "cleaning_active": control_actions["cleaning_active"],
                "eso_disturbance": float(self.hub.ladrc.z2)
            }
            self.state_signal.emit(state_dict)
            
            # Throttle loop to maintain simulation rate
            elapsed = time.perf_counter() - start_t
            sleep_t = max(0.001, dt - elapsed)
            time.sleep(sleep_t)

    def stop(self):
        self.running = False
        self.wait()

class CameraViewWidget(QWidget):
    """
    Custom widget drawing a simulated AI cleanliness assessment feed.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 180)
        self.cleanliness_pct = 0.0
        self.sweep_y = 0.0
        self.cleaning_active = False
        
        # Timer to animate sweep and dust
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)
        
    def update_animation(self):
        if self.cleaning_active:
            # Cleanliness goes up
            self.cleanliness_pct += 0.15
            if self.cleanliness_pct > 99.9:
                self.cleanliness_pct = 99.9
            
            # Wiping sweep line moves down
            self.sweep_y += 1.5
            if self.sweep_y > self.height():
                self.sweep_y = 0.0
        else:
            # Decay slowly or stay
            pass
        self.update()

    def reset_cleanliness(self):
        self.cleanliness_pct = 0.0
        self.sweep_y = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Draw dark glass frame background
        painter.setBrush(QBrush(QColor(24, 28, 38)))
        painter.setPen(QPen(QColor(60, 65, 80), 2))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)
        
        # Draw glass reflection effect (gradual cyan gradient)
        # We will split screen into cleaned (top/right) and dirty (bottom/left)
        # Let's paint the dirty area
        painter.setBrush(QBrush(QColor(40, 38, 30))) # dusty brownish look
        painter.setPen(Qt.NoPen)
        painter.drawRect(2, int(self.sweep_y), w - 4, h - int(self.sweep_y) - 2)
        
        # Draw dirty speckles in the dirty area
        painter.setPen(QPen(QColor(180, 160, 120), 1))
        np.random.seed(42) # fixed seed to keep speckles in place
        for _ in range(50):
            rx = int(np.random.rand() * (w - 10)) + 5
            ry = int(np.random.rand() * (h - 10)) + 5
            if ry > self.sweep_y:
                painter.drawPoint(rx, ry)
                
        # Draw clean area (top)
        painter.setBrush(QBrush(QColor(10, 60, 80, 100))) # transparent clean cyan
        painter.setPen(Qt.NoPen)
        painter.drawRect(2, 2, w - 4, int(self.sweep_y) - 2)
        
        # Draw grid lines of the glass pane
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        for x in range(0, w, 40):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, 40):
            painter.drawLine(0, y, w, y)
            
        # Draw laser sweep line (neon green)
        if self.cleaning_active:
            painter.setPen(QPen(QColor(0, 255, 128), 2))
            painter.drawLine(2, int(self.sweep_y), w - 2, int(self.sweep_y))
            # sweep glow
            painter.setPen(QPen(QColor(0, 255, 128, 40), 6))
            painter.drawLine(2, int(self.sweep_y), w - 2, int(self.sweep_y))

        # Camera HUD text overlay
        painter.setPen(QPen(QColor(0, 240, 255), 1))
        painter.setFont(QFont("Monospace", 8))
        painter.drawText(10, 20, "AI CAM: CLEAN_EVAL_FEED")
        painter.drawText(10, 35, f"RESOLUTION: 1920x1080 @ 30FPS")
        
        # Draw bounding boxes (simulated AI target detection)
        # Draw target window edge
        painter.setPen(QPen(QColor(255, 190, 0, 150), 1, Qt.DashLine))
        painter.drawRect(30, 45, w - 60, h - 70)
        painter.drawText(35, 60, "TARGET AREA: GLASS_PANE_01")
        
        # Cleanliness Display
        painter.setPen(QPen(QColor(0, 255, 128), 1))
        painter.setFont(QFont("微软雅黑", 14, QFont.Bold))
        painter.drawText(w - 150, h - 20, f"CLEAN: {self.cleanliness_pct:.1f}%")

class RobotPoseWidget(QWidget):
    """
    Custom widget displaying the 2D digital twin pose, joints, and cup health.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 250)
        self.robot_pos = QPointF(125, 180)
        self.cup_pressures = [P_ATM] * 4
        self.is_attached = True
        self.control_state = "INIT"
        self.slip_sf = 99.9
        
    def update_pose(self, pos_x: float, pos_y: float, pressures: list, 
                    is_attached: bool, state: str, slip_sf: float):
        # Scale coordinates for widget display
        # Let's map wall coordinates (y = 0..3m) to widget height (bottom to top)
        self.robot_pos.setY(self.height() - 50 - (pos_y * 120) % (self.height() - 100))
        # Keep x centered with slight offset
        self.robot_pos.setX(self.width() / 2.0 + pos_x * 150)
        
        self.cup_pressures = pressures
        self.is_attached = is_attached
        self.control_state = state
        self.slip_sf = slip_sf
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Dark grid background
        painter.setBrush(QBrush(QColor(20, 22, 28)))
        painter.setPen(QPen(QColor(45, 50, 65), 1.5))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)
        
        # Draw glass joint gaps (cross gap line)
        painter.setPen(QPen(QColor(255, 100, 0, 100), 2, Qt.DashLine))
        # Horizontal joint
        painter.drawLine(0, h - 140, w, h - 140)
        painter.setFont(QFont("微软雅黑", 7))
        painter.setPen(QPen(QColor(255, 100, 0, 150), 1))
        painter.drawText(10, h - 145, "GLASS PANEL JOINT GAP (8mm)")
        
        # Draw Robot Body
        rx, ry = self.robot_pos.x(), self.robot_pos.y()
        rw, rh = 70, 90
        
        painter.translate(rx, ry)
        
        # Base body representation
        if not self.is_attached:
            body_color = QColor(255, 59, 48, 80) # Red translucent for fall
        elif self.control_state == "EMERGENCY_LOCK":
            body_color = QColor(255, 150, 0, 150) # Orange lock
        else:
            body_color = QColor(0, 150, 255, 100) # Normal Cyan translucent
            
        painter.setBrush(QBrush(body_color))
        painter.setPen(QPen(QColor(0, 240, 255), 2))
        painter.drawRoundedRect(int(-rw/2), int(-rh/2), rw, rh, 5, 5)
        
        # Draw Center of Gravity circle
        painter.setBrush(QBrush(QColor(0, 240, 255)))
        painter.drawEllipse(QPointF(0, 0), 4, 4)
        
        # Draw Wheels/Tracks (Left/Right rectangles)
        painter.setBrush(QBrush(QColor(50, 50, 60)))
        painter.setPen(QPen(QColor(100, 100, 110), 1))
        painter.drawRect(int(-rw/2 - 8), -35, 8, 70)  # Left Track
        painter.drawRect(int(rw/2), -35, 8, 70)   # Right Track
        
        # Draw 4 Suction Cups
        # Layout: Front-Left, Front-Right, Rear-Left, Rear-Right
        cup_offsets = [
            (-rw/2 + 10, -rh/2 + 15), # FL
            ( rw/2 - 10, -rh/2 + 15), # FR
            (-rw/2 + 10,  rh/2 - 15), # RL
            ( rw/2 - 10,  rh/2 - 15)  # RR
        ]
        
        for idx, (ox, oy) in enumerate(cup_offsets):
            p_abs = self.cup_pressures[idx]
            p_gauge_kpa = (P_ATM - p_abs) / 1000.0
            
            # Suction Cup Color coding based on pressure
            if p_gauge_kpa > 50.0:
                cup_color = QColor(0, 255, 128) # Strong vacuum - Green
            elif p_gauge_kpa > 30.0:
                cup_color = QColor(255, 190, 0) # Normal/Low - Yellow
            else:
                cup_color = QColor(255, 59, 48)  # Critical loss - Red
                
            painter.setBrush(QBrush(cup_color))
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
            painter.drawEllipse(QPointF(ox, oy), 10, 10)
            
            # Draw tiny inner circle
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(ox, oy), 5, 5)
            
        painter.resetTransform()
        
        # GUI Overlay Info
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setFont(QFont("微软雅黑", 8))
        painter.drawText(10, 20, f"POS_Y: {self.robot_pos.y():.1f} px")
        painter.drawText(10, 35, f"SLIP_SF: {self.slip_sf:.2f}")

class MainWindow(QMainWindow):
    def __init__(self, shared_env: SharedEnvironment):
        super().__init__()
        self.shared = shared_env
        self.setWindowTitle("基于数据驱动的高空幕墙机器人数字孪生控制台")
        self.resize(1100, 750)
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1b22; }
            QLabel { color: #d2d2d9; font-family: '微软雅黑'; }
            QPushButton { 
                background-color: #2b2c37; 
                border: 1px solid #4a4b5d; 
                border-radius: 4px; 
                color: #e2e2e7; 
                padding: 6px 12px;
                font-family: '微软雅黑';
            }
            QPushButton:hover { background-color: #3f4052; border-color: #00f0ff; }
            QPushButton:pressed { background-color: #1d1e26; }
            QGroupBox {
                border: 1px solid #3f3f50;
                border-radius: 6px;
                margin-top: 1.2em;
                color: #00f0ff;
                font-weight: bold;
                font-family: '微软雅黑';
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3f3f50;
                height: 6px;
                background: #111218;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00f0ff;
                border: 1px solid #00f0ff;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QProgressBar {
                border: 1px solid #3f3f50;
                border-radius: 4px;
                text-align: center;
                background-color: #111218;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00ff80;
            }
        """)

        # Data arrays for pyqtgraph scrolling curves
        self.time_data = []
        self.pressure_data = []
        self.crit_pressure_data = []
        self.wind_data = []
        self.disturbance_data = []
        self.max_points = 150
        
        self.init_ui()
        
        # Start simulation thread
        self.sim_thread = SimulationThread(self.shared)
        self.sim_thread.state_signal.connect(self.handle_telemetry)
        self.sim_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left Side (Dashboard & Visualization panels)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)
        
        # Top Charts Area
        charts_layout = QHBoxLayout()
        left_layout.addLayout(charts_layout, stretch=1)
        
        # Chart 1: Suction pressure vs Redline
        self.plot_pressures = pg.PlotWidget(title="Real-Time Suction Pressure vs Critical Redline (Pa)")
        self.plot_pressures.setBackground('#1e1f29')
        self.plot_pressures.showGrid(x=True, y=True, alpha=0.3)
        self.plot_pressures.setLabel('left', 'Gauge Pressure', 'Pa')
        self.plot_pressures.setLabel('bottom', 'Time steps')
        self.curve_pressure = self.plot_pressures.plot(pen=pg.mkPen(color='#00f0ff', width=2))
        self.curve_redline = self.plot_pressures.plot(pen=pg.mkPen(color='#ff3b30', width=1.5, style=Qt.DashLine))
        charts_layout.addWidget(self.plot_pressures)
        
        # Chart 2: Wind load & Disturbance estimation
        self.plot_wind = pg.PlotWidget(title="Wind Speed vs ADRC Leakage Disturbance Estimate")
        self.plot_wind.setBackground('#1e1f29')
        self.plot_wind.showGrid(x=True, y=True, alpha=0.3)
        self.plot_wind.setLabel('left', 'Speed / Disturbance')
        self.curve_wind = self.plot_wind.plot(pen=pg.mkPen(color='#ffbe00', width=2))
        self.curve_dist = self.plot_wind.plot(pen=pg.mkPen(color='#a282f9', width=1.5))
        charts_layout.addWidget(self.plot_wind)
        
        # Bottom Visualization panels
        vis_layout = QHBoxLayout()
        left_layout.addLayout(vis_layout, stretch=1)
        
        # Panel 1: Digital twin pose
        self.pose_box = QGroupBox("数字孪生姿态与状态显示")
        pose_box_layout = QVBoxLayout(self.pose_box)
        self.pose_widget = RobotPoseWidget()
        pose_box_layout.addWidget(self.pose_widget)
        vis_layout.addWidget(self.pose_box)
        
        # Panel 2: Camera feedback simulation
        self.cam_box = QGroupBox("AI 洁净度效果评估检测")
        cam_box_layout = QVBoxLayout(self.cam_box)
        self.cam_widget = CameraViewWidget()
        cam_box_layout.addWidget(self.cam_widget)
        vis_layout.addWidget(self.cam_box)
        
        # Right Side (Controls & Environment injection)
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=1)
        
        # Status Card Group
        status_group = QGroupBox("中枢运行状态")
        status_grid = QGridLayout(status_group)
        right_layout.addWidget(status_group)
        
        status_grid.addWidget(QLabel("当前控制状态:"), 0, 0)
        self.lbl_control_state = QLabel("INIT")
        self.lbl_control_state.setStyleSheet("font-weight: bold; color: #00f0ff; font-size: 14px;")
        status_grid.addWidget(self.lbl_control_state, 0, 1)
        
        status_grid.addWidget(QLabel("安全报警码:"), 1, 0)
        self.lbl_alarm = QLabel("0x000 (NORMAL)")
        self.lbl_alarm.setStyleSheet("font-weight: bold; color: #34c759;")
        status_grid.addWidget(self.lbl_alarm, 1, 1)
        
        status_grid.addWidget(QLabel("滑动安全系数:"), 2, 0)
        self.lbl_slip_sf = QLabel("99.9")
        status_grid.addWidget(self.lbl_slip_sf, 2, 1)
        
        status_grid.addWidget(QLabel("倾覆安全系数:"), 3, 0)
        self.lbl_overturn_sf = QLabel("99.9")
        status_grid.addWidget(self.lbl_overturn_sf, 3, 1)
        
        status_grid.addWidget(QLabel("风机实时转速:"), 4, 0)
        self.lbl_fan_rpm = QLabel("0 RPM")
        status_grid.addWidget(self.lbl_fan_rpm, 4, 1)
        
        # User Commands Panel Group
        cmd_group = QGroupBox("控制指令输入")
        cmd_vbox = QVBoxLayout(cmd_group)
        right_layout.addWidget(cmd_group)
        
        # Clean mode button
        self.btn_clean = QPushButton("擦洗清洁模式 (开/关)")
        self.btn_clean.setCheckable(True)
        self.btn_clean.clicked.connect(self.toggle_cleaning)
        cmd_vbox.addWidget(self.btn_clean)
        
        # Climb speed slider
        cmd_vbox.addWidget(QLabel("运动爬升目标速度 (m/s):"))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(-20, 20)  # -0.2 to 0.2 m/s
        self.slider_speed.setValue(0)
        self.lbl_speed_val = QLabel("0.0 m/s")
        self.slider_speed.valueChanged.connect(self.change_speed)
        cmd_vbox.addWidget(self.slider_speed)
        cmd_vbox.addWidget(self.lbl_speed_val)
        
        # Environment Injection Group
        env_group = QGroupBox("外部干扰故障注入")
        env_vbox = QVBoxLayout(env_group)
        right_layout.addWidget(env_group)
        
        # Wind Speed Slider
        env_vbox.addWidget(QLabel("突发大风负荷 (m/s):"))
        self.slider_wind = QSlider(Qt.Horizontal)
        self.slider_wind.setRange(0, 25) # 0 to 25 m/s
        self.slider_wind.setValue(0)
        self.lbl_wind_val = QLabel("0.0 m/s")
        self.slider_wind.valueChanged.connect(self.change_wind)
        env_vbox.addWidget(self.slider_wind)
        env_vbox.addWidget(self.lbl_wind_val)
        
        # Wall roughness Slider
        env_vbox.addWidget(QLabel("壁面微观粗糙度 Ra (微米):"))
        self.slider_rough = QSlider(Qt.Horizontal)
        self.slider_rough.setRange(1, 20) # 1um to 20um
        self.slider_rough.setValue(1)
        self.lbl_rough_val = QLabel("1.0 微米")
        self.slider_rough.valueChanged.connect(self.change_roughness)
        env_vbox.addWidget(self.slider_rough)
        env_vbox.addWidget(self.lbl_rough_val)
        
        # Gap injection button
        self.btn_inject_gap = QPushButton("跨越玻璃拼缝 (注入瞬发漏气)")
        self.btn_inject_gap.setStyleSheet("background-color: #4c2518; border-color: #ff5e00;")
        self.btn_inject_gap.clicked.connect(self.inject_gap)
        env_vbox.addWidget(self.btn_inject_gap)
        
        # Reset Button
        self.btn_reset = QPushButton("系统硬复位 (恢复正常吸附)")
        self.btn_reset.setStyleSheet("background-color: #1a3c26; border-color: #34c759;")
        self.btn_reset.clicked.connect(self.reset_system)
        right_layout.addWidget(self.btn_reset)

    @Slot(dict)
    def handle_telemetry(self, state: dict):
        """
        Processes emitted state values from physics/control thread and updates UI.
        """
        # Update text labels
        self.lbl_control_state.setText(state["control_state"])
        if state["control_state"] == "EMERGENCY_LOCK":
            self.lbl_control_state.setStyleSheet("font-weight: bold; color: #ff3b30; font-size: 14px;")
            self.lbl_alarm.setText(f"0x{state['alarm_code']:03X} (FATAL LOCK)")
            self.lbl_alarm.setStyleSheet("font-weight: bold; color: #ff3b30;")
        else:
            self.lbl_control_state.setStyleSheet("font-weight: bold; color: #00f0ff; font-size: 14px;")
            self.lbl_alarm.setText("0x000 (NORMAL)")
            self.lbl_alarm.setStyleSheet("font-weight: bold; color: #34c759;")
            
        self.lbl_slip_sf.setText(f"{state['slip_sf']:.2f}")
        self.lbl_overturn_sf.setText(f"{state['overturn_sf']:.2f}")
        self.lbl_fan_rpm.setText(f"{state['fan_rpm']:.0f} RPM")
        
        # Color safety factors if critical
        if state["slip_sf"] < 1.5:
            self.lbl_slip_sf.setStyleSheet("color: #ff9500; font-weight: bold;")
        else:
            self.lbl_slip_sf.setStyleSheet("color: #d2d2d9;")
            
        if state["overturn_sf"] < 1.5:
            self.lbl_overturn_sf.setStyleSheet("color: #ff9500; font-weight: bold;")
        else:
            self.lbl_overturn_sf.setStyleSheet("color: #d2d2d9;")

        # Update plots
        # Average vacuum pressure
        avg_gauge_p = np.mean(101325.0 - np.array(state["filtered_pressures"]))
        
        self.pressure_data.append(avg_gauge_p)
        self.crit_pressure_data.append(30000.0) # threshold line
        self.wind_data.append(state["wind_speed"])
        self.disturbance_data.append(state["eso_disturbance"] / 100.0) # scaled for plot visual match
        
        if len(self.pressure_data) > self.max_points:
            self.pressure_data.pop(0)
            self.crit_pressure_data.pop(0)
            self.wind_data.pop(0)
            self.disturbance_data.pop(0)
            
        self.curve_pressure.setData(self.pressure_data)
        self.curve_redline.setData(self.crit_pressure_data)
        
        self.curve_wind.setData(self.wind_data)
        self.curve_dist.setData(self.disturbance_data)
        
        # Update 2D pose view widget
        self.pose_widget.update_pose(
            pos_x=state["pos_x"],
            pos_y=state["pos_y"],
            pressures=state["pressures"],
            is_attached=state["is_attached"],
            state=state["control_state"],
            slip_sf=state["slip_sf"]
        )
        
        # Update Camera Cleaning view widget
        self.cam_widget.cleaning_active = state["cleaning_active"]

    def toggle_cleaning(self):
        state = self.btn_clean.isChecked()
        self.shared.set_cleaning_mode(state)

    def change_speed(self, val):
        speed = val / 100.0 # scale -0.2 to 0.2 m/s
        self.lbl_speed_val.setText(f"{speed:.2f} m/s")
        self.shared.set_climb_speed(speed)

    def change_wind(self, val):
        self.lbl_wind_val.setText(f"{val:.1f} m/s")
        self.shared.set_wind_speed(float(val))

    def change_roughness(self, val):
        rough_m = val * 1e-6
        self.lbl_rough_val.setText(f"{val:.1f} 微米")
        self.shared.set_wall_roughness(rough_m)

    def inject_gap(self):
        self.shared.trigger_gap()

    def reset_system(self):
        self.shared.trigger_reset()
        self.cam_widget.reset_cleanliness()
        self.btn_clean.setChecked(False)
        self.slider_speed.setValue(0)
        self.slider_wind.setValue(0)
        self.slider_rough.setValue(1)
        self.lbl_speed_val.setText("0.0 m/s")
        self.lbl_wind_val.setText("0.0 m/s")
        self.lbl_rough_val.setText("1.0 微米")
        
        self.pressure_data.clear()
        self.crit_pressure_data.clear()
        self.wind_data.clear()
        self.disturbance_data.clear()
        print("System Reset triggered.")

    def closeEvent(self, event):
        self.sim_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    env = SharedEnvironment()
    win = MainWindow(env)
    win.show()
    sys.exit(app.exec())
