"""
Based on Data-driven and Physical Evolution High-Altitude Curtain Wall Robot Digital Twin Control Console.
Adaptive Safety Control System Module (control_system.py)

Implements:
1. Adaptive Kalman Filter for sensor noise reduction (Barometers and IMU).
2. First-Order Linear Active Disturbance Rejection Control (LADRC) for suction fan.
3. Fall Prevention Hard Interrupt & Finite State Machine (FSM).
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, List

# Alarm Codes
ALARM_NONE = 0x000
ALARM_PRESSURE_LEAK = 0xE01        # Single cup pressure loss
ALARM_CRITICAL_VACUUM_LOSS = 0xE02 # Total vacuum pressure below safety margin
ALARM_SLIP_EXCESSIVE = 0xE03       # Wheel/track slippage detected
ALARM_IMU_FALL_DETECTED = 0xE04    # Falling acceleration detected
ALARM_OVERTURN_WARNING = 0xE05     # Tipping angle/moment near limit

class ControlState(Enum):
    INIT = 1
    STANDBY = 2
    CLIMBING = 3
    CLEANING = 4
    SLIP_RECOVERY = 5
    EMERGENCY_LOCK = 6

class BarometerKalmanFilter:
    """
    1D Kalman Filter for smoothing noisy barometer readings (suction cup pressures).
    """
    def __init__(self, initial_p: float = None, q_noise: float = 50.0, r_noise: float = 2500.0):
        # State: P (pressure in Pa)
        self.x = initial_p
        self.p_cov = 100.0
        self.q = q_noise  # Process noise covariance
        self.r = r_noise  # Measurement noise covariance (noisy sensor)
        self.initialized = (initial_p is not None)

    def filter(self, measurement: float) -> float:
        if not self.initialized:
            self.x = measurement
            self.p_cov = self.q  # Start covariance with process noise
            self.initialized = True
            return self.x
            
        # Time Update (Predict)
        self.p_cov = self.p_cov + self.q
        
        # Measurement Update (Correct)
        k_gain = self.p_cov / (self.p_cov + self.r)
        self.x = self.x + k_gain * (measurement - self.x)
        self.p_cov = (1.0 - k_gain) * self.p_cov
        return self.x

class IMUKalmanFilter:
    """
    2D Kalman Filter for smoothing IMU acceleration readings (a_x, a_y).
    """
    def __init__(self, q_noise: float = 0.05, r_noise: float = 0.5):
        # States: [acc_x, acc_y]
        self.x = None
        self.p_cov = np.eye(2) * 0.1
        self.q = np.eye(2) * q_noise
        self.r = np.eye(2) * r_noise
        self.initialized = False

    def filter(self, measurement: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.x = measurement.copy()
            self.initialized = True
            return self.x
            
        # Predict
        self.p_cov = self.p_cov + self.q
        # Correct
        k_gain = self.p_cov @ np.linalg.inv(self.p_cov + self.r)
        self.x = self.x + k_gain @ (measurement - self.x)
        self.p_cov = (np.eye(2) - k_gain) @ self.p_cov
        return self.x

class SuctionLADRC:
    """
    First-Order Linear Active Disturbance Rejection Control (LADRC).
    Regulates the suction fan speed to maintain safety vacuum level,
    actively estimating and compensating for leakages/disturbances.
    """
    def __init__(self, target_gauge_pa: float, b0: float = 0.5, wc: float = 10.0, wo: float = 40.0):
        """
        Args:
            target_gauge_pa: Desired safety vacuum gauge pressure (Pa) (e.g., 60,000 Pa)
            b0: Control gain scaling factor
            wc: Controller bandwidth (rad/s)
            wo: Observer bandwidth (rad/s)
        """
        self.target = target_gauge_pa
        self.b0 = b0
        self.kp = wc  # Controller gain: Kp = wc for 1st-order system
        
        # Extended State Observer (ESO) gains for 1st-order system (2 states: y, f)
        # L = [2*wo, wo^2]^T
        self.beta1 = 2.0 * wo
        self.beta2 = wo ** 2
        
        # Observer states: z1 = estimate of y (pressure), z2 = estimate of disturbance f
        self.z1 = 0.0
        self.z2 = 0.0
        self.initialized = False

    def update(self, y_val: float, u_prev: float, dt: float) -> float:
        """
        Updates the LADRC controller step.
        """
        if not self.initialized:
            self.z1 = y_val
            self.z2 = 0.0
            self.initialized = True
            
        # 1. ESO Update (Euler integration)
        err = self.z1 - y_val
        dz1 = self.z2 + self.b0 * u_prev - self.beta1 * err
        dz2 = -self.beta2 * err
        
        self.z1 += dz1 * dt
        self.z2 += dz2 * dt
        
        # 2. Control Law Calculation
        u0 = self.kp * (self.target - self.z1)
        u = (u0 - self.z2) / self.b0
        
        return u

class SafetyControlHub:
    """
    Curtain Wall Robot Safety Control Hub.
    Coordinates FSM states, filters sensor noise, executes ADRC vacuum regulation,
    and runs the highest priority fall-prevention hard interrupt loop.
    """
    def __init__(self, num_cups: int, max_fan_rpm: float, max_wheel_rpm: float):
        self.num_cups = num_cups
        self.max_fan_rpm = max_fan_rpm
        self.max_wheel_rpm = max_wheel_rpm
        
        # State & FSM
        self.state = ControlState.INIT
        self.alarm_code = ALARM_NONE
        
        # Filters
        self.filters_p = [BarometerKalmanFilter() for _ in range(num_cups)]
        self.filter_imu = IMUKalmanFilter()
        
        # Vacuum ADRC Controller
        # Target vacuum gauge pressure = 65 kPa (absolute P = 36.3 kPa)
        # Safe limit is 30 kPa gauge. If we drop below that, emergency lock triggers.
        self.target_vacuum_gauge = 65000.0
        self.ladrc = SuctionLADRC(target_gauge_pa=self.target_vacuum_gauge, b0=0.2, wc=8.0, wo=35.0)
        self.fan_u_prev = 0.0
        
        # Safe limits
        self.p_atm = 101325.0
        self.crit_vacuum_gauge = 30000.0  # 30 kPa gauge (71 kPa absolute) is critical fallback
        self.backup_valve_locked = False
        
    def filter_sensors(self, raw_pressures: np.ndarray, raw_imu_acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Kalman Filters to smooth noisy telemetry data.
        """
        filtered_p = np.array([self.filters_p[i].filter(raw_pressures[i]) for i in range(self.num_cups)])
        filtered_acc = self.filter_imu.filter(raw_imu_acc)
        return filtered_p, filtered_acc

    def step(self, telemetry: Dict[str, Any], user_commands: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Main control loop executed at each control tick.
        """
        raw_pressures = telemetry.get('pressures', np.ones(self.num_cups) * self.p_atm)
        raw_imu = telemetry.get('imu_accel', np.zeros(2))
        is_attached = telemetry.get('is_attached', True)
        sf_slip = telemetry.get('slip_safety_factor', 99.9)
        sf_overturn = telemetry.get('overturn_safety_factor', 99.9)
        
        # 1. Noise Filtering
        filtered_p, filtered_acc = self.filter_sensors(raw_pressures, raw_imu)
        
        # Average gauge pressure of cups (Pa)
        avg_gauge_p = np.mean(self.p_atm - filtered_p)
        
        # 2. Hard Interrupt & Safety Fall-Prevention Check
        # Triggered immediately if telemetry indicates danger
        danger_detected = False
        
        if not is_attached or sf_overturn < 1.05:
            self.alarm_code = ALARM_OVERTURN_WARNING
            danger_detected = True
        elif avg_gauge_p < self.crit_vacuum_gauge:
            self.alarm_code = ALARM_CRITICAL_VACUUM_LOSS
            danger_detected = True
        elif np.any(self.p_atm - filtered_p < 15000.0): # Any single cup lost pressure
            self.alarm_code = ALARM_PRESSURE_LEAK
            danger_detected = True
        elif sf_slip < 1.2:
            self.alarm_code = ALARM_SLIP_EXCESSIVE
            danger_detected = True
        elif filtered_acc[1] < -3.0: # Extreme downward acceleration detected
            self.alarm_code = ALARM_IMU_FALL_DETECTED
            danger_detected = True
            
        if danger_detected:
            # TRIGGER HARD INTERRUPT SELF-LOCK
            self.state = ControlState.EMERGENCY_LOCK
            self.backup_valve_locked = True # Close auxiliary valves to seal chambers
            
            # Action outputs: Maximize suction, stop movement, cut cleaning reaction forces
            return {
                "target_fan_rpm": float(self.max_fan_rpm),
                "target_wheel_rpm_l": 0.0,
                "target_wheel_rpm_r": 0.0,
                "spray_flow": 0.0,
                "cleaning_active": False,
                "alarm_code": int(self.alarm_code),
                "control_state": self.state.name
            }

        # 3. Normal FSM State Transition
        if self.state == ControlState.EMERGENCY_LOCK:
            # Cannot exit emergency lock unless reset (simulate safe lock)
            pass
        elif self.state == ControlState.INIT:
            self.state = ControlState.STANDBY
        elif self.state == ControlState.STANDBY:
            if user_commands.get('cleaning_mode', False):
                self.state = ControlState.CLEANING
            elif abs(user_commands.get('target_climb_speed', 0.0)) > 0.01:
                self.state = ControlState.CLIMBING
        elif self.state == ControlState.CLIMBING:
            if not user_commands.get('cleaning_mode', False) and abs(user_commands.get('target_climb_speed', 0.0)) < 0.01:
                self.state = ControlState.STANDBY
            elif user_commands.get('cleaning_mode', False):
                self.state = ControlState.CLEANING
        elif self.state == ControlState.CLEANING:
            if not user_commands.get('cleaning_mode', False):
                if abs(user_commands.get('target_climb_speed', 0.0)) > 0.01:
                    self.state = ControlState.CLIMBING
                else:
                    self.state = ControlState.STANDBY

        # 4. Adaptive Suction ADRC Control
        # ADRC computes desired control input (fan RPM) to track target gauge pressure
        adrc_out = self.ladrc.update(avg_gauge_p, self.fan_u_prev, dt)
        
        # Scale and clip output to RPM limits
        fan_rpm_cmd = np.clip(adrc_out, 4000.0, self.max_fan_rpm)
        self.fan_u_prev = fan_rpm_cmd
        
        # 5. Actuator Command Synthesis
        target_v = user_commands.get('target_climb_speed', 0.0)
        r_wheel = 0.04
        wheel_rpm_cmd = (target_v / (2.0 * np.pi * r_wheel)) * 60.0
        wheel_rpm_cmd = np.clip(wheel_rpm_cmd, -self.max_wheel_rpm, self.max_wheel_rpm)
        
        if self.state == ControlState.CLEANING:
            spray_flow_cmd = user_commands.get('spray_flow', 0.0)
            cleaning_active_cmd = True
        else:
            spray_flow_cmd = 0.0
            cleaning_active_cmd = False
            
        return {
            "target_fan_rpm": float(fan_rpm_cmd),
            "target_wheel_rpm_l": float(wheel_rpm_cmd),
            "target_wheel_rpm_r": float(wheel_rpm_cmd),
            "spray_flow": float(spray_flow_cmd),
            "cleaning_active": bool(cleaning_active_cmd),
            "alarm_code": int(self.alarm_code),
            "control_state": self.state.name
        }

if __name__ == "__main__":
    print("Testing Control System Module...")
    # Instantiate control hub
    hub = SafetyControlHub(num_cups=4, max_fan_rpm=15000.0, max_wheel_rpm=120.0)
    
    # 1. Normal state simulation
    telemetry_normal = {
        "pressures": np.array([30000.0, 31000.0, 29000.0, 30500.0]), # Good vacuum (70 kPa gauge)
        "imu_accel": np.array([0.0, 0.0]),
        "is_attached": True,
        "slip_safety_factor": 4.5,
        "overturn_safety_factor": 15.0
    }
    user_cmds = {
        "cleaning_mode": True,
        "target_climb_speed": 0.1,
        "spray_flow": 2.0
    }
    
    out = hub.step(telemetry_normal, user_cmds, dt=0.01)
    print("Normal State Output:")
    for k, v in out.items():
        print(f"  {k}: {v}")
        
    # 2. Simulate leak / sudden pressure loss triggering emergency lock
    print("\nSimulating vacuum loss incident...")
    telemetry_hazard = {
        "pressures": np.array([90000.0, 95000.0, 92000.0, 91000.0]), # Pressure lost (almost atmospheric 101k)
        "imu_accel": np.array([0.0, -9.8]), # Falling IMU acceleration
        "is_attached": True,
        "slip_safety_factor": 0.2,
        "overturn_safety_factor": 0.8
    }
    
    out_emergency = hub.step(telemetry_hazard, user_cmds, dt=0.01)
    print("Emergency Interrupt Output:")
    for k, v in out_emergency.items():
        print(f"  {k}: {v}")
