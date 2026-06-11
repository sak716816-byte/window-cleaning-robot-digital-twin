"""
Based on Data-driven and Physical Evolution High-Altitude Curtain Wall Robot Digital Twin Control Console.
Physics Engine Core Module (physics_engine.py)

This module implements the multi-physics simulation of the wall-climbing robot,
including:
1. Suction cup vacuum decay and flow dynamics (orifice leakage & pump evacuation).
2. Dynamic friction, normal force, and Slip Safety Factor (SF) calculations.
3. Rigid-body overturning moment equilibrium under wind, spray, and brush forces.
4. Parameterized geometric design optimization for the robot mechanism.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

# Physical Constants
G_ACCEL = 9.80665   # Gravity acceleration (m/s^2)
P_ATM = 101325.0    # Standard atmospheric pressure (Pa)
R_GAS = 287.058     # Specific gas constant for air (J/(kg*K))
T_ROOM = 293.15     # Temperature (K)
RHO_AIR_ATM = 1.204  # Air density at standard atmosphere (kg/m^3)
C_D = 0.62          # Discharge coefficient for air leak orifice

@dataclass
class RobotParameters:
    """
    Configuration parameters for the curtain wall robot.
    Defaults represent the Zhongke Luzhou CUBEBOX platform.
    """
    mass: float = 9.0               # Total mass of the robot (kg)
    num_cups: int = 4               # Number of negative pressure suction cups
    cup_diameter: float = 0.05      # Diameter of each circular suction cup (m)
    cup_volume: float = 2.0e-5      # Volume of each suction cup cavity (m^3)
    seal_width: float = 0.005       # Sealing ring lip width in contact with wall (m)
    seal_elasticity: float = 1.5e6  # Elastic modulus of sealing material (Pa)
    cog_height: float = 0.08        # Height of Center of Gravity (CoG) from wall (m)
    length: float = 0.35            # Robot length along body y-axis (m)
    width: float = 0.30             # Robot width along body x-axis (m)
    wheel_radius: float = 0.04      # Radius of driving wheels/tracks (m)
    max_fan_rpm: float = 15000.0    # Max speed of suction fan (RPM)
    max_wheel_rpm: float = 120.0    # Max wheel rotation speed (RPM)
    max_drive_torque: float = 5.0   # Max drive torque per motor (N*m)
    preload_force: float = 40.0     # Mechanical spring preload pressing robot to wall (N)
    cup_positions: np.ndarray = None # Relative (x, y) coordinates of cups from CoG

    def __post_init__(self):
        # Default cup layout: symmetric rectangular placement
        if self.cup_positions is None:
            lx = self.width * 0.35
            ly = self.length * 0.35
            self.cup_positions = np.array([
                [-lx,  ly],  # Front-Left
                [ lx,  ly],  # Front-Right
                [-lx, -ly],  # Rear-Left
                [ lx, -ly]   # Rear-Right
            ])
        else:
            self.cup_positions = np.array(self.cup_positions)

@dataclass
class PhysicsState:
    """
    Current physical state variables of the robot.
    """
    pos_x: float = 0.0              # Position on wall x-axis (m)
    pos_y: float = 0.0              # Position on wall y-axis (m)
    vel_x: float = 0.0              # Velocity along x-axis (m/s)
    vel_y: float = 0.0              # Velocity along y-axis (m/s)
    acc_x: float = 0.0              # Acceleration along x-axis (m/s^2)
    acc_y: float = 0.0              # Acceleration along y-axis (m/s^2)
    
    # Suction cup absolute pressures (Pa). P_ATM = no vacuum.
    cup_pressures: np.ndarray = None 
    
    fan_rpm: float = 0.0            # Current fan RPM
    wheel_rpm_l: float = 0.0        # Left wheel/track RPM
    wheel_rpm_r: float = 0.0        # Right wheel/track RPM
    
    slip_safety_factor: float = 99.9  # Slip Safety Factor (SF)
    overturn_safety_factor: float = 99.9 # Overturning Safety Factor (SF)
    
    is_attached: bool = True        # True if safely attached to the wall
    is_slipping: bool = False       # True if slipping occurs (SF_slip < 1.0)
    is_overturned: bool = False     # True if tipped over (SF_overturn < 1.0)
    
    def __post_init__(self):
        if self.cup_pressures is None:
            # Initialize with atmospheric pressure (no vacuum)
            self.cup_pressures = np.ones(4) * P_ATM

class PhysicsEngine:
    """
    High-Fidelity Physics Engine for Wall-Climbing Curtain Wall Robots.
    Simulates dynamics, aerodynamics, suction pressure kinetics, leakage,
    slipping, and overturning boundaries.
    """
    def __init__(self, params: RobotParameters, initial_state: PhysicsState = None):
        self.p = params
        self.state = initial_state if initial_state is not None else PhysicsState()
        if len(self.state.cup_pressures) != self.p.num_cups:
            self.state.cup_pressures = np.ones(self.p.num_cups) * P_ATM

    def compute_leakage_area(self, cup_idx: int, wall_roughness: float, 
                             wall_gaps: List[Tuple[float, float, float, float]]) -> float:
        """
        Calculates fluid leakage area (m^2) for a specific suction cup.
        Models seal deformation over wall roughness and gaps/joints.
        
        Args:
            cup_idx: Index of the suction cup.
            wall_roughness: Average surface roughness Ra (m).
            wall_gaps: List of gaps as (x_start, y_start, width, depth) in wall coordinates.
            
        Returns:
            Total leakage area (m^2).
        """
        # Cup absolute coordinates
        cup_rel_pos = self.p.cup_positions[cup_idx]
        cup_x = self.state.pos_x + cup_rel_pos[0]
        cup_y = self.state.pos_y + cup_rel_pos[1]
        cup_r = self.p.cup_diameter / 2.0
        cup_perimeter = np.pi * self.p.cup_diameter
        
        # 1. Roughness Leakage
        # The seal deforms under preload. High pressure and soft seals reduce roughness gaps.
        # Height of micro-gaps: h_rough = max(0, Ra - delta_seal)
        # delta_seal = Pressure_Diff * Area / Seal_Stiffness or simple compression
        # We model this phenomenologically:
        seal_compression = 1e-6 * self.p.preload_force / (cup_perimeter * self.p.seal_width * self.p.seal_elasticity)
        h_rough = max(1e-8, wall_roughness - seal_compression)
        area_leak_roughness = cup_perimeter * h_rough
        
        # 2. Wall Gap Leakage (e.g. glass panel joints)
        # If the suction cup seal crosses a linear gap, it creates a leakage path.
        # We simplify the gap as a rectangular slot.
        area_leak_gap = 0.0
        for gx, gy, gw, gd in wall_gaps:
            # We assume vertical or horizontal joints.
            # Vertical gap: runs along constant x = gx, width gw, depth gd
            # Horizontal gap: runs along constant y = gy, width gw, depth gd
            # Check vertical gap overlap with cup seal circle:
            if abs(cup_x - gx) < cup_r:
                # Cup crosses the vertical gap. 
                # The seal ring crosses this gap at two points (entry and exit)
                # Leak area = 2 * gap_width * gap_depth (if seal doesn't fit into the gap)
                area_leak_gap += 2.0 * gw * gd
            
            if abs(cup_y - gy) < cup_r:
                # Cup crosses the horizontal gap.
                area_leak_gap += 2.0 * gw * gd
                
        return area_leak_roughness + area_leak_gap

    def compute_aerodynamic_forces(self, wind_speed: float, wind_angle: float, 
                                   wall_inclination: float) -> Tuple[float, float, float]:
        """
        Computes aerodynamic drag and lift forces due to high-altitude winds.
        
        Args:
            wind_speed: Wind speed (m/s).
            wind_angle: Wind direction in the wall plane (rad).
            wall_inclination: Wall slope angle (rad), pi/2 is vertical.
            
        Returns:
            Tuple of (F_wind_x, F_wind_y, F_wind_normal) where normal is pointing AWAY from wall.
        """
        # Aerodynamic coefficients (typical box shapes)
        c_drag_tangential = 1.05
        c_lift_normal = 0.8  # Lift pulls the robot away from the wall (suction effect)
        
        # Frontal and side areas
        area_x = self.p.length * self.p.cog_height
        area_y = self.p.width * self.p.cog_height
        area_normal = self.p.length * self.p.width
        
        # Dynamic pressure: q = 0.5 * rho * v^2
        q_wind = 0.5 * RHO_AIR_ATM * (wind_speed ** 2)
        
        # Tangential forces (along wall plane)
        f_wind_x = q_wind * area_x * c_drag_tangential * np.cos(wind_angle)
        f_wind_y = q_wind * area_y * c_drag_tangential * np.sin(wind_angle)
        
        # Normal lift force (pulling robot off the wall)
        # Lift is maximized when wind is parallel or slightly angled to the wall
        f_wind_normal = q_wind * area_normal * c_lift_normal * np.sin(wall_inclination)
        
        return f_wind_x, f_wind_y, f_wind_normal

    def compute_cleaning_forces(self, cleaning_active: bool, spray_flow: float) -> Tuple[float, float, float]:
        """
        Computes forces generated by cleaning roller brushes and water spray.
        
        Args:
            cleaning_active: True if roller brush is spinning.
            spray_flow: Water spray flow rate (L/min).
            
        Returns:
            Tuple of (F_clean_x, F_clean_y, F_clean_normal)
        """
        if not cleaning_active:
            return 0.0, 0.0, 0.0
            
        # 1. Roller brush reaction force (roller spins against the direction of travel)
        # Cleaning roller brush generates normal force pressing against wall and tangential drag
        f_brush_normal = 15.0       # Normal reaction force of brush compression (N)
        f_brush_tangential = -8.0   # Friction drag opposite to movement (N) (assume y-direction)
        
        # 2. Water spray reaction force
        # Spray flow rate -> velocity of nozzle jet -> recoil force
        # F = dot_mass * v_jet
        # Assume nozzle diameter = 1.5mm, flow rate = spray_flow (L/min)
        if spray_flow > 0:
            flow_kg_s = (spray_flow / 1000.0) / 60.0 # L/min -> kg/s
            nozzle_area = np.pi * (0.0015 ** 2) / 4.0
            v_jet = flow_kg_s / (1000.0 * nozzle_area) # water density = 1000 kg/m^3
            f_spray_recoil = flow_kg_s * v_jet # Normal force pushing away from wall (N)
        else:
            f_spray_recoil = 0.0
            
        # Total forces (assuming brush acts along body y-axis)
        f_clean_x = 0.0
        f_clean_y = f_brush_tangential
        f_clean_normal = f_brush_normal + f_spray_recoil
        
        return f_clean_x, f_clean_y, f_clean_normal

    def simulate_tick(self, env: Dict[str, Any]) -> PhysicsState:
        """
        Evolves the physical state of the robot by one time step (dt).
        
        Args:
            env: Dictionary containing:
                - 'dt': Time step (s)
                - 'wind_speed': Wind velocity (m/s)
                - 'wind_angle': Wind angle (rad)
                - 'wall_inclination': Slope of the curtain wall (rad, pi/2 = vertical)
                - 'wall_roughness': Wall surface roughness Ra (m)
                - 'wall_gaps': List of gaps [(gx, gy, gw, gd), ...]
                - 'spray_flow': Cleaning water spray flow rate (L/min)
                - 'cleaning_active': Boolean state of roller brushes
                - 'target_fan_rpm': Controller demand for suction fan
                - 'target_wheel_rpm_l': Controller demand for left drive
                - 'target_wheel_rpm_r': Controller demand for right drive
                - 'mu': Surface friction coefficient (default: 0.5)
                
        Returns:
            Updated PhysicsState.
        """
        dt = env.get('dt', 0.01)
        wind_speed = env.get('wind_speed', 0.0)
        wind_angle = env.get('wind_angle', 0.0)
        wall_inclination = env.get('wall_inclination', np.pi / 2.0)
        wall_roughness = env.get('wall_roughness', 1e-6)
        wall_gaps = env.get('wall_gaps', [])
        spray_flow = env.get('spray_flow', 0.0)
        cleaning_active = env.get('cleaning_active', False)
        target_fan_rpm = env.get('target_fan_rpm', self.p.max_fan_rpm)
        target_wheel_rpm_l = env.get('target_wheel_rpm_l', 0.0)
        target_wheel_rpm_r = env.get('target_wheel_rpm_r', 0.0)
        mu = env.get('mu', 0.5)
        
        if not self.state.is_attached:
            # If the robot has already fallen, it stays fallen.
            self.state.vel_x, self.state.vel_y = 0.0, 0.0
            self.state.acc_x, self.state.acc_y = 0.0, -G_ACCEL
            self.state.pos_y += self.state.vel_y * dt + 0.5 * self.state.acc_y * (dt ** 2)
            self.state.cup_pressures.fill(P_ATM)
            return self.state

        # 1. Update Fan and Suction Pressure Dynamics
        # Fan speed follows a first-order lag (inertia of motor)
        tau_fan = 0.15 # Time constant (s)
        self.state.fan_rpm += (target_fan_rpm - self.state.fan_rpm) * (dt / (tau_fan + dt))
        self.state.fan_rpm = np.clip(self.state.fan_rpm, 0.0, self.p.max_fan_rpm)
        
        # Area of each cup
        cup_area = np.pi * (self.p.cup_diameter ** 2) / 4.0
        
        # Pumping capability scales with fan RPM
        # Volumetric suction speed of pump S_pump (m^3/s)
        s_max_single = 0.005 # Max evacuation speed per cup (m^3/s) at max RPM
        normalized_rpm = self.state.fan_rpm / self.p.max_fan_rpm
        
        for i in range(self.p.num_cups):
            p_curr = self.state.cup_pressures[i]
            
            # Compute leakage area
            a_leak = self.compute_leakage_area(i, wall_roughness, wall_gaps)
            
            # Mass flow rate IN (leakage) - Orifice flow model (isentropic/Bernoulli)
            # m_dot_in = C_D * A_leak * sqrt(2 * rho * delta_P)
            dp = max(0.0, P_ATM - p_curr)
            m_dot_in = C_D * a_leak * np.sqrt(2.0 * RHO_AIR_ATM * dp)
            
            # Mass flow rate OUT (pump evacuation)
            # Pump efficiency decreases as vacuum approaches maximum limit
            # Limit absolute pressure P_vacuum_limit = 10% of P_ATM (91 kPa gauge pressure)
            p_limit = 10000.0 # 10 kPa absolute
            p_eff = max(0.0, (p_curr - p_limit) / (P_ATM - p_limit))
            vol_flow_out = s_max_single * normalized_rpm * p_eff
            rho_curr = p_curr / (R_GAS * T_ROOM)
            m_dot_out = vol_flow_out * rho_curr
            
            # Pressure derivative: dP/dt = (R * T / V) * (m_dot_in - m_dot_out)
            dp_dt = (R_GAS * T_ROOM / self.p.cup_volume) * (m_dot_in - m_dot_out)
            
            # Integrate pressure using Euler method
            p_next = p_curr + dp_dt * dt
            self.state.cup_pressures[i] = np.clip(p_next, p_limit, P_ATM)
            
        # Total suction force (normal to wall, pulling robot IN)
        cup_suctions = (P_ATM - self.state.cup_pressures) * cup_area
        f_suction_total = np.sum(cup_suctions)
        
        # 2. Compute Environmental & Cleaning Forces
        f_wind_x, f_wind_y, f_wind_norm = self.compute_aerodynamic_forces(wind_speed, wind_angle, wall_inclination)
        f_clean_x, f_clean_y, f_clean_norm = self.compute_cleaning_forces(cleaning_active, spray_flow)
        
        # Gravity components
        # Wall coordinates: y is UP, x is horizontal (in-plane), z is normal (away from wall)
        f_g_x = 0.0
        f_g_y = -self.p.mass * G_ACCEL * np.sin(wall_inclination)
        f_g_z = -self.p.mass * G_ACCEL * np.cos(wall_inclination) # pulls away from wall if θ > pi/2 (negative incline)
        
        # Total normal force pressing the robot against the wall:
        # F_normal = F_suction + F_preload - F_gravity_z - F_wind_normal - F_clean_normal
        # Positive F_normal means robot is pressed against wall.
        f_normal = f_suction_total + self.p.preload_force + f_g_z - f_wind_norm - f_clean_norm
        
        # 3. Overturning Moment Analysis (Anti-overturning Boundary)
        # We analyze tipping about all 4 edges: Top, Bottom, Left, Right
        # Let's check the bottom edge (y = -length/2) and top edge (y = length/2)
        # Stabilizing torque comes from suction cups and gravity normal components.
        # Overturning torque comes from wind, brush drag, spray recoil, and gravity tangential.
        
        # Distance of CoG to edges:
        L_top = self.p.length / 2.0
        L_bottom = self.p.length / 2.0
        L_left = self.p.width / 2.0
        L_right = self.p.width / 2.0
        
        # Bottom edge pivot: overturning torque (tends to peel top away)
        # Overturning moments:
        # - Gravity tangential: F_g_y * cog_height
        # - Wind tangential: f_wind_y * cog_height
        # - Wind normal: f_wind_norm * L_top (pulls away)
        # - Spray normal recoil: f_clean_norm_spray * L_top
        # - Brush tangential drag: f_clean_y * H_brush
        m_overturn_bottom = (
            abs(f_g_y) * self.p.cog_height + 
            abs(f_wind_y) * self.p.cog_height + 
            abs(f_clean_y) * self.p.cog_height + 
            max(0.0, f_wind_norm) * L_bottom + 
            max(0.0, f_clean_norm) * L_bottom
        )
        
        # Stabilizing moments:
        # - Gravity normal: f_g_z * L_bottom (stabilizes if positive, i.e., θ < pi/2)
        # - Preload force: F_preload * L_preload_pivot (approx centered, so F_preload * L_bottom)
        # - Suction cups: Sum (F_suction_i * (y_cup_i - y_bottom))
        m_stabilize_bottom = max(0.0, f_g_z) * L_bottom + self.p.preload_force * L_bottom
        for i in range(self.p.num_cups):
            cup_y_dist = self.p.cup_positions[i, 1] + L_bottom # distance from bottom edge
            m_stabilize_bottom += cup_suctions[i] * cup_y_dist
            
        # Overturning safety factor
        sf_overturn = m_stabilize_bottom / max(1e-3, m_overturn_bottom)
        self.state.overturn_safety_factor = sf_overturn
        
        # 4. Tangential Load and Slip Safety Factor
        # Drive force from wheels
        self.state.wheel_rpm_l += (target_wheel_rpm_l - self.state.wheel_rpm_l) * (dt / 0.1)
        self.state.wheel_rpm_r += (target_wheel_rpm_r - self.state.wheel_rpm_r) * (dt / 0.1)
        
        avg_wheel_rpm = (self.state.wheel_rpm_l + self.state.wheel_rpm_r) / 2.0
        vel_drive_target = (avg_wheel_rpm / 60.0) * 2.0 * np.pi * self.p.wheel_radius
        
        # Active driving force (limited by motor torque and static friction)
        torque_req = self.p.mass * self.state.acc_y * self.p.wheel_radius # dynamic tracking
        torque_l = np.clip(torque_req / 2.0, -self.p.max_drive_torque, self.p.max_drive_torque)
        torque_r = np.clip(torque_req / 2.0, -self.p.max_drive_torque, self.p.max_drive_torque)
        f_drive_y = (torque_l + torque_r) / self.p.wheel_radius
        
        # Total external tangential load parallel to wall plane
        f_ext_tangential_x = f_wind_x + f_clean_x
        f_ext_tangential_y = f_g_y + f_wind_y + f_clean_y
        
        # Total tangential load
        f_load_tangential = np.sqrt(f_ext_tangential_x**2 + f_ext_tangential_y**2)
        
        # Friction capacity
        max_static_friction = mu * max(0.0, f_normal)
        
        # Slip Safety Factor (SF)
        sf_slip = max_static_friction / max(1e-3, f_load_tangential)
        self.state.slip_safety_factor = sf_slip
        
        # 5. Physics Evolution (Integration)
        if f_normal < 0.0:
            # Normal force is negative: robot detaches from wall!
            self.state.is_attached = False
            self.state.is_slipping = True
            self.state.is_overturned = True
            self.state.acc_x = f_ext_tangential_x / self.p.mass
            self.state.acc_y = -G_ACCEL
        elif sf_overturn < 1.0:
            # Overturned!
            self.state.is_attached = False
            self.state.is_overturned = True
            self.state.acc_y = -G_ACCEL
        elif sf_slip < 1.0:
            # Slipping occurs! Sliding friction acts opposite to slipping direction
            self.state.is_slipping = True
            # Sliding direction unit vector
            slip_dir_x = f_ext_tangential_x / max(1e-3, f_load_tangential)
            slip_dir_y = f_ext_tangential_y / max(1e-3, f_load_tangential)
            
            # Dynamic friction is slightly lower than static
            mu_dynamic = mu * 0.85
            f_friction_x = mu_dynamic * f_normal * slip_dir_x
            f_friction_y = mu_dynamic * f_normal * slip_dir_y
            
            self.state.acc_x = (f_ext_tangential_x - f_friction_x) / self.p.mass
            self.state.acc_y = (f_ext_tangential_y - f_friction_y) / self.p.mass
            
            # Evolve position & velocity
            self.state.vel_x += self.state.acc_x * dt
            self.state.vel_y += self.state.acc_y * dt
            self.state.pos_x += self.state.vel_x * dt
            self.state.pos_y += self.state.vel_y * dt
        else:
            # Safe adhesion: robot tracks the wheel speed input
            self.state.is_slipping = False
            self.state.acc_x = 0.0
            self.state.acc_y = 0.0
            
            # Velocity follows driving speed
            self.state.vel_x = 0.0 # wheels only drive in body-y
            self.state.vel_y = vel_drive_target
            
            self.state.pos_x += self.state.vel_x * dt
            self.state.pos_y += self.state.vel_y * dt
            
        return self.state

def optimize_robot_design(mass: float, num_cups: int, safety_factor_target: float = 4.0, 
                           max_vacuum_gauge_pa: float = 90000.0, mu: float = 0.5) -> Dict[str, Any]:
    """
    Automated design synthesis function. Computes optimal physical parameters
    for the robot based on literature modeling principles.
    
    Formula based on the extracted literature:
        D = sqrt( (1000 * m * g * k) / (pi * n * p_gauge_kpa) ) * 1e-3 (converted to meters)
    which simplifies in SI units to:
        D = sqrt( (4 * m * g * SF) / (pi * n * P_gauge_pa) )
        
    Args:
        mass: Robot mass (kg)
        num_cups: Number of active suction cups
        safety_factor_target: Design safety factor (k or SF)
        max_vacuum_gauge_pa: Operating vacuum gauge pressure (Pa)
        mu: Sliding friction coefficient of suction cup seal on glass
        
    Returns:
        Dictionary of optimal parameters:
            - 'optimal_cup_diameter_m': Optimal cup diameter (m)
            - 'minimum_suction_force_n': Minimum required suction force (N)
            - 'optimal_cup_spacing_m': Suggested spacing (m) to avoid localized overturning
            - 'required_motor_torque_nm': Motor torque needed to climb vertical walls
    """
    # 1. Calculate Cup Diameter
    # D = sqrt( (4 * mass * g * SF) / (pi * num_cups * P_vacuum) )
    numerator = 4.0 * mass * G_ACCEL * safety_factor_target
    denominator = np.pi * num_cups * max_vacuum_gauge_pa
    optimal_diameter = np.sqrt(numerator / denominator)
    
    # 2. Minimum Suction Force
    # F_suction_min = (mass * g / mu) * SF - Preload
    # F_suction_total = num_cups * cup_area * P_vacuum
    preload_assumed = 40.0
    required_suction_force_total = max(0.0, (mass * G_ACCEL / mu) * safety_factor_target - preload_assumed)
    
    # 3. Spacing to balance moments (optimal cup spacing)
    # The spacing should be wide enough that the stabilizing torque covers external wind moments.
    # Spacing is set proportional to the overall length of the robot (standard layout ratio = 0.7)
    suggested_spacing = 0.35 * 0.7 # standard layout: ~24 cm for 35 cm robot
    
    # 4. Motor Torque calculation
    # T = F_load * R_wheel / efficiency
    # F_load = Mass * g (vertical climb) + Rolling_resistance (assume 5% normal load)
    r_wheel = 0.04
    eff = 0.85
    f_normal_est = required_suction_force_total + preload_assumed
    f_load = mass * G_ACCEL + 0.05 * f_normal_est
    required_torque = (f_load * r_wheel) / (eff * 2.0) # divided between 2 drive motors
    
    return {
        "optimal_cup_diameter_m": float(optimal_diameter),
        "minimum_suction_force_n": float(required_suction_force_total),
        "optimal_cup_spacing_m": float(suggested_spacing),
        "required_motor_torque_nm": float(required_torque)
    }

if __name__ == "__main__":
    # Self-test block to verify model code logic
    print("Initializing PhysicsEngine Self-Test...")
    params = RobotParameters(mass=9.0, num_cups=4, cup_diameter=0.05)
    engine = PhysicsEngine(params)
    
    # Simulate a single tick in normal conditions
    env_delta = {
        "dt": 0.01,
        "wind_speed": 5.0,
        "wind_angle": 0.0,
        "wall_inclination": np.pi / 2.0, # Vertical
        "wall_roughness": 1e-6,
        "wall_gaps": [],
        "spray_flow": 0.0,
        "cleaning_active": False,
        "target_fan_rpm": 12000.0,
        "target_wheel_rpm_l": 50.0,
        "target_wheel_rpm_r": 50.0,
        "mu": 0.5
    }
    
    # Run simulation ticks
    print("Running 100 simulation ticks (1 second)...")
    for _ in range(100):
        state = engine.simulate_tick(env_delta)
        
    print(f"Final State: pos_y={state.pos_y:.4f}m, velocity_y={state.vel_y:.4f}m/s")
    print(f"Suction Cup pressures (Pa): {state.cup_pressures}")
    print(f"Slip Safety Factor: {state.slip_safety_factor:.2f}")
    print(f"Overturn Safety Factor: {state.overturn_safety_factor:.2f}")
    
    print("Design optimization test for 9kg robot with 4 cups:")
    opt = optimize_robot_design(mass=9.0, num_cups=4, safety_factor_target=4.0, mu=0.5)
    for k, v in opt.items():
        print(f"  {k}: {v:.4f}")
