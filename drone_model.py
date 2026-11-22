
import numpy as np

# This file contains the physical parameters and equations of motion for the 16-state quadcopter model.
# State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw,
#  p, q, r, omega1, omega2, omega3, omega4]
# Inputs: Four individual motor shaft torques (N*m)
# All calculations use SI units and radians.

# 1. PHYSICAL PARAMETERS

# Drone mass and geometry
MASS = 1.20                 # kg
GRAVITY = 9.81              # m/s^2
ARM_LENGTH = 0.225          # m

# Moments of inertia (kg*m^2)
IX = 0.00887
IY = 0.00887
IZ = 0.01718

# Motor and rotor parameters
THRUST_COEFF = 1.48e-5      # N/(rad/s)^2
DRAG_COEFF = 2.81e-7        # N*m/(rad/s)^2
ROTOR_INERTIA = 2.0e-5      # kg*m^2

# Battery parameters
BATTERY_VOLTAGE = 11.1      # V
BATTERY_CAPACITY = 5.0      # Ah
BATTERY_RESERVE = 0.20      # 20% reserve

# Preliminary electrical assumptions
MOTOR_EFFICIENCY = 0.72
AVIONICS_POWER = 5.0        # W


# 2. ROTATION MATRIX

def rotation_matrix(roll, pitch, yaw):
    """
    Transforms vectors from the drone body frame
    to the inertial ground frame.

    Rotation order: Rz(yaw) * Ry(pitch) * Rx(roll)
    """

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,             cp*cr]
    ])

    return R


# 3. ROTOR THRUST AND BODY MOMENTS

def rotor_forces(rotor_speeds):
    """
    Calculates total thrust and body moments from the
    four rotor angular speeds.

    Rotor layout:
    1 = Front (+x), CCW
    2 = Left  (+y), CW
    3 = Rear  (-x), CCW
    4 = Right (-y), CW
    """

    speeds = np.maximum(rotor_speeds, 0.0)

    # Individual rotor thrusts
    forces = THRUST_COEFF * speeds**2

    F1, F2, F3, F4 = forces

    # Total upward thrust in the body frame
    total_thrust = F1 + F2 + F3 + F4

    # Roll moment
    torque_x = ARM_LENGTH * (F2 - F4)

    # Pitch moment
    torque_y = ARM_LENGTH * (F3 - F1)

    # Yaw moment
    torque_z = DRAG_COEFF * (
        -speeds[0]**2
        + speeds[1]**2
        - speeds[2]**2
        + speeds[3]**2
    )

    body_torques = np.array([
        torque_x,
        torque_y,
        torque_z
    ])

    return total_thrust, body_torques

# 4. HOVER EQUILIBRIUM

def hover_conditions():
    """
    Calculates the equilibrium rotor speed and motor torque
    needed for stationary, level hover.
    """

    # Each rotor supplies one quarter of the drone's weight
    thrust_per_rotor = MASS * GRAVITY / 4

    # F = k * omega^2
    hover_speed = np.sqrt(
        thrust_per_rotor / THRUST_COEFF
    )

    # Motor torque balances aerodynamic drag torque
    hover_torque = DRAG_COEFF * hover_speed**2

    return hover_speed, hover_torque

# 5. COMPLETE 16-STATE EQUATIONS OF MOTION

def drone_derivatives(time, state, motor_torques):
    """
    Calculates the time derivatives of all 16 drone states.

    This function is passed to scipy.integrate.solve_ivp.
    """

    # Position in ground coordinates
    x, y, z = state[0:3]

    # Velocity in ground coordinates
    vx, vy, vz = state[3:6]

    # Euler angles (radians)
    roll, pitch, yaw = state[6:9]

    # Body angular velocities (rad/s)
    p, q, r = state[9:12]

    # Individual rotor speeds (rad/s)
    rotor_speeds = state[12:16]

    # A. ROTOR THRUST AND MOMENTS

    total_thrust, body_torques = rotor_forces(
        rotor_speeds
    )

    torque_x, torque_y, torque_z = body_torques

    # B. TRANSLATIONAL DYNAMICS

    R = rotation_matrix(roll, pitch, yaw)

    # All thrust acts along the drone's body +z axis
    thrust_body = np.array([
        0.0,
        0.0,
        total_thrust
    ])

    # Transform thrust into ground coordinates
    thrust_ground = R @ thrust_body

    # Newton's second law
    acceleration = thrust_ground / MASS

    # Subtract gravity in the ground z direction
    acceleration[2] -= GRAVITY

    ax, ay, az = acceleration

    # C. EULER ANGLE KINEMATICS

    roll_rate = (
        p
        + q * np.sin(roll) * np.tan(pitch)
        + r * np.cos(roll) * np.tan(pitch)
    )

    pitch_rate = (
        q * np.cos(roll)
        - r * np.sin(roll)
    )

    yaw_rate = (
        q * np.sin(roll) / np.cos(pitch)
        + r * np.cos(roll) / np.cos(pitch)
    )

    # D. ROTATIONAL DYNAMICS
    
    # Euler's rigid-body equations
    p_rate = (
        torque_x - (IZ - IY) * q * r
    ) / IX

    q_rate = (
        torque_y - (IX - IZ) * p * r
    ) / IY

    r_rate = (
        torque_z - (IY - IX) * p * q
    ) / IZ

    # E. MOTOR AND ROTOR DYNAMICS

    # Jr * omega_dot = motor_torque - aerodynamic_torque
    rotor_acceleration = (
        motor_torques
        - DRAG_COEFF * np.maximum(rotor_speeds, 0.0)**2
    ) / ROTOR_INERTIA

    # F. COMPLETE STATE DERIVATIVE VECTOR

    derivatives = np.array([
        vx,
        vy,
        vz,
        ax,
        ay,
        az,
        roll_rate,
        pitch_rate,
        yaw_rate,
        p_rate,
        q_rate,
        r_rate,
        rotor_acceleration[0],
        rotor_acceleration[1],
        rotor_acceleration[2],
        rotor_acceleration[3]
    ])

    return derivatives

# 6. ELECTRICAL POWER ESTIMATE

def electrical_power(rotor_speeds):
    
    # Shaft power = b * omega^3

    speeds = np.maximum(rotor_speeds, 0.0)

    shaft_power = DRAG_COEFF * speeds**3

    total_shaft_power = np.sum(shaft_power)

    electrical_power_total = (
        total_shaft_power / MOTOR_EFFICIENCY
        + AVIONICS_POWER
    )

    return electrical_power_total
