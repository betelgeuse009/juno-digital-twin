#!/usr/bin/env python3
from pytrinamic.connections import ConnectionManager
from pytrinamic.modules import TMCM1260
from time import time

start_deviation_error = 0

def unwrap_position_signed(current_position, previous_position, max_position=2**32):
    """
    Converts encoder reading into a continuous signed value, handling rollover.
    """
    if current_position >= max_position // 2:
        current_signed = current_position - max_position
    else:
        current_signed = current_position

    if previous_position >= max_position // 2:
        previous_signed = previous_position - max_position
    else:
        previous_signed = previous_position

    diff = current_signed - previous_signed
    if diff > max_position // 2:
        return previous_signed - (max_position - diff)
    elif diff < -max_position // 2:
        return previous_signed + (max_position + diff)
    else:
        return current_signed

class Stepper:
    """
    High-level controller for TMCM-1260 stepper motor over CAN or USB.
    Handles configuration, absolute positioning, braking, and encoder sync.
    """

    def __init__(self, interface, data_rate, module_id,
                 max_velocity, max_acc, MaxDeceleration, V1, A1, D1):
        global start_deviation_error

        # Setup communication interface
        if interface == "can":
            communication = "socketcan_tmcl"
        elif interface == "usb":
            communication = "usb_tmcl"
        else:
            raise ValueError(f"Unsupported interface: {interface}")

        # Connect to the module
        self.interface = ConnectionManager(
            f"--interface {communication} --port can1 --data-rate {data_rate}"
        ).connect()

        self.module = TMCM1260(self.interface, module_id=module_id)
        self.motor = self.module.motors[0]
        self.module_id = module_id

        # Configure motor motion parameters
        self.motor.set_axis_parameter(self.motor.AP.MaxVelocity, max_velocity)
        self.motor.set_axis_parameter(self.motor.AP.MaxAcceleration, max_acc)
        self.motor.set_axis_parameter(self.motor.AP.MaxDeceleration, MaxDeceleration)
        self.motor.set_axis_parameter(self.motor.AP.V1, V1)
        self.motor.set_axis_parameter(self.motor.AP.A1, A1)
        self.motor.set_axis_parameter(self.motor.AP.D1, D1)
        self.motor.set_axis_parameter(self.motor.AP.StartVelocity, 1000)
        self.motor.set_axis_parameter(self.motor.AP.StopVelocity, 1000)
        self.motor.set_axis_parameter(self.motor.AP.RampWaitTime, 0)
        self.motor.set_axis_parameter(self.motor.AP.MaxCurrent, 200)
        self.motor.set_axis_parameter(self.motor.AP.StandbyCurrent, 100)
        self.motor.set_axis_parameter(self.motor.AP.SG2Threshold, 11)
        self.motor.set_axis_parameter(self.motor.AP.SG2FilterEnable, 0)
        self.motor.set_axis_parameter(self.motor.AP.SmartEnergyStallVelocity, 0)
        self.motor.set_axis_parameter(self.motor.AP.SmartEnergyHysteresis, 15)
        self.motor.set_axis_parameter(self.motor.AP.SmartEnergyHysteresisStart, 0)
        self.motor.set_axis_parameter(self.motor.AP.SECUS, 1)
        self.motor.set_axis_parameter(self.motor.AP.SmartEnergyThresholdSpeed, 7999774)

        # Driver settings
        self.motor.drive_settings.boost_current = 0
        self.motor.drive_settings.microstep_resolution = self.motor.ENUM.MicrostepResolution256Microsteps

        # Initialize position tracking
        self.virtual_position = self.motor.get_actual_position()
        self.motor.set_axis_parameter(self.motor.AP.EncoderPosition, 0)
        self.effective_position = self.motor.get_axis_parameter(self.motor.AP.EncoderPosition)
        start_deviation_error = self.effective_position - self.virtual_position

        # Setup encoder unwrap tracking
        self.last_encoder_position = self.effective_position
        self.unwrapped_position = 0

        print(f"[Stepper] Initialized on port can1 id={module_id}")
        print(self.motor.drive_settings)

    def move_to_position(self, target: int):
        """
        Move to an absolute position and update internal virtual position tracker.
        """
        self.motor.move_to(target)
        self.virtual_position = target

    def update_effective_position(self):
        """
        Updates the internal effective position (encoder value).
        """
        self.effective_position = self.motor.get_axis_parameter(self.motor.AP.EncoderPosition)

    def update_unwrapped_position(self):
        """
        Returns continuous encoder position, unwrapped across rollovers.
        """
        current_encoder = self.motor.get_axis_parameter(self.motor.AP.EncoderPosition)
        self.unwrapped_position = unwrap_position_signed(current_encoder, self.last_encoder_position)
        self.last_encoder_position = current_encoder
        return self.unwrapped_position

    def brake(self):
        """
        Executes a timed braking maneuver:
        1. Moves back by 70,000 ticks from current encoder position.
        2. Monitors encoder during 5-second hold and reissues target if deviation exceeds 200 ticks.
        3. Returns to the original start position.
        """
        print("[Stepper] Starting timed brake...")

        retract_ticks = 70000
        brake_duration = 5.0  # seconds
        align_threshold = 200  # ticks

        # --- Step 1: Move to brake position ---
        start_pos = self.motor.get_axis_parameter(self.motor.AP.EncoderPosition)
        target_pos = start_pos - retract_ticks
        print(f"[Stepper] Moving from {start_pos} → {target_pos}")
        self.move_to_position(target_pos)

        while not self.motor.get_position_reached():
            pass

        # --- Step 2: Monitor alignment during brake ---
        print(f"[Stepper] Holding brake for {brake_duration:.1f} seconds with live re-alignment...")
        t_start = time()
        while (time() - t_start) < brake_duration:
            actual_pos = self.motor.get_axis_parameter(self.motor.AP.ActualPosition)
            encoder_pos = self.motor.get_axis_parameter(self.motor.AP.EncoderPosition)
            error = abs(actual_pos - encoder_pos)

            if error > align_threshold:
                print(f"[Stepper] Drift detected (Δ={error}). Reissuing move to {target_pos}")
                self.move_to_position(target_pos)

        # --- Step 3: Return to original position ---
        print(f"[Stepper] Returning to start position: {start_pos}")
        self.move_to_position(start_pos)

        while not self.motor.get_position_reached():
            pass

        print("[Stepper] Timed brake completed.")

    def move_stepper(self, step):
        """
        Moves the stepper to an absolute position.
        """
        print(f"[Stepper] Moving to position: {step}")
        self.motor.move_to(step)

    def disconnect_motor(self):
        """
        Stops the motor and closes the communication interface.
        """
        self.motor.stop()
        self.interface.close()
        print("[Stepper] Disconnected")

