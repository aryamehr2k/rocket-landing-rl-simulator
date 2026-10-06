/* Generated from electric_hopper.yaml and hop_50m.yaml. All angles in radians. */
#ifndef HOP_PARAMS_H
#define HOP_PARAMS_H

#define HOP_LAUNCH_TIME 2.0f
#define HOP_MISSION_INIT {.target_altitude = 50.0f, .hover_time = 10.0f, .hover_margin = 2.0f, .climb_speed = 4.0f, .climb_acceleration = 1.5f, .descent_speed = 3.0f, .final_height = 4.0f, .final_speed = 0.7f}
#define HOP_VEHICLE_INIT {.control_dt = 0.02f, .hover_throttle = 0.562592026f, .max_gimbal = 0.174532925f, .max_gimbal_rate = 2.61799388f, .max_throttle_rate = 2.0f, .abort_tilt = 0.698131701f, .geofence_radius = 40.0f, .gravity = 9.80665f, .att_kp = 0.6f, .att_ki = 0.0f, .att_kd = 0.12f, .att_max_integral = 0.0523598776f, .pos_kp = 0.034906585f, .pos_kd = 0.0698131701f, .max_tilt_command = 0.261799388f, .alt_height_gain = 1.0f, .alt_max_speed_correction = 2.0f, .alt_speed_gain = 2.0f, .alt_integral_gain = 0.5f, .alt_max_integral_accel = 2.0f, .roll_kp = 0.5f, .roll_kd = 0.08f, .steering_policy = 1, .throttle_policy = 1}

#endif
