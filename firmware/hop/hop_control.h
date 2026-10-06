/* One control step of the electric vehicle: trained network or PID, mission plan, safety limits.
 *
 * Call hop_control_step once per control period (1 / control_rate_hz) with the board's own state
 * estimate. It follows rocketsim/hop/computer.py: at the launch time the mission plan starts (end
 * your estimator's pad calibration at the same moment), the network sees the state and the
 * reference from the previous step, the reference is then advanced, the PID runs, the network's commands replace
 * (direct mode) or correct (residual mode) the PID's, and everything passes the safety limits.
 * A network output or input that is not a finite number falls back to the PID for that step.
 * The returned command is meant for the NEXT control period (one step of delay on the board).
 */
#ifndef HOP_CONTROL_H
#define HOP_CONTROL_H

#include "hop_mission.h"

typedef struct {
    float height;          /* landing feet above the pad, m */
    float vertical_speed;  /* m/s, positive up */
    float position[2];     /* x, y from the pad, m (world frame: x, y horizontal, z up) */
    float velocity[2];     /* vx, vy, m/s */
    float tilt[2];         /* lean of the nose toward +x and toward +y, rad */
    float tilt_rate[2];    /* rad/s */
    float total_tilt;      /* angle between the body axis and vertical, rad */
    float roll;            /* rad, right hand about the body axis */
    float roll_rate;       /* body rate about the body axis, rad/s */
} hop_estimate_t;

typedef struct {
    float control_dt;           /* s */
    float hover_throttle;       /* mass x g / max thrust */
    float max_gimbal;           /* rad */
    float max_gimbal_rate;      /* rad/s */
    float max_throttle_rate;    /* 1/s */
    float abort_tilt;           /* rad */
    float geofence_radius;      /* m */
    float gravity;              /* m/s^2 */
    /* attitude PID (per plane): tilt error to gimbal */
    float att_kp, att_ki, att_kd, att_max_integral;
    /* position loop: lateral position and speed to a tilt command */
    float pos_kp, pos_kd, max_tilt_command;
    /* altitude PID: height error to speed, speed error to acceleration */
    float alt_height_gain, alt_max_speed_correction, alt_speed_gain, alt_integral_gain, alt_max_integral_accel;
    /* roll PD */
    float roll_kp, roll_kd;
    int steering_policy;        /* 1: the network steers, 0: the PID */
    int throttle_policy;        /* 1: the network sets the throttle, 0: the PID */
} hop_vehicle_t;

typedef struct {
    float gimbal_pitch;  /* servo command, rad */
    float gimbal_yaw;    /* servo command, rad */
    float throttle;      /* 0 to 1 */
    float roll_torque;   /* N m */
    int armed;           /* 1 while the motor may run */
    int fallback;        /* number of planes the PID took over this step */
} hop_command_t;

typedef struct {
    hop_guidance_t guidance;
    float launch_time;         /* s, the mission starts on the first step at or after this time */
    float last_gimbal[2];      /* the network's previous world-plane gimbal commands, rad */
    float att_integral[2];
    float speed_integral;
    int aborted;
    hop_command_t last;        /* the command that went out last step */
} hop_controller_t;

/* launch_time: when the mission starts, after the pad calibration (set it far ahead and call
 * hop_controller_set_launch_time later to launch on a ground command). */
void hop_controller_init(hop_controller_t *c, const hop_mission_t *mission, float launch_time);
void hop_controller_set_launch_time(hop_controller_t *c, float launch_time);
hop_command_t hop_control_step(hop_controller_t *c, const hop_vehicle_t *v, const hop_estimate_t *est, float t);

#endif
