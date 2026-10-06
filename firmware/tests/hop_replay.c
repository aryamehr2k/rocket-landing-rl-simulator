/* Desktop harness: replays recorded estimates through hop_control_step and prints the commands.
 * Input lines: t height vz x y vx vy tilt_x tilt_y rate_x rate_y total_tilt roll roll_rate.
 * Output lines: pitch yaw throttle roll_torque armed fallback.
 * The mission, vehicle and model come from hop_params.h and policy_config.h generated with the model. */
#include <stdio.h>

#include "../hop/hop_control.h"
#include "hop_params.h"

#define FIELDS 14

int main(void)
{
    hop_mission_t mission = HOP_MISSION_INIT;
    hop_vehicle_t vehicle = HOP_VEHICLE_INIT;
    hop_controller_t controller;
    hop_controller_init(&controller, &mission, HOP_LAUNCH_TIME);
    float t;
    hop_estimate_t e;
    while (scanf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f", &t, &e.height, &e.vertical_speed,
                 &e.position[0], &e.position[1], &e.velocity[0], &e.velocity[1], &e.tilt[0], &e.tilt[1],
                 &e.tilt_rate[0], &e.tilt_rate[1], &e.total_tilt, &e.roll, &e.roll_rate) == FIELDS) {
        hop_command_t c = hop_control_step(&controller, &vehicle, &e, t);
        printf("%.9g %.9g %.9g %.9g %d %d\n", c.gimbal_pitch, c.gimbal_yaw, c.throttle, c.roll_torque, c.armed, c.fallback);
    }
    return 0;
}
