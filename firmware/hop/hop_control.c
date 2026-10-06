#include "hop_control.h"

#include <math.h>

#include "../policy/policy.h"
#include "policy_config.h"

#define MIN_COS_TILT 0.5f
#define TIME_TOLERANCE 1e-6f
#define PLANES 2
#define OUT_GIMBAL 0
#define OUT_THROTTLE 1

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static int flying(hop_phase_t phase)
{
    return phase == HOP_ASCENT || phase == HOP_HOVER || phase == HOP_DESCENT || phase == HOP_LANDING;
}

void hop_controller_init(hop_controller_t *c, const hop_mission_t *mission, float launch_time)
{
    hop_command_t zero = {0.0f, 0.0f, 0.0f, 0.0f, 0, 0};
    hop_guidance_init(&c->guidance, mission);
    c->launch_time = launch_time;
    c->last_gimbal[0] = c->last_gimbal[1] = 0.0f;
    c->att_integral[0] = c->att_integral[1] = 0.0f;
    c->speed_integral = 0.0f;
    c->aborted = 0;
    c->last = zero;
}

void hop_controller_set_launch_time(hop_controller_t *c, float launch_time)
{
    c->launch_time = launch_time;
}

static float field_value(int id, const hop_controller_t *c, const hop_estimate_t *e, int plane)
{
    const hop_guidance_t *g = &c->guidance;
    switch (id) {
    case HOP_FIELD_HEIGHT_ERROR: return g->height - e->height;
    case HOP_FIELD_VERTICAL_SPEED_ERROR: return g->speed - e->vertical_speed;
    case HOP_FIELD_REFERENCE_SPEED: return g->speed;
    case HOP_FIELD_HEIGHT: return e->height;
    case HOP_FIELD_VERTICAL_SPEED: return e->vertical_speed;
    case HOP_FIELD_LATERAL_POSITION: return e->position[plane];
    case HOP_FIELD_LATERAL_SPEED: return e->velocity[plane];
    case HOP_FIELD_TILT: return e->tilt[plane];
    case HOP_FIELD_TILT_RATE: return e->tilt_rate[plane];
    case HOP_FIELD_ROLL: return e->roll;
    case HOP_FIELD_GIMBAL: return c->last_gimbal[plane];
    case HOP_FIELD_THROTTLE: return c->last.throttle;
    case HOP_FIELD_PHASE_ASCENT: return g->phase == HOP_ASCENT ? 1.0f : 0.0f;
    case HOP_FIELD_PHASE_HOVER: return g->phase == HOP_HOVER ? 1.0f : 0.0f;
    case HOP_FIELD_PHASE_DESCENT: return g->phase == HOP_DESCENT ? 1.0f : 0.0f;
    case HOP_FIELD_PHASE_LANDING: return g->phase == HOP_LANDING ? 1.0f : 0.0f;
    default: return NAN;
    }
}

/* Runs the network for one plane; returns 0 and fills out[2] when inputs and outputs are finite. */
static int network(const hop_controller_t *c, const hop_estimate_t *e, int plane, float *out)
{
    float obs[POLICY_INPUT_SIZE];
    for (int i = 0; i < POLICY_INPUT_SIZE; i++) {
        obs[i] = field_value(POLICY_FIELD_IDS[i], c, e, plane) / POLICY_FIELD_SCALES[i];
    }
    if (policy_forward(obs, out) != POLICY_OK) {
        return 1;
    }
    return (isfinite(out[OUT_GIMBAL]) && isfinite(out[OUT_THROTTLE])) ? 0 : 1;
}

static float plane_pid(hop_controller_t *c, const hop_vehicle_t *v, const hop_estimate_t *e, int plane)
{
    float wanted = -(v->pos_kp * e->position[plane] + v->pos_kd * e->velocity[plane]);
    float tilt_command = clampf(wanted, -v->max_tilt_command, v->max_tilt_command);
    float error = tilt_command - e->tilt[plane];
    if (v->att_ki > 0.0f) {
        float windup = v->att_max_integral / v->att_ki;
        c->att_integral[plane] = clampf(c->att_integral[plane] + error * v->control_dt, -windup, windup);
    }
    return v->att_kp * error + v->att_ki * c->att_integral[plane] - v->att_kd * e->tilt_rate[plane];
}

static float altitude_pid(hop_controller_t *c, const hop_vehicle_t *v, const hop_estimate_t *e)
{
    const hop_guidance_t *g = &c->guidance;
    float wanted = g->speed;
    if (g->phase != HOP_LANDING) {
        float correction = v->alt_height_gain * (g->height - e->height);
        wanted += clampf(correction, -v->alt_max_speed_correction, v->alt_max_speed_correction);
    }
    float error = wanted - e->vertical_speed;
    if (v->alt_integral_gain > 0.0f) {
        float limit = v->alt_max_integral_accel / v->alt_integral_gain;
        c->speed_integral = clampf(c->speed_integral + error * v->control_dt, -limit, limit);
    }
    float accel = v->alt_speed_gain * error + v->alt_integral_gain * c->speed_integral;
    float cos_tilt = cosf(e->total_tilt);
    return v->hover_throttle * (1.0f + accel / v->gravity) / (cos_tilt > MIN_COS_TILT ? cos_tilt : MIN_COS_TILT);
}

static float limit(float value, float previous, float rate_step, float lo, float hi)
{
    value = clampf(value, lo, hi);
    return clampf(value, previous - rate_step, previous + rate_step);
}

hop_command_t hop_control_step(hop_controller_t *c, const hop_vehicle_t *v, const hop_estimate_t *e, float t)
{
    hop_command_t command = {0.0f, 0.0f, 0.0f, 0.0f, 0, 0};
    float gimbal[PLANES];
    int bad[PLANES];
    float votes = 0.0f;
    int voters = 0;
    if (c->guidance.phase == HOP_PAD && t + TIME_TOLERANCE >= c->launch_time) {
        hop_guidance_start(&c->guidance, t);
    }
    /* The network sees this step's estimate and the reference of the previous step. */
    for (int p = 0; p < PLANES; p++) {
        float out[2];
        bad[p] = network(c, e, p, out);
        if (!bad[p]) {
            gimbal[p] = clampf(out[OUT_GIMBAL], -1.0f, 1.0f) * (POLICY_RESIDUAL ? POLICY_RESIDUAL_GIMBAL : v->max_gimbal);
            c->last_gimbal[p] = gimbal[p];
            votes += clampf(out[OUT_THROTTLE], -1.0f, 1.0f);
            voters++;
        }
    }
    int launched = c->guidance.phase != HOP_PAD;
    hop_guidance_update(&c->guidance, t, v->control_dt);
    if (!flying(c->guidance.phase) || c->aborted) {
        command.armed = launched && !c->aborted;
        c->last = command;
        return command;
    }
    if (e->total_tilt > v->abort_tilt || hypotf(e->position[0], e->position[1]) > v->geofence_radius) {
        c->aborted = 1;
        c->last = command;
        return command;
    }
    float delta[PLANES];
    for (int p = 0; p < PLANES; p++) {
        delta[p] = plane_pid(c, v, e, p);
        if (bad[p]) {
            command.fallback++;
        } else if (v->steering_policy) {
            delta[p] = POLICY_RESIDUAL ? delta[p] + gimbal[p] : gimbal[p];
        }
    }
    float throttle = altitude_pid(c, v, e);
    if (v->throttle_policy && voters > 0) {
        float wanted = votes / (float)voters * POLICY_THROTTLE_RANGE;
        if (POLICY_RESIDUAL) {
            throttle += wanted;
        } else {
            float cos_tilt = cosf(e->total_tilt);
            throttle = wanted + v->hover_throttle / (cos_tilt > MIN_COS_TILT ? cos_tilt : MIN_COS_TILT);
        }
    }
    /* World-plane commands to servo commands: rotate by minus the roll angle. */
    float cr = cosf(-e->roll), sr = sinf(-e->roll);
    float pitch = cr * delta[0] - sr * delta[1];
    float yaw = sr * delta[0] + cr * delta[1];
    float gimbal_step = v->max_gimbal_rate * v->control_dt;
    command.gimbal_pitch = limit(pitch, c->last.gimbal_pitch, gimbal_step, -v->max_gimbal, v->max_gimbal);
    command.gimbal_yaw = limit(yaw, c->last.gimbal_yaw, gimbal_step, -v->max_gimbal, v->max_gimbal);
    command.throttle = limit(throttle, c->last.throttle, v->max_throttle_rate * v->control_dt, 0.0f, 1.0f);
    command.roll_torque = -(v->roll_kp * e->roll + v->roll_kd * e->roll_rate);
    command.armed = 1;
    c->last = command;
    return command;
}
