#include "hop_mission.h"

#include <math.h>

#define ARRIVAL_DISTANCE 0.05f
#define TIME_TOLERANCE 1e-6f /* s; times summed from many steps are not exact */

static float minf(float a, float b) { return a < b ? a : b; }
static float maxf(float a, float b) { return a > b ? a : b; }

static void enter(hop_guidance_t *g, hop_phase_t phase, float t)
{
    g->phase = phase;
    g->phase_start = t;
}

void hop_guidance_init(hop_guidance_t *g, const hop_mission_t *mission)
{
    g->mission = *mission;
    g->phase = HOP_PAD;
    g->height = 0.0f;
    g->speed = 0.0f;
    g->phase_start = 0.0f;
}

void hop_guidance_start(hop_guidance_t *g, float t)
{
    enter(g, HOP_ASCENT, t);
}

void hop_guidance_update(hop_guidance_t *g, float t, float dt)
{
    const hop_mission_t *m = &g->mission;
    if (g->phase == HOP_ASCENT) {
        float room = maxf(m->target_altitude - g->height, 0.0f);
        float braking = sqrtf(2.0f * m->climb_acceleration * room);
        g->speed = minf(minf(m->climb_speed, g->speed + m->climb_acceleration * dt), braking);
        g->height += g->speed * dt;
        if (m->target_altitude - g->height <= ARRIVAL_DISTANCE) {
            g->height = m->target_altitude;
            g->speed = 0.0f;
            enter(g, HOP_HOVER, t);
        }
    } else if (g->phase == HOP_HOVER) {
        if (t - g->phase_start + TIME_TOLERANCE >= m->hover_time + m->hover_margin) {
            enter(g, HOP_DESCENT, t);
        }
    } else if (g->phase == HOP_DESCENT) {
        g->speed = maxf(-m->descent_speed, g->speed - m->climb_acceleration * dt);
        g->height += g->speed * dt;
        if (g->height <= m->final_height) {
            enter(g, HOP_LANDING, t);
        }
    } else if (g->phase == HOP_LANDING) {
        g->speed = minf(-m->final_speed, g->speed + m->climb_acceleration * dt);
        g->height = maxf(0.0f, g->height + g->speed * dt);
    }
}
