/* Mission plan of the electric vehicle: reference height and vertical speed of the landing feet,
 * stepped once per control step. Same rules as rocketsim/hop/mission.py (Guidance). */
#ifndef HOP_MISSION_H
#define HOP_MISSION_H

typedef enum { HOP_PAD, HOP_ASCENT, HOP_HOVER, HOP_DESCENT, HOP_LANDING, HOP_LANDED, HOP_ABORT } hop_phase_t;

typedef struct {
    float target_altitude;    /* m */
    float hover_time;         /* s, required hold */
    float hover_margin;       /* s, the reference hovers this much longer */
    float climb_speed;        /* m/s */
    float climb_acceleration; /* m/s^2 */
    float descent_speed;      /* m/s */
    float final_height;       /* m, below this the reference slows to the final speed */
    float final_speed;        /* m/s */
} hop_mission_t;

typedef struct {
    hop_mission_t mission;
    hop_phase_t phase;
    float height;       /* reference height, m */
    float speed;        /* reference vertical speed, m/s */
    float phase_start;  /* s */
} hop_guidance_t;

void hop_guidance_init(hop_guidance_t *g, const hop_mission_t *mission);
void hop_guidance_start(hop_guidance_t *g, float t);
void hop_guidance_update(hop_guidance_t *g, float t, float dt);

#endif
