/* Inputs the network of the electric vehicle may use, by name. The order the network expects is
 * listed in policy_config.h (POLICY_FIELD_IDS), generated with the model. */
#ifndef HOP_FIELDS_H
#define HOP_FIELDS_H

typedef enum {
    HOP_FIELD_HEIGHT_ERROR,
    HOP_FIELD_VERTICAL_SPEED_ERROR,
    HOP_FIELD_REFERENCE_SPEED,
    HOP_FIELD_HEIGHT,
    HOP_FIELD_VERTICAL_SPEED,
    HOP_FIELD_LATERAL_POSITION,
    HOP_FIELD_LATERAL_SPEED,
    HOP_FIELD_TILT,
    HOP_FIELD_TILT_RATE,
    HOP_FIELD_ROLL,
    HOP_FIELD_GIMBAL,
    HOP_FIELD_THROTTLE,
    HOP_FIELD_PHASE_ASCENT,
    HOP_FIELD_PHASE_HOVER,
    HOP_FIELD_PHASE_DESCENT,
    HOP_FIELD_PHASE_LANDING
} hop_field_t;

#endif
