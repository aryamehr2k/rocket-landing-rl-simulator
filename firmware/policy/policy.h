/* Forward pass of the trained landing policy. Plain C99, no dependencies, no allocation.
 *
 * The weights, the input normalisation and the layer sizes live in policy_weights.h, which
 * scripts/export_policy.py writes from a training run. Call policy_forward once per plane per
 * control step with the observation fields in the order listed in policy_weights.h, already
 * divided by their scales; it returns the gimbal command in [-1, 1] and the ignite output.
 */
#ifndef ROCKET_POLICY_H
#define ROCKET_POLICY_H

#include "policy_weights.h"

#define POLICY_OK 0
#define POLICY_BAD_INPUT 1

/* Returns POLICY_BAD_INPUT and leaves out untouched when an input is not a finite number. */
int policy_forward(const float *observation, float *out);

#endif
