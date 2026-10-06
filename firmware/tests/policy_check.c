/* Desktop harness: reads observations from stdin (one per line, POLICY_INPUT_SIZE floats),
 * prints the policy outputs (gimbal, ignite) per line. scripts/export_policy.py uses it to
 * compare the C forward pass with the Python one. */
#include <stdio.h>

#include "../policy/policy.h"

int main(void)
{
    float observation[POLICY_INPUT_SIZE];
    float out[POLICY_OUTPUT_SIZE];
    while (1) {
        for (int i = 0; i < POLICY_INPUT_SIZE; i++) {
            if (scanf("%f", &observation[i]) != 1) {
                return 0;
            }
        }
        if (policy_forward(observation, out) != POLICY_OK) {
            for (int j = 0; j < POLICY_OUTPUT_SIZE; j++) {
                printf("nan%s", j + 1 < POLICY_OUTPUT_SIZE ? " " : "\n");
            }
            continue;
        }
        for (int j = 0; j < POLICY_OUTPUT_SIZE; j++) {
            printf("%.9g%s", out[j], j + 1 < POLICY_OUTPUT_SIZE ? " " : "\n");
        }
    }
}
