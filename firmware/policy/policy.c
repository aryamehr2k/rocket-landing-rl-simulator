#include "policy.h"

#include <math.h>

static float clampf(float value, float low, float high)
{
    return value < low ? low : (value > high ? high : value);
}

static float activate(float value)
{
    return POLICY_ACTIVATION_TANH ? tanhf(value) : (value > 0.0f ? value : 0.0f);
}

/* One dense layer: out[j] = sum_i in[i] * weights[j * in_size + i] + bias[j]. */
static void dense(const float *in, int in_size, const float *weights, const float *bias, int out_size, float *out)
{
    for (int j = 0; j < out_size; j++) {
        float sum = bias[j];
        const float *row = weights + j * in_size;
        for (int i = 0; i < in_size; i++) {
            sum += in[i] * row[i];
        }
        out[j] = sum;
    }
}

int policy_forward(const float *observation, float *out)
{
    float buffer_a[POLICY_MAX_WIDTH];
    float buffer_b[POLICY_MAX_WIDTH];
    float *current = buffer_a;
    float *next = buffer_b;

    for (int i = 0; i < POLICY_INPUT_SIZE; i++) {
        if (!isfinite(observation[i])) {
            return POLICY_BAD_INPUT;
        }
        float normalised = (observation[i] - POLICY_OBS_MEAN[i]) / POLICY_OBS_STD[i];
        current[i] = clampf(normalised, -POLICY_OBS_CLIP, POLICY_OBS_CLIP);
    }
    int width = POLICY_INPUT_SIZE;
    for (int layer = 0; layer < POLICY_HIDDEN_LAYERS; layer++) {
        int out_width = POLICY_LAYER_SIZES[layer + 1];
        dense(current, width, POLICY_WEIGHTS[layer], POLICY_BIASES[layer], out_width, next);
        for (int j = 0; j < out_width; j++) {
            next[j] = activate(next[j]);
        }
        float *swap = current;
        current = next;
        next = swap;
        width = out_width;
    }
    dense(current, width, POLICY_WEIGHTS[POLICY_HIDDEN_LAYERS], POLICY_BIASES[POLICY_HIDDEN_LAYERS], POLICY_OUTPUT_SIZE, next);
    for (int j = 0; j < POLICY_OUTPUT_SIZE; j++) {
        out[j] = clampf(next[j], POLICY_OUTPUT_LOW, POLICY_OUTPUT_HIGH);
    }
    return POLICY_OK;
}
