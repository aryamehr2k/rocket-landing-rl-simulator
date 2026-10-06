# Desktop build of the flight controller code and its checks.
#   make            builds the replay and policy checks
#   make check      replays a recorded flight (replay.txt, one estimate per line) through the controller
# The model headers (policy_weights.h, policy_config.h, hop_params.h) come from a model folder:
#   python scripts/export_policy.py runs/<run> --install

CC ?= gcc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra
INCLUDES = -Ipolicy -Ihop

all: build/hop_replay build/policy_check

build:
	mkdir -p build

build/hop_replay: tests/hop_replay.c hop/hop_control.c hop/hop_mission.c policy/policy.c | build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ -lm

build/policy_check: tests/policy_check.c policy/policy.c | build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ -lm

check: build/hop_replay
	./build/hop_replay < replay.txt

clean:
	rm -rf build

.PHONY: all check clean
