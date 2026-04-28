from __future__ import annotations
import argparse
import json
import numpy as np
from phase2_rl.landing_env import Phase2RLConfig, RocketLandingEnv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Deterministic environment seed.")
    parser.add_argument("--steps", type=int, default=10, help="Number of random-action steps to run.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    env = RocketLandingEnv(Phase2RLConfig(), seed=args.seed)
    observation, info = env.reset(seed=args.seed)
    rewards: list[float] = []
    terminated = False
    truncated = False
    last_info = info
    for _ in range(args.steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, last_info = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    summary = {
        "initial_info": info,
        "observation_shape": list(observation.shape),
        "steps_executed": len(rewards),
        "reward_sum": float(np.sum(rewards)) if rewards else 0.0,
        "terminated": terminated,
        "truncated": truncated,
        "last_info": last_info,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
