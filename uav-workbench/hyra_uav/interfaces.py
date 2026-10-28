"""Engineering seams for shared records and a future trainer; no policy training."""
from dataclasses import dataclass
from .core import Config, Memory, Pose, Task


def merge_arrived(memory: Memory, observations, now):
    """At a common clock, ties prefer the smallest agent ID then first input.

    Process all observations through Memory's permission checks. Equal-time
    conflicts are deterministic regardless of agent iteration order.
    """
    ordered = sorted(enumerate(observations), key=lambda pair: (
        pair[1]["arrived_s"], pair[1]["observed_s"], pair[1].get("agent_id", 0), pair[0]))
    for _, observation in ordered:
        memory.accept(observation, now)
    return memory


@dataclass(frozen=True)
class RewardConfig:
    distance_per_m: float = 0.001
    turn_per_s: float = 0.001


class TaskEnv:
    """reset/step returns legal policy views. Evaluator truth stays in Task.

    Reward is the change in current map quality minus actual motion costs.
    This is an explicit engineering default, not a frozen P4 learning objective.
    """
    def __init__(self, process, config=Config(), pose=Pose(0, 0), reward_config=RewardConfig()):
        self.process, self.config, self.pose = process, config, pose
        self.reward_config = reward_config
        self.task = None

    def reset(self, seed=None, options=None):
        if options:
            raise ValueError("本接口暂不接受隐式环境覆盖参数")
        self.task = Task(self.process, self.config, self.pose)
        return self.task.view(), dict(seed=seed, time_s=0.0, source_sha256=self.process.digest)

    def step(self, action):
        if self.task is None:
            raise ValueError("请先reset")
        previous = self.task.snapshots[-1]
        view, done = self.task.step(action)
        current = self.task.snapshots[-1]
        reward = (current["metrics"]["correct_coverage"]-previous["metrics"]["correct_coverage"]
                  - self.reward_config.distance_per_m*(current["distance_m"]-previous["distance_m"])
                  - self.reward_config.turn_per_s*(current["turn_s"]-previous["turn_s"]))
        return view, float(reward), False, done, dict(time_s=view.time_s)
