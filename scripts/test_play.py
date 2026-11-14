"""CPU checks for play retries, terminal state preservation, and saved metrics."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import play
import test_recurrent_ppo as support
import torch


class FakeSim:
    def __init__(self):
        self.playing = True
        self.pause_calls = 0
        self.render_calls = 0

    def pause(self):
        self.playing = False
        self.pause_calls += 1

    def play(self):
        self.playing = True

    def is_playing(self):
        return self.playing

    def render(self):
        self.render_calls += 1


class FakeEnv:
    def __init__(self, outcomes=("static_collision", "time_out", "success")):
        self.unwrapped = self
        self.device = "cpu"
        self.step_dt = 0.1
        self.max_episode_length = 3
        self.outcomes = outcomes
        self.attempts = 0
        self.reset_calls = 0
        self.sim = FakeSim()
        self.robot = SimpleNamespace(
            find_bodies=lambda *args, **kwargs: ([0], ["body"]),
            data=SimpleNamespace(body_link_pos_w=torch.zeros(1, 1, 3)),
        )
        self.scene = {"robot": self.robot}
        self.command_manager = SimpleNamespace(get_command=lambda name: torch.tensor([[3.0, 6.0, 6.0]]))
        self.termination_manager = SimpleNamespace(
            active_terms=["static_collision", "success", "time_out"],
            get_term=lambda name: torch.tensor([self.done and name == self.outcomes[self.attempts - 1]]),
        )

    def observation(self):
        return {
            "policy": {
                "state": torch.zeros(1, 9),
                "goal": torch.ones(1, 4),
                "depth": torch.ones(1, 1, 24, 32),
            }
        }

    def _reset_idx(self, env_ids):
        self.reset_calls += 1
        self.robot.data.body_link_pos_w.fill_(999.0)

    def reset(self):
        self._reset_idx(torch.tensor([0]))
        self.attempts += 1
        self.steps, self.done = 0, False
        self.robot.data.body_link_pos_w.zero_()
        return self.observation(), {}

    def step(self, action):
        assert action.shape == (1, 3) and torch.isfinite(action).all()
        self.steps += 1
        self.robot.data.body_link_pos_w[0, 0] = torch.tensor([1.0, 2.0, 2.0]) * self.steps
        self.done = self.steps == 3
        if self.done:
            self._reset_idx(torch.tensor([0]))
        truncated = self.done and self.outcomes[self.attempts - 1] == "time_out"
        return (
            self.observation(),
            torch.ones(1),
            torch.tensor([self.done and not truncated]),
            torch.tensor([truncated]),
            {},
        )


class PlayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.aliases = patch.dict(
            "sys.modules",
            {
                "nav.tasks.manager_based.nav.agents.common": support.common,
                "nav.tasks.manager_based.nav.agents.history_buffer": support.history_module,
            },
        )
        self.aliases.start()
        self.addCleanup(self.aliases.stop)
        self.app = SimpleNamespace(is_running=lambda: True)

    def test_terminal_position_metrics_and_recurrent_reset(self):
        for algo in ("lstm_ppo", "ppo"):
            agent = support.make_agent(algo)
            agent.eval()
            env = FakeEnv()
            for stochastic in (False, True):
                points, summary = play.collect_episode(env, agent, self.app, stochastic=stochastic)
                self.assertEqual(len(points), 4)
                self.assertEqual(points[-1], (3.0, 6.0, 6.0))
                self.assertEqual(summary["path_length_m"], 9.0)
                self.assertAlmostEqual(summary["path_length_xy_m"], 3 * math.sqrt(5))
                self.assertAlmostEqual(summary["duration_seconds"], 0.3)
                self.assertTrue(summary["terminal_position_recorded"])
                self.assertNotIn("_reset_idx", env.__dict__)
                if agent.is_recurrent:
                    self.assertTrue(all(torch.count_nonzero(v).item() == 0 for v in agent._memory))
                    self.assertTrue(agent._episode_start.all())
            self.assertEqual(env.reset_calls, 2)

    def test_retry_fail_fail_success_and_saved_attempts(self):
        env = FakeEnv()
        options = SimpleNamespace(episodes=None, max_attempts=0, stochastic=False)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = play.run_playback(env, support.make_agent(), self.app, output, options, {})
            self.assertEqual(result["attempted_episodes"], 3)
            self.assertEqual(result["completed_episodes"], 3)
            self.assertEqual(result["failed_episodes"], 2)
            self.assertEqual(result["success_episode"], 3)
            self.assertEqual(result["success_path_length_m"], 9.0)
            self.assertAlmostEqual(result["success_duration_seconds"], 0.3)
            self.assertEqual(json.loads((output / "summary.json").read_text()), result)
            with (output / "attempts.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["outcome"] for row in rows], list(env.outcomes))
            self.assertEqual(json.loads((output / "trajectory.json").read_text())["episode"], 3)
            with (output / "trajectory.csv").open() as file:
                points = list(csv.DictReader(file))
            self.assertEqual(len(points), 4)
            self.assertEqual(float(points[-1]["x"]), 3.0)
            self.assertTrue((output / "trajectory.svg").is_file())
        self.assertFalse(env.sim.is_playing())
        self.assertEqual(env.reset_calls, 3)
        self.assertEqual(env.robot.data.body_link_pos_w[0, 0].tolist(), [3.0, 6.0, 6.0])

    def test_optional_attempt_limit_does_not_report_success(self):
        options = SimpleNamespace(episodes=None, max_attempts=1, stochastic=False)
        with tempfile.TemporaryDirectory() as tmp:
            result = play.run_playback(FakeEnv(), support.make_agent(), self.app, Path(tmp), options, {})
            self.assertEqual(result["outcome"], "max_attempts_reached")
            self.assertEqual(result["attempted_episodes"], 1)
            self.assertIsNone(result["success_path_length_m"])
            self.assertFalse((Path(tmp) / "trajectory.json").exists())

    def test_reset_override_is_restored_on_exception(self):
        env = FakeEnv()
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with play.defer_episode_reset(env, lambda: (1, 2, 3)):
                raise RuntimeError("test exception")
        self.assertNotIn("_reset_idx", env.__dict__)
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(env.reset_calls, 1)

    def test_hold_renders_paused_scene_without_stepping(self):
        env = FakeEnv()
        app = SimpleNamespace(is_running=lambda: env.sim.render_calls < 3)
        play.hold_success(env, app)
        self.assertEqual(env.sim.render_calls, 3)
        self.assertFalse(env.sim.is_playing())
        self.assertEqual(env.reset_calls, 0)


if __name__ == "__main__":
    unittest.main()
