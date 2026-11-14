"""CPU regression checks; run with the project's PyTorch/TensorDict interpreter."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "source/nav/nav/tasks/manager_based/nav/agents"
# Import the standalone policy package without starting Isaac Sim via nav.__init__.
spec = importlib.util.spec_from_file_location(
    "_nav_test_agents", AGENTS / "__init__.py", submodule_search_locations=[str(AGENTS)]
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
common = importlib.import_module(f"{spec.name}.common")
history_module = importlib.import_module(f"{spec.name}.history_buffer")
factory = importlib.import_module(f"{spec.name}.factory")
recurrent = importlib.import_module(f"{spec.name}.lstm_ppo")


def namespace(value):
    return SimpleNamespace(**{key: namespace(item) if isinstance(item, dict) else item for key, item in value.items()})


def config(algo="lstm_ppo"):
    return namespace(
        {
            "policy_type": algo,
            "feature_extractor": {
                "learning_rate": 0.0005,
                "history_length": 1 if algo == "lstm_ppo" else 6,
                "history_stride": 1 if algo == "lstm_ppo" else 6,
                "token_dim": 16,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
            },
            "recurrent": {"sequence_length": 3},
            "actor": {"learning_rate": 0.0005, "clip_ratio": 0.1, "action_limit": 2.0},
            "critic": {"learning_rate": 0.0005, "clip_ratio": 0.1, "value_loss_coefficient": 1.0},
            "entropy_loss_coefficient": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "max_grad_norm": 5.0,
            "training_frame_num": 5,
            "training_epoch_num": 1,
            "num_minibatches": 2,
        }
    )


def make_agent(algo="lstm_ppo"):
    return factory.make_policy(algo, config(algo), {"state": (9,), "goal": (4,), "depth": (1, 24, 32)}, 3, "cpu")


def script_functions(filename, names, scope):
    """Exercise the actual script functions while skipping CLI/AppLauncher startup."""
    tree = ast.parse((ROOT / "scripts" / filename).read_text())
    definitions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec"), scope)
    return scope


class FakeEnv:
    """Parallel auto-reset episodes with different horizons and terminal reasons."""

    def __init__(self):
        self.unwrapped = self
        self.num_envs, self.device, self.max_episode_length = 3, "cpu", 4
        self.horizons = torch.tensor([2, 3, 4])
        self.termination_manager = self
        self.reward_manager = self
        self.active_terms = ["success", "static_collision", "dynamic_collision", "out_of_bounds", "time_out"]
        self.reset()

    def observation(self):
        return {
            "state": self.steps[:, None].float().expand(-1, 9).clone(),
            "goal": torch.ones(3, 4),
            "depth": torch.rand(3, 1, 24, 32),
        }

    def reset(self):
        self.steps = torch.zeros(3, dtype=torch.long)
        self.done = torch.zeros(3, dtype=torch.bool)
        return self.observation()

    def step(self, action):
        assert action.shape == (3, 3) and action.abs().max() <= 2.0
        self.steps += 1
        self.done = self.steps >= self.horizons
        self.steps[self.done] = 0
        truncated = self.done & (torch.arange(3) == 2)
        return self.observation(), torch.ones(3), self.done & ~truncated, truncated, {}

    def get_term(self, name):
        env_index = {"static_collision": 0, "success": 1, "time_out": 2}.get(name, -1)
        return self.done & (torch.arange(3) == env_index)

    def get_term_cfg(self, name):
        raise ValueError("Fake environment has no reward component instrumentation")


class RecurrentPPOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.train = script_functions(
            "train.py",
            {
                "RolloutEnvStatistics",
                "RolloutRewardComponentStatistics",
                "EpisodeReturnTracker",
                "attach_history",
                "collect_ppo_rollout",
            },
            {"torch": torch, "TensorDict": TensorDict, "obs_to_tensordict": common.obs_to_tensordict},
        )

    def setUp(self):
        torch.manual_seed(7)

    def collect(self, agent):
        env = FakeEnv()
        obs = common.obs_to_tensordict(env.reset(), 3, "cpu")
        buffer = history_module.HistoryBuffer(
            3, "cpu", agent.cfg.feature_extractor.history_length, agent.cfg.feature_extractor.history_stride
        )
        buffer.append(obs["agents", "observation"])
        self.train["attach_history"](obs, buffer.window())
        tracker = self.train["EpisodeReturnTracker"](3, "cpu")
        return self.train["collect_ppo_rollout"](env, agent, obs, agent.cfg, tracker, buffer)

    def test_sequence_replay_padding_update_and_bootstrap(self):
        agent = make_agent()
        rollout, obs, _ = self.collect(agent)
        before = tuple(value.clone() for value in agent._memory)
        agent.bootstrap_value(obs)
        for actual, expected in zip(agent._memory, before):
            torch.testing.assert_close(actual, expected)
        transition = rollout["transitions"]
        for env_ids, starts in recurrent.make_recurrent_minibatches(3, 5, 3, 2, "cpu"):
            batch, inputs, memory, starts_mask, valid = agent._sequence_batch(rollout, env_ids, starts)
            with torch.no_grad():
                feature, _ = agent.feature_extractor.forward_sequence(inputs, memory, starts_mask)
                log_prob, _ = agent.actor.evaluate_action(feature[valid], batch["agents", "action_normalized"][valid])
                torch.testing.assert_close(log_prob, batch["sample_log_prob"][valid], atol=2e-5, rtol=2e-5)
            self.assertEqual(valid.sum().item(), sum(min(3, 5 - start) for start in starts.tolist()))
        old_weights = agent.feature_extractor.lstm.weight_ih_l0.detach().clone()
        info = agent.update(rollout)
        self.assertTrue(all(math.isfinite(value) for value in info.values()))
        self.assertFalse(torch.equal(old_weights, agent.feature_extractor.lstm.weight_ih_l0))
        for actual, expected in zip(agent._memory, before):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(transition["episode_start"][0, 2].item())
        self.assertEqual(rollout["history"]["sequence_action"][0, 2].abs().sum().item(), 0.0)

    def test_reset_is_per_environment(self):
        agent = make_agent()
        self.collect(agent)
        before = tuple(value.clone() for value in agent._memory)
        agent.reset_memory(torch.tensor([False, True, False]))
        for actual, expected in zip(agent._memory, before):
            self.assertEqual(actual[:, 1].abs().sum().item(), 0.0)
            torch.testing.assert_close(actual[:, [0, 2]], expected[:, [0, 2]])

    def test_sampler_covers_every_transition_once(self):
        for sequence_length in (1, 3, 8):
            covered = []
            for env_ids, starts in recurrent.make_recurrent_minibatches(3, 5, sequence_length, 32, "cpu"):
                for env_id, start in zip(env_ids.tolist(), starts.tolist()):
                    covered.extend((env_id, step) for step in range(start, min(start + sequence_length, 5)))
            self.assertEqual(sorted(covered), [(env_id, step) for env_id in range(3) for step in range(5)])

    def test_checkpoints_and_original_ppo(self):
        with tempfile.TemporaryDirectory() as directory:
            for algo in ("ppo", "lstm_ppo"):
                agent = make_agent(algo)
                rollout, _, _ = self.collect(agent)
                self.assertTrue(all(math.isfinite(value) for value in agent.update(rollout).values()))
                path = Path(directory) / f"{algo}.pt"
                agent.save(path)
                restored = make_agent(algo)
                restored.load(path)
                for key, value in agent.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[key])
                other = make_agent("ppo" if algo == "lstm_ppo" else "lstm_ppo")
                with self.assertRaises(ValueError):
                    other.load(path)
                if algo == "ppo":
                    data = torch.load(path, weights_only=True)
                    del data["policy_type"], data["agent_config"]
                    torch.save(data, path)
                    restored.load(path)

    def test_evaluation_uses_complete_parallel_episodes(self):
        scope = script_functions(
            "eval.py",
            {"get_term_mask", "classify_termination", "evaluate"},
            {
                "torch": torch,
                "HistoryBuffer": history_module.HistoryBuffer,
                "obs_to_tensordict": common.obs_to_tensordict,
            },
        )
        for stochastic in (False, True):
            records = scope["evaluate"](FakeEnv(), make_agent(), 2, 1 / 60, stochastic=stochastic)
            self.assertEqual(len(records), 6)
            for row in records:
                self.assertEqual(row["length_steps"], row["env_id"] + 2)
            self.assertEqual(
                {row["termination_reason"] for row in records}, {"static_collision", "success", "time_out"}
            )


if __name__ == "__main__":
    unittest.main()
