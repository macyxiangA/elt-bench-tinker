"""Scripted policy: replay fixed assistant messages through a real env, no model needed.

Used by the credential-free test and the official-task rollout script; only the
model is replaced, the env/tools/reward are the training code.
"""

from __future__ import annotations

import json

from tinker_cookbook.renderers.base import Message, ToolCall


# Reference solution for ELT-Bench `trains` (format with ns=<namespace>, open_roof='none').
TRAINS_TRANSFORM = """
CREATE TABLE {ns}.trains AS
SELECT t.id AS train_id,
       t.direction,
       CASE WHEN t.direction = 'east' AND COUNT(c.id) < 4 THEN 1 ELSE 0 END
         AS run_east_and_has_less_than_4_cars,
       CASE WHEN SUM(CASE WHEN c.len = 'long' THEN 1 ELSE 0 END) > 2 THEN 1 ELSE 0 END
         AS has_more_than_2_long_cars,
       MAX(CASE WHEN c.position = 3 AND c.sides = 'double' THEN 1 ELSE 0 END)
         AS has_double_sided_cars_in_3rd_position,
       MAX(CASE WHEN c.position = 1 AND c.shape = 'rectangle' THEN 1 ELSE 0 END)
         AS has_rectangular_shaped_head_cars,
       COALESCE(SUM(CASE WHEN c.roof = '{open_roof}' THEN 1 ELSE 0 END), 0)
         AS number_of_open_roof_car
FROM {ns}.train t LEFT JOIN {ns}.cars c ON c.train_id = t.id
GROUP BY t.id, t.direction
"""


def call(name: str, **args: str) -> Message:
    tc = ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)))
    return {"role": "assistant", "content": "", "tool_calls": [tc]}


class PromptOnlyRenderer:
    """Stands in for a tokenizer-backed renderer; only builds the prompt prefix."""

    def create_conversation_prefix_with_tools(self, tools, system_prompt: str = ""):
        return [{"role": "system", "content": system_prompt}]

    def get_stop_sequences(self) -> list[str]:
        return []


async def run_scripted(env, script: list[Message]) -> tuple[float, dict]:
    """Step `env` (from build_agent_tool_env) with `script`; return (reward, metrics)."""
    msg_env = env.message_env
    await msg_env.initial_observation()
    for message in script:
        result = await msg_env.step(message)
        if result.episode_done:
            return result.reward, result.metrics
    raise AssertionError("script ended without finishing the episode")
