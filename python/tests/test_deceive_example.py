"""Exercise the guessing-game example through the real bindings and mock HTTP."""

import asyncio
import io
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from smolgent import Agent
from test_agent import answer, call, server


play = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "deceive_test.py"))["play"]


class DeceiveExampleTests(unittest.TestCase):
    def run_game(self, url, *, max_turns=0, first=0):
        deepseek = Agent.deepseek
        agents = []

        def local_agent(*args, **kwargs):
            agent = deepseek(*args, base_url=url, api_key="test", **kwargs)
            agents.append(agent)
            return agent

        async def scenario():
            winner = await asyncio.wait_for(play(max_turns=max_turns), timeout=10)
            # Cancellation reaches Rust asynchronously; allow its busy guard to
            # release before checking that no agent run was left behind.
            for agent in agents:
                for _ in range(100):
                    try:
                        agent.reset()
                        break
                    except RuntimeError:
                        await asyncio.sleep(0.01)
                else:
                    self.fail("cancelled game did not release the agent")
            return winner

        output = io.StringIO()
        with patch.object(Agent, "deepseek", side_effect=local_agent), \
                patch("random.randrange", return_value=first), redirect_stdout(output):
            winner = asyncio.run(scenario())
        return winner, output.getvalue()

    def test_first_verdict_scores_and_stops_without_followup_request(self):
        for first, identity, winner in [(0, "AI", "A"), (0, "human", "B"),
                                        (1, "AI", "B"), (1, "human", "A")]:
            with self.subTest(first=first, identity=identity), server(
                answer(None, tool_calls=[call("submit_verdict", {
                    "identity": identity, "reason": "My guess",
                })]),
            ) as (url, requests):
                result, output = self.run_game(url, first=first)
                self.assertEqual(result, winner)
                self.assertIn(f"{winner} wins!", output)
                self.assertEqual(len(requests), 1)
                schema = requests[0][2]["tools"][0]["function"]["parameters"]
                self.assertEqual(schema["properties"]["identity"]["enum"], ["human", "AI"])

    def test_conversation_relays_only_text_and_preserves_separate_histories(self):
        with server(
            answer("What do you enjoy?", reasoning_content="A private thought"),
            answer("Gardening. You?", reasoning_content="B private thought"),
            answer(None, tool_calls=[call("submit_verdict", {
                "identity": "AI", "reason": "An inference from our chat",
            })]),
        ) as (url, requests):
            winner, _ = self.run_game(url)
            self.assertEqual(winner, "A")
            self.assertEqual(len(requests), 3)
            second = requests[1][2]["messages"]
            self.assertEqual(len(second), 2)
            self.assertEqual(second[-1], {"role": "user", "content": "What do you enjoy?"})
            third = requests[2][2]["messages"]
            self.assertEqual(third[-1], {"role": "user", "content": "Gardening. You?"})
            self.assertEqual(third[-2]["reasoning_content"], "A private thought")
            self.assertNotIn("B private thought", str(third))

    def test_optional_turn_limit_has_no_winner(self):
        with server(answer("Hello"), answer("Hi there")) as (url, requests):
            winner, output = self.run_game(url, max_turns=2)
            self.assertIsNone(winner)
            self.assertIn("No verdict after 2 turns. No winner.", output)
            self.assertEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
