"""DeepSeek transport tests against the compiled bindings and a local HTTP server."""

import asyncio
import json
import os
import unittest
from unittest.mock import patch

from smolgent import Agent, File, Image, MultimodalResult, SmolgentError, tool
from test_agent import answer, call, server


class DeepSeekTests(unittest.TestCase):
    def test_defaults_credentials_and_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                Agent.deepseek()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}):
            with server(answer()) as (url, requests):
                agent = Agent.deepseek(base_url=url, compaction=False)
                self.assertEqual(agent.run("Hello").text, "Hello")
                path, headers, body = requests[0]
                self.assertEqual(path, "/chat/completions")
                self.assertEqual(headers["authorization"], "Bearer env-key")
                self.assertEqual(body["model"], "deepseek-flash")
                self.assertNotIn("reasoning", body)
                self.assertNotIn("thinking", body)
            Agent.deepseek(keyring_id="my-app/deepseek")
        for options in [
            {"thinking": "yes"}, {"reasoning_effort": "invalid"},
            {"thinking": False, "reasoning_effort": "high"},
            {"thinking": True, "reasoning_effort": "none"},
            {"api_key": ""}, {"api_key": "key", "keyring_id": "id"},
            {"base_url": "file:///tmp"},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                Agent.deepseek(**({"api_key": "key"} | options))
        with self.assertRaisesRegex(ValueError, "DeepSeek options"):
            Agent.openrouter("model", api_key="key", thinking=True)

    def test_thinking_and_prefixed_urls(self):
        for options, expected in [
            ({"thinking": False}, {"thinking": {"type": "disabled"}}),
            ({"thinking": True, "reasoning_effort": "max"},
             {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}),
            ({"reasoning_effort": "low"}, {"reasoning_effort": "low"}),
            ({"reasoning_effort": "none"}, {"reasoning_effort": "none"}),
        ]:
            with self.subTest(options=options), server(answer()) as (url, requests):
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ignored"}):
                    agent = Agent.deepseek("custom-model", base_url=url + "/proxy/v1/",
                                           api_key="explicit", compaction=False, **options)
                agent.run("Hi")
                path, headers, body = requests[0]
                self.assertEqual(path, "/proxy/v1/chat/completions")
                self.assertEqual(headers["authorization"], "Bearer explicit")
                self.assertEqual(body.pop("model"), "custom-model")
                body.pop("messages")
                self.assertEqual(body, expected)

    def test_reasoning_replayed_across_tool_rounds_and_user_turns(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        @tool
        async def double(value: int) -> int:
            """Double a number."""
            return value * 2

        with server(
            answer(None, reasoning_content="Add first", tool_calls=[call("add", {"a": 2, "b": 3})]),
            answer("", reasoning_content="Then double", tool_calls=[call("double", {"value": 5}, "call_2")]),
            answer("10", reasoning_content="Finished", tool_calls=None),
            answer("Still 10", reasoning_content="Remembered", tool_calls=None),
        ) as (url, requests):
            agent = Agent.deepseek(base_url=url, api_key="test", tools=[add, double], compaction=False)
            self.assertEqual(asyncio.run(agent.arun("Add and double")).text, "10")
            response = agent.run("What was the answer?")
            self.assertEqual(response.reasoning["reasoning_content"], "Remembered")
            messages = requests[-1][2]["messages"]
            self.assertEqual([m["reasoning_content"] for m in messages if m["role"] == "assistant"],
                             ["Add first", "Then double", "Finished"])
            self.assertEqual([(m["tool_call_id"], m["content"]) for m in messages if m["role"] == "tool"],
                             [("call_1", "5"), ("call_2", "10")])
            self.assertEqual(len(requests[0][2]["tools"]), 2)
            self.assertEqual(agent.messages[2]["reasoning_content"], "Add first")

    def test_image_and_file_wire_format_preserves_history_and_tool_schema(self):
        picture = Image.from_url("https://example.com/image.png")
        attachment = File.from_bytes(b"image", mime_type="image/png", filename="image.png")

        @tool(parameters={"type": "object", "properties": {},
                          "description": json.dumps({"type": "file", "file": {"file_data": "schema"}})})
        def get_image():
            return MultimodalResult(["Image:", picture, attachment])

        with server(answer(None, reasoning_content="Inspect", tool_calls=[call("get_image", {})]),
                    answer("Done", tool_calls=None)) as (url, requests):
            agent = Agent.deepseek(base_url=url, api_key="key", tools=[get_image], compaction=False)
            agent.run(["Describe", picture, attachment])
            for message in requests[-1][2]["messages"]:
                if message["role"] in {"user", "tool"}:
                    self.assertEqual(message["content"][1], picture.to_content_part())
                    self.assertEqual(message["content"][2], {
                        "type": "file", "file_data": attachment.file_data, "filename": "image.png"})
            self.assertIn("file", agent.messages[1]["content"][2])
            self.assertIn("file", agent.messages[3]["content"][2])
            self.assertEqual(requests[0][2]["tools"][0]["function"]["parameters"], get_image.parameters)

    def test_api_error_keeps_history_unchanged(self):
        with server((401, {"error": {"message": "Invalid API key"}})) as (url, requests):
            agent = Agent.deepseek(base_url=url, api_key="invalid", compaction=False)
            with self.assertRaisesRegex(SmolgentError, "401.*Invalid API key"):
                agent.run("Hello")
            self.assertEqual(len(agent.messages), 1)


if __name__ == "__main__":
    unittest.main()
