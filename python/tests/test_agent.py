"""Integration tests exercise the compiled extension; no API keys or network services needed."""

import asyncio
import contextvars
import json
import os
import queue
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from smolgent import Agent, SmolgentError, Tool, tool


def answer(text="Hello", **metadata):
    return {"choices": [{"message": {"role": "assistant", "content": text, **metadata}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


@contextmanager
def server(*responses):
    pending = queue.Queue()
    for response in responses:
        pending.put(response)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            try:
                response = pending.get_nowait()
            except queue.Empty:
                response = (500, {"error": "unexpected request"})
            if callable(response):
                response = response(body)
            status, body = response if isinstance(response, tuple) else (200, response)
            payload = json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # Expected when the caller cancels an in-flight request.

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def local_agent(base_url, **kwargs):
    return Agent.compatible("test-model",
                            chat_completions_url=base_url + "/v1/chat/completions",
                            compaction=False, **kwargs)


class AgentTests(unittest.TestCase):
    def test_chat_history_reasoning_and_reset(self):
        with server(answer("First", reasoning_content="private reasoning",
                           annotations=[{"type": "test"}]), answer("Second")) as (url, requests):
            agent = local_agent(url, api_key="test-key", system_prompt="Be concise")
            response = agent.run("Hi")
            self.assertEqual(str(response), "First")
            self.assertEqual(response.reasoning["reasoning_content"], "private reasoning")
            self.assertEqual(response.raw["usage"]["prompt_tokens"], 10)
            snapshot = agent.messages
            snapshot.clear()
            self.assertEqual(len(agent.messages), 3)
            self.assertEqual(agent.run("Again").text, "Second")
            path, headers, body = requests[1]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual({k.lower(): v for k, v in headers.items()}["authorization"],
                             "Bearer test-key")
            self.assertEqual(body["model"], "test-model")
            self.assertEqual(len(body["messages"]), 4)
            self.assertEqual(body["messages"][2]["reasoning_content"], "private reasoning")
            self.assertEqual(body["messages"][2]["annotations"], [{"type": "test"}])
            agent.reset()
            self.assertEqual(agent.messages, [{"role": "system", "content": "Be concise"}])

    def test_sync_and_async_tools_preserve_context_and_loop(self):
        request_id = contextvars.ContextVar("request_id")
        seen = []

        @tool(parameters={"type": "object", "properties": {"x": {"type": "integer"}},
                          "required": ["x"]})
        def double(x):
            """Double a number."""
            seen.append(("sync", request_id.get()))
            return x * 2

        async def scenario(url):
            request_id.set("test-context")
            loop = asyncio.get_running_loop()

            @tool(parameters={"type": "object", "properties": {}})
            async def report():
                self.assertIs(asyncio.get_running_loop(), loop)
                seen.append(("async", request_id.get()))
                await asyncio.sleep(0)
                return {"ok": True}

            return await local_agent(url, tools=[double, report]).arun("Use both tools")

        with server(answer(None, tool_calls=[call("double", {"x": 4}),
                                             call("report", {}, "call_2")]),
                    answer("Done")) as (url, requests):
            self.assertEqual(asyncio.run(scenario(url)).text, "Done")
            results = [m for m in requests[1][2]["messages"] if m["role"] == "tool"]
            self.assertEqual(results[0]["content"], "8")
            self.assertEqual(json.loads(results[1]["content"]), {"ok": True})
            self.assertEqual(seen, [("sync", "test-context"), ("async", "test-context")])

    def test_tool_errors_are_reported_to_model(self):
        def fail():
            raise ValueError("cannot calculate")

        broken = Tool("broken", "Fails", {"type": "object"}, fail)
        with server(answer(None, tool_calls=[call("broken", {})]), answer("Recovered")) as (url, requests):
            self.assertEqual(local_agent(url, tools=[broken]).run("Try").text, "Recovered")
            self.assertIn("cannot calculate", requests[1][2]["messages"][-1]["content"])

    def test_read_tool_and_root_enforcement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "allowed"
            allowed.mkdir()
            (allowed / "notes.txt").write_text("known file content", encoding="utf-8")
            (root / "secret.txt").write_text("hidden content", encoding="utf-8")
            with server(answer(None, tool_calls=[
                call("read", {"path": str(allowed / "notes.txt")}),
                call("read", {"path": str(root / "secret.txt")}, "call_2"),
            ]), answer("Read complete")) as (url, requests):
                agent = local_agent(url, read_roots=[allowed])
                self.assertEqual(agent.run("Read files").text, "Read complete")
                names = {t["function"]["name"] for t in requests[0][2]["tools"]}
                self.assertEqual(names, {"read", "ripgrep"})
                results = [m["content"] for m in requests[1][2]["messages"] if m["role"] == "tool"]
                self.assertIn("known file content", results[0])
                self.assertIn("outside the allowed read roots", results[1])
                self.assertNotIn("hidden content", results[1])

    def test_no_filesystem_tools_by_default(self):
        with server(answer()) as (url, requests):
            local_agent(url).run("Hi")
            self.assertFalse(requests[0][2].get("tools"))

    def test_write_root_grants_read_and_confines_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "allowed"
            allowed.mkdir()
            with server(answer(None, tool_calls=[
                call("create_file", {"path": str(allowed / "new.txt"), "content": "created"}),
                call("create_file", {"path": str(root / "outside.txt"), "content": "denied"}, "call_2"),
            ]), answer("Done")) as (url, requests):
                local_agent(url, write_roots=[allowed]).run("Create files")
                self.assertEqual((allowed / "new.txt").read_text(), "created")
                self.assertFalse((root / "outside.txt").exists())
                names = {t["function"]["name"] for t in requests[0][2]["tools"]}
                self.assertEqual(names, {"read", "ripgrep", "create_file", "delete_file", "apply_patch"})
                self.assertIn("outside the allowed write roots",
                              requests[1][2]["messages"][-1]["content"])

    def test_request_timeout_leaves_history_intact(self):
        release = threading.Event()

        def delayed(_):
            release.wait(timeout=3)
            return answer()

        with server(delayed) as (url, _):
            agent = local_agent(url, timeout=0.1)
            initial = agent.messages
            try:
                with self.assertRaises(SmolgentError):
                    agent.run("Wait")
                self.assertEqual(agent.messages, initial)
            finally:
                release.set()

    def test_multimodal_and_llama_cpp_path(self):
        content = [{"type": "text", "text": "Describe"},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]
        with server(answer("Image")) as (url, requests):
            agent = Agent.llama_cpp("vision", base_url=url + "/prefix", compaction=False)
            self.assertEqual(agent.run(content).text, "Image")
            self.assertEqual(requests[0][0], "/prefix/v1/chat/completions")
            self.assertEqual(requests[0][2]["messages"][-1]["content"], content)

    def test_http_error_and_max_rounds_leave_history_intact(self):
        with server((401, {"error": "invalid key"}),
                    answer(None, tool_calls=[call("missing", {})]), answer("Retry")) as (url, _):
            agent = local_agent(url, max_tool_rounds=0)
            initial = agent.messages
            with self.assertRaisesRegex(SmolgentError, "401"):
                agent.run("Fail")
            self.assertEqual(agent.messages, initial)
            with self.assertRaisesRegex(SmolgentError, "stopped after 0 tool rounds"):
                agent.run("Fail again")
            self.assertEqual(agent.messages, initial)
            self.assertEqual(agent.run("Retry").text, "Retry")

    def test_cancellation_busy_guard_and_reuse(self):
        entered = threading.Event()
        release = threading.Event()

        def delayed(_):
            entered.set()
            release.wait(timeout=5)
            return answer("Cancelled answer")

        async def scenario(url):
            agent = local_agent(url)
            initial = agent.messages
            task = asyncio.create_task(agent.arun("Wait"))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            with self.assertRaisesRegex(RuntimeError, "already running"):
                await agent.arun("Overlap")
            with self.assertRaisesRegex(RuntimeError, "already running"):
                agent.reset()
            with self.assertRaisesRegex(RuntimeError, "already running"):
                _ = agent.messages
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            # Cancellation is forwarded to the Rust runtime asynchronously.
            for _ in range(100):
                try:
                    current = agent.messages
                    break
                except RuntimeError:
                    await asyncio.sleep(0.01)
            else:
                self.fail("cancelled run did not release the session")
            self.assertEqual(current, initial)
            release.set()
            self.assertEqual((await agent.arun("Retry")).text, "Retry answer")

        with server(delayed, answer("Retry answer")) as (url, _):
            try:
                asyncio.run(scenario(url))
            finally:
                release.set()

    def test_run_inside_event_loop_has_clear_error(self):
        async def scenario():
            with self.assertRaisesRegex(RuntimeError, "await agent.arun"):
                local_agent("http://127.0.0.1:1").run("Hi")
        asyncio.run(scenario())

    def test_cancelling_run_cancels_active_async_tool(self):
        async def scenario(url):
            entered = asyncio.Event()
            cancelled = asyncio.Event()

            @tool(parameters={"type": "object"})
            async def wait_for_cancel():
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            agent = local_agent(url, tools=[wait_for_cancel])
            task = asyncio.create_task(agent.arun("Wait"))
            await asyncio.wait_for(entered.wait(), timeout=3)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(cancelled.wait(), timeout=1)

        with server(answer(None, tool_calls=[call("wait_for_cancel", {})])) as (url, _):
            asyncio.run(scenario(url))

    def test_openrouter_credentials_and_validation(self):
        # Capture constructor input without making an external request or touching a keyring.
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=True):
            with patch("smolgent.NativeAgent") as native:
                Agent.openrouter("openrouter/auto")
                config = json.loads(native.call_args.args[0])
                self.assertEqual(config["api_key"], "env-key")
                self.assertEqual(config["provider"], "openrouter")
                Agent.openrouter("model", keyring_id="my-app/openrouter")
                config = json.loads(native.call_args.args[0])
                self.assertIsNone(config["api_key"])
                self.assertEqual(config["keyring_id"], "my-app/openrouter")
                Agent.openrouter("model", api_key="explicit-key")
                self.assertEqual(json.loads(native.call_args.args[0])["api_key"], "explicit-key")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                Agent.openrouter("model")
        for options in ({"timeout": 0}, {"timeout": float("inf")}, {"max_tool_rounds": -1},
                        {"api_key": "a", "keyring_id": "b"}, {"read_roots": "."}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                local_agent("http://127.0.0.1:1", **options)
        duplicate = Tool("read", "", {"type": "object"}, lambda: None)
        with self.assertRaisesRegex(ValueError, "reserved"):
            local_agent("http://127.0.0.1:1", read_roots=["."], tools=[duplicate])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            local_agent("http://127.0.0.1:1", tools=[duplicate, duplicate])


if __name__ == "__main__":
    unittest.main()
