"""Streaming integration tests against the compiled Rust extension and a local SSE server."""
import asyncio
import json
import queue
import threading
import unittest
from contextlib import aclosing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from smolgent import Agent, Image, SmolgentError, tool


def frame(delta=None, finish=None, **extra):
    value = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **extra}
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\r\n\r\n").encode()


DONE = b"data: [DONE]\n\n"


def reply(text="Hello"):
    return [frame({"content": text}), frame(finish="stop"), DONE]


@contextmanager
def server(*scripts, status=200, content_type="text/event-stream"):
    pending = queue.Queue()
    for script in scripts:
        pending.put(script)
    requests = []
    gates = [part for script in scripts for part in script if isinstance(part, threading.Event)]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            try:
                script = pending.get_nowait()
            except queue.Empty:
                script = [b'data: {"error":"unexpected request"}\n\n']
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                for part in script:
                    if isinstance(part, threading.Event):
                        part.wait(5)
                    else:
                        self.wfile.write(part)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests
    finally:
        for gate in gates:
            gate.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


def agent_at(url, **kwargs):
    return Agent.deepseek("deepseek-flash", base_url=url, api_key="test-key",
                          compaction=False, **kwargs)


async def released(agent):
    # Cancellation crosses into the Rust runtime asynchronously.
    for _ in range(100):
        try:
            return agent.messages
        except RuntimeError:
            await asyncio.sleep(.01)
    raise AssertionError("agent did not release its session")


class StreamingTests(unittest.TestCase):
    def test_sync_tool_loop_and_reasoning_replay(self):
        calls = []

        @tool
        def add(a: int, b: int) -> int:
            """Add numbers."""
            calls.append((a, b))
            return a + b

        first = b": keep-alive\r\n\r\n" + frame({"reasoning_content": "Check "}) + frame({
            "reasoning_content": "sum", "content": "Calculating…", "tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "ad", "arguments": '{"a":123,'},
            }],
        }) + frame({"tool_calls": [{"index": 0, "function": {"name": "d", "arguments": '"b":456}'}}]}, "tool_calls") + DONE
        fragmented = [first[i:i+3] for i in range(0, len(first), 3)]
        second = [frame({"content": "579 🙂"}), frame(finish="stop"),
                  b'data: {"choices":[],"usage":{"completion_tokens":9}}\n\n', DONE]
        with server(fragmented, second, reply("Next")) as (url, requests):
            agent = agent_at(url, tools=[add], thinking=False)
            events = list(agent.stream("Add"))
            self.assertEqual(calls, [(123, 456)])
            self.assertEqual("".join(e.text for e in events if e.type == "text_delta"), "Calculating…579 🙂")
            self.assertEqual("".join(e.text for e in events if e.type == "reasoning_delta"), "Check sum")
            types = [e.type for e in events]
            self.assertLess(types.index("model_completed"), types.index("tool_started"))
            self.assertLess(types.index("tool_started"), types.index("tool_result"))
            self.assertEqual(types.count("completed"), 1)
            self.assertEqual(events[-1].response.text, "579 🙂")
            self.assertEqual(events[-1].response.raw["usage"]["completion_tokens"], 9)
            self.assertEqual(len(agent.messages), 5)
            self.assertTrue(all(request["stream"] for request in requests))
            self.assertEqual(requests[0]["thinking"], {"type": "disabled"})
            assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
            self.assertEqual(assistant["reasoning_content"], "Check sum")
            self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], '{"a":123,"b":456}')
            self.assertEqual(list(agent.stream("Again"))[-1].response.text, "Next")

    def test_async_receives_text_before_completion_and_cancels(self):
        gate = threading.Event()
        with server([frame({"content": "early"}), gate, frame(finish="stop"), DONE]) as (url, requests):
            agent = agent_at(url)
            before = agent.messages

            async def run():
                async with aclosing(agent.astream("Hi")) as stream:
                    async def first_text():
                        async for event in stream:
                            if event.type == "text_delta":
                                return event.text
                    self.assertEqual(await asyncio.wait_for(first_text(), 2), "early")
                    self.assertFalse(gate.is_set())
                self.assertEqual(await released(agent), before)
            asyncio.run(run())
            self.assertEqual(len(requests), 1)

    def test_stream_errors_do_not_commit_or_execute_partial_tools(self):
        calls = []

        @tool
        def act() -> str:
            """Perform an action."""
            calls.append(True)
            return "ok"

        tool_frame = frame({"tool_calls": [{"index": 0, "id": "x", "function": {"name": "act", "arguments": "{}"}}]}, "tool_calls")
        for script in [
            [frame({"content": "partial"})],
            [tool_frame],  # No [DONE]: never execute even a fully assembled call.
            [b'data: {"error":{"message":"overloaded"}}\n\n'],
            [b'data: not-json\n\n'],
            [frame({"tool_calls": [{"index": 0, "id": "x", "function": {"name": "act", "arguments": "{"}}]}, "tool_calls"), DONE],
            [frame({"content": "partial"}), DONE],  # No terminal choice.
        ]:
            with self.subTest(script=script), server(script) as (url, _):
                agent = agent_at(url, tools=[act])
                before = agent.messages
                with self.assertRaises(SmolgentError):
                    list(agent.stream("Hi"))
                self.assertEqual(agent.messages, before)
        self.assertEqual(calls, [])

    def test_http_errors_and_timeout(self):
        with server([b'{"error":"unauthorized"}'], status=401, content_type="application/json") as (url, _):
            with self.assertRaisesRegex(SmolgentError, "401.*unauthorized"):
                list(agent_at(url).stream("Hi"))
        gate = threading.Event()
        with server([b": keep-alive\n\n", gate, *reply()]) as (url, _):
            agent = agent_at(url, timeout=.1)
            before = agent.messages
            with self.assertRaisesRegex(SmolgentError, "timed out"):
                list(agent.stream("Hi"))
            self.assertEqual(agent.messages, before)

    def test_cancel_async_tool_and_release_backpressure(self):
        async def run(url):
            started, cancelled = asyncio.Event(), asyncio.Event()

            @tool
            async def wait() -> str:
                """Wait for something."""
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()

            agent = agent_at(url, tools=[wait])
            before = agent.messages
            stream = agent.astream("Hi")
            async def consume():
                async for _ in stream:
                    pass
            task = asyncio.create_task(consume())
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await stream.aclose()
            await asyncio.wait_for(cancelled.wait(), 2)
            self.assertEqual(await released(agent), before)

        script = [frame({"tool_calls": [{"index": 0, "id": "x", "function": {"name": "wait", "arguments": "{}"}}]}, "tool_calls"), DONE]
        with server(script) as (url, requests):
            asyncio.run(run(url))
            self.assertEqual(len(requests), 1)

        async def slow_consumer(url):
            agent = agent_at(url)
            before = agent.messages
            stream = agent.astream("Hi")
            await anext(stream)
            await asyncio.sleep(.1)  # Let both bounded queues fill.
            await asyncio.wait_for(stream.aclose(), 2)
            self.assertEqual(await released(agent), before)
        with server([frame({"content": "x"}) * 1000, frame(finish="stop"), DONE]) as (url, _):
            asyncio.run(slow_consumer(url))

    def test_async_completion_and_sync_guard(self):
        with server(reply()) as (url, _):
            agent = agent_at(url)
            async def run():
                with self.assertRaisesRegex(RuntimeError, "astream"):
                    list(agent.stream("Hi"))
                events = [event async for event in agent.astream("Hi")]
                self.assertEqual(events[-1].response.text, "Hello")
                self.assertEqual(len(agent.messages), 3)
            asyncio.run(run())

    def test_multimodal_tool_result_and_tool_error_recovery(self):
        @tool
        def image() -> Image:
            """Fetch an image."""
            return Image.from_url("https://example.com/tool.png")

        @tool
        def fail() -> str:
            """A failing tool."""
            raise ValueError("failed intentionally")

        calls = [frame({"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "fail", "arguments": "{}"}},
            {"index": 1, "id": "b", "function": {"name": "image", "arguments": "{}"}},
        ]}, "tool_calls"), DONE]
        with server(calls, reply("Recovered")) as (url, requests):
            agent = agent_at(url, tools=[fail, image])
            events = list(agent.stream(["Look", Image.from_url("https://example.com/input.png")]))
            self.assertEqual(events[-1].response.text, "Recovered")
            results = [e.data for e in events if e.type == "tool_result"]
            self.assertIn("failed intentionally", results[0]["content"])
            self.assertEqual(results[1]["content"][0]["type"], "image_url")
            user = requests[0]["messages"][-1]
            self.assertEqual(user["content"][1]["type"], "image_url")
            self.assertEqual(requests[1]["messages"][-1]["content"], results[1]["content"])

    def test_close_after_model_completed_before_final_event_discards_history(self):
        with server(reply()) as (url, _):
            agent = agent_at(url)
            before = agent.messages
            stream = agent.stream("Hi")
            for event in stream:
                if event.type == "model_completed":
                    break
            stream.close()
            # Synchronous close has drained callbacks; native cancellation may release later.
            self.assertEqual(asyncio.run(released(agent)), before)


if __name__ == "__main__":
    unittest.main()
