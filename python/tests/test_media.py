"""Media encoding and transport through the compiled Python/Rust agent loop."""

import asyncio
import base64
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from smolgent import Agent, Audio, File, Image, MultimodalResult, Video, tool
from smolgent._media import serialize_prompt
from test_agent import answer, call, local_agent, server

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


class MediaTests(unittest.TestCase):
    def test_files_bytes_urls_and_content_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("image.png", "audio.wav", "clip.mp4", "report.pdf"):
                (root / filename).write_bytes(b"transport fixture")
            image = Image.from_file(root / "image.png", detail="high")
            audio = Audio.from_file(root / "audio.wav")
            video = Video.from_file(root / "clip.mp4")
            file = File.from_file(root / "report.pdf")
        # Files are read eagerly, so objects survive file closure/deletion.
        encoded = base64.b64encode(b"transport fixture").decode()
        self.assertEqual(image.to_content_part(), {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + encoded, "detail": "high"}})
        self.assertEqual(audio.to_content_part(), {"type": "input_audio", "input_audio": {
            "data": encoded, "format": "wav"}})
        self.assertEqual(video.to_content_part(), {"type": "video_url", "video_url": {
            "url": "data:video/mp4;base64," + encoded}})
        self.assertEqual(file.to_content_part(), {"type": "file", "file": {
            "filename": "report.pdf", "file_data": "data:application/pdf;base64," + encoded}})
        self.assertEqual(Image.from_bytes(bytearray(b"transport fixture"), mime_type="image/png").url,
                         image.url)
        self.assertEqual(Audio.from_bytes(memoryview(b"transport fixture"), format="WAV"), audio)
        for cls in (Image, Video, File):
            media = cls.from_url("https://example.com/media")
            self.assertIn("https://example.com/media", json.dumps(media.to_content_part()))

    def test_media_validation(self):
        invalid = [
            lambda: Image.from_url("photo.png"),
            lambda: Video.from_bytes(b"video", mime_type="image/png"),
            lambda: Image.from_bytes(b"", mime_type="image/png"),
            lambda: Audio.from_bytes([0, 1], format="wav"),
            lambda: Audio.from_bytes(b"audio", format=""),
            lambda: File.from_bytes(b"pdf", mime_type="pdf"),
            lambda: Image.from_url("https://example.com/a", detail="invalid"),
            lambda: MultimodalResult("text"),
            lambda: serialize_prompt([42]),
            lambda: Audio.from_file("unknown.extension"),
            lambda: Image.from_file("unknown.extension"),
        ]
        for operation in invalid:
            with self.subTest(operation=operation), self.assertRaises((ValueError, TypeError)):
                operation()

    def test_media_without_pillow_and_missing_extra_error(self):
        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            self.assertEqual(json.loads(serialize_prompt(["Listen", Audio.from_bytes(b"wav", format="wav")]))[1]
                             ["type"], "input_audio")
            self.assertEqual(json.loads(serialize_prompt(Image.from_bytes(b"png", mime_type="image/png")))[0]
                             ["type"], "image_url")
            with self.assertRaisesRegex(ImportError, "smolgent\\[images\\]"):
                Image.from_pil(object())

    def test_prompt_helpers_preserve_order_and_raw_dictionaries(self):
        audio = Audio.from_bytes(b"audio", format="wav")
        video = Video.from_url("https://example.com/clip.mp4")
        raw = {"type": "text", "text": "Also watch"}
        parts = ["Listen", audio, raw, video]
        expected = [{"type": "text", "text": "Listen"}, audio.to_content_part(), raw,
                    video.to_content_part()]
        self.assertEqual(json.loads(serialize_prompt(parts)), expected)
        self.assertEqual(json.loads(serialize_prompt(MultimodalResult(parts))), expected)
        self.assertEqual(json.loads(serialize_prompt("hello")), "hello")

    def test_multimodal_tools_and_history_replay(self):
        image = Image.from_bytes(b"image", mime_type="image/png")
        audio = Audio.from_bytes(b"audio", format="wav")
        video = Video.from_bytes(b"video", mime_type="video/mp4")
        file = File.from_bytes(b"pdf", mime_type="application/pdf", filename="report.pdf")

        @tool
        def picture():
            return image

        @tool
        async def recording() -> Audio:
            await asyncio.sleep(0)
            return audio

        @tool
        def clip() -> Video:
            return video

        @tool
        def document() -> File:
            return file

        @tool
        def mixed() -> MultimodalResult:
            return MultimodalResult(["Evidence:", image, audio, video, file])

        tools = [picture, recording, clip, document, mixed]
        with server(answer(None, tool_calls=[call(item.name, {}, str(i)) for i, item in enumerate(tools)]),
                    answer("Reviewed"), answer("Remembered")) as (url, requests):
            agent = local_agent(url, tools=tools)
            self.assertEqual(agent.run(["Review", image]).text, "Reviewed")
            results = [m for m in agent.messages if m["role"] == "tool"]
            expected = [[part.to_content_part()] for part in (image, audio, video, file)]
            expected.append([{"type": "text", "text": "Evidence:"}] +
                            [part.to_content_part() for part in (image, audio, video, file)])
            self.assertEqual([m["content"] for m in results], expected)
            self.assertEqual([m["tool_call_id"] for m in results], [str(i) for i in range(5)])
            self.assertEqual(requests[1][2]["messages"][-5:], results)
            agent.run("Recall the attachments")
            self.assertEqual([m for m in requests[2][2]["messages"] if m["role"] == "tool"], results)

    def test_json_tool_results_remain_text_including_content_lookalikes(self):
        values = ["plain text", 42, True, None, {"ok": True},
                  [{"type": "image_url", "image_url": {"url": "https://example.com/image"}}],
                  {"kind": "content", "value": [{"type": "text", "text": "still JSON"}]}]

        @tool
        def value(index: int):
            return values[index]

        with server(answer(None, tool_calls=[call("value", {"index": i}, str(i)) for i in range(len(values))]),
                    answer("Done")) as (url, requests):
            local_agent(url, tools=[value]).run("Use tools")
            results = [m["content"] for m in requests[1][2]["messages"] if m["role"] == "tool"]
            self.assertTrue(all(isinstance(result, str) for result in results))
            self.assertEqual(results[0], values[0])
            self.assertEqual([json.loads(result) for result in results[1:]], values[1:])

    def test_llama_cpp_video_translation_for_prompts_and_tool_results(self):
        video = Video.from_url("https://example.com/clip.mp4")

        @tool
        def clip():
            return video

        with server(answer(None, tool_calls=[call("clip", {})]), answer("Done")) as (url, requests):
            agent = Agent.llama_cpp("test", base_url=url, tools=[clip], compaction=False)
            asyncio.run(agent.arun(MultimodalResult(["Watch", video])))
            self.assertEqual(requests[0][2]["messages"][-1]["content"][1],
                             {"type": "input_video", "input_video": {"url": video.url}})
            self.assertEqual(requests[1][2]["messages"][-1]["content"][0]["type"], "input_video")
            self.assertEqual(agent.messages[-2]["content"], [video.to_content_part()])

    def test_invalid_media_tool_result_is_model_visible_error(self):
        @tool
        def broken():
            return MultimodalResult([{"type": "unsupported"}])

        with server(answer(None, tool_calls=[call("broken", {})]), answer("Recovered")) as (url, requests):
            self.assertEqual(local_agent(url, tools=[broken]).run("Try").text, "Recovered")
            self.assertIn("Tool error:", requests[1][2]["messages"][-1]["content"])


@unittest.skipIf(PILImage is None, "install smolgent[images] to test Pillow integration")
class PillowTests(unittest.TestCase):
    def test_png_and_jpeg_encoding_do_not_modify_source(self):
        source = PILImage.new("RGBA", (80, 40), (255, 0, 0, 0))
        png = Image.from_pil(source)
        jpeg = Image.from_pil(source, format="JPEG", max_size=(20, 20), quality=90)
        self.assertEqual(source.size, (80, 40))
        self.assertEqual(source.mode, "RGBA")
        self.assertEqual(source.getpixel((0, 0)), (255, 0, 0, 0))
        with PILImage.open(io.BytesIO(base64.b64decode(png.url.split(",", 1)[1]))) as decoded:
            self.assertEqual(decoded.format, "PNG")
            self.assertEqual(decoded.size, (80, 40))
            self.assertEqual(decoded.mode, "RGBA")
        with PILImage.open(io.BytesIO(base64.b64decode(jpeg.url.split(",", 1)[1]))) as decoded:
            self.assertEqual(decoded.format, "JPEG")
            self.assertEqual(decoded.size, (20, 10))
            self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))

    def test_pil_prompt_and_sync_async_tool_returns(self):
        source = PILImage.new("RGB", (3, 2), "red")

        @tool
        def picture() -> PILImage.Image:
            return source

        @tool
        async def mixed():
            return MultimodalResult(["Image:", source])

        with server(answer(None, tool_calls=[call("picture", {}), call("mixed", {}, "second")]),
                    answer("Red")) as (url, requests):
            local_agent(url, tools=[picture, mixed]).run(["Describe", source])
            prompt_part = requests[0][2]["messages"][-1]["content"][1]
            self.assertEqual(prompt_part["type"], "image_url")
            self.assertEqual(requests[1][2]["messages"][-2]["content"], [prompt_part])
            self.assertEqual(requests[1][2]["messages"][-1]["content"],
                             [{"type": "text", "text": "Image:"}, prompt_part])

    def test_encoding_runs_off_event_loop(self):
        source = PILImage.new("RGB", (2, 2))
        threads = []
        original = Image.from_pil

        def record_thread(image, **kwargs):
            threads.append(threading.get_ident())
            return original(image, **kwargs)

        @tool
        async def picture():
            return source

        main_thread = threading.get_ident()
        with server(answer(None, tool_calls=[call("picture", {})]), answer("Done")) as (url, _):
            with patch.object(Image, "from_pil", side_effect=record_thread):
                asyncio.run(local_agent(url, tools=[picture]).arun(source))
        self.assertEqual(len(threads), 2)
        self.assertTrue(all(thread != main_thread for thread in threads))


if __name__ == "__main__":
    unittest.main()
