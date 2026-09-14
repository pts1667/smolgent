"""Return a media file. Set OPENROUTER_API_KEY and choose a tool/media-capable model.

python python/examples/multimodal_tool.py recording.wav --kind audio --model YOUR_MODEL
python python/examples/multimodal_tool.py clip.mp4 --kind video --model YOUR_MODEL
"""

import argparse
from pathlib import Path

from smolgent import Agent, Audio, File, Image, MultimodalResult, Video, tool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--kind", choices=["image", "audio", "video", "file"], required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    media_type = {"image": Image, "audio": Audio, "video": Video, "file": File}[args.kind]

    @tool
    def get_media() -> MultimodalResult:
        """Retrieve the selected media for inspection."""
        return MultimodalResult([
            f"Attachment: {args.path.name}",
            media_type.from_file(args.path),
        ])

    agent = Agent.openrouter(args.model, tools=[get_media])
    print(agent.run("Call get_media and describe or summarize its contents."))


if __name__ == "__main__":
    main()
