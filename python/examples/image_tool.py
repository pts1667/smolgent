"""Return a Pillow image. Set OPENROUTER_API_KEY; pass a vision/tool-capable model."""

import argparse

from PIL import Image, ImageDraw
from smolgent import Agent, tool


@tool
def drawing() -> Image.Image:
    """Retrieve a drawing for visual inspection."""
    image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(image).ellipse((40, 40, 216, 216), fill="blue")
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    agent = Agent.openrouter(args.model, tools=[drawing])
    print(agent.run("Call drawing and describe the shape and its color."))
