"""Media inputs and tool results, using the Rust core's content-part format."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Union

if TYPE_CHECKING:
    from PIL import Image as PILImage

Bytes = bytes | bytearray | memoryview
FilePath = str | os.PathLike[str]


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _encoded(data: Bytes) -> str:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must contain encoded bytes, not a path or raw samples")
    if not data:
        raise ValueError("media data must not be empty")
    return base64.b64encode(data).decode("ascii")


def _mime(value: str, family: str | None = None) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[\w.+-]+/[\w.+-]+", value):
        raise ValueError("mime_type must be a media type such as 'video/mp4'")
    value = value.lower()
    if family and not value.startswith(family + "/"):
        raise ValueError(f"mime_type must start with '{family}/'")
    return value


def _file_mime(path: FilePath, mime_type: str | None, family: str | None = None) -> str:
    if mime_type is None:
        mime_type = mimetypes.guess_type(os.fspath(path))[0]
        if mime_type is None and family is None:
            mime_type = "application/octet-stream"
        if mime_type is None:
            raise ValueError("cannot infer this file's media type; supply mime_type=")
    return _mime(mime_type, family)


def _url(value: str) -> str:
    _nonempty(value, "url")
    if not value.startswith(("https://", "http://", "data:")):
        raise ValueError("url must be an HTTP(S) URL or data URL; use from_file for local files")
    return value


@dataclass(frozen=True)
class Image:
    """An encoded image or remote image URL. Pillow is needed only by from_pil."""

    url: str = field(repr=False)
    detail: str | None = None

    def __post_init__(self) -> None:
        _url(self.url)
        if self.detail not in (None, "auto", "low", "high"):
            raise ValueError("detail must be 'auto', 'low', or 'high'")

    @classmethod
    def from_url(cls, url: str, *, detail: str | None = None) -> Image:
        return cls(url, detail)

    @classmethod
    def from_bytes(cls, data: Bytes, *, mime_type: str,
                   detail: str | None = None) -> Image:
        return cls(f"data:{_mime(mime_type, 'image')};base64,{_encoded(data)}", detail)

    @classmethod
    def from_file(cls, path: FilePath, *, mime_type: str | None = None,
                  detail: str | None = None) -> Image:
        mime_type = _file_mime(path, mime_type, "image")
        return cls.from_bytes(Path(path).read_bytes(), mime_type=mime_type, detail=detail)

    @classmethod
    def from_pil(cls, image: PILImage.Image, *, format: str = "PNG",
                 max_size: tuple[int, int] | None = None, quality: int = 85,
                 detail: str | None = None) -> Image:
        """Encode a snapshot without modifying the source. JPEG flattens alpha onto white."""
        try:
            from PIL import Image as PILImage, ImageOps
        except ImportError as error:
            raise ImportError("Pillow is required; install 'smolgent[images]'") from error
        if not isinstance(image, PILImage.Image):
            raise TypeError("image must be a PIL.Image.Image")
        format = format.upper()
        if format not in ("PNG", "JPEG"):
            raise ValueError("format must be 'PNG' or 'JPEG'")
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError("quality must be an integer between 1 and 100")
        if max_size is not None and (
            len(max_size) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
                                      for n in max_size)
        ):
            raise ValueError("max_size must contain two positive integers")
        snapshot = ImageOps.exif_transpose(image)
        if max_size is not None:
            snapshot.thumbnail(max_size)
        if format == "JPEG":
            rgba = snapshot.convert("RGBA")
            snapshot = PILImage.new("RGB", rgba.size, "white")
            snapshot.paste(rgba, mask=rgba.getchannel("A"))
        elif snapshot.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"):
            snapshot = snapshot.convert("RGB")
        buffer = io.BytesIO()
        snapshot.save(buffer, format=format, **({"quality": quality} if format == "JPEG" else {}))
        return cls.from_bytes(buffer.getvalue(), mime_type="image/" + format.lower(), detail=detail)

    def to_content_part(self) -> dict[str, Any]:
        image = {"url": self.url}
        if self.detail is not None:
            image["detail"] = self.detail
        return {"type": "image_url", "image_url": image}


@dataclass(frozen=True)
class Audio:
    """Encoded audio; format names and codec support depend on the provider/model."""

    data: str = field(repr=False)
    format: str

    def __post_init__(self) -> None:
        _nonempty(self.data, "data")
        if not isinstance(self.format, str) or not re.fullmatch(r"[a-z0-9_+-]+", self.format):
            raise ValueError("format must be a lowercase format name such as 'wav' or 'mp3'")

    @classmethod
    def from_bytes(cls, data: Bytes, *, format: str) -> Audio:
        return cls(_encoded(data), _nonempty(format, "format").lower())

    @classmethod
    def from_file(cls, path: FilePath, *, format: str | None = None) -> Audio:
        if format is None:
            extension = Path(path).suffix.lower().lstrip(".")
            formats = {"wav", "mp3", "flac", "ogg", "opus", "aac", "m4a", "aiff", "aif"}
            if extension not in formats:
                raise ValueError("cannot infer this audio file's format; supply format=")
            format = "aiff" if extension == "aif" else extension
        return cls.from_bytes(Path(path).read_bytes(), format=format)

    def to_content_part(self) -> dict[str, Any]:
        return {"type": "input_audio", "input_audio": {"data": self.data, "format": self.format}}


@dataclass(frozen=True)
class Video:
    """An encoded video or remote video URL; this helper does not transcode."""

    url: str = field(repr=False)

    def __post_init__(self) -> None:
        _url(self.url)

    @classmethod
    def from_url(cls, url: str) -> Video:
        return cls(url)

    @classmethod
    def from_bytes(cls, data: Bytes, *, mime_type: str) -> Video:
        return cls(f"data:{_mime(mime_type, 'video')};base64,{_encoded(data)}")

    @classmethod
    def from_file(cls, path: FilePath, *, mime_type: str | None = None) -> Video:
        mime_type = _file_mime(path, mime_type, "video")
        return cls.from_bytes(Path(path).read_bytes(), mime_type=mime_type)

    def to_content_part(self) -> dict[str, Any]:
        return {"type": "video_url", "video_url": {"url": self.url}}


@dataclass(frozen=True)
class File:
    """A document attachment (for example a PDF), subject to provider support."""

    file_data: str = field(repr=False)
    filename: str | None = None

    def __post_init__(self) -> None:
        _url(self.file_data)
        if self.filename is not None:
            _nonempty(self.filename, "filename")

    @classmethod
    def from_url(cls, url: str, *, filename: str | None = None) -> File:
        return cls(url, filename)

    @classmethod
    def from_bytes(cls, data: Bytes, *, mime_type: str, filename: str | None = None) -> File:
        return cls(f"data:{_mime(mime_type)};base64,{_encoded(data)}", filename)

    @classmethod
    def from_file(cls, path: FilePath, *, mime_type: str | None = None,
                  filename: str | None = None) -> File:
        mime_type = _file_mime(path, mime_type)
        return cls.from_bytes(Path(path).read_bytes(), mime_type=mime_type,
                              filename=Path(path).name if filename is None else filename)

    def to_content_part(self) -> dict[str, Any]:
        file = {"file_data": self.file_data}
        if self.filename is not None:
            file["filename"] = self.filename
        return {"type": "file", "file": file}


Media = Image | Audio | Video | File
Part = Union[str, dict[str, Any], Media, "PILImage.Image"]


@dataclass(frozen=True, init=False)
class MultimodalResult:
    """Ordered text and attachments. Also accepted as a prompt.

    Plain dict/list tool returns remain JSON text. Use this container to mark
    their contents as media parts. Pillow images are encoded when submitted.
    """

    parts: tuple[Part, ...] = field(repr=False)

    def __init__(self, parts: Iterable[Part]) -> None:
        if isinstance(parts, (str, bytes, dict)):
            raise TypeError("parts must be a sequence, for example MultimodalResult(['text', image])")
        object.__setattr__(self, "parts", tuple(parts))


Content = Union[str, Media, MultimodalResult, list[Part], "PILImage.Image"]


def _is_pil_image(value: Any) -> bool:
    # An existing Pillow image necessarily has its module loaded. Keep imports
    # optional and avoid loading Pillow just to serialize ordinary JSON results.
    module = sys.modules.get("PIL.Image")
    return module is not None and isinstance(value, module.Image)


def _part(value: Part) -> dict[str, Any]:
    if isinstance(value, str):
        return {"type": "text", "text": value}
    if isinstance(value, (Image, Audio, Video, File)):
        return value.to_content_part()
    if _is_pil_image(value):
        return Image.from_pil(value).to_content_part()
    if isinstance(value, dict):
        return value  # The Rust ContentPart deserializer validates wire-format dictionaries.
    raise TypeError("content parts must be text, media wrappers, Pillow images, or content dictionaries")


def serialize_prompt(prompt: Content) -> str:
    if isinstance(prompt, str):
        value = prompt
    else:
        parts = prompt.parts if isinstance(prompt, MultimodalResult) else (
            prompt if isinstance(prompt, list) else [prompt]
        )
        value = [_part(part) for part in parts]
    return json.dumps(value, allow_nan=False)


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, (Image, Audio, Video, File, MultimodalResult)) or _is_pil_image(result):
        parts = result.parts if isinstance(result, MultimodalResult) else [result]
        return json.dumps({"kind": "content", "value": [_part(part) for part in parts]},
                          allow_nan=False)
    return json.dumps({"kind": "json", "value": result}, allow_nan=False)
