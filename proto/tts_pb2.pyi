from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TTSRequest(_message.Message):
    __slots__ = ("message", "language", "speed")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    message: str
    language: str
    speed: float
    def __init__(self, message: _Optional[str] = ..., language: _Optional[str] = ..., speed: _Optional[float] = ...) -> None: ...

class TTSResponse(_message.Message):
    __slots__ = ("audio_chunk", "end_of_audio")
    AUDIO_CHUNK_FIELD_NUMBER: _ClassVar[int]
    END_OF_AUDIO_FIELD_NUMBER: _ClassVar[int]
    audio_chunk: bytes
    end_of_audio: bool
    def __init__(self, audio_chunk: _Optional[bytes] = ..., end_of_audio: bool = ...) -> None: ...
