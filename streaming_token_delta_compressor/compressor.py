"""Core Streaming Token Delta Compressor implementation."""
import struct
import io
from typing import Dict, List, Optional, Tuple


class CompressedFrame:
    def __init__(self, token_id: int, delta_text: str, is_dict_hit: bool):
        self.token_id = token_id
        self.delta_text = delta_text
        self.is_dict_hit = is_dict_hit


class TokenDeltaCompressor:
    def __init__(self, dict_size_limit: int = 256):
        self.dict_size_limit = dict_size_limit
        self.encoder_dict: Dict[str, int] = {}  # phrase -> code
        self.decoder_dict: Dict[int, str] = {}  # code -> phrase
        self.next_code = 1

    def compress_token(self, token_text: str) -> bytes:
        # Check dictionary
        if token_text in self.encoder_dict:
            code = self.encoder_dict[token_text]
            # Flag 1: dict code (code uint16)
            return struct.pack("!BH", 1, code)
        else:
            # Register in dict if space
            if self.next_code < self.dict_size_limit:
                code = self.next_code
                self.encoder_dict[token_text] = code
                self.decoder_dict[code] = token_text
                self.next_code += 1

            # Flag 0: raw bytes (length uint8 + raw utf-8)
            encoded = token_text.encode("utf-8")
            return struct.pack("!BB", 0, len(encoded)) + encoded

    def decompress_frame(self, frame_bytes: bytes) -> Tuple[str, int]:
        stream = io.BytesIO(frame_bytes)
        flag = stream.read(1)
        if not flag:
            return "", 0

        is_dict = struct.unpack("!B", flag)[0]
        if is_dict == 1:
            code = struct.unpack("!H", stream.read(2))[0]
            consumed = 3
            return self.decoder_dict.get(code, ""), consumed
        else:
            length = struct.unpack("!B", stream.read(1))[0]
            raw_text = stream.read(length).decode("utf-8", errors="replace")
            # Register in client dict
            if self.next_code < self.dict_size_limit:
                code = self.next_code
                self.decoder_dict[code] = raw_text
                self.next_code += 1
            consumed = 2 + length
            return raw_text, consumed
