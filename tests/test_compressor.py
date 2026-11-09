"""Unit tests for streaming-token-delta-compressor."""
import unittest
from streaming_token_delta_compressor.compressor import TokenDeltaCompressor


class TestTokenDeltaCompressor(unittest.TestCase):
    def setUp(self):
        self.enc = TokenDeltaCompressor()
        self.dec = TokenDeltaCompressor()

    def test_compression_and_decompression(self):
        stream = ["Hello", " world", " from", " agent", " Hello", " world"]
        out = []
        for s in stream:
            b = self.enc.compress_token(s)
            t, _ = self.dec.decompress_frame(b)
            out.append(t)
        self.assertEqual(out, stream)
