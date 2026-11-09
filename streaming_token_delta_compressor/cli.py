"""CLI for streaming-token-delta-compressor."""
import sys
import argparse
from .compressor import TokenDeltaCompressor


def main():
    parser = argparse.ArgumentParser(description="Streaming Token Delta Compressor CLI")
    parser.add_argument("--demo", action="store_true", help="Run token delta compression demo")
    args = parser.parse_args()

    if args.demo:
        tokens = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", " the", " fox"]
        enc = TokenDeltaCompressor()
        dec = TokenDeltaCompressor()

        total_raw = sum(len(t.encode()) for t in tokens)
        total_comp = 0
        reconstructed = []

        for t in tokens:
            b = enc.compress_token(t)
            total_comp += len(b)
            decomp, _ = dec.decompress_frame(b)
            reconstructed.append(decomp)

        print("Tokens Reconstructed:", "".join(reconstructed))
        print(f"Raw Bytes: {total_raw} | Compressed Bytes: {total_comp}")
    else:
        parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
