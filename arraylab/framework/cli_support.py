"""Shared argument errors and pipe-safe JSON output; no business imports."""
import argparse
import json


class UsageError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(message)


def emit(value, stream):
    if stream is not None:
        # Console pipes remain ASCII-safe; JSON files use their own UTF-8 writer.
        stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + '\n')
        stream.flush()
