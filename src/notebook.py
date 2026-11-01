"""Запуск CLI с живым stdout/stderr в ячейке Jupyter; только standard library."""

import codecs
import os
import signal
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], *, cwd: Path, env: dict | None = None) -> None:
    """Передавать в ячейку и строки, и tqdm с carriage return; прерывать дочерний CLI."""
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if env:
        environment.update(env)
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    ) as process:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while chunk := process.stdout.read1(4096):
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()
            sys.stdout.write(decoder.decode(b"", final=True))
            code = process.wait()
        except BaseException:
            # uv is a launcher: stop its training child as well.
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)
