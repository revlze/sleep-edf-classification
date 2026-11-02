"""Запуск CLI с живым stdout/stderr в ячейке Jupyter; только standard library."""

import codecs
import html
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

PREFIX = "__SLEEP_PROGRESS__"


class _Output:
    """Заменять один display-output Jupyter, а не печатать каждое обновление."""

    def __init__(self):
        self.pending = ""
        self.handle = None
        try:
            from IPython import get_ipython

            self.notebook = getattr(get_ipython(), "kernel", None) is not None
        except ImportError:
            self.notebook = False

    def progress(self, state: dict) -> None:
        n, total = state["n"], state["total"]
        detail = html.escape(state["detail"])
        description = html.escape(state["description"])
        elapsed = int(state["elapsed"])
        status = " · ошибка" if state["status"] == "failed" else ""
        content = (
            f"<div><b>{description}{status}</b> &nbsp; {n}/{total} &nbsp; "
            f"прошло {elapsed // 60}:{elapsed % 60:02d}<br>"
            f'<progress value="{n}" max="{max(total, 1)}" style="width:100%"></progress>'
            f"<br>{detail}</div>"
        )
        if self.notebook:
            from IPython.display import HTML, display

            if self.handle is None:
                self.handle = display(HTML(content), display_id=True)
            else:
                self.handle.update(HTML(content))
        else:
            text = (
                f"{state['description']}: {n}/{total} | {elapsed}s | {state['detail']}"
            )
            sys.stdout.write("\r" + text + "\x1b[K")
            if state["status"] != "running":
                sys.stdout.write("\n")
            sys.stdout.flush()

    def write(self, text: str, *, final: bool = False) -> None:
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.startswith(PREFIX):
                self.progress(json.loads(line[len(PREFIX) :]))
            else:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
        if final and self.pending:
            sys.stdout.write(self.pending)
            sys.stdout.flush()
            self.pending = ""


def run_command(command: list[str], *, cwd: Path, env: dict | None = None) -> None:
    """Передавать в ячейку и строки, и tqdm с carriage return; прерывать дочерний CLI."""
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "SLEEP_PROGRESS_JSON": "1",
    }
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
        output = _Output()
        try:
            while chunk := process.stdout.read1(4096):
                output.write(decoder.decode(chunk))
            output.write(decoder.decode(b"", final=True), final=True)
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
