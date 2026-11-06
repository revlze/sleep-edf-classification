import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

FINAL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "offline_runtime_test", FINAL / "offline_runtime.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_archive(root, *, link=False):
    path = root / "data.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("site-packages/example.py")
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside.py"
            tar.addfile(info)
        else:
            payload = b"value = 42\n"
            info.size = len(payload)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
    data = path.read_bytes()
    entry = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "regular_files": 0 if link else 1,
        "uncompressed_bytes": 0 if link else len(payload),
    }
    (root / "offline_manifest.json").write_text(
        json.dumps({"kind": module.KIND, "files": {path.name: entry}})
    )
    return module.RuntimeInput(root)


def test_local_archive_sha_and_inventory(tmp_path):
    source = make_archive(tmp_path)
    target = tmp_path / "restored"
    module.extract_checked(source, "data.tar.gz", target, lambda _: None)
    assert (target / "site-packages/example.py").read_text() == "value = 42\n"
    source.manifest["files"]["data.tar.gz"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        module.extract_checked(source, "data.tar.gz", tmp_path / "bad", lambda _: None)


def test_runtime_rejects_links(tmp_path):
    source = make_archive(tmp_path, link=True)
    with pytest.raises(ValueError, match="link/device"):
        module.extract_checked(
            source, "data.tar.gz", tmp_path / "restored", lambda _: None
        )
    assert not (tmp_path / "outside.py").exists()


def test_notebook_has_no_online_installation():
    notebook = json.loads(
        (FINAL / "notebooks/06_final.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert '"sync"' not in code and '"install"' not in code
    assert "runtime_module.prepare" in code and '"UV_OFFLINE": "1"' in code
    assert "sleep_edf_final_v2" in code
