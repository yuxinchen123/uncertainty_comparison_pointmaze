"""Install the upstream alternative sprite archive without declaring ROM ownership."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

URL = "https://drive.google.com/uc?export=download&id=1yODhHqXMvSbMlVeCM8vfekU1lPbkgbCb"


def install(destination):
    """Fetch the upstream alternative archive into a study-owned data directory."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rnd_atari_sprites_") as scratch:
        archive_path = Path(scratch) / "alternative_sprites.zip"
        with urllib.request.urlopen(URL, timeout=90) as response:
            with archive_path.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive_path) as archive:
            # Reject traversal before extracting the upstream archive.
            root = (Path(scratch) / "extracted").resolve()
            for item in archive.infolist():
                if not (root / item.filename).resolve().is_relative_to(root):
                    raise ValueError("the sprite archive contains a path outside its extraction directory")
            archive.extractall(root)
        sources = sorted((path for path in root.rglob("*")
                          if path.is_dir() and path.name in ("sprites", "custom_sprites")),
                         key=lambda path: len(path.parts))
        if not sources:
            raise ValueError("the upstream alternative archive has no sprite directory")
        target = destination / "jaxtari"
        shutil.copytree(sources[0], target / "sprites", dirs_exist_ok=True)
        (target / ".alternative_sprites_installed").touch()
        (target / "sprites" / ".version").write_text("1\n")
        record = {"source_url": URL, "archive_sha256": digest,
                  "mode": "alternative_sprites", "sprite_version": 1}
        (target / "asset_source.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    install(parser.parse_args().destination)
