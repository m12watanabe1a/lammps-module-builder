#!/usr/bin/env python3

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone

VERSION = "0.0.0"
FILEPATH = "build_info.json"


@dataclass
class BuildFile:
    path: str


@dataclass
class BuildInfo:
    build_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    files: list[BuildFile] = field(default_factory=list)
    version: str = VERSION

    def add_file(self, filepath: str | os.PathLike[str]) -> None:
        self.files.append(BuildFile(path=str(filepath)))

    def save(self) -> None:
        with open(FILEPATH, "w", encoding="utf-8") as f:
            json.dump(self, f, default=lambda o: o.__dict__, indent=2)

    def delete(self) -> None:
        for file in self.files:
            if os.path.exists(file.path):
                if os.path.isdir(file.path):
                    shutil.rmtree(file.path)
                else:
                    os.remove(file.path)
        if os.path.exists(FILEPATH):
            os.remove(FILEPATH)

    @staticmethod
    def exists() -> bool:
        return os.path.exists(FILEPATH)

    @classmethod
    def load(cls) -> "BuildInfo":
        if not os.path.exists(FILEPATH):
            return cls()
        with open(FILEPATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        files = [BuildFile(**file) for file in data.get("files", [])]
        return cls(
            build_time=data.get("build_time", ""),
            files=files,
            version=data.get("version"),
        )
