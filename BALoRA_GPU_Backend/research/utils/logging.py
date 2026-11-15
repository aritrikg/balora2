"""Small JSONL event logger for auditable experiment metadata."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any
class JSONLLogger:
    def __init__(self, path: str | Path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def log(self, event: str, **fields: Any) -> None:
        row={"time_unix":time.time(),"event":str(event),**fields}
        with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,default=str,allow_nan=False)+"\n")
