# rook_orchestrator/tools/task_api.py
from typing import Dict, Any
import uuid

class TaskAPI:
    def __init__(self):
        self.tasks: Dict[str, Dict[str,Any]] = {}

    def create_task(self, payload: Dict[str,Any]) -> Dict[str,Any]:
        tid = payload.get("task_id") or str(uuid.uuid4())
        self.tasks[tid] = payload
        return {"ok": True, "task_id": tid}

    def reassign(self, task_id: str, to: str) -> Dict[str,Any]:
        if task_id in self.tasks:
            self.tasks[task_id]['assignee'] = to
            return {"ok": True, "task_id": task_id, "new_assignee": to}
        return {"ok": False, "reason": "not_found"}
