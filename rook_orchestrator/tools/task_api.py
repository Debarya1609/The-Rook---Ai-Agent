# rook_orchestrator/tools/task_api.py
from typing import Dict, Any, Optional
import uuid

class TaskAPI:
    def __init__(self):
        # tasks: id -> payload (payload includes 'assignee' if provided)
        self.tasks: Dict[str, Dict[str,Any]] = {}

    def create_task(self, payload: Dict[str,Any]) -> Dict[str,Any]:
        # allow explicit task_id or generate one
        tid = payload.get("task_id") or str(uuid.uuid4())
        # normalize payload fields
        stored = {
            "task_id": tid,
            "task": payload.get("task"),
            "assignee": payload.get("assignee"),
            "due": payload.get("due"),
            "meta": payload.get("meta", {})
        }
        self.tasks[tid] = stored
        return {"ok": True, "task_id": tid, "task": stored}

    def reassign(self, task_id: str, to: str) -> Dict[str,Any]:
        if task_id in self.tasks:
            self.tasks[task_id]['assignee'] = to
            return {"ok": True, "task_id": task_id, "new_assignee": to}
        return {"ok": False, "reason": "not_found"}

    def find_task_by_assignee(self, assignee: str) -> Optional[str]:
        """
        Try to find a task id assigned to `assignee`.
        Matching is case-insensitive and allows partial matches.
        Returns task_id or None.
        """
        if not assignee:
            return None
        target = assignee.lower()
        # exact match first
        for tid, payload in self.tasks.items():
            a = payload.get("assignee")
            if a and a.lower() == target:
                return tid
        # partial / contains match
        for tid, payload in self.tasks.items():
            a = payload.get("assignee")
            if a and target in a.lower():
                return tid
        return None
