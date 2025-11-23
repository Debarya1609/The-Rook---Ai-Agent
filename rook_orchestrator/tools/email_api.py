# rook_orchestrator/tools/email_api.py
from typing import Dict, Any

class EmailAPI:
    def send(self, to: str, subject: str, body: str) -> Dict[str,Any]:
        # deterministic mock: return a fake message id
        return {"ok": True, "to": to, "subject": subject, "message_id": "msg_12345"}
