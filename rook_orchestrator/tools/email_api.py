# rook_orchestrator/tools/email_api.py
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Any

class EmailClient:
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str, use_tls: bool = True):
        self.smtp_host = smtp_host #smtp.gmail.com -  we can read these from the .env
        self.smtp_port = smtp_port #587  we can read these from the .env
        self.username = username  #we can read these from the .env
        self.password = password # we can read these from the .env
        self.use_tls = use_tls

    def send(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Sends an email and returns a result dictionary.
        """
        msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = self.username
        msg["To"] = to
        msg["Subject"] = subject

        try:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)

            if self.use_tls:
                server.starttls()

            server.login(self.username, self.password)
            server.sendmail(self.username, [to], msg.as_string())
            server.quit()

            return {
                "success": True,
                "message": "Email sent successfully",
                "to": to,
                "subject": subject
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "to": to,
                "subject": subject
            }

