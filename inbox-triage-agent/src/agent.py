"""
Agent / Action Layer for Inbox Triage Agent.
Decoupled from the classification engine (Email -> Category, Category -> Action).

Specification Rule Mapping (Section 6 & 22):
  - URGENT      -> Flag the email and move it to a priority folder
  - NEEDS-REPLY -> Draft a suggested reply
  - FYI         -> Auto-archive the email silently
  - SPAM        -> Delete the email automatically
  - DELEGATE    -> Display classification & placeholder indicating automated action is to be defined
"""

from typing import Dict, Any
from datetime import datetime


class AgentActionEngine:
    """
    Executes automated triage rules based strictly on predicted email classification.
    All actions are clearly tagged as SIMULATED in compliance with Project Rules 7 & 8.
    """

    def __init__(self):
        self.action_history = []

    def generate_suggested_reply(self, email: Dict[str, Any]) -> str:
        """
        Drafts an intelligent, contextual suggested reply template for NEEDS-REPLY emails.
        """
        sender = email.get("sender", "Colleague")
        sender_name = sender.split("@")[0].replace(".", " ").title() if "@" in sender else "Colleague"
        subject = email.get("subject", "")

        # Dynamic draft generation based on subject patterns
        subj_lower = subject.lower()
        if "proposal" in subj_lower or "deck" in subj_lower:
            reply = (
                f"Hi {sender_name},\n\n"
                f"Thank you for sharing the proposal. I have reviewed the key points and will provide "
                f"detailed inline feedback by end of day today.\n\n"
                f"Best regards,\nInbox Triage Agent (on behalf of User)"
            )
        elif "sync" in subj_lower or "availability" in subj_lower or "meeting" in subj_lower:
            reply = (
                f"Hi {sender_name},\n\n"
                f"Thanks for reaching out. Both options work on my calendar, but Tuesday afternoon is "
                f"slightly preferred. Feel free to place a calendar invite.\n\n"
                f"Best regards,\nInbox Triage Agent (on behalf of User)"
            )
        elif "feedback" in subj_lower or "review" in subj_lower:
            reply = (
                f"Hi {sender_name},\n\n"
                f"Thanks for following up on this. I've noted the request and will submit my evaluation notes "
                f"shortly.\n\n"
                f"Best regards,\nInbox Triage Agent (on behalf of User)"
            )
        else:
            reply = (
                f"Hi {sender_name},\n\n"
                f"Thank you for your message regarding '{subject}'. I am currently looking into this and "
                f"will follow up with a detailed response shortly.\n\n"
                f"Best regards,\nInbox Triage Agent (on behalf of User)"
            )
        return reply

    def execute_action(self, email: Dict[str, Any], category: str, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Applies predefined action rule based on predicted category.
        Returns detailed action execution result.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        email_id = email.get("email_id", "EML-DEMO")
        subject = email.get("subject", "No Subject")

        if category == "URGENT":
            action_result = {
                "category": "URGENT",
                "action_type": "FLAG_AND_PRIORITIZE",
                "action_description": "Flag the email and move it to a priority folder",
                "simulated_execution": {
                    "is_simulated": True,
                    "target_folder": "Inbox/Priority_Urgent",
                    "flagged": True,
                    "priority_level": "P0 - High Urgency",
                    "notification_sent": True,
                    "status": "Moved to Priority Folder & Starred"
                },
                "details": f"Email flagged as high urgency (confidence: {confidence:.1%}). Moved to Priority folder for immediate attention."
            }

        elif category == "NEEDS-REPLY":
            suggested_reply = self.generate_suggested_reply(email)
            action_result = {
                "category": "NEEDS-REPLY",
                "action_type": "DRAFT_REPLY",
                "action_description": "Draft a suggested reply",
                "simulated_execution": {
                    "is_simulated": True,
                    "target_folder": "Drafts",
                    "draft_created": True,
                    "draft_id": f"DRAFT-{email_id}",
                    "suggested_reply": suggested_reply,
                    "status": "Suggested Reply Drafted and Saved to Drafts"
                },
                "details": f"Suggested reply generated and saved to Drafts folder (confidence: {confidence:.1%}). Awaiting user review/send."
            }

        elif category == "FYI":
            action_result = {
                "category": "FYI",
                "action_type": "AUTO_ARCHIVE",
                "action_description": "Auto-archive the email silently",
                "simulated_execution": {
                    "is_simulated": True,
                    "target_folder": "Archive",
                    "marked_as_read": True,
                    "silent_action": True,
                    "notifications_suppressed": True,
                    "status": "Silently Archived"
                },
                "details": f"Email marked as informational FYI (confidence: {confidence:.1%}). Silently archived without triggering user notifications."
            }

        elif category == "SPAM":
            action_result = {
                "category": "SPAM",
                "action_type": "DELETE_AUTOMATICALLY",
                "action_description": "Delete the email automatically",
                "simulated_execution": {
                    "is_simulated": True,
                    "target_folder": "Trash",
                    "auto_purge_days": 30,
                    "sender_quarantined": True,
                    "status": "Moved to Trash (Scheduled for Deletion)"
                },
                "details": f"Identified as unsolicited spam/phishing (confidence: {confidence:.1%}). Automatically routed to Trash."
            }

        elif category == "DELEGATE":
            # Per Rule 9 & Section 6: Do not invent automated action; provide clearly marked placeholder
            action_result = {
                "category": "DELEGATE",
                "action_type": "DELEGATION_REQUIRED",
                "action_description": "Classify as an email requiring delegation handling",
                "simulated_execution": {
                    "is_simulated": True,
                    "target_folder": "Inbox/Pending_Delegation",
                    "action_defined": False,
                    "placeholder_notice": (
                        "NOTE: The project specification explicitly marks the automated action for DELEGATE as to be defined. "
                        "No automated action was invented per Project Rule 9. Email held in Pending Delegation queue for manual routing."
                    ),
                    "status": "Classified as DELEGATE (Automated Action Rule: To Be Defined)"
                },
                "details": "Classified as DELEGATE. Awaiting organization-specific delegation rule definition."
            }
        else:
            action_result = {
                "category": category,
                "action_type": "UNKNOWN",
                "action_description": "No rule defined for this category",
                "simulated_execution": {"is_simulated": True, "status": "No Action"},
                "details": "Unrecognized category."
            }

        action_result["timestamp"] = timestamp
        action_result["email_id"] = email_id
        action_result["subject"] = subject
        action_result["confidence"] = confidence

        self.action_history.append(action_result)
        return action_result
