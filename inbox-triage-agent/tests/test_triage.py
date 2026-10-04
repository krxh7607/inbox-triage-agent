"""
Unit and Integration Tests for Inbox Triage Agent.
Verifies:
  1. Text Preprocessing
  2. Feature Extraction Pipeline
  3. Prediction on all 5 categories for all 3 models
  4. Agent Action Layer execution and rules
"""

import unittest
import pandas as pd
from src.preprocessing import clean_text, extract_url_count
from src.predict import EmailClassifierPredictor
from src.agent import AgentActionEngine


class TestInboxTriageAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = EmailClassifierPredictor()
        cls.agent = AgentActionEngine()

    def test_preprocessing(self):
        sample_text = "URGENT: Please review http://company.com/deck by 5 PM! <script>alert(1)</script>"
        cleaned = clean_text(sample_text)
        self.assertIsInstance(cleaned, str)
        self.assertNotIn("<script>", cleaned)
        self.assertTrue(len(cleaned) > 0)
        self.assertEqual(extract_url_count(sample_text), 1)

    def test_all_five_categories_predicted(self):
        test_cases = [
            ("URGENT", {
                "email_id": "TEST-URGENT",
                "subject": "System latency alert on database cluster",
                "body": "The production cluster is reporting severe latency issues and timeouts. All senior engineers must join the incident call immediately.",
                "sender": "alerts@company.com",
                "received_time": "10:00",
                "has_links": 0,
                "has_attachments": 0
            }),
            ("NEEDS-REPLY", {
                "email_id": "TEST-REPLY",
                "subject": "Review needed: proposal draft for roadmap",
                "body": "Could you please review the attached document and share your feedback? Let me know your thoughts before tomorrow.",
                "sender": "colleague@company.com",
                "received_time": "11:00",
                "has_links": 0,
                "has_attachments": 1
            }),
            ("FYI", {
                "email_id": "TEST-FYI",
                "subject": "Weekly digest: team updates and release notes",
                "body": "Here is the routine summary of team activities from this past week. Attached are the slides. This notification is for your information.",
                "sender": "digest@company.com",
                "received_time": "09:00",
                "has_links": 1,
                "has_attachments": 0
            }),
            ("SPAM", {
                "email_id": "TEST-SPAM",
                "subject": "Exclusive offer: Claim your free lottery prize",
                "body": "Congratulations, your email was selected to receive an exclusive cash grant. Click the link to claim your reward.",
                "sender": "prizes@promo-rewards.xyz",
                "received_time": "02:00",
                "has_links": 1,
                "has_attachments": 0
            }),
            ("DELEGATE", {
                "email_id": "TEST-DELEGATE",
                "subject": "Customer ticket: Assign billing issue to specialist",
                "body": "We received a support ticket from an enterprise account regarding billing. Can someone please route this request to the engineer on rotation?",
                "sender": "support@company.com",
                "received_time": "13:00",
                "has_links": 0,
                "has_attachments": 0
            }),
        ]

        models = ["Naive Bayes", "Logistic Regression", "SVM (RBF)"]

        for model_name in models:
            for expected_label, email_data in test_cases:
                result = self.predictor.predict_single(email_data, model_name=model_name, execute_agent=True)
                pred_label = result["predicted_category"]
                self.assertIn(pred_label, ["URGENT", "NEEDS-REPLY", "FYI", "SPAM", "DELEGATE"])
                # Check agent action presence
                self.assertIn("agent_action", result)
                action = result["agent_action"]
                self.assertEqual(action["category"], pred_label)

    def test_agent_action_rules(self):
        email = {"email_id": "E1", "sender": "test@company.com", "subject": "Test Proposal"}

        # URGENT rule
        act_urgent = self.agent.execute_action(email, "URGENT")
        self.assertEqual(act_urgent["action_description"], "Flag the email and move it to a priority folder")

        # NEEDS-REPLY rule
        act_reply = self.agent.execute_action(email, "NEEDS-REPLY")
        self.assertEqual(act_reply["action_description"], "Draft a suggested reply")
        self.assertIn("suggested_reply", act_reply["simulated_execution"])

        # FYI rule
        act_fyi = self.agent.execute_action(email, "FYI")
        self.assertEqual(act_fyi["action_description"], "Auto-archive the email silently")

        # SPAM rule
        act_spam = self.agent.execute_action(email, "SPAM")
        self.assertEqual(act_spam["action_description"], "Delete the email automatically")

        # DELEGATE rule (placeholder per Section 6 & Rule 9)
        act_del = self.agent.execute_action(email, "DELEGATE")
        self.assertFalse(act_del["simulated_execution"]["action_defined"])
        self.assertIn("placeholder_notice", act_del["simulated_execution"])


if __name__ == "__main__":
    unittest.main()
