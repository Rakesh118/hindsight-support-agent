import unittest
from unittest.mock import patch, MagicMock
import os
import json

from app import app, MODEL_NAME

class TestChatApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Support Agent', response.data)
        self.assertIn(b'Customer ID:', response.data)

    def test_chat_route_missing_parameters(self):
        response = self.client.post('/api/chat', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

        response = self.client.post('/api/chat', json={"customer_id": "cust_123"})
        self.assertEqual(response.status_code, 400)

        response = self.client.post('/api/chat', json={"message": "hello"})
        self.assertEqual(response.status_code, 400)

    @patch.dict(os.environ, {}, clear=True)
    def test_chat_route_missing_env_vars(self):
        response = self.client.post('/api/chat', json={
            "customer_id": "cust_123",
            "message": "Hello!"
        })
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn("error", data)

    @patch('app.Groq')
    @patch('app.HindsightClient')
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq_key", "HINDSIGHT_API_KEY": "fake_hindsight_key"})
    def test_chat_route_success(self, mock_hindsight_class, mock_groq_class):
        # Mock Hindsight client
        mock_hindsight_instance = MagicMock()
        mock_hindsight_class.return_value = mock_hindsight_instance

        mock_recall_result = MagicMock()
        mock_recall_result.text = "Customer previously ordered Widget A."
        mock_recall_response = MagicMock()
        mock_recall_response.results = [mock_recall_result]
        mock_hindsight_instance.recall.return_value = mock_recall_response

        # Mock Groq client
        mock_groq_instance = MagicMock()
        mock_groq_class.return_value = mock_groq_instance

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello! I see you previously ordered Widget A. How can I help with that?"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_groq_instance.chat.completions.create.return_value = mock_completion

        # Call /api/chat
        payload = {
            "customer_id": "cust_101",
            "message": "Where is my order?"
        }
        response = self.client.post('/api/chat', json=payload)
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data["reply"], "Hello! I see you previously ordered Widget A. How can I help with that?")
        self.assertEqual(data["recalled_context"], ["Customer previously ordered Widget A."])

        # Verify Hindsight recall was called
        mock_hindsight_instance.recall.assert_called_once_with(
            bank_id="cust_101",
            query="Where is my order?"
        )

        # Verify Groq chat completion was called with the model and context prompt
        mock_groq_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_groq_instance.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], MODEL_NAME)
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Customer previously ordered Widget A.", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "Where is my order?"})

        # Verify Hindsight retain was called
        mock_hindsight_instance.retain.assert_called_once_with(
            bank_id="cust_101",
            content="User message: Where is my order?\nAssistant reply: Hello! I see you previously ordered Widget A. How can I help with that?"
        )

if __name__ == '__main__':
    unittest.main()
