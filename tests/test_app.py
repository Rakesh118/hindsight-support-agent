import pytest
from unittest.mock import MagicMock, patch
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Customer Support Chat" in response.data

def test_chat_missing_message(client):
    response = client.post('/chat', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Message cannot be empty" in data["error"]

def test_chat_missing_groq_api_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.post('/chat', json={"message": "Hello"})
    assert response.status_code == 500
    data = response.get_json()
    assert "GROQ_API_KEY environment variable is missing" in data["error"]

@patch("app.get_groq_client")
@patch("app.get_hindsight_client")
def test_chat_successful_flow(mock_get_hindsight, mock_get_groq, client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy_groq_key")
    monkeypatch.setenv("HINDSIGHT_API_KEY", "dummy_hindsight_key")
    monkeypatch.setenv("HINDSIGHT_API_URL", "http://localhost:8080")

    # Mock Groq client
    mock_groq = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Hello! How can I help you today?"
    mock_groq.chat.completions.create.return_value = mock_completion
    mock_get_groq.return_value = mock_groq

    # Mock Hindsight client
    mock_hindsight = MagicMock()
    mock_recall_res = MagicMock()
    mock_recall_res.to_prompt_string.return_value = "Past context: User prefers fast shipping."
    mock_recall_res.results = ["Past context: User prefers fast shipping."]
    mock_hindsight.recall.return_value = mock_recall_res
    mock_get_hindsight.return_value = mock_hindsight

    response = client.post('/chat', json={
        "message": "I need help with my order",
        "customer_id": "cust_999"
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data["reply"] == "Hello! How can I help you today?"
    assert data["customer_id"] == "cust_999"
    assert "Past context: User prefers fast shipping." in data["recalled_context"]

    # Verify Hindsight recall was called with expected customer bank_id and query
    mock_hindsight.recall.assert_called_once_with(bank_id="cust_999", query="I need help with my order")

    # Verify Groq chat completion model was called with openai/gpt-oss-120b
    mock_groq.chat.completions.create.assert_called_once()
    call_args = mock_groq.chat.completions.create.call_args[1]
    assert call_args["model"] == "openai/gpt-oss-120b"
    assert any("User prefers fast shipping" in msg["content"] for msg in call_args["messages"] if msg["role"] == "system")

    # Verify Hindsight retain was called with conversation details
    mock_hindsight.retain.assert_called_once()
    retain_args = mock_hindsight.retain.call_args[1]
    assert retain_args["bank_id"] == "cust_999"
    assert "Customer message: I need help with my order" in retain_args["content"]
    assert "Support response: Hello! How can I help you today?" in retain_args["content"]
