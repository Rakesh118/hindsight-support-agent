import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

GROQ_MODEL = "openai/gpt-oss-120b"

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)

def get_hindsight_client():
    api_key = os.environ.get("HINDSIGHT_API_KEY")
    base_url = os.environ.get("HINDSIGHT_API_URL") or os.environ.get("HINDSIGHT_BASE_URL")
    if not base_url or not api_key:
        return None
    from hindsight_client import Hindsight
    return Hindsight(base_url=base_url, api_key=api_key)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    customer_id = data.get("customer_id", "").strip() or "default_customer"

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    groq_client = get_groq_client()
    if not groq_client:
        return jsonify({
            "error": "GROQ_API_KEY environment variable is missing. Please set GROQ_API_KEY."
        }), 500

    hindsight_client = get_hindsight_client()

    # Step 1: Check Hindsight memory for past customer context
    past_context = ""
    recalled_items = []
    if hindsight_client:
        try:
            recall_response = hindsight_client.recall(bank_id=customer_id, query=message)
            if hasattr(recall_response, "to_prompt_string") and callable(recall_response.to_prompt_string):
                past_context = recall_response.to_prompt_string()
            elif hasattr(recall_response, "results") and recall_response.results:
                past_context = "\n".join([str(res) for res in recall_response.results])

            if hasattr(recall_response, "results") and recall_response.results:
                recalled_items = [str(res) for res in recall_response.results]
        except Exception as e:
            app.logger.error(f"Error recalling memory from Hindsight: {e}")

    # Step 2: Build prompt with system instructions and customer context
    system_prompt = (
        "You are a helpful and polite customer support AI agent for our service. "
        "Use the provided customer background information to personalize your answer if relevant."
    )
    if past_context:
        system_prompt += f"\n\nContext about this customer (Customer ID: {customer_id}):\n{past_context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    # Step 3: Get reply from Groq API
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        app.logger.error(f"Error calling Groq API: {e}")
        return jsonify({"error": f"Failed to get reply from AI model: {str(e)}"}), 500

    # Step 4: Save new conversation information to Hindsight
    if hindsight_client:
        try:
            conversation_record = f"Customer message: {message}\nSupport response: {reply}"
            hindsight_client.retain(bank_id=customer_id, content=conversation_record)
        except Exception as e:
            app.logger.error(f"Error retaining memory to Hindsight: {e}")

    return jsonify({
        "reply": reply,
        "customer_id": customer_id,
        "recalled_context": past_context,
        "recalled_items": recalled_items
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
