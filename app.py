import os
from flask import Flask, render_template, request, jsonify
from groq import Groq
from hindsight import HindsightClient

app = Flask(__name__)

# Model to use as requested
MODEL_NAME = "openai/gpt-oss-120b"

def get_hindsight_client():
    hindsight_api_key = os.environ.get("HINDSIGHT_API_KEY")
    hindsight_base_url = os.environ.get("HINDSIGHT_BASE_URL", "http://localhost:8888")
    if not hindsight_api_key:
        raise ValueError("HINDSIGHT_API_KEY environment variable is missing.")
    return HindsightClient(base_url=hindsight_base_url, api_key=hindsight_api_key)

def get_groq_client():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")
    return Groq(api_key=groq_api_key)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    customer_id = data.get("customer_id")
    user_message = data.get("message")

    if not customer_id or not user_message:
        return jsonify({"error": "Both 'customer_id' and 'message' are required."}), 400

    try:
        groq_client = get_groq_client()
        hindsight_client = get_hindsight_client()
    except ValueError as err:
        return jsonify({"error": str(err)}), 500

    # 1. Check Hindsight (memory system) for past information about this customer
    recalled_context_items = []
    try:
        recall_response = hindsight_client.recall(bank_id=customer_id, query=user_message)
        if recall_response and getattr(recall_response, "results", None):
            for res in recall_response.results:
                if hasattr(res, "text") and res.text:
                    recalled_context_items.append(res.text)
    except Exception as e:
        app.logger.warning(f"Could not recall memory from Hindsight: {e}")

    context_str = "\n".join(recalled_context_items) if recalled_context_items else "No previous memory found for this customer."

    # 2. Construct system prompt incorporating recalled context
    system_prompt = (
        "You are a helpful and polite customer support agent. "
        "Below is known context/past information about this customer retrieved from memory:\n\n"
        f"--- Customer Context ---\n{context_str}\n------------------------\n\n"
        "Use this context to personalize your response when relevant."
    )

    # 3. Call Groq API to get reply
    try:
        completion = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        ai_reply = completion.choices[0].message.content
    except Exception as e:
        app.logger.error(f"Error calling Groq API: {e}")
        return jsonify({"error": f"Failed to get response from Groq API: {str(e)}"}), 500

    # 4. Save new information/conversation into Hindsight for this customer
    new_memory_content = f"User message: {user_message}\nAssistant reply: {ai_reply}"
    try:
        hindsight_client.retain(bank_id=customer_id, content=new_memory_content)
    except Exception as e:
        app.logger.warning(f"Could not save memory to Hindsight: {e}")

    return jsonify({
        "reply": ai_reply,
        "recalled_context": recalled_context_items
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
