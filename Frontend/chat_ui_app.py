# chat_ui_app.py
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# We'll assume your main backend runs at "http://localhost:9000"
# with endpoints like:
#   /meetAndConnect/query
#   /meetAndConnect/recommendedQueries
#   /primaryWorkDeskData/query
#   /primaryWorkDeskData/recommendedQueries

@app.route('/')
def index():
    """Render the main chat UI."""
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    """
    Receives the user's message + selected dataset from the front-end,
    calls the corresponding endpoint on the main server,
    and returns the AI's response (and optionally image).
    """
    user_message = request.form.get('message', '')
    dataset_type = request.form.get('dataset_type', 'meetAndConnect')  # default

    # Construct the correct URL based on the dataset
    if dataset_type == 'meetAndConnect':
        query_url = "http://localhost:9000/meetAndConnect/query"
    else:
        query_url = "http://localhost:9000/primaryWorkDeskData/query"

    payload = {
        "query": user_message
    }

    try:
        # Increase timeout if queries can take a while (e.g. 90s)
        response = requests.post(query_url, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()

        # LLM server returns { "response": ..., "image": ... }
        ai_text = data.get('response', '')
        # if the "response" is a nested dict { "response": "text", "metadata": ... }, handle that:
        if isinstance(ai_text, dict):
            ai_text = ai_text.get('response', '')

        ai_image = data.get('image', None)

        return jsonify({
            "status": "success",
            "response_text": ai_text,
            "response_image": ai_image
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get_recommended_queries', methods=['POST'])
def get_recommended_queries():
    """
    When the user switches dataset, we fetch recommended queries from the main server
    depending on the dataset selected. 
    Expects JSON with {"dataset_type": "meetAndConnect" or "primaryWorkDeskData"}.
    """
    data = request.get_json()
    dataset_type = data.get('dataset_type', 'meetAndConnect')
    
    if dataset_type == 'meetAndConnect':
        recommended_url = "http://localhost:9000/meetAndConnect/recommendedQueries"
    else:
        recommended_url = "http://localhost:9000/primaryWorkDeskData/recommendedQueries"
    
    try:
        resp = requests.get(recommended_url, timeout=10)
        resp.raise_for_status()
        rec_data = resp.json()  # Should have {"queries": [...]} 
        return jsonify(rec_data)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run this front-end on port 2000, while backend is on 9000
    app.run(debug=True, port=2000)