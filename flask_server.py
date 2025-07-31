from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/precaution", methods=["POST"])
def generate_precaution():
    try:
        disease = request.json["disease"]
        prompt = f"What are some prevention and treatment tips for {disease} in plants?"
        response = model.generate_content(prompt)
        return jsonify({"Precaution": response.text})
    except Exception as e:
        return jsonify({"Precaution": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
