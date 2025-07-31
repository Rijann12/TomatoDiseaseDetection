from flask import Flask,request,jsonify
import google.generativeai as genai

app = Flask(__name__)
#Insert Gemini API key 
genai.configure(api_key="AIzaSyBhZHzQJ2a5Ll35ICau_LRFhSkpbN9r-B0")

model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/precaution', methods=['POST'])
def get_precaution():
    data= request.json
    print(" Received request:", data)

    disease = data.get("disease","")

    
    if not disease:
        return jsonify({"error": "Disease name is required"}),400 
    
    prompt = f"Precaution steps and remedies for the plant disease: '{disease}'?"
    try:
        response = model.generate_content(prompt)
        print(" Gemini response:", response.text)
        return jsonify({"Precaution": response.text})
    
    except Exception as e:
        print(" Gemini suggestion:", precaution_text)
        return jsonify({"error":str(e)}), 500
if __name__=='__main__' :
    app.run(host='0.0.0.0', port=5001, debug=True)
  
    