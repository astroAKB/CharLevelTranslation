from flask import Flask, render_template, request, jsonify
from utils import Sample_text
from class import CharRNN
import pickle
import tensorflow as tf
 

try:
    with open('trained_model.pkl', 'rb') as f:
        rnn_model = pickle.load(f)
except FileNotFoundError:
    print("Error: trained_model.pkl not found.")
    exit()

app = Flask(__name__)

# Dummy text generator (you'll replace this with your model)
def generate_text(input_text):
    return Sample_text(rnn_model, start_string=input_text, length=500, temperature=0.6)
    return ans

@app.route('/')
def home():
    return render_template('index.html')  # Renders the HTML page

@app.route('/generate', methods=['POST']) 
def generate():
    data = request.get_json()
    input_text = data.get('input_text', '')
    generated_text = generate_text(input_text)
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

