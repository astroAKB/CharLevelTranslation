from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
 

# try:
#     with open('trained_model.pkl', 'rb') as f:
#         rnn_model = pickle.load(f)
# except FileNotFoundError:
#     print("Error: trained_model.pkl not found.")
#     exit()

try:
    with open('trained_model.pkl', 'rb') as f:
        rnn_model = pickle.load(f)
    with open('char2int.pkl', 'rb') as f:
        char2int = pickle.load(f)
    
except FileNotFoundError:
    print("Error: char2int.pkl or int2char.pkl not found. Make sure they are in the same directory or provide the correct path.")

app = Flask(__name__)

import numpy as np

ch_array = np.array(['\n',' ', '!', '&', "'", '(', ')', ',', '-', '.', '1', ':', ';', '?', 'A', 'B', 'C', 'D',
                     'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                     'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                     'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

print(ch_array)
print(ch_array.shape)
def sample(model, start_str, len_gen=500, max_input_len=40, scale_factor=1.0):
    encoded_input = [char2int[s] for s in start_str]
    encoded_input = tf.reshape(encoded_input, (1, -1))

    generated = start_str
    # model.reset_states()
    
    for i in range(len_gen):
        logits = model(encoded_input)
        logits = tf.squeeze(logits, 0)

        scaled_logits = logits*scale_factor
        new_char_idx = tf.random.categorical(scaled_logits, num_samples=1)
        new_char_idx = tf.squeeze(new_char_idx)[-1].numpy()

        generated += str(ch_array[new_char_idx])

        new_char_idx = tf.expand_dims([new_char_idx], 0)
        encoded_input = tf.concat([encoded_input, new_char_idx], axis=1)
        encoded_input = encoded_input[:, -max_input_len:]
    
    return generated

# Dummy text generator (you'll replace this with your model)
def generate_text(input_text):
    print(input_text)
    s = sample(rnn_model, start_str=input_text, len_gen=100)
    print("done")
    return s

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

