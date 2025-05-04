import tensorflow as tf
import pickle

try:
    with open('char2int.pkl', 'rb') as f:
        char2int = pickle.load(f)
    with open('int2char.pkl', 'rb') as f:
        int2char = pickle.load(f)
except FileNotFoundError:
    print("Error: char2int.pkl or int2char.pkl not found. Make sure they are in the same directory or provide the correct path.")


def Sample_text(model, start_string, length=500, temperature=0.5):
    input_eval = [char2int[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)  # shape: (1, seq_length)
    result = list(start_string)

    for _ in range(length):
        predictions = model(input_eval)  # shape: (1, vocab_size)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(tf.math.log(predictions), num_samples=1)[0, 0].numpy()

        result.append(int2char[predicted_id])
        input_eval = tf.expand_dims([predicted_id], 0)  # shape: (1, 1)

    return ''.join(result)