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