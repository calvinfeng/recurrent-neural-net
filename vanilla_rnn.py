# Author(s) Calvin Feng, 2018
import numpy as np
import time
from numpy import dot, tanh, exp
from pdb import set_trace


class VanillaRNNModel(object):
    """Vanilla recurrent neural network with no fancy optimization technique.

    :param integer hidden_dim: Size of the hidden layer of neurons
    :param integer input_dim: Please note that input dimension is the same as output dimension for this character model
    """
    def __init__(self, input_dim, hidden_dim):
        self.input_dim, self.output_dim = input_dim, input_dim
        self.hidden_dim = hidden_dim
        self.Wxh = np.random.randn(self.hidden_dim, self.input_dim) * 0.01
        self.Whh = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
        self.Why = np.random.randn(self.output_dim, self.hidden_dim) * 0.01
        self.Bh = np.zeros((self.hidden_dim, 1))  # hidden bias
        self.By = np.zeros((self.output_dim, 1))  # output bias

    def sample_chars(self, hidden_state, seed_idx, num_seq):
        """Sample a sequence of characters from the current model, this is primarily used for test time"""
        x = np.zeros((self.input_dim, 1))
        x[seed_idx] = 1
        indices = []
        for t in xrange(num_seq):
            hidden = tanh(dot(self.Wxh, x) + dot(self.Whh, hidden_state) + self.Bh)
            output = dot(self.Why, hidden) + self.By
            prob = exp(output) / np.sum(exp(output))
            idx = np.random.choice(range(self.input_dim), p=prob.ravel())  # ravel() flattens the matrix
            x = np.zeros((self.input_dim, 1))
            x[idx] = 1
            indices.append(idx)

        return indices

    def loss(self, input_list, target_list, prev_hidden_state):
        """
        Usage:
        Given vocabs = ['a', 'b', 'c', 'd'], if input sequence was 'dad', then the input list is interpretered as
        [3, 0, 3]; Same applies for target list.

        :param []int input_list: List of integers that represent indices of the characters from an input sequence
        :param []int target_list: List of integers that represent indices of the characters from a target sequence
        """
        input_states, hidden_states, output_states, prob_states = {}, {}, {}, {}
        hidden_states[-1] = np.copy(prev_hidden_state)

        # Perform forward pass
        loss = 0
        for t in xrange(len(input_list)):
            # Encode input state in 1-of-k representation
            input_states[t] = np.zeros((self.input_dim, 1))
            input_states[t][input_list[t]] = 1

            # Compute hidden state
            hidden_states[t] = tanh(dot(self.Wxh, input_states[t]) + dot(self.Whh, hidden_states[t-1]) + self.Bh)

            # Compute output state a.k.a. unnomralized log probability using current hidden state
            output_states[t] = dot(self.Why, hidden_states[t]) + self.By

            # Compute softmax probability state using the output state
            prob_states[t] = exp(output_states[t]) / np.sum(exp(output_states[t]))

            loss += -np.log(prob_states[t][target_list[t], 0])  # Remember that prob is an (O, 1) vector.

        # Perform back propagation
        grad_prev_h = np.zeros((self.hidden_dim, 1))
        grad_Wxh, grad_Whh, grad_Why = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        grad_Bh, grad_By = np.zeros_like(self.Bh), np.zeros_like(self.By)

        for t in reversed(xrange(len(input_list))):
            # Compute softmax gradient
            grad_output = np.copy(prob_states[t])
            grad_output[target_list[t]] -= 1

            # Backward pass into the weights
            grad_Why += dot(grad_output, hidden_states[t].T)  # (O, 1)(1, H) => (O, H)
            grad_By += grad_output  # (O, 1) + (O, 1)

            # Compute gradient of next hidden state, which is the output of the tanh non-linearity
            grad_h = dot(self.Why.T, grad_output) + grad_prev_h  # (H, O)(O, H) => (H, H)

            # Take derivative of tanh, using u-substitutions, in this case, u = dot(Whh, h[t-1]) + dot(Wxh, x[t]) + Bh
            grad_u = (1 - hidden_states[t] * hidden_states[t]) * grad_h
            grad_Bh += grad_u

            # Compute gradient of the last two W's
            grad_Wxh += dot(grad_u, input_states[t].T)
            grad_Whh += dot(grad_u, hidden_states[t-1].T)

            # Finally, compute grad_prev_h, i.e. gradient of loss with respect to h[t - 1], because we need this for the
            # t - 1 step just like what we did at line 86.
            grad_prev_h = dot(self.Whh.T, grad_u)

        for grad_param in [grad_Wxh, grad_Whh, grad_Why, grad_By, grad_Bh]:
            np.clip(grad_param, -5, -5, out=grad_param)  # Clip to mitigate exploding gradients

        return loss, grad_Wxh, grad_Whh, grad_Why, grad_By, grad_Bh, hidden_states[len(input_list) - 1]


def load_dictionary(filepath):
    with open(filepath, 'r') as file:
        text_data = file.read()
        chars = list(set(text_data))
        num_chars, num_unique_chars = len(text_data), len(chars)

        # Create a mapping from character to idx
        char_to_idx = dict()
        for i, ch in enumerate(chars):
            char_to_idx[ch] = i

        # Create a mapping from idx to character
        idx_to_char = dict()
        for i, ch in enumerate(chars):
            idx_to_char[i] = ch

        print "text document contains %d characters and has %d unique characters" % (num_chars, num_unique_chars)
        return text_data, char_to_idx, idx_to_char


def adagrad_optimize(model, grads, mem, learning_rate=1e-1):
    for param in mem:
        mem[param] += grads[param] * grads[param]
        updated_param = getattr(model, param) - (learning_rate * grads[param] / np.sqrt(mem[param] + 1e-8))
        setattr(model, param, updated_param)


def main():
    hidden_size = 100
    seq_length = 50
    learning_rate = 1e-3
    text_data, char_to_idx, idx_to_char = load_dictionary("datasets/word_dictionary.txt")
    model = VanillaRNNModel(len(char_to_idx), hidden_size)

    # Create memory variables for Adagrad
    mem = {
        "Wxh": np.zeros_like(model.Wxh),
        "Whh": np.zeros_like(model.Whh),
        "Why": np.zeros_like(model.Why),
        "Bh": np.zeros_like(model.Bh),
        "By": np.zeros_like(model.By)
    }

    step, pointer, epoch_size, grads, smooth_loss = 0, 0, 100, dict(), -np.log(1.0/len(char_to_idx))*seq_length
    while True:
        if step == 0 or pointer + seq_length + 1 >= len(text_data):
            prev_hidden_state = np.zeros((hidden_size, 1))  # Reset RNN memory
            pointer = 0  # Reset the pointer

        # Since we are trying to predict next letter in the sequence, the target is simply pointer + 1
        input_list = [char_to_idx[ch] for ch in text_data[pointer:pointer+seq_length]]
        target_list = [char_to_idx[ch] for ch in text_data[pointer+1: pointer+seq_length+1]]

        if step % epoch_size == 0:
            sampled_indices = model.sample_chars(prev_hidden_state, input_list[0], 200)
            predicted_text = ''.join(idx_to_char[idx] for idx in sampled_indices)
            print "-------------\n%s\n-------------" % predicted_text

        # Forward Prop
        loss, grads['Wxh'], grads['Whh'], grads['Why'], grads['By'], grads['Bh'], prev_hidden_state = model.loss(input_list,
                                                                                                            target_list,
                                                                                                            prev_hidden_state)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if step % epoch_size == 0:
            print "iter. step %d, loss: %f" % (step, smooth_loss)

        adagrad_optimize(model, grads, mem, learning_rate)

        step += 1
        pointer += seq_length


if __name__ == '__main__':
    main()
