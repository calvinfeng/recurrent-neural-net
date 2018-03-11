# Author(s) Calvin Feng, 2018
import numpy as np
import time
from random import uniform
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
        self.params = {
            'Wxh': np.random.randn(self.hidden_dim, self.input_dim) * 0.01,
            'Whh': np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01,
            'Why': np.random.randn(self.output_dim, self.hidden_dim) * 0.01,
            'Bh': np.zeros((self.hidden_dim, 1)),
            'By': np.zeros((self.output_dim, 1))
        }

    def sample_chars(self, hidden_state, seed_idx, num_seq):
        """Sample a sequence of characters from the current model, this is primarily used for test time"""
        x = np.zeros((self.input_dim, 1))
        x[seed_idx] = 1
        indices = []
        for t in xrange(num_seq):
            hidden_state = tanh(dot(self.params['Wxh'], x) + dot(self.params['Whh'], hidden_state) + self.params['Bh'])
            output = dot(self.params['Why'], hidden_state) + self.params['By']
            prob = exp(output) / np.sum(exp(output))
            idx = np.random.choice(range(self.input_dim), p=prob.ravel())  # ravel() flattens the matrix
            x = np.zeros((self.input_dim, 1))
            x[idx] = 1
            indices.append(idx)

        return hidden_state, indices

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
            hidden_states[t] = tanh(dot(self.params['Wxh'], input_states[t]) +
                                    dot(self.params['Whh'], hidden_states[t-1]) +
                                    self.params['Bh'])

            # Compute output state a.k.a. unnomralized log probability using current hidden state
            output_states[t] = dot(self.params['Why'], hidden_states[t]) + self.params['By']

            # Compute softmax probability state using the output state
            prob_states[t] = exp(output_states[t]) / np.sum(exp(output_states[t]))

            loss += -np.log(prob_states[t][target_list[t], 0])  # Remember that prob is an (O, 1) vector.

        # Perform back propagation
        grads = dict()
        grad_prev_h = np.zeros((self.hidden_dim, 1))

        for name in self.params:
            grads[name] = np.zeros_like(self.params[name])

        for t in reversed(xrange(len(input_list))):
            # Compute softmax gradient
            grad_output = np.copy(prob_states[t])
            grad_output[target_list[t]] -= 1

            # Backward pass into the weights
            grads['Why'] += dot(grad_output, hidden_states[t].T)  # (O, 1)(1, H) => (O, H)
            grads['By'] += grad_output  # (O, 1) + (O, 1)

            # Compute gradient of next hidden state, which is the output of the tanh non-linearity
            grad_h = dot(self.params['Why'].T, grad_output) + grad_prev_h # (H, O)(O, H) => (H, H)

            # Take derivative of tanh, using u-substitutions, in this case, u = dot(Whh, h[t-1]) + dot(Wxh, x[t]) + Bh
            grad_u = (1 - hidden_states[t] * hidden_states[t]) * grad_h
            grads['Bh'] += grad_u

            # Compute gradient of the last two W's
            grads['Wxh'] += dot(grad_u, input_states[t].T)
            grads['Whh'] += dot(grad_u, hidden_states[t-1].T)

            # Finally, compute grad_prev_h, i.e. gradient of loss with respect to h[t - 1], because we need this for the
            # t - 1 step just like what we did at line 86.
            grad_prev_h = dot(self.params['Whh'].T, grad_u)

        for name in grads:
            np.clip(grads[name], -5, 5, out=grads[name])  # Clip to mitigate exploding gradients

        return loss, grads, hidden_states[len(input_list) - 1]

    def gradient_check(self, input_list, target_list, prev_hidden_state):
        num_checks, delta = 10, 1e-6
        _, grads, _ = self.loss(input_list, target_list, prev_hidden_state)
        for name in grads:
            if grads[name].shape != self.params[name].shape:
                raise "matrix dimensions don't match: %s and %s" % (grads[name].shape, self.params[name].shape)

            print name
            for i in xrange(num_checks):
                rand_idx = int(uniform(0, self.params[name].size))

                old_val = self.params[name].flat[rand_idx]  # flatten the matrix and use integer index to retrieve value

                self.params[name].flat[rand_idx] = old_val + delta
                fxpd, _, _ = self.loss(input_list, target_list, prev_hidden_state)

                self.params[name].flat[rand_idx] = old_val - delta
                fxmd, _, _ = self.loss(input_list, target_list, prev_hidden_state)

                analytical_grad = grads[name].flat[rand_idx]
                numerical_grad = (fxpd - fxmd) / (2 * delta)
                rel_error = abs(analytical_grad - numerical_grad) / abs(analytical_grad + numerical_grad)
                print "analytical gradient value: %f, numerical gradient value: %f error => %e" % (analytical_grad,
                                                                                                   numerical_grad,
                                                                                                   rel_error)


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
    for name in model.params:
        mem[name] += grads[name] * grads[name]
        model.params[name] += -1* (learning_rate * grads[name] / np.sqrt(mem[name] + 1e-8))


def main():
    hidden_size = 100
    seq_length = 20
    learning_rate = 1e-1
    text_data, char_to_idx, idx_to_char = load_dictionary("datasets/word_dictionary.txt")
    model = VanillaRNNModel(len(char_to_idx), hidden_size)

    # Create memory variables for Adagrad
    mem = dict()
    for name in model.params:
        mem[name] = np.zeros_like(model.params[name])

    step, pointer, epoch_size, smooth_loss = 0, 0, 100, -np.log(1.0/len(char_to_idx))*seq_length

    while True:
        if step == 0 or pointer + seq_length + 1 >= len(text_data):
            prev_hidden_state = np.zeros((hidden_size, 1))  # Reset RNN memory
            pointer = 0  # Reset the pointer

        # Since we are trying to predict next letter in the sequence, the target is simply pointer + 1
        input_list = [char_to_idx[ch] for ch in text_data[pointer:pointer+seq_length]]
        target_list = [char_to_idx[ch] for ch in text_data[pointer+1: pointer+seq_length+1]]

        # Optional gradient check
        # model.gradient_check(input_list, target_list, prev_hidden_state)

        if step % epoch_size == 0:
            _, sampled_indices = model.sample_chars(prev_hidden_state, input_list[0], 500)
            predicted_text = ''.join(idx_to_char[idx] for idx in sampled_indices)
            print "-------------\n%s\n-------------" % predicted_text

        # Forward Prop
        loss, grads, prev_hidden_state = model.loss(input_list, target_list, prev_hidden_state)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if step % epoch_size == 0:
            print "iter. step %d, loss: %f" % (step, smooth_loss)

        adagrad_optimize(model, grads, mem, learning_rate)

        step += 1
        pointer += seq_length


if __name__ == '__main__':
    main()
