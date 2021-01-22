# -*- coding:utf-8 -*-

import numpy as np
import scipy.special


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes
        self.lr = learningrate

        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        # print(output_errors.shape)
        # print(np.sum(output_errors**2))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


def do_train(mydnn, train_file, max_steps=100000):
    with open(train_file, "rt") as fp:
        lines = fp.readlines()

    for i, line in enumerate(lines):
        if i > max_steps:
            break
        lst_v = line.split(",")
        inputs = np.asfarray(lst_v[1:]) / 255.0 * 0.99 + 0.01
        targets = np.zeros(10) + 0.01
        targets[int(lst_v[0])] = 0.99
        mydnn.train(inputs, targets)
        if i % 10000 == 0:
            print("i : ", i)


def do_test(mydnn, test_file, max_steps=100):
    with open(test_file, "rt") as fp:
        lines = fp.readlines()

    lst_res = list()
    for i, line in enumerate(lines):
        if i > max_steps:
            break
        lst_v = line.split(",")
        inputs = np.asfarray(lst_v[1:]) / 255.0 * 0.99 + 0.01
        right_num = int(lst_v[0])
        outputs = mydnn.query(inputs)
        lst_res.append(1 if right_num == np.argmax(outputs) else 0)
        # print(outputs)
        # print(np.argmax(outputs))
    print("正确率 ： ", float(sum(lst_res))/len(lst_res))

def main():
    inodes = 28*28
    hnodes = 100
    onodes = 10
    lr = 0.3
    mydnn = neuralNetwork(inodes, hnodes, onodes, lr)
    do_train(mydnn, "./csv/mnist_train.csv", max_steps=100000)
    do_test(mydnn, "./csv/mnist_test.csv", max_steps=100)


if __name__ == "__main__":
    print("hi")

    main()
