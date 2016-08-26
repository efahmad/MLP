import neural_network
import numpy as np
import random
import matplotlib.pyplot as plt


def get_class_title(class_vector):
    bin_index = ''
    str_vector = [str(x) for x in class_vector]
    bin_index = bin_index.join(str_vector)
    if bin_index.__contains__("-"):
        return ''
    index = int(bin_index, base=2)
    if 0 <= index < len(class_titles):
        return class_titles[index]
    return ''


def get_class_vector(value):
    index = class_titles.index(value)
    index_binary_list = bin(index)[2:]
    binary_list = [0 for x in bin(len(class_titles))[2:]]
    binary_list[-len(index_binary_list):] = [int(x) for x in index_binary_list]
    return binary_list


def run_and_print_results(x, y, epochs=10000):
    # normalize the values to bring them into the range 0-1
    # x_min = x.min()
    # x_max = x.max()
    # x = (x - x_min) / float(x_max - x_min)

    nn.fit(x, y, epochs=epochs)
    # print("\n\nFinal Weights:")
    # print(nn.weights)
    # print("\n\n")
    all_count = len(validation_data)
    errors_count = 0
    for i in range(all_count):
        result = nn.predict(validation_data_attributes[i])
        result = [int(round(x)) for x in result]
        if result != validation_data_classes[i]:
            errors_count += 1
        # print("Validation number " + str(i + 1) + ": " + str(validation_data_attributes[i]))
        # print("calculated result: " + str(result) + " --> " + get_class_title(result))
        # print("target:            " + str(validation_data_classes[i])
        #       + " --> " + get_class_title(validation_data_classes[i]))
        # print("")

    print("----------------- Epochs: " + str(epochs) + " -----------------")
    print("All validations: " + str(all_count))
    print("Errors: " + str(errors_count))
    performance = (all_count - errors_count) * 100 / all_count
    print("Performance: " + str(performance) + "%")
    print("\n")
    return performance


examples = []
# Read data from file
for line in open("data/iris.dat", 'r'):
    # strip off newline and any other trailing whitespace
    line = line.strip()
    cols = line.split(",")
    examples.append([float(numeric_string) for numeric_string in cols[0:-1]])
    examples[-1].append(cols[-1].strip())

features_count = len(examples[0]) - 1
# Take 4/5 of all data for training
training_data_count = len(examples) * 4 / 5

# Get class titles
class_titles = [example[-1] for example in examples]
class_titles = set(class_titles)
class_titles = list(class_titles)

# Convert class titles to numbers
for example in examples:
    example[-1] = get_class_vector(example[-1])

# Determine training and validation parts of data
training_data = []
validation_data = []
rand_indices = random.sample(range(0, len(examples)), training_data_count)
for i in range(len(examples)):
    if i in rand_indices:
        training_data.append(examples[i])
    else:
        validation_data.append(examples[i])

# Separate features from classes in training and validation data
training_data_attributes = [training[0:-1] for training in training_data]
training_data_classes = [training[-1] for training in training_data]
validation_data_attributes = [validation[0:-1] for validation in validation_data]
validation_data_classes = [validation[-1] for validation in validation_data]

# Init neural network
nn = neural_network.NeuralNetwork(
        [features_count, features_count + 2, len(validation_data_classes[0])], 'tanh')  # logistic
# print("\n\nInitial Weights:")
# print(nn.weights)
initialWeights = np.copy(nn.weights)
attributes = np.array(training_data_attributes)
targets = np.array(training_data_classes)

# Run the neural network & print the results
performances = []
epochs = [5000, 10000, 15000, 20000]
for epoch in epochs:
    nn.weights = np.copy(initialWeights)
    performances.append(run_and_print_results(attributes, targets, epoch))
# Draw and show the plot
plt.plot(epochs, performances, linewidth=2.0)
plt.grid(True)
plt.axis([0, 25000, 0, 110])
plt.show()
