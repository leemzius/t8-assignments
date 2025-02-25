# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Numpy
import numpy as np
# Pandas
import pandas as pd
# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_dataset(excel_file_path='dataset.xlsx'):
    df = pd.read_excel(excel_file_path)
    x1 = df['x1'].values
    x2 = df['x2'].values
    y = df['class'].values
    x = np.column_stack([x1, x2])
    return x1, x2, x, y


def plot_dataset(min_val, max_val, train_val1_list, train_val2_list, train_outputs):
    # Initialize plot
    fig = plt.figure(figsize=(10, 7))

    # Scatter plot
    markers = {0: "x", 1: "o", 2: "P"}
    colors = {0: "r", 1: "g", 2: "b"}
    indexes_0 = np.where(train_outputs == 0)[0]
    v1_0 = train_val1_list[indexes_0]
    v2_0 = train_val2_list[indexes_0]
    indexes_1 = np.where(train_outputs == 1)[0]
    v1_1 = train_val1_list[indexes_1]
    v2_1 = train_val2_list[indexes_1]
    indexes_2 = np.where(train_outputs == 2)[0]
    v1_2 = train_val1_list[indexes_2]
    v2_2 = train_val2_list[indexes_2]
    plt.scatter(v1_0, v2_0, c=colors[0], marker=markers[0])
    plt.scatter(v1_1, v2_1, c=colors[1], marker=markers[1])
    plt.scatter(v1_2, v2_2, c=colors[2], marker=markers[2])


#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.title("Dataset Visualization")
#     plt.legend()

#     plt.show()


# Test Function for Dataset object
def test_dataset_object(dataset):
    # Test case 1
    print("--- Test case 1 (dataset): Implemented the correct __getitem__ method.")
    print("Sample with index 384 was drawn from dataset.")
    index = 384
    true1 = (torch.tensor([0.9429, 0.2715], dtype=torch.float32), torch.tensor(1., dtype=torch.float32))
    try:
        test1 = dataset[index]
    except:
        test1 = "Something went wrong when retrieving sample 384."
    print("Retrieved: {}".format(test1))
    print("Expected: {}".format(true1))

    try:
        c1 = torch.allclose(test1[0], true1[0], atol=1e-4)
        c2 = torch.allclose(test1[1], true1[1], atol=1e-4)
        val = "Passed" if c1 and c2 else "Failed"
    except:
        val = "Failed"
    print("Test case 1: {}".format(val))

    # Test case 2
    print("--- Test case 2 (dataset): Implemented the correct __len__ method.")
    print("Asking for dataset length.")
    true2 = 1000
    try:
        test2 = len(dataset)
    except:
        test2 = "Something went wrong when asking for dataset length."
    print("Retrieved: {}".format(test2))
    print("Expected: {}".format(true2))
    try:
        test2 = len(dataset)
        val = "Passed" if test2 == true2 else "Failed"
    except:
        val = "Failed"
    print("Test case 2: {}".format(val))


# Test Function for Dataloader object
def test_dataloader_object(dataloader):
    # Test case 1
    print("--- Test case 1 (dataloader): Using the correct batch size.")
    print("Asking for batch size.")
    true1 = 128
    try:
        test1 = dataloader.batch_size
    except:
        test1 = "Something went wrong when checking batch size."
    print("Retrieved: {}".format(test1))
    print("Expected: {}".format(true1))
    try:
        val = "Passed" if test1 == true1 else "Failed"
    except:
        val = "Failed"
    print("Test case 1: {}".format(val))

    # Test case 2
    print("--- Test case 2 (dataloader): Dataloader is shuffling the dataset, as requested.")
    print("Asking if Dataloader will be shuffling.")
    true2 = True
    try:
        test2 = "torch.utils.data.sampler.RandomSampler object at" in str(dataloader.sampler)
    except:
        test2 = "Something went wrong when checking shuffling."
    print("Retrieved: {}".format(test2))
    print("Expected: {}".format(true2))
    try:
        val = "Passed" if test2 == true2 else "Failed"
    except:
        val = "Failed"
    print("Test case 2: {}".format(val))


# Test Function for WeirdActivation object
def test_act_object(act_fun):
    # Test case: Checking for correct output shape
    print("--- Test case (activation function): Checking for correct output shape.")
    print("Testing forward on a Tensor of values.")

    true_shape = (1, 10)  # Assuming we expect the shape to be (1, 3)

    try:
        # Create a tensor input with 1 row and 2 columns
        x = torch.tensor([[0.5, -0.5]])

        # Pass the input through the activation function
        out = act_fun.forward(x).cpu().detach()

        # Check if the shape of the output matches the expected shape
        output_shape = out.shape
        print("Retrieved shape: {}".format(output_shape))
        print("Expected shape: {}".format(true_shape))

        if output_shape == true_shape:
            val = "Passed"
        else:
            val = "Failed"
    except Exception as e:
        val = f"Failed due to error: {str(e)}"
        output_shape = "Error in computation"

    print("Test case result: {}".format(val))

