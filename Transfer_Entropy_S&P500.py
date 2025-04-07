# %%
# Import necessary libraries
import numpy as np
import os
import pandas as pd
from statistics import mode
from collections import Counter
import matplotlib.pyplot as plt
from itertools import permutations, product
from math import log2
from tabulate import tabulate  # For pretty-printing tables

# %%
def extract_data(main_folder):
    """
    Extracts stock data from CSV files within a given folder structure.

    Args:
        main_folder (str): Path to the main folder containing category folders.

    Returns:
        dict: Dictionary with company names as keys and their corresponding 
              'Open' stock values as a list of values.
    """
    stock_data = {}
    dirs = [d for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
    for category in dirs:
        category_path = os.path.join(main_folder, category)
        num_data_ls = []
        for company_file in os.listdir(category_path):
            if company_file.endswith('.csv'):
                company_name = company_file.split('.')[0]
                company_path = os.path.join(category_path, company_file)
                df = pd.read_csv(company_path)
                df = df[['Date', 'Open', 'Close']]
                num_data_ls.append(len(df['Open'].values))
                stock_data[company_name] = df['Open'].values
        num_data = mode(num_data_ls)
        stock_data = {company: data for company, data in stock_data.items() if len(data) == num_data}
    return stock_data


# %%
def ordinal_pattern(stock_data, D):
    """
    Computes ordinal patterns for time series data of given companies.

    Args:
        stock_data (dict): Dictionary with stock prices of companies.
        D (int): Embedding dimension.

    Returns:
        dict: Ordinal patterns for each company.
    """
    dimension = [i for i in range(D)]
    perm = list(product(dimension, repeat=D))
    labels = [i+1 for i in range(D**D)]
    perm_map = {p: label for label, p in zip(labels, perm)}
    company_names = stock_data.keys()
    patterns = {}

    for idx, company in enumerate(company_names):
        patterns[idx] = []
        prices = stock_data[company]
        if D > len(prices):
            raise ValueError("Embedding dimension D must be smaller than the length of the time series.")

        windows = np.lib.stride_tricks.sliding_window_view(prices, D)
        for window in windows:
            sorted_list = window[np.argsort(window)]
            defined_dict = {value: index for index, value in enumerate(sorted_list)}
            result = [defined_dict[value] for value in window]
            patterns[idx].append(perm_map[tuple(result)])
    return patterns, perm_map


# %%
def Estimate_Margin_PDF(data):
    """
    Estimates marginal probabilities from data using a counter.

    Args:
        data (list): Input data.

    Returns:
        function: Function to get probability of a specific number.
    """
    counts = Counter(data)
    total_count = len(data)
    margin_probabilities = {k: v / total_count for k, v in counts.items()}

    return lambda x: margin_probabilities.get(int(x), 0)


# %%
def Estimate_Joint_PDF(data):
    """
    Estimates joint probabilities from stacked data.

    Args:
        data (list): List of data arrays to combine.

    Returns:
        function: Function to get probability of a specific row in the data.
    """
    joint_data = np.vstack(data).T
    tuple_data = [tuple(row) for row in joint_data]
    counts = Counter(tuple_data)
    total_count = len(tuple_data)
    joint_probabilities = {k: v / total_count for k, v in counts.items()}

    return lambda x: joint_probabilities.get(tuple(x), 0)



# %%
def Entropy_transfer(patterns, perm_map, delta):
    """
    Calculates entropy transfer matrix and direction matrix.

    Args:
        patterns (dict): Ordinal patterns for each company.
        delta (int): Lag for entropy transfer calculation.

    Returns:
        np.ndarray, np.ndarray: Entropy transfer and direction matrices.
    """
    num_companies = len(patterns)
    T_matrix = np.zeros((num_companies, num_companies))
    Direction = np.zeros((num_companies, num_companies))

    for i, pattern_i in patterns.items():
        for j, pattern_j in patterns.items():
            if i == j:
                T_matrix[j, i] = 0
                continue
            data = [
                pattern_i[delta:],   # X_delta
                pattern_i[:-delta],  # X
                pattern_j[:-delta],  # Y
            ]            
            
            p_xxy = Estimate_Joint_PDF(data)
            p_xy = Estimate_Joint_PDF(data[1:])
            p_xx = Estimate_Joint_PDF(data[:2])
            p_x = Estimate_Margin_PDF(data[1])

            t_ji = 0
            for x_delta in perm_map.values():
                for x in perm_map.values():
                    for y in perm_map.values():                        
                        pxxy = p_xxy([x_delta, x, y])
                        pxx = p_xx([x_delta, x])
                        pxy = p_xy([x, y])
                        px = p_x(x)
                        if pxxy > 0 and pxx > 0 and pxy > 0 and px > 0:
                            t_ji += pxxy * log2(pxxy * px / (pxy * pxx))
            T_matrix[j, i] = t_ji   # T(j → i)

    # Calculate Direction matrix
    for i in range(num_companies):
        for j in range(num_companies):
            Tij, Tji = T_matrix[i, j], T_matrix[j, i]
            if Tij + Tji == 0:
                Direction[i, j] = 0
                continue
            Direction[i, j] = (Tij - Tji) / (Tij + Tji)


    return T_matrix, Direction

# %%
def heat_map(T_matrix, title):
    """
    Displays a heatmap of the given matrix.

    Args:
        T_matrix (np.ndarray): Matrix to plot.
        title (str): Title of the heatmap.
    """
    plt.figure(figsize=(8, 6))
    max_num, min_num = np.max(T_matrix), np.min(T_matrix)
    plt.imshow(T_matrix, cmap='seismic', vmin=min_num, vmax=max_num)
    plt.colorbar(label="Entropy Transfer Intensity")
    plt.title(title)
    plt.xlabel("Company Index")
    plt.ylabel("Company Index")
    plt.xticks([],[])
    plt.yticks([],[])
    plt.tight_layout()
    plt.savefig(f'{title}', dpi=300)
    plt.show()


# %%
# File path to the data folder
file_path = r"YOUR FILE PATH"

# Extract data
stock_data = extract_data(file_path)

# Compute ordinal patterns
patterns, perm_map = ordinal_pattern(stock_data, D=3)

# Calculate Entropy Transfer Matrix and Direction Matrix
T_matrix, Direction = Entropy_transfer(patterns, perm_map, delta=1)

# Display the Entropy Transfer Matrix as a heatmap
print("Entropy Transfer Matrix:")
print(tabulate(T_matrix, tablefmt="grid", floatfmt=".3f"))
heat_map(T_matrix, title="Entropy Transfer Matrix")

# Display the Direction Matrix as a heatmap
print("\nDirection Matrix:")
print(tabulate(Direction, tablefmt="grid", floatfmt=".3f"))
heat_map(Direction, title="Entropy Transfer Direction")


