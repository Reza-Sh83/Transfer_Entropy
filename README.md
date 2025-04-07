# 🔄 Transfer Entropy Analysis of S&P 500 Stocks

This project computes and visualizes **Transfer Entropy** between stock price time series. Transfer Entropy is a powerful method for understanding directional information flow in complex systems like financial markets. The code uses ordinal pattern analysis to uncover causal relationships between companies based on their historical stock prices.

---

## 🧠 Motivation

In financial systems, understanding not just the correlation but the **direction of influence** between assets can provide crucial insights. This project explores how information transfers from one stock to another over time, using Transfer Entropy.

---

## 🚀 Features

- Loads and processes open price data from CSV files
- Computes ordinal patterns with adjustable embedding dimension
- Calculates pairwise **Transfer Entropy** between companies
- Computes a **Direction Matrix** indicating the net direction of information flow
- Visualizes results as heatmaps for better interpretability

---

## 📆 Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- tabulate

Install dependencies via pip:

```bash
pip install numpy pandas matplotlib tabulate
```

---

## 🛠️ Usage

1. **Place your data**  
   Organize stock price data in subfolders by sector or category.

2. **Set parameters**  
   In the script, modify:
   ```python
   file_path = "YOUR FILE PATH"
   patterns, perm_map = ordinal_pattern(stock_data, D=3)  # Embedding dimension
   T_matrix, Direction = Entropy_transfer(patterns, perm_map, delta=1)  # Lag
   ```

3. **Run the script**  

```bash
python Transfer_Entropy_S&P500.py
```

The output will include:
- **Transfer Entropy Matrix** (saved heatmap + printed values)
- **Direction Matrix** (saved heatmap + printed values)

---

## 📊 Example Output

### Transfer Entropy Matrix
A heatmap where each cell shows how much information transfers from one company to another.

### Direction Matrix
A signed matrix showing the **net direction** of influence between pairs:
- Positive value: row company influences column company
- Negative value: column company influences row company

<img src="Entropy Transfer Matrix.png" width="450">
<img src="Entropy Transfer Direction.png" width="450">

---

## 🤝 Contributing

Suggestions and contributions are welcome! Feel free to fork the project and submit pull requests to add new analysis tools or optimize performance.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

