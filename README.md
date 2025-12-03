# 📈 Basket Trading with Bayesian Optimization  
*A machine-learning approach to improving cointegration-based trading strategies*

This project explores how to enhance traditional cointegration-based basket trading by using **Bayesian Optimization (Optuna)** to directly maximize out-of-sample trading performance.

Classical cointegration methods (like the Johansen test) estimate statistically valid weights, but these weights often fail to generate profitable trading signals in real markets due to drift, noise, and unstable relationships.

This project reframes cointegration trading as a black-box optimization problem, using Bayesian Optimization to search for:

- Optimal basket weights
- Optimal trading thresholds** (entry/exit/stop)
- Configurations that maximize Sharpe ratio, return, and stability

The result is a significant improvement in profitability and risk-adjusted performance.
---
## 🚀 Key Results

This project demonstrates that Bayesian Optimization dramatically improves the performance of a cointegration-based basket trading strategy.

### 🔹 Johansen Baseline (Traditional Approach)
- ❌ **–11.7% total return**
- ❌ **Sharpe ≈ –0.03**
- ❌ **Max drawdown ≈ –29%**
- ❌ High volatility and unstable mean reversion

### 🔹 Bayesian Optimization — Weights Only
- ✔️ Directly optimized cointegration weights
- ✔️ **+38% total return**
- ✔️ **Sharpe ≈ 0.77**
- ✔️ Drawdown reduced by half

### 🔹 Strategy BO — Weights + Trading Thresholds
- ⭐ Best overall configuration
- ⭐ Optimizes weights *and* entry/exit/stop levels  
- ⭐ Strongest Sharpe  
- ⭐ Most stable equity curve  
- ⭐ Most robust out-of-sample behavior

These findings show that statistically derived cointegration weights do not translate into optimal trading performance, but Bayesian Optimization does.
---
## 📊 Visual Results

Below are the key visualizations that demonstrate the performance difference between the classical Johansen strategy and the Bayesian-optimized strategies.

### **1️⃣ Equity Curve Comparison**
This plot compares the cumulative equity for:
- Johansen baseline  
- Bayesian Optimization (weights only)  
- Full Strategy BO (weights + thresholds)

> **Bayesian Optimization produces a smoother, higher-return, lower-risk equity curve.**

<img width="1000" height="498" alt="Screenshot 2025-12-03 at 12 05 52 AM" src="https://github.com/user-attachments/assets/f12afa12-bfa9-42db-a59d-e12129f94563" />

---

### **2️⃣ Spread & Z-Score Behavior (Strategy BO)**
The optimized strategy uses a z-score–based mean-reversion engine.

This chart shows:
- The spread  
- Z-score  
- BO-optimized entry/exit/stop levels  
- Clear mean-reversion patterns identified by BO

<img width="997" height="594" alt="Screenshot 2025-12-03 at 12 06 22 AM" src="https://github.com/user-attachments/assets/4104dbb5-3cb8-4dd2-93c1-fa568b0fe818" />

---
## 🏗️ Project Structure

The repository follows a clean, modular design to separate data loading, statistical modeling, optimization, backtesting, and visualization.

basket-bo/
├─ src/
│  ├─ __init__.py
│  ├─ data_loader.py        # Fetches historical price data (yfinance)
│  ├─ cointegration.py      # Johansen test + basket weight extraction
│  ├─ backtest.py           # Mean-reversion backtester + z-score logic
│  ├─ bayes_opt.py          # Bayesian Optimization for weights & thresholds
│  ├─ plotting.py           # Visualization helpers (equity, spread, z-score)
│  └─ evaluate.py           # Rolling-window out-of-sample evaluation
│
├─ test_plot.py             # Generates all visuals used in README
├─ requirements.txt         # Python dependencies
├─ README.md                # This file
└─ .gitignore               # Ensures clean version control

---
## 💡 Why This Project Matters

This project goes beyond implementing an algorithm — it demonstrates the ability to:

### **1. Reframe a traditional statistical problem as a machine-learning optimization problem**
Classic cointegration assumes:
- stable markets  
- no structural breaks  
- mean-reverting spreads  

But real markets drift.  
By using Bayesian Optimization, we shift from:
> “Find statistically significant weights”
to:
> “Find weights that actually trade well out-of-sample.”

This thinking is directly aligned with modern ML engineering principles.

---

### **2. Build complete end-to-end systems**
The project includes:
- data ingestion  
- statistical modeling  
- backtesting engine  
- optimization loop  
- rolling window evaluation  
- visual diagnostics  

This mirrors the workflow of production ML systems:
> data → model → evaluation → iteration → deployment

---

### **3. Apply ML to noisy, non-differentiable real-world objectives**
Sharpe ratio cannot be optimized analytically.  
It is:
- noisy  
- discontinuous  
- non-convex  

Bayesian Optimization is specifically designed for these problems, and this project shows the ability to apply the right tool for the right task.

---

### **4. Demonstrate meaningful measurable improvement**
The optimized strategy shows:
- Higher return  
- Higher Sharpe  
- Lower drawdowns  
- More stable performance  
- Better generalization across time  

This reflects the ability to **quantitatively measure and validate model improvements** — critical for any ML or engineering role.

---

### **5. Communicate insights clearly (plots, explanations, code structure)**
Readable code, clear plots, and well-organized modules show engineering maturity and the ability to make complex topics understandable.

This is a core value in teams like Shopify’s Dev Degree:  
clear thinking → clear code → clear communication.

---

## 🔮 Future Improvements

There are several natural extensions that can make this project even more powerful and production-ready:

---

### **1️⃣ Add Transaction Costs & Slippage**
All current results are frictionless.  
A realistic model would incorporate:
- trading commissions  
- bid–ask spreads  
- partial fills  
- slippage during volatility  

This tests whether strategies remain profitable in real markets.

---

### **2️⃣ Expand the Asset Universe**
Currently optimized for a 3-asset tech basket.  
Future work includes:
- sector ETFs  
- international equities  
- FX pairs  
- crypto baskets  
- volatility-adjusted baskets  

This allows testing the robustness of BO across asset classes.

---

### **3️⃣ Multi-Objective Bayesian Optimization**
Instead of optimizing only Sharpe ratio, we can jointly optimize:
- return  
- volatility  
- drawdown  
- turnover  
- stability  

Multi-objective BO can find the **Pareto-optimal frontier** of trading strategies.

---

### **4️⃣ Regime Detection & Adaptive Optimization**
Markets behave differently during:
- high volatility  
- low volatility  
- trending regimes  
- mean-reverting regimes  

A future system can:
- detect the regime  
- run BO per regime  
- switch weights dynamically  

This moves the strategy closer to professional quant systems.

---

### **5️⃣ Deploy as an Interactive Dashboard**
Using **Streamlit** or **React + FastAPI**, we could build:
- live visualizations  
- parameter controls  
- real-time optimization demos  
- equity curve displays  

This creates a user-friendly interface for demonstrating the strategy.

---

### **6️⃣ Publish a Research Paper**
The methodology and results are strong enough to be turned into a:
- university research paper  
- arXiv preprint  
- SSRN submission  
- Medium / Towards Data Science article  

This adds academic credibility and professional polish.

---

### **7️⃣ Explore Other Optimization Frameworks**
For comparison:
- Genetic algorithms  
- CMA-ES  
- Simulated annealing  
- Particle swarm optimization  

Useful to validate whether BO is consistently superior across markets.

---

## 📬 Contact

If you have questions about the methodology, optimization approach, or implementation details, feel free to reach out:

**Dhir Champaneri**  
📧 Email: dhirchampaneri@gmail.com
📍 Toronto, Canada
🌐 GitHub: https://github.com/DhirChampaneri

---

## 📝 Final Notes

This project demonstrates how machine learning — specifically **Bayesian Optimization** — can significantly enhance traditional statistical trading strategies such as cointegration.

It showcases:
- full end-to-end system design  
- optimization under uncertainty  
- clean implementation  
- strong use of Python, statistics, and ML engineering  
- clear communication and visualization  

This work reflects a practical, research-driven engineering mindset and serves as a foundation for future exploration in algorithmic trading, quantitative finance, and machine learning optimization.

If you're reviewing this as part of an internship or program application, thank you for taking the time to explore the project!



