import numpy as np
import pygad
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Data Preparation
# Fetch real historical data from Yahoo Finance
num_assets = 5
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Download the adjusted closing prices for the selected tickers
data = yf.download(tickers, start="2020-01-01", end="2024-01-01")['Adj Close']

# Drop tickers that failed to download
data = data.dropna(axis=1, how='all')

# Update the number of assets to reflect actual data downloaded
num_assets = data.shape[1]
tickers = data.columns.tolist()

# Calculate the mean returns and covariance matrix of the returns
returns = data.pct_change().mean().values
cov_matrix = data.pct_change().cov().values

# Step 2: Define Fitness Function
def fitness_function(ga_instance, solution, solution_idx):
    try:
        # Calculate portfolio return
        portfolio_return = np.sum(solution * returns)
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(solution.T, np.dot(cov_matrix, solution))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Ensure that portfolio risk is not zero to prevent division errors
        if portfolio_risk == 0:
            return -np.inf  # Invalid solution

        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_risk
        return sharpe_ratio

    except Exception as e:
        print(f"Error calculating fitness for solution {solution_idx}: {e}")
        return -np.inf  # Penalize any solution that fails

# Step 3: Set Up GA
ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=10,
                       fitness_func=fitness_function,
                       sol_per_pop=20,
                       num_genes=num_assets,
                       init_range_low=0,
                       init_range_high=1,
                       parent_selection_type="tournament",
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_num_genes=1,  # Ensures at least one gene is mutated
                       gene_space=[{'low': 0, 'high': 1} for _ in range(num_assets)],
                       on_generation=lambda instance: print(f"Generation {instance.generations_completed} completed"))

# Step 4: Run GA
ga_instance.run()

# Step 5: Output the Best Solution
try:
    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Best Portfolio: {solution}")
    print(f"Fitness (Sharpe Ratio): {solution_fitness}")

    # Visualization: Display allocation
    plt.bar(range(num_assets), solution)
    plt.xlabel("Assets")
    plt.ylabel("Allocation")
    plt.title("Optimized Portfolio Allocation")
    plt.xticks(ticks=range(num_assets), labels=tickers)
    plt.show()

except Exception as e:
    print(f"Error finding the best solution: {e}")
