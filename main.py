import numpy as np
import pygad
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Data Preparation
# Fetch real historical data from Yahoo Finance
num_assets = 5
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Download the adjusted closing prices for the selected tickers
data = yf.download(tickers, start="2018-01-01", end="2024-01-01")['Adj Close']

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
        # Ensure sum of weights equals to 1 (or penalize heavily if not)
        if not np.isclose(np.sum(solution), 1):
            return -np.inf  # Penalize invalid solutions

        # Limit individual asset allocation to a max of 50% to encourage diversity
        if np.any(solution > 0.5):
            return -np.inf  # Penalize solutions that allocate more than 50% to a single asset

        # Calculate portfolio return
        portfolio_return = np.sum(solution * returns)

        # Calculate portfolio variance
        portfolio_variance = np.dot(solution.T, np.dot(cov_matrix, solution))
        portfolio_risk = np.sqrt(portfolio_variance)

        # Dynamic risk tolerance based on generation count
        max_risk_threshold = 0.25 + (ga_instance.generations_completed / ga_instance.num_generations) * 0.1

        # Ensure that portfolio risk is within a specific range to balance risk and return
        if portfolio_risk == 0 or portfolio_risk > max_risk_threshold:
            return -np.inf  # Invalid solution or too risky

        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_risk

        # Reward diversification: more assets with non-zero weights is better
        num_nonzero_allocations = np.count_nonzero(solution)
        diversification_reward = num_nonzero_allocations / num_assets

        # Additional penalty for concentration beyond 40%
        concentration_penalty = -np.sum(solution[solution > 0.4])

        # Combine Sharpe Ratio with diversification reward and concentration penalty
        fitness_value = sharpe_ratio + diversification_reward + concentration_penalty

        return fitness_value

    except Exception as e:
        print(f"Error calculating fitness for solution {solution_idx}: {e}")
        return -np.inf  # Penalize any solution that fails

# Step 3: Normalize Solutions
def normalize_solution(solution):
    total = np.sum(solution)
    if total == 0:
        # Assign equal weight to all assets if the total is zero
        return np.full_like(solution, 1.0 / len(solution))
    return solution / total

# Step 4: Set Up GA with Early Stopping
best_fitness_values = []

# Update Early Stopping with Higher Sensitivity and Larger Window
def normalize_population(ga_instance):
    global best_fitness_values

    for i in range(ga_instance.sol_per_pop):
        ga_instance.population[i] = normalize_solution(ga_instance.population[i])

    best_fitness = ga_instance.best_solution()[1]
    best_fitness_values.append(best_fitness)

    # Early stopping: check if the fitness value has plateaued
    if len(best_fitness_values) > 30:  # Increased window size
        recent_fitness = best_fitness_values[-30:]
        if np.std(recent_fitness) < 1e-6:  # Increased sensitivity
            ga_instance.stop()  # Stop the evolution if no significant improvement

    # Adaptive mutation: adjust mutation rate based on convergence
    diversity = np.mean([np.std(ind) for ind in ga_instance.population])
    try:
        mutation_percent_genes = float(ga_instance.mutation_percent_genes)
    except ValueError:
        mutation_percent_genes = 10  # Set a default value if conversion fails

    if diversity < 0.1:
        ga_instance.mutation_percent_genes = min(100, mutation_percent_genes + 5)
    else:
        ga_instance.mutation_percent_genes = max(5, mutation_percent_genes - 5)

    print(f"Generation {ga_instance.generations_completed} completed with best fitness: {best_fitness}")


# Increase population size, number of generations, and modify crossover strategy for better exploration
ga_instance = pygad.GA(num_generations=3000,  # Increased number of generations for further exploration
                       num_parents_mating=20,
                       fitness_func=fitness_function,
                       sol_per_pop=100,  # Increased population size for more diversity
                       num_genes=num_assets,
                       init_range_low=0,
                       init_range_high=1,
                       parent_selection_type="tournament",
                       crossover_type="uniform",  # Changed to uniform crossover
                       crossover_probability=0.8,  # Adjusted crossover rate
                       mutation_type="random",
                       mutation_num_genes=3,  # Increase mutation for more exploration
                       gene_space=[{'low': 0, 'high': 1} for _ in range(num_assets)],
                       on_generation=normalize_population)

# Step 5: Run GA
ga_instance.run()


# Step 6: Output the Best Solution
try:
    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Best Portfolio: {solution}")
    print(f"Fitness (Sharpe Ratio + Diversification Reward): {solution_fitness}")

    # Visualization: Display allocation
    plt.bar(range(num_assets), solution)
    plt.xlabel("Assets")
    plt.ylabel("Allocation")
    plt.title("Optimized Portfolio Allocation")
    plt.xticks(ticks=range(num_assets), labels=tickers)
    plt.show()

except Exception as e:
    print(f"Error finding the best solution: {e}")
