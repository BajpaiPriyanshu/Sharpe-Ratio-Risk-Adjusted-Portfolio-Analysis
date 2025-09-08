import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

print("=" * 60)
print("    SHARPE RATIO & RISK-ADJUSTED RETURNS ANALYSIS")
print("=" * 60)

print("\n1. GENERATING SAMPLE STOCK DATA")
print("-" * 40)
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
n_days = 252
dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

np.random.seed(42)
stock_data = {}
for stock in stocks:
    start_price = np.random.uniform(100, 300)
    if stock == 'TSLA':
        daily_returns = np.random.normal(0.001, 0.03, n_days)
    elif stock in ['AAPL', 'MSFT']:
        daily_returns = np.random.normal(0.0008, 0.018, n_days)
    else:
        daily_returns = np.random.normal(0.0006, 0.02, n_days)
    
    price_series = start_price * (1 + daily_returns).cumprod()
    stock_data[stock] = price_series

prices_df = pd.DataFrame(stock_data, index=dates)
print(f"Generated price data for {len(stocks)} stocks over {n_days} days")
print(f"Date range: {prices_df.index[0].strftime('%Y-%m-%d')} to {prices_df.index[-1].strftime('%Y-%m-%d')}")

print("\nFirst 5 rows of stock prices:")
print(prices_df.head())

print("\n\n2. CALCULATING DAILY RETURNS")
print("-" * 40)

returns_df = prices_df.pct_change().dropna()

print("Daily returns calculated using percentage change method")
print("Formula: (Price_today - Price_yesterday) / Price_yesterday")
print("\nFirst 5 rows of daily returns:")
print(returns_df.head())

print("\nDaily Returns Summary Statistics:")
print(returns_df.describe())

print("\n\n3. SETTING UP RISK-FREE RATE")
print("-" * 40)

annual_risk_free_rate = 0.04
daily_risk_free_rate = annual_risk_free_rate / 252

print(f"Annual Risk-Free Rate: {annual_risk_free_rate:.2%}")
print(f"Daily Risk-Free Rate: {daily_risk_free_rate:.6f}")
print("Risk-free rate represents return on 'safe' investments like Treasury bonds")

print("\n\n4. CALCULATING EXCESS RETURNS")
print("-" * 40)

excess_returns_df = returns_df - daily_risk_free_rate

print("Excess returns = Stock returns - Risk-free rate")
print("This shows how much extra return we get for taking risk")
print("\nFirst 5 rows of excess returns:")
print(excess_returns_df.head())

print("\n\n5. CALCULATING SHARPE RATIOS")
print("-" * 40)

sharpe_ratios = {}

print("Sharpe Ratio Formula: (Mean Excess Return) / (Std Dev of Excess Returns)")
print("Higher Sharpe Ratio = Better risk-adjusted performance\n")

for stock in stocks:
    mean_excess_return = excess_returns_df[stock].mean()
    std_excess_return = excess_returns_df[stock].std()
    daily_sharpe = mean_excess_return / std_excess_return
    annual_sharpe = daily_sharpe * np.sqrt(252)
    
    sharpe_ratios[stock] = annual_sharpe
    print(f"{stock}:")
    print(f"  Mean Daily Excess Return: {mean_excess_return:.6f} ({mean_excess_return*252:.4f} annually)")
    print(f"  Std Dev Daily Excess Return: {std_excess_return:.6f} ({std_excess_return*np.sqrt(252):.4f} annually)")
    print(f"  Sharpe Ratio: {annual_sharpe:.4f}")
    print()

print("\n6. RANKING STOCKS BY RISK-ADJUSTED PERFORMANCE")
print("-" * 50)
sharpe_df = pd.DataFrame(list(sharpe_ratios.items()), columns=['Stock', 'Sharpe_Ratio'])
sharpe_df = sharpe_df.sort_values('Sharpe_Ratio', ascending=False)
sharpe_df['Rank'] = range(1, len(sharpe_df) + 1)

print("Stock Rankings (Best to Worst Risk-Adjusted Performance):")
print(sharpe_df.to_string(index=False))

print("\n\n7. PORTFOLIO ANALYSIS")
print("-" * 40)

portfolios = {
    'Equal_Weight': [0.2, 0.2, 0.2, 0.2, 0.2],
    'Top_3_Sharpe': [0.4, 0.3, 0.3, 0.0, 0.0],
    'Conservative': [0.3, 0.3, 0.25, 0.15, 0.0],
    'Aggressive': [0.1, 0.1, 0.1, 0.2, 0.5]
}

portfolio_sharpe = {}

print("Analyzing different portfolio strategies:")
print("Weights correspond to stocks in order:", stocks)
print()

for portfolio_name, weights in portfolios.items():
    portfolio_returns = (returns_df * weights).sum(axis=1)
    portfolio_excess = portfolio_returns - daily_risk_free_rate
    portfolio_mean_excess = portfolio_excess.mean()
    portfolio_std = portfolio_excess.std()
    portfolio_sharpe_daily = portfolio_mean_excess / portfolio_std
    portfolio_sharpe_annual = portfolio_sharpe_daily * np.sqrt(252)
    
    portfolio_sharpe[portfolio_name] = portfolio_sharpe_annual
    print(f"{portfolio_name}:")
    print(f"  Weights: {weights}")
    print(f"  Sharpe Ratio: {portfolio_sharpe_annual:.4f}")
    print(f"  Annual Return: {portfolio_returns.mean() * 252:.2%}")
    print(f"  Annual Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
    print()

print("\n8. CREATING VISUALIZATIONS")
print("-" * 40)

plt.figure(figsize=(10, 6))
plt.plot(prices_df.index, prices_df)
plt.title('Stock Price Evolution Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(stocks)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
colors = ['green' if x > 1 else 'orange' if x > 0 else 'red' for x in sharpe_ratios.values()]
bars = plt.bar(sharpe_ratios.keys(), sharpe_ratios.values(), color=colors)
plt.title('Individual Stock Sharpe Ratios', fontsize=14, fontweight='bold')
plt.ylabel('Sharpe Ratio')
plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Good Performance (1.0)')
plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
colors2 = ['darkgreen' if x > 1 else 'darkorange' if x > 0 else 'darkred' for x in portfolio_sharpe.values()]
bars2 = plt.bar(portfolio_sharpe.keys(), portfolio_sharpe.values(), color=colors2)
plt.title('Portfolio Strategy Sharpe Ratios', fontsize=14, fontweight='bold')
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=45)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Good Performance (1.0)')
plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
annual_returns = returns_df.mean() * 252
annual_volatility = returns_df.std() * np.sqrt(252)

scatter = plt.scatter(annual_volatility, annual_returns, s=150, c=list(sharpe_ratios.values()),
                     cmap='RdYlGn', alpha=0.8, edgecolors='black', linewidth=1)

plt.title('Risk vs Return Profile', fontsize=14, fontweight='bold')
plt.xlabel('Annual Volatility (Risk)')
plt.ylabel('Annual Return')
plt.grid(True, alpha=0.3)

for i, stock in enumerate(stocks):
    plt.annotate(stock, (annual_volatility[i], annual_returns[i]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=11, fontweight='bold')

cbar = plt.colorbar(scatter)
cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20, fontsize=12)

plt.tight_layout()
plt.show()

print("All 4 figures have been displayed separately!")
print("\n\n9. SUMMARY AND KEY INSIGHTS")
print("=" * 50)

best_stock = sharpe_df.iloc[0]['Stock']
best_portfolio = max(portfolio_sharpe, key=portfolio_sharpe.get)

print(f"BEST INDIVIDUAL STOCK: {best_stock}")
print(f"   Sharpe Ratio: {sharpe_ratios[best_stock]:.4f}")
print(f"   This stock provides the best risk-adjusted returns")
print()

print(f" BEST PORTFOLIO STRATEGY: {best_portfolio}")
print(f"   Sharpe Ratio: {portfolio_sharpe[best_portfolio]:.4f}")
print(f"   This diversification strategy offers superior risk-adjusted performance")
print()

print(" KEY LEARNINGS:")
print("   • Sharpe Ratio > 1.0 indicates good risk-adjusted performance")
print("   • Sharpe Ratio > 2.0 indicates excellent performance")
print("   • Higher volatility doesn't always mean better returns")
print("   • Diversification can improve risk-adjusted returns")
print("   • Risk-free rate is crucial for relative performance evaluation")
print()

print(" INTERPRETATION GUIDE:")
print("   • Sharpe Ratio measures excess return per unit of risk")
print("   • It helps compare investments with different risk profiles")
print("   • Essential metric for portfolio optimization")
print("   • Used by professional fund managers for performance evaluation")

print("Project Complete!!!...")
