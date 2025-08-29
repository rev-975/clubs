import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AI Financial Market Data.csv")
df.drop_duplicates(inplace=True) # removes any duplicate rows
df.dropna(inplace=True) # removes rows with missing values


#print(df.head(5))
#print(df.info())

# check for missing values
print(df.isnull().sum())

#print(f"\nCompanies: {df['Company'].unique()}")
#print(f"\nEvents: {df['Event'].unique()}\n")

# basic stat analysis

avg_rnd = df.groupby("Company")["R&D_Spending_USD_Mn"].mean()
avg_rev = df.groupby("Company")["AI_Revenue_USD_Mn"].mean()
avg_stock_impact = df.groupby("Company")["Stock_Impact_%"].mean()

# print averages

print(f"Average RnD Spending: \n{avg_rnd.round(2)}\n")
print(f"Average AI Revenue: \n{avg_rev.round(2)}\n")
print(f"Average Stock Impact: \n{avg_stock_impact.round(2)}\n")

# highest total RnD spending

top_rnd = df.groupby('Company')["R&D_Spending_USD_Mn"].sum()
print(f"Company with highest RnD Spending: {top_rnd.idxmax()} (${top_rnd.max()})")

# highest growth

ai_growth = df.groupby("Company")["AI_Revenue_Growth_%"].mean()
print(f"The company with the fastest AI Revenue growth is {ai_growth.idxmax()} (${ai_growth.max()}).\n")

# Corr

print("Correlation matrix: ")
corr = df[['R&D_Spending_USD_Mn','AI_Revenue_USD_Mn', 'AI_Revenue_Growth_%', 'Stock_Impact_%']].corr()
print(corr)

# 1. heatmap
plt.figure(figsize=(12, 12))
corr = df[['R&D_Spending_USD_Mn','AI_Revenue_USD_Mn', 'AI_Revenue_Growth_%', 'Stock_Impact_%']].corr()
sns.heatmap(corr, annot = True, cmap = 'coolwarm', center = 0)
plt.title('Corr Heatmap')
plt.yticks(rotation=45)
plt.show()

# 2.1 rnd spending vs ai revenue
sns.lmplot(x= 'R&D_Spending_USD_Mn', y = 'AI_Revenue_USD_Mn', hue='Company', data = df, fit_reg=True)
plt.title('R&D Spending vs AI Revenue Trendlines')
plt.xlabel('R&D Spending (USD Mn)')
plt.ylabel('AI Revenue (USD Mn)')
plt.show()

# 2.2 rnd vs ai revenue over time 
for company in df["Company"].unique():
    subset = df[df["Company"] == company]
    plt.plot(subset["Date"], subset["AI_Revenue_USD_Mn"], label=f"{company} Revenue")
    plt.plot(subset["Date"], subset["R&D_Spending_USD_Mn"], linestyle="--", label=f"{company} R&D")
plt.legend()
plt.title("R&D vs AI Revenue over Time")
plt.show()


# 2.3 top 3 companies with most roi(rev/rnd_spending)
df['ROI'] = df['AI_Revenue_USD_Mn']/df['R&D_Spending_USD_Mn']
avg_roi_by_company = df.groupby('Company')['ROI'].mean().sort_values(ascending=False)
print(f'Top 3 companies by ROI: \n{avg_roi_by_company.head(3)}')

avg_roi_by_company.plot(kind='bar', color=['green', 'blue', 'orange', 'red', 'purple'])
plt.title('avg ROI by Company')
plt.xlabel('Company')
plt.ylabel('ROI')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. relation between rnd and stock impact
plt.scatter(df['R&D_Spending_USD_Mn'], df['Stock_Impact_%'])
plt.xlabel('R&D Spending (USD)')
plt.ylabel('Stock Impact (%)')
plt.title('R&D Spending vs Stock Impact')
plt.show()


# 4. top 2 events where max stock impact seen
top_positive = df.nlargest(2, "Stock_Impact_%")[["Date", "Company", "Event", "Stock_Impact_%"]]
top_negative = df.nsmallest(2, "Stock_Impact_%")[["Date", "Company", "Event", "Stock_Impact_%"]]
print("\nTop Positive Stock Impact Events:\n", top_positive)
print("\nTop Negative Stock Impact Events:\n", top_negative)

# 4.2 company that reacts strongest
strongest_company = df["Stock_Impact_%"].abs().groupby(df["Company"]).mean()
print(f"\nStrongest reacting company: {strongest_company.idxmax()} ({strongest_company.max()})")

