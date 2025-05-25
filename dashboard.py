import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Trader Performance Dashboard")
st.markdown(
    "<h1 style='text-align: center; color: #4F8BF9;'>Trader Performance Analysis Across Market Sentiment Regimes</h1>",
    unsafe_allow_html=True,
)

st.divider()

st.subheader("Overview")
st.markdown(
    """ This dashboard analyzes the performances of 32 different traders across different market sentiment regimes, using the Fear and Greed Index as a sentiment indicator. The analysis includes metrics such as Closed PnL, Win Rate, Sharpe Ratio, and more, segmented by market sentiment classifications: Extreme Fear, Fear, Neutral, Greed, and Extreme Greed.
    The goal is to understand how traders perform under varying market conditions and identify patterns in their trading behavior.""")

st.divider()

# --- Data Loading ---
df_market = pd.read_csv("fear_greed_index.csv")
df_trader = pd.read_csv("historical_data.csv")

# --- Data Preprocessing ---
df_trader['Timestamp IST'] = pd.to_datetime(df_trader['Timestamp IST'], dayfirst=True)
df_trader['date'] = df_trader['Timestamp IST'].dt.date
df_market['date'] = pd.to_datetime(df_market['date']).dt.date
df = df_trader.merge(df_market, on='date', how='left')
df = df.dropna(subset=['value', 'classification'])
df = df.drop(columns=['Timestamp IST', 'timestamp', 'Timestamp'], errors='ignore')
df.drop(columns=['Transaction Hash', 'Order ID', 'Trade ID'], inplace=True, errors='ignore')

# --- Helper Functions ---
def compute_trading_metrics(df, group_by='classification'):
    df = df.copy()
    df['is_win'] = df['Closed PnL'] > 0
    df['ROI'] = df['Closed PnL'] / df['Size USD']
    grouped = df.groupby(group_by)
    metrics = grouped.agg(
        Trades=('Closed PnL', 'count'),
        WinRate=('is_win', 'mean'),
        AvgPnL=('Closed PnL', 'mean'),
        TotalPnL=('Closed PnL', 'sum'),
        StdPnL=('Closed PnL', 'std'),
        AvgROI=('ROI', 'mean'),
        TotalROI=('ROI', 'sum')
    ).reset_index()
    metrics['Sharpe'] = metrics['AvgPnL'] / metrics['StdPnL']
    def profit_factor(sub_df):
        profit = sub_df[sub_df['Closed PnL'] > 0]['Closed PnL'].sum()
        loss = abs(sub_df[sub_df['Closed PnL'] < 0]['Closed PnL'].sum())
        return profit / loss if loss > 0 else float('inf')
    pf = grouped.apply(profit_factor).reset_index(name='ProfitFactor')
    metrics = metrics.merge(pf, on=group_by)
    def max_drawdown(sub_df):
        cumsum = sub_df['Closed PnL'].cumsum()
        running_max = cumsum.cummax()
        drawdown = running_max - cumsum
        return drawdown.max() if not drawdown.empty else 0
    dd = grouped.apply(max_drawdown).reset_index(name='MaxDrawdown')
    metrics = metrics.merge(dd, on=group_by)
    return metrics

# ...existing code...

sentiment_metrics = compute_trading_metrics(df, group_by='classification')
trader_metrics = compute_trading_metrics(df, group_by='Account')
combo_metrics = compute_trading_metrics(df, group_by=['Account', 'classification'])

# Add this block so trader_stats is available everywhere
trader_stats = df.groupby(['Account', 'classification']).agg({
    'Closed PnL': ['mean', 'sum', 'count'],
    'Size USD': 'sum',
    'Fee': 'sum'
}).reset_index()
trader_stats.columns = ['_'.join(col).strip('_') for col in trader_stats.columns.values]

# --- Plot style for smaller font and uniform size ---
PLOT_SIZE = (5, 3)
TICK_FONT = 7
LABEL_FONT = 8
TITLE_FONT = 10

# Row 1: Mean Closed PnL by Classification & Violin Plot
col1, col2 = st.columns(2)
with col1:
    st.subheader("Mean Closed PnL by Classification")
    st.markdown("Closed PnL is high in extreme greed followed by fear and low in neutral.This suggests that traders tend to achieve the best average profits during periods of Extreme Greed, and also perform well during Fear and Greed. Performance is weaker during Extreme Fear and Neutral sentiment, indicating these are less favorable or more challenging market conditions for traders.")

    means = df.groupby('classification')['Closed PnL'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    means.plot(kind='bar', ax=ax, color='#4F8BF9')
    ax.set_ylabel('Mean of Closed PnL')
    ax.set_title('Analysis of Closed PnL by Classification', fontsize=TITLE_FONT)
    for i, val in enumerate(means):
        ax.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=LABEL_FONT)
    ax.tick_params(axis='x', labelsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    st.pyplot(fig)
    

with col2:
    st.subheader("Violin Plot of Closed PnL by Market Sentiment")
    st.markdown("While the typical trade outcome (median) is similar across sentiment regimes, the risk (variability and outliers) may be higher during Greed and Extreme Greed periods.")
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.violinplot(data=df, x='classification', y='Closed PnL', order=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'], ax=ax)
    ax.set_title('Violin Plot of Closed PnL by Market Sentiment', fontsize=TITLE_FONT)
    ax.tick_params(axis='x', labelsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    st.pyplot(fig)

st.divider()

# Row 2: Time Series & Total PnL by Market Sentiment
col3, col4 = st.columns(2)
with col3:
    st.subheader("Total Closed PnL Over Time")
    st.markdown("The PnL remained relatively stable for much of the early period, but starting in late 2024, there is a sharp increase in both gains and losses, indicating a period of high trading activity and volatility. Such spikes may reflect changes in market conditions, trading strategies, or trader participation.")
    df['date'] = pd.to_datetime(df['date'])
    daily_pnl = df.groupby('date')['Closed PnL'].sum().reset_index()
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.plot(daily_pnl['date'], daily_pnl['Closed PnL'])
    ax.set_title('Total Closed PnL Over Time', fontsize=TITLE_FONT)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Closed PnL')
    ax.tick_params(axis='x', labelsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    st.pyplot(fig)

with col4:
    st.subheader("Total PnL by Market Sentiment")
    st.markdown("The total PnL is highest in Extreme Greed, followed by Fear and Greed, while it is lowest in Neutral and Extreme Fear. This suggests that traders tend to perform better during periods of high market sentiment (Greed) and worse during periods of low sentiment (Fear). This might be because the traders may be more active or successful during volatile or emotionally charged market conditions (Fear, Extreme Greed), while periods of Extreme Fear or Neutral sentiment see less opportunity or more losses.")
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.barplot(data=sentiment_metrics.sort_values(by='TotalPnL',ascending=False), x='classification', y='TotalPnL', ax=ax)
    ax.set_title('Total PnL by Market Sentiment', fontsize=TITLE_FONT)
    ax.set_xlabel('Market Sentiment')
    ax.set_ylabel('Total PnL')
    ax.tick_params(axis='x', labelsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    st.pyplot(fig)

st.divider()

# Row 3: Total PnL by Trader (Top 10) & Top 5 Traders by Total Closed PnL
col5, col6 = st.columns(2)
with col5:
    st.subheader("Total PnL by Trader (Top 10)")
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.barplot(
        data=trader_metrics.sort_values(by='TotalPnL', ascending=False).head(10),
        x='Account', y='TotalPnL', ax=ax
    )
    ax.set_title('Total PnL by Trader', fontsize=TITLE_FONT)
    ax.set_xlabel('Trader')
    ax.set_ylabel('Total PnL')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=85, fontsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    st.pyplot(fig)

with col6:
    st.markdown(
        """
        <div style='display: flex; align-items: center; justify-content: center; height: 350px;'>
            <div style='text-align: center; font-size: 1.1em;'>
                The x-axis lists the top 10 trader accounts (likely anonymized as wallet addresses).<br>
                The y-axis shows the total profit and loss (PnL) for each trader.<br>
                There is a clear leader: the first trader has a much higher total PnL than the others.<br>
                The top few traders generate significantly more profit than the rest, indicating a strong performance gap.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# Row 4: Top 5 Traders in Each Sentiment Regime (2 per row, always fill left to right)
top5_combo = combo_metrics.sort_values(['classification', 'TotalPnL'], ascending=[True, False]).groupby('classification').head(5)
sentiments = list(top5_combo['classification'].unique())
num_sentiments = len(sentiments)
i = 0
while i < num_sentiments:
    cols = st.columns(2)
    # First plot in the row
    sentiment = sentiments[i]
    subset = top5_combo[top5_combo['classification'] == sentiment]
    with cols[0]:
        st.subheader(f"**Top 5 Traders in {sentiment}**")
        fig, ax = plt.subplots(figsize=PLOT_SIZE)
        sns.barplot(data=subset, x='Account', y='TotalPnL', ax=ax)
        ax.set_title(f'Top 5 Traders in {sentiment}', fontsize=TITLE_FONT)
        ax.set_xlabel('Account', fontsize=LABEL_FONT)
        ax.set_ylabel('TotalPnL', fontsize=LABEL_FONT)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=85, ha='right', fontsize=TICK_FONT)
        ax.tick_params(axis='y', labelsize=TICK_FONT)
        st.pyplot(fig)
    # Second plot in the row, if available
    if i + 1 < num_sentiments:
        sentiment2 = sentiments[i + 1]
        subset2 = top5_combo[top5_combo['classification'] == sentiment2]
        with cols[1]:
            st.subheader(f"**Top 5 Traders in {sentiment2}**")
            fig2, ax2 = plt.subplots(figsize=PLOT_SIZE)
            sns.barplot(data=subset2, x='Account', y='TotalPnL', ax=ax2)
            ax2.set_title(f'Top 5 Traders in {sentiment2}', fontsize=TITLE_FONT)
            ax2.set_xlabel('Account', fontsize=LABEL_FONT)
            ax2.set_ylabel('TotalPnL', fontsize=LABEL_FONT)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=85, ha='right', fontsize=TICK_FONT)
            ax2.tick_params(axis='y', labelsize=TICK_FONT)
            st.pyplot(fig2)
    i += 2
    st.divider()

# ...existing code above...

# Row 5: Correlation Bar Plot (left) & Content (right)
col7, col8 = st.columns(2)
with col7:
    st.subheader("Correlation: Sentiment vs Metrics")
    sentiment_order = {
        'Extreme Fear': 0,
        'Fear': 1,
        'Neutral': 2,
        'Greed': 3,
        'Extreme Greed': 4
    }
    sentiment_metrics['sentiment_score'] = sentiment_metrics['classification'].map(sentiment_order)
    metrics_to_check = ['AvgPnL', 'WinRate', 'StdPnL', 'ProfitFactor', 'Sharpe', 'MaxDrawdown', 'Trades', 'AvgROI']
    correlations = {}
    for metric in metrics_to_check:
        corr = sentiment_metrics[['sentiment_score', metric]].corr().iloc[0,1]
        correlations[metric] = corr
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.bar(correlations.keys(), correlations.values())
    ax.set_ylabel('Correlation with Sentiment Score', fontsize=LABEL_FONT)
    ax.set_title('Correlation: Sentiment vs Metrics', fontsize=TITLE_FONT)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    plt.tight_layout()
    st.pyplot(fig)
with col8:
    st.markdown(""" 
    - AvgPnL, WinRate, ProfitFactor, Sharpe, and AvgROI all have strong positive correlations with sentiment score. This means as market sentiment becomes more positive (moves from Extreme Fear to Extreme Greed), these metrics tend to improve.
    - StdPnL (standard deviation of PnL) has a negative correlation, suggesting that higher sentiment is associated with more consistent (less volatile) trading results.
    - MaxDrawdown and Trades have weak positive correlations, indicating only a slight increase in these metrics with higher sentiment.
    - The strongest positive correlation is with AvgROI, indicating that average return on investment increases the most as sentiment becomes more positive.
    """)

st.divider()

# Row 6: Correlation Heatmap (left) & Content (right)
col9, col10 = st.columns(2)
with col9:
    st.subheader("Correlation Heatmap")
    corr = sentiment_metrics[['sentiment_score', 'AvgPnL', 'WinRate', 'StdPnL', 'Sharpe', 'ProfitFactor', 'MaxDrawdown', 'AvgROI']].corr()
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=TITLE_FONT)
    st.pyplot(fig)
with col10:
    st.markdown(""" 
    - The heatmap shows correlations between sentiment score and various trading metrics.
    - AvgPnL, WinRate, ProfitFactor, Sharpe, and AvgROI all have strong positive correlations with sentiment score, meaning they improve as market sentiment becomes more positive.
    - StdPnL has a negative correlation, indicating that trading results are less volatile in positive sentiment.
    - MaxDrawdown and Trades have weak correlations.
    - Interpretation: Traders generally perform better and more consistently in positive sentiment regimes, with higher returns and less risk.
    """)

st.divider()

# Row 7: Risk-Seeking Traders (left) & Content (right)
col11, col12 = st.columns(2)
with col11:
    st.subheader("Top 5 Risk-Seeking Traders (by StdPnL)")
    risk_traders = trader_metrics.sort_values('StdPnL', ascending=False).head(5)
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.barplot(data=risk_traders, x='Account', y='StdPnL', ax=ax)
    ax.set_title('Risk-Seeking Traders', fontsize=TITLE_FONT)
    ax.set_xlabel('Trader', fontsize=LABEL_FONT)
    ax.set_ylabel('StdPnL', fontsize=LABEL_FONT)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=85, fontsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    st.pyplot(fig)
with col12:
    st.markdown("""
    - This plot ranks traders by the standard deviation of their PnL (StdPnL), a measure of volatility in their results.
    - The top trader has much higher StdPnL than the rest, indicating very volatile performance.
    - Interpretation: These traders take bigger risks, leading to larger swings in their profits and losses. While this can result in high rewards, it also exposes them to greater potential losses. Their trading style is aggressive and less predictable.
    """)

st.divider()

# Row 8: Stable Traders (left) & Content (right)
col13, col14 = st.columns(2)
with col13:
    st.subheader("Top 5 Stable Traders (by Sharpe Ratio)")
    stable_traders = trader_metrics.sort_values('Sharpe', ascending=False).head(5)
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.barplot(data=stable_traders, x='Account', y='Sharpe', ax=ax)
    ax.set_title('Stable Traders', fontsize=TITLE_FONT)
    ax.set_xlabel('Trader', fontsize=LABEL_FONT)
    ax.set_ylabel('Sharpe Ratio', fontsize=LABEL_FONT)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=85, fontsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    st.pyplot(fig)
with col14:
    st.markdown("""
    - This plot shows the top 5 traders ranked by Sharpe ratio, a measure of risk-adjusted return.
    - All five traders have similar Sharpe ratios, indicating they achieve good returns for the risk they take.
    - Interpretation: These traders are efficient at converting risk into reward, likely due to strong risk management and consistent performance. They may not always have the highest profits, but their results are stable and reliable.
    """)

st.divider()

# Row 9: Consistent Traders (left) & Content (right)
col15, col16 = st.columns(2)
with col15:
    st.subheader("Top 5 Consistent Traders Across All Regimes")
    pivot = trader_stats.pivot_table(index='Account', columns='classification', values='Closed PnL_mean')
    consistent = pivot.dropna().loc[(pivot > 0).all(axis=1)]
    consistent['mean_across_regimes'] = consistent.mean(axis=1)
    sorted_consistent = consistent.sort_values(by='mean_across_regimes', ascending=False)
    top_5_consistent_traders = sorted_consistent.head()
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    top_5_consistent_traders['mean_across_regimes'].plot(kind='bar', ax=ax, color='#4F8BF9', title='Top 5 Consistent Traders')
    ax.set_ylabel('Mean Closed PnL Across Regimes', fontsize=LABEL_FONT)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=85, fontsize=TICK_FONT)
    st.pyplot(fig)
with col16:
    st.markdown("""
    - This plot highlights traders who have positive average PnL across all sentiment regimes, not just in favorable conditions.
    - The top two traders are especially consistent, maintaining strong average PnL regardless of market mood.
    - Interpretation: These traders are highly adaptable and resilient, able to perform well in both bullish and bearish markets. Their consistency is rare and valuable, indicating robust strategies and disciplined risk management.
    """)

st.divider()

# Row 10: Contrarian Traders (left) & Content (right)
col17, col18 = st.columns(2)
with col17:
    st.subheader("Top 5 Contrarian Traders (Positive PnL in Fear/Extreme Fear)")
    contrarian = pivot[(pivot['Extreme Fear'] > 0) & (pivot['Fear'] > 0)]
    contrarian['Mean across Fears'] = contrarian[['Extreme Fear', 'Fear']].mean(axis=1)
    contrarian = contrarian.sort_values(by='Mean across Fears', ascending=False)
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    contrarian.head()['Mean across Fears'].plot(kind='bar', ax=ax, color='#F9A826', title='Top 5 Contrarian Traders')
    ax.set_ylabel('Mean Closed PnL in Fear Regimes', fontsize=LABEL_FONT)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=85, fontsize=TICK_FONT)
    st.pyplot(fig)
with col18:
    st.markdown("""
    - This plot shows the top 5 traders who achieved the highest average closed PnL during periods of market Fear and Extreme Fear.
    - The first trader stands out with a much higher mean PnL in fear regimes than the others.
    - These traders are able to profit even when the market is fearful, suggesting they use strategies that benefit from downturns or market overreactions.
    - Interpretation: These are "contrarian" traders who succeed when most others struggle, possibly by taking positions against prevailing sentiment or exploiting panic-driven price moves.
    """)

st.success("You've reached the end of the Trader Performance Dashboard! This dashboard provides a comprehensive analysis of trader performance across different market sentiment regimes, highlighting key metrics and insights. Thank you for exploring the data with us!")
# --- Footer ---
st.markdown(
    "<div style='text-align: center;'>Trader Performance Dashboard | Developed by Chidwila Kotnana</div>",
    unsafe_allow_html=True
)