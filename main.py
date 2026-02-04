import pandas as pd
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class AdvancedAIProjectAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_metrics = None
        self.results = None

    def load_data(self):
        try:
            # Note: Encoding and separator kept as per your original file structure
            self.df = pd.read_csv(self.file_path, encoding='cp1254', sep=';')
            self.df.columns = self.df.columns.str.strip()
            print(f"Data Loaded: {self.df.shape[0]} records found.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def feature_engineering(self):
        # Numeric conversions
        cols = ['Cost', 'Investment', 'Fraud', 'CSAT', 'ProcessingTime']
        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.replace(',', '.')
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        self.df['Budget_Deviation'] = abs((self.df['Cost'] - self.df['Investment']) / (self.df['Investment'] + 1e-5))
        # Efficiency Score: Relation between Fraud/CSAT and Cost
        self.df['Efficiency_Score'] = ((self.df['Fraud'] * self.df['CSAT']) / (self.df['Cost'] + 1e-5)) * 1_000_000

    def calculate_trends(self):
        def get_slope(group):
            if len(group) > 1:
                return linregress(group['Year'], group['Efficiency_Score'])[0]
            return 0.0

        trend_data = self.df.groupby('Project Name').apply(get_slope).reset_index(name='Trend_Slope')
        avg_metrics = self.df.groupby('Project Name')[['Efficiency_Score', 'Budget_Deviation']].mean().reset_index()
        self.df_metrics = pd.merge(avg_metrics, trend_data, on='Project Name')

    def apply_advanced_algorithms(self):
        features = ['Efficiency_Score', 'Trend_Slope', 'Budget_Deviation']
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(self.df_metrics[features]), columns=features)

        hc = AgglomerativeClustering(n_clusters=3)
        self.df_metrics['Cluster_HC'] = hc.fit_predict(X_scaled)

        cluster_means = self.df_metrics.groupby('Cluster_HC')[features].mean()
        cluster_means['Quality_Index'] = cluster_means['Efficiency_Score'] + cluster_means['Trend_Slope']

        best_cluster = cluster_means['Quality_Index'].idxmax()
        worst_cluster = cluster_means['Quality_Index'].idxmin()

        def label_cluster(c):
            if c == best_cluster:
                return 'STAR (High Perf.)'
            elif c == worst_cluster:
                return 'RISKY (Low Perf.)'
            else:
                return 'STANDARD'

        self.df_metrics['Category'] = self.df_metrics['Cluster_HC'].map(label_cluster)

        iso = IsolationForest(contamination=0.15, random_state=42)
        self.df_metrics['Is_Anomaly'] = iso.fit_predict(X_scaled)  # -1 indicates anomaly

        X_scaled['Budget_Score'] = 1 - X_scaled['Budget_Deviation']

        self.df_metrics['Final_Score'] = (
                (X_scaled['Trend_Slope'] * 0.50) +
                (X_scaled['Efficiency_Score'] * 0.30) +
                (X_scaled['Budget_Score'] * 0.20)
        )

        self.results = self.df_metrics.sort_values('Final_Score', ascending=False)

    def generate_report(self):
        print("\nADVANCED AI PROJECT ANALYSIS REPORT")

        best = self.results.iloc[0]
        worst = self.results.iloc[-1]
        anomalies = self.results[self.results['Is_Anomaly'] == -1]['Project Name'].tolist()

        print(f"\nTOP PERFORMING PROJECT: {best['Project Name']}")
        print(f"   • Final Score: {best['Final_Score']:.4f}")
        print(f"   • Trend Slope: {best['Trend_Slope']:.2f}")
        print(f"   • Efficiency: {best['Efficiency_Score']:.2f}")

        print(f"\nHIGHEST RISK PROJECT: {worst['Project Name']}")
        print(f"   • Final Score: {worst['Final_Score']:.4f}")
        print(f"   • Category: {worst['Category']}")

        print(f"\nDETECTED OUTLIERS (Anomalies): {', '.join(anomalies) if anomalies else 'None'}")
        print("(These projects show unusual behavior, potentially representing high-impact breakthroughs or failures.)")

        print("\n--- RANKED PROJECT LIST ---")
        cols = ['Project Name', 'Category', 'Final_Score', 'Trend_Slope', 'Efficiency_Score']
        print(self.results[cols].to_string(index=False))

        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=self.results, x='Trend_Slope', y='Efficiency_Score',
                        hue='Category', size='Final_Score', sizes=(100, 500), palette='viridis')

        anom_data = self.results[self.results['Is_Anomaly'] == -1]
        if not anom_data.empty:
            plt.scatter(anom_data['Trend_Slope'], anom_data['Efficiency_Score'],
                        color='red', marker='x', s=150, label='Anomaly')

        plt.title('AI Projects: Score and Anomaly Analysis')
        plt.xlabel('Trend Slope (Growth)')
        plt.ylabel('Efficiency Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('Advanced_AI_Analysis.png')
        print("\nVisualization saved as 'Advanced_AI_Analysis.png'.")


if __name__ == "__main__":
    analyzer = AdvancedAIProjectAnalyzer('AIProjectDataSet.csv')
    analyzer.load_data()
    if analyzer.df is not None:
        analyzer.feature_engineering()
        analyzer.calculate_trends()
        analyzer.apply_advanced_algorithms()
        analyzer.generate_report()
