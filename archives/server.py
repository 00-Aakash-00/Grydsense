import os
import logging
import shutil
import base64
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Large-scale data handling
import polars as pl

# LLM-based capabilities
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

# For advanced Pandas-based queries (via PandasAI)
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasOpenAI

# Visualization
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Web Search & Scraping
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from llama_index.llms.groq import Groq

# ----------------------------------------------------------------------------
# 1. Environment & Logging
# ----------------------------------------------------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 2. Data Loading
# ----------------------------------------------------------------------------
# Replace 'data.parquet' with your actual large dataset path (Parquet, CSV, etc.)
DATA_FILE_PATH = os.path.join("data", "data.parquet")

def load_data() -> pl.DataFrame:
    """
    Loads the entire dataset as a Polars DataFrame.
    For extremely large data, consider lazy scanning with pl.scan_parquet().
    """
    df = pl.read_parquet(DATA_FILE_PATH)
    return df

# ----------------------------------------------------------------------------
# 3. Enhanced Plotting Class
# ----------------------------------------------------------------------------
class EnhancedPlotting:
    export_dir = 'exports'
    
    def __init__(self):
        # Seaborn style & context for aesthetics
        sns.set_style("whitegrid")
        sns.set_context("talk")

    def plot_bar(self, x_data, y_data, xlabel, ylabel, title="Bar Chart"):
        """
        Creates a more advanced, aesthetic bar chart using Seaborn & Matplotlib.
        """
        try:
            filename = 'img.png'
            if not os.path.exists(self.export_dir):
                os.makedirs(self.export_dir)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            palette = sns.color_palette("coolwarm", len(x_data))

            # Optionally sort data by y_data for a structured look:
            # sorted_indices = np.argsort(y_data)
            # x_data = [x_data[i] for i in sorted_indices]
            # y_data = [y_data[i] for i in sorted_indices]

            # Plot using Seaborn
            sns.barplot(
                x=x_data,
                y=y_data,
                palette=palette,
                edgecolor='black',
                ax=ax
            )

            ax.set_xlabel(xlabel, fontsize=14, color='darkslategray')
            ax.set_ylabel(ylabel, fontsize=14, color='darkslategray')
            ax.set_title(title, fontsize=16, color='darkslategray', pad=15)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            # Optional average line
            avg_value = np.mean(y_data)
            ax.axhline(avg_value, color='red', linewidth=1.2, linestyle='--', alpha=0.7)
            ax.text(
                len(x_data) - 0.5,
                avg_value + 0.5,
                f'Avg: {avg_value:.2f}',
                color='red',
                fontsize=12,
                ha='right'
            )

            # Annotate bar values
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f'{height:.2f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=12, color='black',
                    xytext=(0, 5),
                    textcoords='offset points'
                )

            plt.tight_layout()
            filepath = os.path.join(self.export_dir, filename)
            plt.savefig(filepath, dpi=120)
            plt.close()
            return "success"
        except Exception as e:
            return f"error - {str(e)}"

    def plot_pie(self, labels, sizes, title="Pie Chart"):
        """
        Generates a more advanced, aesthetic donut chart (pie chart with a hole).
        """
        try:
            filename = 'img.png'
            if not os.path.exists(self.export_dir):
                os.makedirs(self.export_dir)

            fig, ax = plt.subplots(figsize=(7, 7))
            palette = sns.color_palette("Spectral", len(labels))

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=palette,
                autopct='%1.1f%%',
                startangle=140,
                pctdistance=0.8,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                textprops={'color': 'black', 'fontsize': 12},
            )

            # Donut hole
            centre_circle = plt.Circle((0, 0), 0.65, fc='white')
            fig.gca().add_artist(centre_circle)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_weight('bold')

            ax.set_title(title, fontsize=16, color='darkslategray', pad=20)

            # Optional legend outside
            plt.legend(
                wedges,
                labels,
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )

            plt.tight_layout()
            filepath = os.path.join(self.export_dir, filename)
            plt.savefig(filepath, dpi=120, bbox_inches='tight')
            plt.close()
            return "success"
        except Exception as e:
            return f"error - {str(e)}"

# ----------------------------------------------------------------------------
# 4. SMARTBuildingDataRetriever Class
# ----------------------------------------------------------------------------
class SMARTBuildingDataRetriever(BaseToolSpec):
    """
    Advanced retriever with additional methods for deeper insights into workspace usage,
    occupant flow, capacity vs. usage, correlation analysis, etc.
    """
    spec_functions = [
        "summaryByRoomType",
        "topNMostUsedSpaces",
        "usageTrendsByHour",
        "occupancyVsCapacityAnalysis",
        "correlationOccupantCountUsage",
        "plot_bar",
        "plot_pie",
        "misc",
        "webBrowsing"
    ]
    export_dir = 'exports'

    def __init__(self):
        # We won’t store data in self. We'll load it as needed via Polars.
        self.plotter = EnhancedPlotting()

    def summaryByRoomType(self, start_date=None, end_date=None):
        """
        Returns total 'Occupied Minutes' grouped by Sub Category Name (Meeting Room, Phone Booth, etc.)
        in an optional date range.
        """
        df = load_data()

        if start_date:
            df = df.filter(pl.col("Timestamp") >= pl.lit(start_date))
        if end_date:
            df = df.filter(pl.col("Timestamp") <= pl.lit(end_date))

        grouped = df.groupby("Sub Category Name").agg([
            pl.col("Occupied Minutes").sum().alias("Total Occupied Minutes")
        ])
        return grouped.to_pandas().to_dict(orient="records")

    def topNMostUsedSpaces(self, n=5, start_date=None, end_date=None):
        """
        Finds the top N spaces by total Occupied Minutes in a date range.
        """
        df = load_data()
        if start_date:
            df = df.filter(pl.col("Timestamp") >= pl.lit(start_date))
        if end_date:
            df = df.filter(pl.col("Timestamp") <= pl.lit(end_date))

        grouped = (
            df.groupby("Entity Name")
              .agg(pl.col("Occupied Minutes").sum().alias("TotalOccupied"))
              .sort("TotalOccupied", descending=True)
        )
        top_spaces = grouped.head(n)
        return top_spaces.to_pandas().to_dict(orient="records")

    def usageTrendsByHour(self, date=None):
        """
        Aggregates occupant usage or Occupied Minutes by hour of day, for a given date.
        """
        df = load_data()

        if date:
            df = df.filter(pl.col("Date") == pl.lit(date))  # depends on your data structure

        df = df.with_column(pl.col("Timestamp").dt.hour().alias("HourOfDay"))
        grouped = df.groupby("HourOfDay").agg([
            pl.col("Occupied Minutes").mean().alias("AvgOccupiedMin"),
            pl.col("Occupied Count").mean().alias("AvgOccupantCount")
        ]).sort("HourOfDay")

        return grouped.to_pandas().to_dict(orient="records")

    def occupancyVsCapacityAnalysis(self, start_date=None, end_date=None):
        """
        Compares average occupant count vs. seating capacity across rooms 
        to see how well spaces are utilized.
        """
        df = load_data()
        if start_date:
            df = df.filter(pl.col("Timestamp") >= pl.lit(start_date))
        if end_date:
            df = df.filter(pl.col("Timestamp") <= pl.lit(end_date))

        grouped = df.groupby("Entity Name").agg([
            pl.col("Occupied Count").mean().alias("AvgOccupantCount"),
            pl.col("Seating Capacity").first().alias("SeatingCapacity")
        ])
        grouped = grouped.with_column(
            (pl.col("AvgOccupantCount") / pl.col("SeatingCapacity")).alias("UtilizationRate")
        )

        return grouped.to_pandas().to_dict(orient="records")

    def correlationOccupantCountUsage(self, start_date=None, end_date=None):
        """
        Analyzes correlation between 'Occupied Count' and 'Usage Frequency' or 'Occupied Minutes'.
        """
        df = load_data()
        if start_date:
            df = df.filter(pl.col("Timestamp") >= pl.lit(start_date))
        if end_date:
            df = df.filter(pl.col("Timestamp") <= pl.lit(end_date))

        pdf = df.select(["Occupied Count", "Usage Frequency", "Occupied Minutes"]).to_pandas()
        corr_count_freq = pdf["Occupied Count"].corr(pdf["Usage Frequency"])
        corr_count_minutes = pdf["Occupied Count"].corr(pdf["Occupied Minutes"])

        return {
            "OccupiedCount_vs_UsageFrequency": corr_count_freq,
            "OccupiedCount_vs_OccupiedMinutes": corr_count_minutes
        }

    # ---------------------------- PLOTTING METHODS --------------------------------
    def plot_bar(self, x_data, y_data, xlabel, ylabel, title="Bar Chart"):
        """
        Use the EnhancedPlotting class to generate a bar chart.
        """
        return self.plotter.plot_bar(x_data, y_data, xlabel, ylabel, title)

    def plot_pie(self, labels, sizes, title="Pie Chart"):
        """
        Use the EnhancedPlotting class to generate a pie (donut) chart.
        """
        return self.plotter.plot_pie(labels, sizes, title)

    # ---------------------------- LLM - DATAFRAME METHODS ------------------------
    def misc(self, prompt):
        """
        For advanced queries using PandasAI. We convert Polars to pandas.
        """
        df_polars = load_data()
        pdf = df_polars.to_pandas()
        llm = PandasOpenAI(api_token=openai_api_key)
        sdf = SmartDataframe(df=pdf, config={"llm": llm, "verbose": True, "save_charts": True})
        resp = sdf.chat(prompt)
        return resp

    def webBrowsing(self, query):
        """
        For queries not related to the building dataset (DuckDuckGo + Groq).
        """
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, region='wt-wt', safesearch='Moderate', timelimit='y')
            results = list(results)
            if not results:
                return "No results found for the query.", []

            sources = []
            texts = []
            for result in results[:3]:
                link = result.get("href") or result.get("url")
                if link:
                    sources.append(link)
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        page = requests.get(link, headers=headers, timeout=10)
                        soup = BeautifulSoup(page.content, 'html.parser')
                        text = soup.get_text(separator=' ', strip=True)
                        texts.append(text)
                    except Exception as e:
                        print(f"Could not retrieve content from {link}: {e}")

            if not texts:
                return "Could not retrieve content from any sources.", sources

            combined_text = ' '.join(texts)[:5000]
            llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)
            prompt = f"""
            Based on the following information from the sources, answer the question concisely:

            Information:
            {combined_text}

            Question:
            {query}

            Answer:
            """
            response = llm.complete(prompt)
            return response, sources
        except Exception as e:
            return f"An error occurred: {str(e)}", []

# ----------------------------------------------------------------------------
# 5. Helper Functions
# ----------------------------------------------------------------------------
def clear_exports_folder(folder_path='exports'):
    folder_path = os.path.join(os.getcwd(), folder_path)
    if not os.path.exists(folder_path):
        return
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                os.remove(file_path)
            except PermissionError:
                logger.warning(f"Permission denied: {file_path}")
            except Exception as e:
                logger.warning(f"Error deleting {file_path}: {e}")
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                shutil.rmtree(dir_path)
            except PermissionError:
                logger.warning(f"Permission denied: {dir_path}")
            except Exception as e:
                logger.warning(f"Error deleting {dir_path}: {e}")

def find_first_png(folder_path='exports'):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                return os.path.join(root, file)
    return None

# ----------------------------------------------------------------------------
# 6. Global LLM and Context
# ----------------------------------------------------------------------------
llm = OpenAI(model="gpt-4o", api_key=openai_api_key)
current_date = datetime.now().strftime("%Y-%m-%d")
context = f"""
    You are a Senior Smart Building AI assistant.
    Your goal is to provide advanced analytics on occupancy, usage trends,
    capacity vs. utilization, correlation analysis, etc.
    The current date is {current_date}.
"""

# ----------------------------------------------------------------------------
# 7. Flask Routes
# ----------------------------------------------------------------------------
@app.route('/query', methods=['POST'])
def handle_query():
    """
    Expects JSON with at least {"query": "..."}.
    Optional: "start_date", "end_date", or other parameters if relevant.
    """
    try:
        data = request.get_json()
        user_query = data.get('query')
        if not user_query:
            return jsonify({'error': 'Please provide a valid query.'}), 400

        clear_exports_folder('exports')

        # Initialize the data retriever
        sb_data_retriever = SMARTBuildingDataRetriever()

        # Create the agent
        agent = OpenAIAgent.from_tools(
            sb_data_retriever.to_tool_list(),
            llm=llm,
            verbose=True,
            context=context
        )

        response = agent.query(user_query)

        # Check for generated chart
        image_path = find_first_png('exports')
        image_data = None
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

        logger.info(f"Query: {user_query}")
        logger.info(f"Response: {response}")

        return jsonify({'response': response, 'image': image_data})

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/recommendedQueries', methods=['GET'])
def recommended_queries():
    """
    Returns a list of more advanced queries that showcase the data and new features.
    """
    queries = [
        "Show me the top 10 most utilized conference rooms",
        "Generate a bar chart comparing average occupant count by hour for 2024-03-01",
        "What is the correlation between 'Occupied Count' and 'Occupied Minutes'?",
        "Which spaces have the highest occupancy vs. capacity ratio?",
        "Plot a pie chart of total occupied minutes by sub category name",
        "Give me a correlation analysis between occupant count and usage frequency",
        "Perform a capacity utilization summary from 2024-03-01 to 2024-03-15"
    ]
    return jsonify({'queries': queries})


@app.route('/status', methods=['GET'])
def status():
    return 'active', 200

# ----------------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # In production, consider using gunicorn or another WSGI server
    app.run(debug=False, host='0.0.0.0', port=9000)