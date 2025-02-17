import os
import logging
import shutil
import base64
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import polars as pl

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasOpenAI

import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------------------------------------------------------
# 1. Environment & Setup
# ----------------------------------------------------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 2. File Paths
# ----------------------------------------------------------------------------
MEET_AND_CONNECT_DATA_FILE = os.path.join("data", "data.parquet")
PRIMARY_WORK_DESK_DATA_FILE = os.path.join("data", "Primary_day_wise.parquet")

# ----------------------------------------------------------------------------
# 3. Data Loading
# ----------------------------------------------------------------------------
def load_meet_and_connect_data() -> pl.DataFrame:
    """
    Loads the 'Meet & Connect' dataset as a Polars DataFrame.
    Returns:
        pl.DataFrame
    """
    logger.info("Loading Meet & Connect data...")
    df_parquet = pl.read_parquet(MEET_AND_CONNECT_DATA_FILE)
    df = pl.DataFrame(df_parquet)  # Ensure a Polars DataFrame

    # If needed, parse timestamps/dates or rename columns here
    # e.g., df = df.with_columns(pl.col("Timestamp").str.strptime(pl.Datetime, "..."))
    logger.info(f"Meet&Connect columns: {df.columns}")
    return df

def load_primary_workdesk_data() -> pl.DataFrame:
    """
    Loads the 'Primary Work Desk' dataset as a Polars DataFrame.
    Returns:
        pl.DataFrame
    """
    logger.info("Loading Primary Work Desk data...")
    df_parquet = pl.read_parquet(PRIMARY_WORK_DESK_DATA_FILE)
    df = pl.DataFrame(df_parquet)

    # Rename columns to unify naming
    df = df.rename({
        "desk ID": "desk_id",
        "Date": "date",
        "Business Unit": "business_unit",
        "Floor Name": "floor_name",
        "Week day": "week_day",
        "Occupancy minutes": "occupancy_minutes",
    })

    # If needed, parse date
    # df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%m-%d-%Y", strict=False))

    logger.info(f"PrimaryDesk columns: {df.columns}")
    return df

# ----------------------------------------------------------------------------
# 4. Enhanced Plotting
# ----------------------------------------------------------------------------
class EnhancedPlotting:
    export_dir = 'exports'
    
    def __init__(self):
        sns.set_style("whitegrid")
        sns.set_context("talk")

    def plot_bar(self, x_data, y_data, xlabel, ylabel, title="Bar Chart"):
        """Creates a bar chart and saves to 'exports/img.png'."""
        try:
            if not os.path.exists(self.export_dir):
                os.makedirs(self.export_dir)

            filename = 'img.png'
            fig, ax = plt.subplots(figsize=(8, 6))
            palette = sns.color_palette("coolwarm", len(x_data))

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

            if len(y_data) > 0:
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
        """Creates a donut chart and saves to 'exports/img.png'."""
        try:
            if not os.path.exists(self.export_dir):
                os.makedirs(self.export_dir)

            filename = 'img.png'
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

            centre_circle = plt.Circle((0, 0), 0.65, fc='white')
            fig.gca().add_artist(centre_circle)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_weight('bold')

            ax.set_title(title, fontsize=16, color='darkslategray', pad=20)

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
# 5. MeetAndConnectDataRetriever
# ----------------------------------------------------------------------------
class MeetAndConnectDataRetriever(BaseToolSpec):
    """
    Tools for analyzing the 'Meet & Connect' dataset.
    """
    spec_functions = [
        "summaryByRoomType",
        "topNMostUsedSpaces",
        "occupancyVsCapacityAnalysis",
        "correlationOccupantCountUsage",
        "usageTrendsByHour",
        "plot_bar",
        "plot_pie",
        "misc"
    ]
    export_dir = 'exports'

    def __init__(self):
        self.plotter = EnhancedPlotting()

    def summaryByRoomType(self, start_date=None, end_date=None):
        """
        Aggregates total 'Occupied Minutes' grouped by 'Sub Category Name'.
        """
        df = load_meet_and_connect_data()

        # If you need to filter by date, e.g. on "Timestamp":
        # if start_date:
        #     df = df.filter(pl.col("Timestamp") >= start_date)
        # if end_date:
        #     df = df.filter(pl.col("Timestamp") <= end_date)

        # IMPORTANT: use group_by, not groupby
        grouped = df.group_by("Sub Category Name").agg(
            pl.col("Occupied Minutes").sum().alias("Total Occupied Minutes")
        )
        return grouped.to_pandas().to_dict(orient="records")

    def topNMostUsedSpaces(self, n=5, start_date=None, end_date=None):
        """
        Finds top N spaces by total 'Occupied Minutes'.
        """
        df = load_meet_and_connect_data()
        # if start_date: ...
        grouped = (
            df.group_by("Entity Name")
              .agg(pl.col("Occupied Minutes").sum().alias("TotalOccupied"))
              .sort("TotalOccupied", descending=True)
        )
        return grouped.head(n).to_pandas().to_dict(orient="records")

    def occupancyVsCapacityAnalysis(self, start_date=None, end_date=None):
        """
        Average occupant count vs seating capacity -> utilization rate.
        """
        df = load_meet_and_connect_data()
        grouped = df.group_by("Entity Name").agg([
            pl.col("Occupied Count").mean().alias("AvgOccupantCount"),
            pl.col("Seating Capacity").first().alias("SeatingCapacity")
        ])
        grouped = grouped.with_column(
            (pl.col("AvgOccupantCount") / pl.col("SeatingCapacity")).alias("UtilizationRate")
        )
        return grouped.to_pandas().to_dict(orient="records")

    def correlationOccupantCountUsage(self, start_date=None, end_date=None):
        """
        Checks correlation between occupant count and usage frequency/minutes.
        """
        df = load_meet_and_connect_data()
        pdf = df.select(["Occupied Count", "Usage Frequency", "Occupied Minutes"]).to_pandas()
        corr_count_freq = pdf["Occupied Count"].corr(pdf["Usage Frequency"])
        corr_count_minutes = pdf["Occupied Count"].corr(pdf["Occupied Minutes"])
        return {
            "OccupiedCount_vs_UsageFrequency": corr_count_freq,
            "OccupiedCount_vs_OccupiedMinutes": corr_count_minutes
        }

    def usageTrendsByHour(self, date=None):
        """
        Aggregates occupant usage by hour of day.
        """
        df = load_meet_and_connect_data()
        # e.g., if date:
        #     df = df.filter(pl.col("Date") == date)
        df = df.with_column(pl.col("Timestamp").dt.hour().alias("HourOfDay"))

        grouped = df.group_by("HourOfDay").agg([
            pl.col("Occupied Minutes").mean().alias("AvgOccupiedMin"),
            pl.col("Occupied Count").mean().alias("AvgOccupantCount")
        ]).sort("HourOfDay")
        return grouped.to_pandas().to_dict(orient="records")

    def plot_bar(self, x_data, y_data, xlabel, ylabel, title="Bar Chart"):
        return self.plotter.plot_bar(x_data, y_data, xlabel, ylabel, title)

    def plot_pie(self, labels, sizes, title="Pie Chart"):
        return self.plotter.plot_pie(labels, sizes, title)

    def misc(self, prompt):
        """
        Open-ended PandasAI queries.
        """
        df_polars = load_meet_and_connect_data()
        pdf = df_polars.to_pandas()
        llm_local = PandasOpenAI(api_token=openai_api_key)
        sdf = SmartDataframe(df=pdf, config={"llm": llm_local, "verbose": True, "save_charts": True})
        return sdf.chat(prompt)

# ----------------------------------------------------------------------------
# 6. PrimaryWorkDeskDataRetriever
# ----------------------------------------------------------------------------
class PrimaryWorkDeskDataRetriever(BaseToolSpec):
    """
    Tools for analyzing the 'Primary Work Desk' dataset.
    """
    spec_functions = [
        "summaryByDeskID",
        "topNDesksByOccupancy",
        "usageTrendsByDayOfWeek",
        "usageTrendsOverDateRange",
        "plot_bar",
        "plot_pie",
        "misc",
    ]
    export_dir = 'exports'

    def __init__(self):
        self.plotter = EnhancedPlotting()

    def summaryByDeskID(self, start_date=None, end_date=None):
        df = load_primary_workdesk_data()
        # Filter if needed by "date"
        grouped = df.group_by("desk_id").agg(
            pl.col("occupancy_minutes").sum().alias("total_occupancy_minutes")
        )
        return grouped.to_pandas().to_dict(orient="records")

    def topNDesksByOccupancy(self, n=5, start_date=None, end_date=None):
        df = load_primary_workdesk_data()
        # Filter if needed
        grouped = (
            df.group_by("desk_id")
              .agg(pl.col("occupancy_minutes").sum().alias("total_occupancy"))
              .sort("total_occupancy", descending=True)
        )
        return grouped.head(n).to_pandas().to_dict(orient="records")

    def usageTrendsByDayOfWeek(self, desk_id=None):
        df = load_primary_workdesk_data()
        if desk_id:
            df = df.filter(pl.col("desk_id") == desk_id)

        grouped = df.group_by("week_day").agg(
            pl.col("occupancy_minutes").mean().alias("avg_occupancy_minutes")
        ).sort("week_day")
        return grouped.to_pandas().to_dict(orient="records")

    def usageTrendsOverDateRange(self, desk_id=None):
        df = load_primary_workdesk_data()
        if desk_id:
            df = df.filter(pl.col("desk_id") == desk_id)

        grouped = df.group_by("date").agg(
            pl.col("occupancy_minutes").sum().alias("total_occupancy_minutes")
        ).sort("date")
        return grouped.to_pandas().to_dict(orient="records")

    def plot_bar(self, x_data, y_data, xlabel, ylabel, title="Bar Chart"):
        return self.plotter.plot_bar(x_data, y_data, xlabel, ylabel, title)

    def plot_pie(self, labels, sizes, title="Pie Chart"):
        return self.plotter.plot_pie(labels, sizes, title)

    def misc(self, prompt):
        df_polars = load_primary_workdesk_data()
        pdf = df_polars.to_pandas()
        llm_local = PandasOpenAI(api_token=openai_api_key)
        sdf = SmartDataframe(df=pdf, config={"llm": llm_local, "verbose": True, "save_charts": True})
        return sdf.chat(prompt)

# ----------------------------------------------------------------------------
# 7. Helpers
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
# 8. LLM & Flask
# ----------------------------------------------------------------------------
llm = OpenAI(model="gpt-4o", api_key=openai_api_key)
current_date = datetime.now().strftime("%Y-%m-%d")
context = f"""
    You are a Senior Smart Building AI assistant.
    Your goal is to provide advanced analytics on occupancy data,
    usage trends, capacity vs. utilization, correlation analysis, etc.
    The current date is {current_date}.
"""

@app.route('/meetAndConnect/query', methods=['POST'])
def meet_and_connect_query():
    try:
        data = request.get_json()
        user_query = data.get('query')
        if not user_query:
            return jsonify({'error': 'Please provide a valid query.'}), 400

        clear_exports_folder('exports')
        retriever = MeetAndConnectDataRetriever()
        agent = OpenAIAgent.from_tools(
            retriever.to_tool_list(),
            llm=llm,
            verbose=True,
            context=context
        )

        response = agent.query(user_query)
        image_path = find_first_png('exports')
        image_data = None
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({'response': response, 'image': image_data})
    except Exception as e:
        logger.error(f"MeetAndConnect query error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/meetAndConnect/recommendedQueries', methods=['GET'])
def meet_and_connect_recommended_queries():
    queries = [
        "Show total occupied minutes by room type",
        "Which are the top 5 most-used meeting rooms?",
        "Show me the occupancy vs. capacity analysis for the last quarter",
        "Compute correlation between occupant count and usage frequency",
        "Generate a bar chart of total occupied minutes by Sub Category Name",
    ]
    return jsonify({'queries': queries})

@app.route('/meetAndConnect/status', methods=['GET'])
def meet_and_connect_status():
    return 'MeetAndConnect dataset endpoints are active', 200

@app.route('/primaryWorkDeskData/query', methods=['POST'])
def primary_workdesk_query():
    try:
        data = request.get_json()
        user_query = data.get('query')
        if not user_query:
            return jsonify({'error': 'Please provide a valid query.'}), 400

        clear_exports_folder('exports')
        retriever = PrimaryWorkDeskDataRetriever()
        agent = OpenAIAgent.from_tools(
            retriever.to_tool_list(),
            llm=llm,
            verbose=True,
            context=context
        )

        response = agent.query(user_query)
        image_path = find_first_png('exports')
        image_data = None
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({'response': response, 'image': image_data})
    except Exception as e:
        logger.error(f"PrimaryWorkDesk query error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/primaryWorkDeskData/recommendedQueries', methods=['GET'])
def primary_workdesk_recommended_queries():
    queries = [
        "Give me the occupancy data by business units",
        "Show me the top 5 desks by total occupancy in November 2024",
        "Generate a bar chart of average occupancy by weekday",
        "Plot a pie chart of occupancy distribution by floor_name",
        "Compute the daily total occupancy for desk 003.A01",
    ]
    return jsonify({'queries': queries})

@app.route('/primaryWorkDeskData/status', methods=['GET'])
def primary_workdesk_status():
    return 'PrimaryWorkDeskData dataset endpoints are active', 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9000)