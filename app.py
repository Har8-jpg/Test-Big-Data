import os
import sys

# Make sure Spark uses the same Python environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import streamlit as st
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# -------------------------------------------------
# Step 1: Start Spark Session
# -------------------------------------------------
@st.cache_resource
def create_spark_session():
    spark = SparkSession.builder \
        .appName("Hacker KMeans Clustering") \
        .master("local[1]") \
        .config("spark.sql.shuffle.partitions", "1") \
        .config("spark.default.parallelism", "1") \
        .config("spark.python.worker.reuse", "false") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


spark = create_spark_session()


# -------------------------------------------------
# Streamlit Interface
# -------------------------------------------------
st.title("TechnologyKu Hacker Attack Clustering System")

st.write("""
This web-based system uses PySpark and K-Means clustering to group hacker attack
sessions based on similar behaviour patterns.
""")


# -------------------------------------------------
# Step 2: Upload CSV File
# -------------------------------------------------
uploaded_file = st.file_uploader("hack_data.csv", type=["csv"])

if uploaded_file is not None:

    try:
        # Read CSV using pandas
        pandas_df = pd.read_csv(uploaded_file)

        st.subheader("1. Dataset Preview")
        st.dataframe(pandas_df.head())

        st.subheader("2. Dataset Shape")
        st.write("Number of rows:", pandas_df.shape[0])
        st.write("Number of columns:", pandas_df.shape[1])

        st.subheader("3. Dataset Columns")
        st.write(list(pandas_df.columns))

        st.subheader("4. Data Types")
        st.write(pandas_df.dtypes.astype(str))

        st.subheader("5. Missing Values")
        st.write(pandas_df.isnull().sum())


        # -------------------------------------------------
        # Step 3: Select numeric columns
        # -------------------------------------------------
        numeric_cols = []

        for column_name in pandas_df.columns:
            if pd.api.types.is_numeric_dtype(pandas_df[column_name]):
                numeric_cols.append(column_name)

        st.subheader("6. Numeric Columns Selected")
        st.write(numeric_cols)

        if len(numeric_cols) < 2:
            st.error("Not enough numeric columns for K-Means clustering.")
            st.stop()


        # -------------------------------------------------
        # Step 4: Remove missing values
        # -------------------------------------------------
        cleaned_pandas_df = pandas_df.dropna(subset=numeric_cols)

        st.subheader("7. Data Cleaning")
        st.write("Rows before cleaning:", len(pandas_df))
        st.write("Rows after removing missing values:", len(cleaned_pandas_df))

        if len(cleaned_pandas_df) == 0:
            st.error("No data left after removing missing values.")
            st.stop()


        # -------------------------------------------------
        # Step 5: Convert cleaned pandas dataframe to Spark dataframe
        # -------------------------------------------------
        df = spark.createDataFrame(cleaned_pandas_df)
        df = df.repartition(1)


        # -------------------------------------------------
        # Step 6: Create feature vector using VectorAssembler
        # -------------------------------------------------
        assembler = VectorAssembler(
            inputCols=numeric_cols,
            outputCol="features",
            handleInvalid="skip"
        )

        assembled_df = assembler.transform(df)

        st.subheader("8. Feature Vector")
        st.write("Selected numeric columns were combined into one feature vector using VectorAssembler.")


        # -------------------------------------------------
        # Step 7: Scale features using StandardScaler
        # -------------------------------------------------
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )

        scaler_model = scaler.fit(assembled_df)
        scaled_df = scaler_model.transform(assembled_df)

        st.subheader("9. Feature Scaling")
        st.write("StandardScaler was applied to standardize the numeric features.")


        # -------------------------------------------------
        # Step 8: Select K value
        # -------------------------------------------------
        st.subheader("10. K-Means Clustering")

        k_value = st.selectbox(
            "Select number of clusters (k)",
            [2, 3]
        )


        # -------------------------------------------------
        # Step 9: Train K-Means model
        # -------------------------------------------------
        kmeans = KMeans(
            featuresCol="scaled_features",
            predictionCol="cluster",
            k=k_value,
            seed=42
        )

        model = kmeans.fit(scaled_df)
        clustered_df = model.transform(scaled_df)

        st.success(f"K-Means clustering completed using k = {k_value}")


        # -------------------------------------------------
        # Step 10: Evaluate clustering using Silhouette Score
        # -------------------------------------------------
        evaluator = ClusteringEvaluator(
            featuresCol="scaled_features",
            predictionCol="cluster",
            metricName="silhouette"
        )

        silhouette_score = evaluator.evaluate(clustered_df)

        st.subheader("11. Clustering Evaluation")
        st.write("Silhouette Score:", round(silhouette_score, 4))

        st.info("""
        A higher silhouette score means the clusters are more clearly separated.
        Compare k = 2 and k = 3 to decide whether two or three hackers were involved.
        """)


        # -------------------------------------------------
        # Step 11: Convert result to pandas for display
        # -------------------------------------------------
        result_df = clustered_df.select(*numeric_cols, "cluster").toPandas()

        st.subheader("12. Clustered Dataset")
        st.dataframe(result_df.head(30))


        # -------------------------------------------------
        # Step 12: Cluster distribution
        # -------------------------------------------------
        st.subheader("13. Cluster Distribution")

        cluster_count = result_df["cluster"].value_counts().sort_index()
        cluster_count_df = cluster_count.reset_index()
        cluster_count_df.columns = ["Cluster", "Number of Sessions"]

        st.dataframe(cluster_count_df)

        # Streamlit built-in bar chart, no matplotlib needed
        st.bar_chart(
            cluster_count_df.set_index("Cluster")
        )


        # -------------------------------------------------
        # Step 13: Cluster behaviour summary
        # -------------------------------------------------
        st.subheader("14. Cluster Behaviour Summary")

        cluster_summary = result_df.groupby("cluster")[numeric_cols].mean()
        st.dataframe(cluster_summary)

        st.write("""
        The table above shows the average behaviour for each cluster.
        For example, one cluster may show higher bytes transferred, while another
        cluster may show higher pages corrupted or typing speed.
        """)


        # -------------------------------------------------
        # Step 14: Simple interpretation
        # -------------------------------------------------
        st.subheader("15. Simple Interpretation")

        if k_value == 2:
            st.write("""
            For k = 2, the dataset is divided into two main behaviour groups.
            If the clusters are balanced and clearly different, it may suggest that
            two hackers were mainly involved.
            """)
        else:
            st.write("""
            For k = 3, the dataset is divided into three behaviour groups.
            If all three clusters have meaningful differences, it may support the
            possibility that a third hacker was involved.
            """)

        st.write("""
        Final decision should be made by comparing the silhouette score,
        cluster distribution, and cluster behaviour summary for k = 2 and k = 3.
        """)

    except Exception as e:
        st.error("An error occurred while running the clustering system.")
        st.exception(e)

else:
    st.info("Please upload a CSV file to start the clustering process.")