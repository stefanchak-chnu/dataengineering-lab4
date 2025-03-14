import os
from datetime import timedelta

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, month, dayofmonth, desc, row_number, to_timestamp
from pyspark.sql.window import Window


def create_spark_session():
    return SparkSession.builder \
        .appName("Bike Trips Analysis") \
        .getOrCreate()


def load_data(spark, file_path):
    return spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)


def save_result(df, output_dir, question_name):
    full_path = os.path.join("out", output_dir)
    os.makedirs(full_path, exist_ok=True)

    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(full_path)
    print(f"Result for {question_name} saved to {full_path}")


def preprocess_data(df):
    # Convert time columns
    df = df.withColumn("start_date", to_timestamp(col("start_time")))
    df = df.withColumn("end_date", to_timestamp(col("end_time")))

    return df


def average_trip_duration_per_day(df):
    """
    Question A: What is the average trip duration per day?
    """
    result = df.groupBy(dayofmonth(col("start_date")).alias("day")) \
        .agg(avg(col("duration_sec")).alias("avg_duration_sec"))

    return result


def trips_count_per_day(df):
    """
    Question B: How many trips were made each day?
    """
    result = df.groupBy(dayofmonth(col("start_date")).alias("day")) \
        .agg(count("*").alias("trips_count"))

    return result


def popular_start_station_per_month(df):
    """
    Question C: What was the most popular start station for each month?
    """
    # Group by month and start station, count trips
    station_counts = df.groupBy(month(col("start_date")).alias("month"),
                                col("from_station_name").alias("start_station")) \
        .agg(count("*").alias("trips_count"))

    # Define window for finding maximum count per month
    window_spec = Window.partitionBy("month").orderBy(desc("trips_count"))

    # Add rank column and filter for rank 1
    result = station_counts.withColumn("rank", row_number().over(window_spec)) \
        .filter(col("rank") == 1) \
        .select("month", "start_station", "trips_count")

    return result


def top_three_stations_last_two_weeks(df):
    """
    Question D: Which stations are in the top 3 for trips each day during the last two weeks?
    """
    # Find the latest date in the dataset
    max_date = df.agg({"start_date": "max"}).collect()[0][0]

    # Calculate the date two weeks ago
    two_weeks_ago = max_date - timedelta(days=14)

    # Filter for the last two weeks
    recent_df = df.filter(col("start_date") >= two_weeks_ago)

    # Group by day and start station, count trips
    station_counts = recent_df.groupBy(dayofmonth(col("start_date")).alias("day"),
                                       col("from_station_name").alias("start_station")) \
        .agg(count("*").alias("trips_count"))

    # Define window for ranking stations by day
    window_spec = Window.partitionBy("day").orderBy(desc("trips_count"))

    # Add rank column and filter for top 3
    result = station_counts.withColumn("rank", row_number().over(window_spec)) \
        .filter(col("rank") <= 3) \
        .select("day", "start_station", "trips_count", "rank")

    return result


def average_duration_by_gender(df):
    """
    Question E: Do men or women ride longer on average?
    """
    # Use the gender column
    result = df.groupBy(col("gender").alias("gender")) \
        .agg(avg(col("duration_sec")).alias("avg_duration_sec"))

    return result


def main():
    # Create Spark session
    spark = create_spark_session()

    df = load_data(spark, "/jobs/Divvy_Trips_2019_Q4.csv")

    processed_df = preprocess_data(df)

    # Process and save results for each question
    save_result(average_trip_duration_per_day(processed_df), "question_a", "Average trip duration per day")
    save_result(trips_count_per_day(processed_df), "question_b", "Trips count per day")
    save_result(popular_start_station_per_month(processed_df), "question_c","Most popular start station per month")
    save_result(top_three_stations_last_two_weeks(processed_df), "question_d", "Top 3 stations for last two weeks")
    save_result(average_duration_by_gender(processed_df), "question_e", "Average duration by gender")

    spark.stop()


if __name__ == "__main__":
    main()