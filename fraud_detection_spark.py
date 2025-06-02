# fraud_detection.py
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.sql.functions import lower, col, regexp_replace, trim, desc

from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, DecisionTreeClassificationModel, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.spark import SparkXGBClassifier
import shutil
import os

def initialize_spark():
    """Initialize and configure Spark session"""
    return SparkSession.builder \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.1") \
        .appName("FraudDetection") \
        .getOrCreate()

def load_and_clean_data(spark, url):
    """Load data with schema validation and cleaning"""
    schema = StructType([
        StructField("dialogue", StringType(), True),
        StructField("personality", StringType(), True),
        StructField("type", StringType(), True),
        StructField("labels", StringType(), True)
    ])
    
    df = spark.createDataFrame(pd.read_csv(url), schema)
    df = df.filter(trim(col("labels")).isin(["0", "1"]))
    df = df.withColumn("labels", col("labels").cast(DoubleType()))
    return df.withColumn(
        "clean_text", 
        regexp_replace(lower(col("dialogue")), "[^a-zA-Z ]", "")
    ).filter(col("clean_text") != "")

def build_feature_pipeline():
    """Feature engineering pipeline stages"""
    return [
        Tokenizer(inputCol="clean_text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered_words"),
        CountVectorizer(inputCol="filtered_words", outputCol="raw_features", vocabSize=20000),
        IDF(inputCol="raw_features", outputCol="features")
    ]

def train_models(train_df, feature_stages):
    """Train multiple classifiers with hyperparameter tuning"""
    # Common classifier configurations
    dt = DecisionTreeClassifier(
        featuresCol="features", 
        labelCol="labels",
        maxDepth=5,
        probabilityCol="probability",
        rawPredictionCol="rawPrediction"
    )
    
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="labels",
        numTrees=100,
        maxDepth=5,
        seed=42,
        featureSubsetStrategy="auto"
    )
    
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="labels",
        num_workers=4,
        max_depth=5,
        n_estimators=100,
        eval_metric="auc"
    )
    
    models = {
        "DecisionTree": Pipeline(stages=feature_stages + [dt]),
        "RandomForest": Pipeline(stages=feature_stages + [rf]),
        "XGBoost": Pipeline(stages=feature_stages + [xgb])
    }
    
    return {name: model.fit(train_df) for name, model in models.items()}

def evaluate_model(model, datasets):
    """Evaluate model on multiple datasets with comprehensive metrics"""
    results = {}
    
    for name, data in datasets.items():
        preds = model.transform(data)
        
        # Binary classification metrics
        bin_eval = BinaryClassificationEvaluator(
            labelCol="labels",
            rawPredictionCol="rawPrediction"
        )
        auc = bin_eval.evaluate(preds)
        
        # Multiclass metrics
        multi_eval = MulticlassClassificationEvaluator(labelCol="labels")
        metrics = {
            "Accuracy": multi_eval.setMetricName("accuracy").evaluate(preds),
            "Precision": multi_eval.setMetricName("weightedPrecision").evaluate(preds),
            "Recall": multi_eval.setMetricName("weightedRecall").evaluate(preds),
            "F1": multi_eval.setMetricName("f1").evaluate(preds),
            "AUC": auc
        }
        
        # Confusion matrix
        cm = preds.crosstab("labels", "prediction").toPandas()
        cm = cm.set_index(f"labels_prediction")
        
        results[name] = {"metrics": metrics, "confusion_matrix": cm}
    
    return results

def plot_with_annotations(ax, data, xlabel, ylabel, title, rotation=45):
    """Helper function to create plots with data labels"""
    plot = sns.barplot(data=data, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    
    # Add data labels
    for p in plot.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha="center", va="bottom")

def visualize_results(results):
    """Generate comprehensive visualizations with data labels"""
    # 1. Metrics comparison across models and datasets
    metrics_data = []
    for model_name, res in results.items():
        for dataset_name, values in res.items():
            for metric, score in values["metrics"].items():
                metrics_data.append({
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Metric": metric,
                    "Score": score
                })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot metrics comparison
    plt.figure(figsize=(15, 8))
    g = sns.FacetGrid(metrics_df, col="Metric", hue="Dataset", 
                      col_wrap=3, height=4, sharey=False)
    g.map(sns.barplot, "Model", "Score")
    g.add_legend()
    g.fig.suptitle("Model Performance Comparison Across Datasets", y=1.02)
    
    # Add data labels
    for ax in g.axes.flat:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha="center", va="bottom")
    
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.show()
    
    # 2. Confusion matrices
    for model_name, res in results.items():
        fig, axes = plt.subplots(1, len(res), figsize=(15, 6))  # Slightly taller figure
        fig.suptitle(f"{model_name} - Confusion Matrices", y=1.05, fontsize=16, fontweight='bold')

        # Set consistent font sizes
        title_fontsize = 15
        label_fontsize = 15
        annotation_fontsize = 15
        accuracy_fontsize = 14

        for i, (dataset_name, values) in enumerate(res.items()):
            ax = axes[i] if len(res) > 1 else axes
            
            # Create heatmap with larger annotations
            heatmap = sns.heatmap(
                values["confusion_matrix"], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                cbar=False,
                square=True,
                ax=ax,
                annot_kws={"size": annotation_fontsize}  # Larger annotation text
            )
            
            # Set title and labels with larger font
            ax.set_title(dataset_name, fontsize=title_fontsize, pad=12)
            ax.set_xlabel("Predicted", fontsize=label_fontsize)
            ax.set_ylabel("Actual", fontsize=label_fontsize)
            
            # Adjust tick labels size
            ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
            
            # Add accuracy annotation with larger font
            accuracy = values["metrics"]["Accuracy"]
            ax.text(0.5, -0.2,  # Slightly lower position
                    f"Accuracy: {accuracy:.4f}", 
                    ha="center", 
                    va="center",
                    transform=ax.transAxes,
                    fontsize=accuracy_fontsize,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

        plt.tight_layout()
        plt.savefig(f"confusion_matrices_{model_name.lower()}.png", dpi=300, bbox_inches='tight')
        plt.show()

def analyze_word_associations(spark, model, df, vocab, top_n=10):
    """Analyze word-class associations for any model"""
    # Process text through feature pipeline
    processed_df = model.stages[0].transform(df)  # Tokenizer
    processed_df = model.stages[1].transform(processed_df)  # StopWordsRemover
    
    # Get feature importances
    if isinstance(model.stages[-1], RandomForestClassificationModel):
        try: 
            importances = model.stages[-1].featureImportances
        except Exception as e:
            print(f"Could not get feature importances: {str(e)}")
            importances = None
        # Convert to numpy array for easier indexing
        importances_array = importances.toArray()
    elif isinstance(model.stages[-1], DecisionTreeClassificationModel):
        try:
            importances = model.stages[-1].featureImportances
        except Exception as e:
            print(f"Could not get feature importances: {str(e)}")
            importances = None
        # Convert to numpy array for easier indexing
        importances_array = importances.toArray()
    else:
        print("Model type not supported for feature importance analysis")
        return None
    
    print(importances_array)
    # Get top important features
    important_feature_indices = np.argsort(importances_array)[-top_n:][::-1]
    
    word_stats = []
    for idx in important_feature_indices:
        word = vocab[int(idx)]
        
        # Count word occurrences by label
        stats = processed_df.groupBy("labels").agg(
            F.sum(F.when(F.array_contains(F.col("filtered_words"), word), 1).otherwise(0)).alias("count")
        ).collect()
        
        # Get counts for each class
        scam_count = next((row["count"] for row in stats if row["labels"] == 1), 0)
        non_scam_count = next((row["count"] for row in stats if row["labels"] == 0), 0)
        ratio = scam_count / (scam_count + non_scam_count) if (scam_count + non_scam_count) > 0 else 0.0
        
        word_stats.append((word, int(scam_count), int(non_scam_count), round(float(ratio), 3), round(float(importances_array[idx]), 3)))
    
    # Create DataFrame
    word_stats_df = spark.createDataFrame(
        word_stats,
        ["word", "scam_count", "non_scam_count", "scam_ratio", "importance"]
    ).orderBy(desc("importance"))
    
    return word_stats_df

def plot_word_associations(word_stats_df, model_name):
    """Enhanced visualization of word-class associations"""
    pdf = word_stats_df.toPandas()
    
    # Create figure with two subplots
    plt.figure(figsize=(16, 6))
    
    # Plot 1: Word counts by class
    plt.subplot(1, 2, 1)
    melted_df = pdf.melt(id_vars="word", value_vars=["scam_count", "non_scam_count"])
    ax = sns.barplot(data=melted_df, x="word", y="value", hue="variable")
    plt.title(f"Word Frequency - {model_name}")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    
    # Add data labels
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom')
    
    # Plot 2: Scam ratio and importance
    plt.subplot(1, 2, 2)
    pdf = pdf.sort_values("scam_ratio", ascending=False)
    ax = sns.barplot(data=pdf, x="word", y="scam_ratio", color="salmon")
    ax2 = ax.twinx()
    sns.lineplot(data=pdf, x="word", y="importance", color="blue", marker="o", ax=ax2)
    
    plt.title(f"Scam Ratio vs Importance - {model_name}")
    plt.xlabel("Words")
    plt.ylabel("Scam Ratio")
    ax2.set_ylabel("Feature Importance", color="blue")
    plt.xticks(rotation=45, ha='right')
    
    # Add data labels
    for i, (_, row) in enumerate(pdf.iterrows()):
        ax.text(i, row["scam_ratio"] + 0.02, f"{row['scam_ratio']:.2f}", 
                ha='center', va='bottom')
        ax2.text(i, row["importance"] + 0.01, f"{row['importance']:.3f}", 
                 ha='center', va='bottom', color='blue')
    
    plt.tight_layout()
    plt.savefig(f"word_associations_{model_name.lower()}.png")
    plt.show()

def main():
    spark = initialize_spark()
    
    try:
        # Data pipeline
        url = "https://huggingface.co/datasets/BothBosu/multi-agent-scam-conversation/raw/main/agent_conversation_all.csv"
        df = load_and_clean_data(spark, url)
        
        # Save model to path
        model_path = "dialogue_classification_model"
        
        # Train-Validation-Test split (70-10-20)
        train_df, temp_df = df.randomSplit([0.7, 0.3], seed=42)
        val_df, test_df = temp_df.randomSplit([1/3, 2/3], seed=42)
        
        print(f"\nData Split Counts:")
        print(f"Training: {train_df.count()}")
        print(f"Validation: {val_df.count()}")
        print(f"Test: {test_df.count()}")
        
        # Feature engineering
        feature_stages = build_feature_pipeline()
        
        # Model training
        models = train_models(train_df, feature_stages)
        
        # Evaluation on both validation and test sets
        results = {}
        for name, model in models.items():
            results[name] = evaluate_model(
                model, 
                {"Validation": val_df, "Test": test_df}
            )
        
        # Print metrics
        print("\nModel Evaluation Results:")
        for model_name, res in results.items():
            print(f"\n{model_name}:")
            for dataset_name, values in res.items():
                print(f"\n{dataset_name} Set:")
                for metric, score in values["metrics"].items():
                    print(f"{metric}: {score:.4f}")
        
        # Visualization
        visualize_results(results)
        
        # Feature importance analysis
        if "RandomForest" in models:
            rf_model = models["RandomForest"]
            vocab = rf_model.stages[2].vocabulary
            rf_word_stats = analyze_word_associations(spark, rf_model, df, vocab)
            
            if rf_word_stats:
                print("\nRandom Forest - Top Words and Associations:")
                rf_word_stats.show()
                plot_word_associations(rf_word_stats, "RandomForest")

        # Similarly for Decision Tree if needed
        if "DecisionTree" in models:
            dt_model = models["DecisionTree"]
            vocab = dt_model.stages[2].vocabulary
            
            # Delete existing model if it exists
            model_path = "fraud_detection_model"
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                
            dt_model.save(model_path)
            dt_word_stats = analyze_word_associations(spark, dt_model, df, vocab)
            
            if dt_word_stats:
                print("\nDecision Tree - Top Words and Associations:")
                dt_word_stats.show()
                plot_word_associations(dt_word_stats, "DecisionTree")
           
    finally:
        spark.stop()

if __name__ == "__main__":
    main()