# customersegmentation
# README Section
# Enhanced Customer Segmentation with K-means Clustering
# This script generates synthetic customer data (~1,000 transactions) with multiple purchases per customer and applies K-means clustering to segment customers based on Recency, Frequency, Monetary, Customer Lifetime, and Purchase Interval. It produces a report with dynamic marketing strategies.
#
# Key Features:
# - Realistic data with product-specific pricing, seasonal patterns (Q4 boost, summer slump), and customer journey metrics (1–5 transactions per customer).
# - Advanced clustering with elbow method, silhouette scores, and temporal validation (train/test split).
# - Star schema for BI integration (Power BI/Tableau) with dimension (products, regions, customers) and fact tables.
# - Exports in Parquet, Excel, and PDF formats; includes radar chart data and model persistence.
# - Modular design with error handling, logging, and data validation for 50,000+ data points.
#
# How to Use:
# 1. Run the script to generate, cluster, and export data.
# 2. Import 'customer_segmentation_fact.parquet' or 'customer_segmentation_report.xlsx' into Power BI/Tableau.
# 3. Create visualizations (e.g., scatter plots, radar charts) or view 'segmentation_report.pdf'.
#
# Requirements:
# - Python 3.8+ with pandas, numpy, sklearn, joblib, fpdf, openpyxl, matplotlib.
#
# Output Files:
# - customer_segmentation_fact.parquet: Transaction data with segments.
# - product_dimension.parquet: Product lookup.
# - region_dimension.parquet: Region metadata.
# - customer_dimension.parquet: Customer segments with deciles.
# - radar_chart_data.parquet: Radar chart data.
# - customer_segmentation_report.xlsx: Transactions, clusters, segments.
# - segmentation_report.pdf: Cluster summary.
# - segmentation_pipeline.pkl: Saved model for inference.
#
# Inference:
# - Use `predict_segment(new_data)` to classify new customers after loading 'segmentation_pipeline.pkl'.
#
# This script showcases data engineering, clustering, and marketing analytics for business optimization, aligned with financial analysis goals.
