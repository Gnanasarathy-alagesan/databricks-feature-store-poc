# Databricks Feature Store POC

A complete machine learning workflow demonstration using Databricks Feature Store, showcasing centralized feature management and real-time inference capabilities.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd databricks-feature-store-poc
   ```

2. **Upload to Databricks Workspace**
   - Import notebooks from the `notebooks/` folder
   - Upload CSV files from the `data/` folder to Unity Catalog Volume: `/Volumes/workspace/prediction/customer_product/`

3. **Run notebooks in sequence**
   - Execute `01_feature_table_creation.py`
   - Execute `02_model_training.py`

## 📁 Project Structure

```
databricks-feature-store-poc/
├── 📓 notebooks/
│   ├── 01_feature_table_creation.py    # Creates feature tables from CSV data
│   └── 02_model_training.py            # Trains ML model using Feature Store
├── 📊 data/
│   ├── customer_features.csv           # Customer demographics & purchase history
│   ├── product_features.csv            # Product characteristics & metadata
│   ├── training_labels.csv             # Historical purchase outcomes
│   └── inference_data.csv              # Sample data for predictions
└── README.md                           # This file
```

## 📋 Prerequisites

### Infrastructure Requirements
- Databricks workspace (Runtime 13.0+ recommended)
- Feature Store enabled in your workspace
- MLflow for model tracking and registry
- Model Serving capability

### Data Requirements
Ensure all CSV files in the `data/` folder contain the following structure:

**customer_features.csv**
- `customer_id` (string) - Primary key
- `total_purchase_7d` (float) - Purchase amount last 7 days
- `total_purchase_30d` (float) - Purchase amount last 30 days
- Additional customer features...

**product_features.csv**
- `product_id` (string) - Primary key
- `category` (string) - Product category
- Additional product features...

**training_labels.csv**
- `customer_id` (string) - Foreign key
- `product_id` (string) - Foreign key
- `label` (int) - Target variable (0/1 for binary classification)

**inference_data.csv**
- `customer_id` (string) - Customer identifier
- `product_id` (string) - Product identifier

## 🔧 Setup Instructions

### Step 1: Data Upload
1. Upload all CSV files from `data/` folder to Unity Catalog Volume:
   ```python
   # In Databricks notebook
   dbutils.fs.cp("file:/path/to/local/file.csv", "/Volumes/workspace/prediction/customer_product/file.csv")
   ```
   
   Or using the Databricks UI:
   - Navigate to **Catalog Explorer**
   - Go to `workspace` → `prediction` → `customer_product` volume
   - Upload CSV files directly through the UI

### Step 2: Feature Table Creation
1. Open `01_feature_table_creation.py` in Databricks
2. Update file paths if necessary
3. Run all cells to create feature tables

### Step 3: Model Training
1. Open `02_model_training.py` in Databricks
2. Ensure feature tables are created successfully
3. Run all cells to train and register the model

### Step 4: Enable Online Tables (Manual)
1. Navigate to Feature Store in Databricks UI
2. Find your created feature tables
3. Enable online serving for real-time inference

## 📊 What This POC Demonstrates

### Core Capabilities
- **Feature Store Integration**: Centralized feature management with versioning
- **Automated Feature Joins**: Seamless integration during model training
- **Online/Offline Consistency**: Real-time serving with offline training
- **Model Registry**: MLflow integration for model versioning
- **Production Serving**: Auto-scaling inference endpoints

### Technical Workflow
1. **Data Ingestion**: CSV files → Feature Tables
2. **Feature Engineering**: Automated feature lookups and joins
3. **Model Training**: CatBoost classifier with Feature Store integration
4. **Model Registry**: Automated model versioning and metadata tracking
5. **Online Serving**: Real-time inference with sub-10ms feature retrieval

## 🎯 Expected Outcomes

### Performance Metrics
- **Feature Retrieval**: < 10ms latency
- **End-to-End Inference**: < 100ms
- **Model Training**: ~5 minutes with automated joins
- **Table Creation**: ~2 minutes for both feature tables

### Deliverables
- ✅ Production-ready feature tables
- ✅ Trained ML model with comprehensive metrics
- ✅ Registered model in MLflow Model Registry
- ✅ Online tables enabled for real-time serving
- ✅ Complete ML pipeline from data to inference

## 🔍 Key Files Explained

### `01_feature_table_creation.py`
**Purpose**: Creates centralized feature tables from raw CSV data

**Key Functions**:
- Loads and validates CSV files
- Creates `sales.customer_features` and `sales.product_features` tables
- Implements data quality checks and schema validation
- Sets up primary key constraints

**Outputs**:
- Feature tables ready for ML workflows
- Data quality reports
- Schema documentation

### `02_model_training.py`
**Purpose**: Trains machine learning model using Feature Store features

**Key Functions**:
- Configures feature lookups with automatic joins
- Trains CatBoost classifier for purchase prediction
- Evaluates model performance with comprehensive metrics
- Registers model in MLflow Model Registry

**Outputs**:
- Trained CatBoost model
- Feature importance analysis
- Model performance metrics
- MLflow experiment tracking

## 🚀 Next Steps

After completing this POC, consider:

1. **Enable Online Tables**: Manual step in Databricks UI for real-time serving
2. **Deploy Model Serving**: Set up REST API endpoint for live inference
3. **Add Monitoring**: Implement feature drift and model performance monitoring
4. **Scale Up**: Add more feature tables and streaming capabilities
5. **Production Hardening**: Add security, logging, and alerting

## 📞 Support

### Troubleshooting
- **File Not Found Errors**: Verify CSV files are uploaded to correct DBFS paths
- **Permission Issues**: Ensure workspace has Feature Store and MLflow enabled
- **Schema Errors**: Check CSV files match expected column names and types

### Resources
- [Unity Catalog Volumes Documentation](https://docs.databricks.com/en/connect/unity-catalog/volumes.html)
- [Databricks Feature Store Documentation](https://docs.databricks.com/en/machine-learning/feature-store/index.html)
- [MLflow Model Registry Guide](https://docs.databricks.com/en/mlflow/model-registry.html)
- [CatBoost Documentation](https://catboost.ai/en/docs/)

## 📄 License

This project is provided as-is for demonstration purposes. Please ensure compliance with your organization's data governance and security policies.

---

**Project Status**: ✅ POC Complete | **Next Phase**: Production Implementation

For questions or issues, please refer to the project documentation or contact the development team.
