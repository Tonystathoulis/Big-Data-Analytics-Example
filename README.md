# Big Data Processing and Analytics

This project implements a PySpark-based big data analytics workflow for analyzing product sales performance across different regions. It loads a fact table and multiple dimension tables, performs data cleaning and transformation, builds a star schema-style fact table, exports the final dataset, and applies machine learning techniques to predict high-value customers.

## Datasets

The project uses the following datasets:

- FactInternetSales.csv
- DimCustomer.csv
- DimDate.csv
- DimGeography.csv
- DimProduct.csv
- DimProductCategory.csv
- DimProductSubcategory.csv

These files are expected to be stored in the DataSet_final directory located in the same folder as the main Python script.

## Requirements

The project was developed and tested with:

- Python 3.10.x
- Java 11
- PySpark 3.5.1

Additional Python libraries:

- numpy
- matplotlib
- seaborn
- scikit-learn

## Setup Instructions
### Manual setup

If you prefer to set up the environment manually, install the required libraries using:

```bash
pip install pyspark==3.5.1
pip install numpy==1.26.4
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install scikit-learn==1.3.2
```

## Running the Project

Run the main script:

```bash
python 100675768_implementation.py
```

The script will:

- Load the datasets
- Perform cleaning and transformation
- Generate a final fact table
- Export the output to final_fact_table.csv
- Display visualizations and model evaluation results

## Output Files

The script generates:

- final_fact_table.csv: the cleaned and transformed final fact table
- Interactive or displayed plots for sales and product analysis

## Notes

- The dataset folder must remain in the same directory as the Python script.
- The implementation is designed for educational and analytical purposes and demonstrates a full big data pipeline using PySpark.
