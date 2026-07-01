#IMPORTS OF PYSPARK MODULES
from pyspark.sql import SparkSession #FOR SPARK SESSION CREATION
import os   #FOR FILE PATH MANAGEMENT
import sys  #FOR PYTHON ENVIRONMENT MANAGEMENT
import matplotlib.pyplot as plt #FOR PLOTTING
import seaborn as sns   
from pyspark.sql.functions import (     
    col, year, month, dayofmonth, when, to_date, sum as spark_sum, rand  #FOR DATE EXTRACTION, CONDITIONAL OPERATIONS AND DATA CLEANING
) 
from pyspark.ml.feature import StringIndexer, OneHotEncoder #FOR FEATURE ENCODING
from pyspark.ml import Pipeline #FOR MULTIPLE DATA PROCESSING STEPS

#FOR USE OF THE SAME PYTHON ENVIRONMENT AS SPARK
PYTHON_PATH = sys.executable  
os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH

#GET CURRENT DIRECTORY OF THE FILE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#BUILD PATH TO DATASET FOLDER
data_path = os.path.join(BASE_DIR, "DataSet_final") + "/"

#CREATE A SPARK SESSION
spark = SparkSession.builder \
    .appName("DataWarehouse100675768") \
    .getOrCreate()

#LOAD OF FACT TABLE
fact_int_sales = spark.read.csv(
    data_path + "FactInternetSales.csv",
    header=True,
    inferSchema=True
)

#LOAD DIMPRODUCT TABLE
dim_product = spark.read.csv(
    data_path + "DimProduct.csv",
    header=True,
    inferSchema=True
)

#LOAD DIMPRODUCTSUBCATEGORY TABLE (CREATES PRODUCT HIERARCHY)
dim_product_sub = spark.read.csv(
    data_path + "DimProductSubcategory.csv",
    header=True,
    inferSchema=True
)

#LOAD DIMPRODUCTCATEGORY TABLE
dim_product_cat = spark.read.csv(
    data_path + "DimProductCategory.csv",
    header=True,
    inferSchema=True
)

#LOAD DIMCUSTOMER TABLE (PROVIDES CUSTOMER DEMOGRAPHIC AND GEOGRAPHIC INFORMATION)
dim_customer = spark.read.csv(
    data_path + "DimCustomer.csv",
    header=True,
    inferSchema=True
)

#LOAD DIMDATE TABLE (ALLOWS TIME BASED ANALYSIS SUCH AS YEARLY AND MONTHLY SALES)
dim_date = spark.read.csv(
    data_path + "DimDate.csv",
    header=True,
    inferSchema=True
)

#LOAD DIMGEOGRAPHY TABLE
dim_geo = spark.read.csv(
    data_path + "DimGeography.csv",
    header=True,
    inferSchema=True
)

#SHOW SAMPLE ROWS TO CONFIRM TABLES LOADED CORRECTLY
print("FACT INTERNET SALES:")
fact_int_sales.show(5)

print("DIM PRODUCT:")
dim_product.show(5)

print("DIM PRODUCT SUBCATEGORY:")
dim_product_sub.show(5)

print("DIM PRODUCT CATEGORY:")
dim_product_cat.show(5)

print("DIM CUSTOMER:")
dim_customer.show(5)

print("DIM DATE:")
dim_date.show(5)

print("DIM GEOGRAPHY:")
dim_geo.show(5)

print("\n      DATA QUALITY CHECKS        \n")
#CALCULATION OF MISSING VALUE SUMMARY
def missing_value_summary(df, name):
    missing_cols = df.select([
        spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns
    ])
    
    #SUM IN ALL COLUMNS TO GET TOTAL MISSING VALUES
    total_missing = missing_cols.select(
        sum(missing_cols[c] for c in missing_cols.columns).alias("total")
    ).first()["total"]

    print(f"{name}: {total_missing} missing values")

print("\nMISSING VALUE SUMMARY:")
missing_value_summary(fact_int_sales, "FACT INTERNET SALES")
missing_value_summary(dim_product, "DIM PRODUCT")
missing_value_summary(dim_product_sub, "DIM PRODUCT SUBCATEGORY")
missing_value_summary(dim_product_cat, "DIM PRODUCT CATEGORY")
missing_value_summary(dim_customer, "DIM CUSTOMER")
missing_value_summary(dim_date, "DIM DATE")
missing_value_summary(dim_geo, "DIM GEOGRAPHY")

#CLEAN OF DUPLICATE COUNTS
def duplicate_count(df, name):
    print(f"{name}: {df.count() - df.dropDuplicates().count()} duplicates")

duplicate_count(fact_int_sales, "FACT INTERNET SALES")
duplicate_count(dim_product, "DIM PRODUCT")
duplicate_count(dim_product_sub, "DIM PRODUCT SUBCATEGORY")
duplicate_count(dim_product_cat, "DIM PRODUCT CATEGORY")
duplicate_count(dim_customer, "DIM CUSTOMER")
duplicate_count(dim_date, "DIM DATE")
duplicate_count(dim_geo, "DIM GEOGRAPHY")

#CHECK FOREIGN KEY INTEGRITY
print("\nINVALID FOREIGN KEYS:")

invalid_product_fk = fact_int_sales.join(dim_product, "ProductKey", "left_anti").count() 
print(f"Invalid ProductKey: {invalid_product_fk}")

invalid_customer_fk = fact_int_sales.join(dim_customer, "CustomerKey", "left_anti").count()
print(f"Invalid CustomerKey: {invalid_customer_fk}")
#CHECK ORDERDATEKEY FOREIGN KEY
invalid_date_fk = fact_int_sales.join(
    dim_date,
    fact_int_sales["OrderDateKey"] == dim_date["DateKey"],
    "left_anti"
).count()
print(f"Invalid DateKey: {invalid_date_fk}")


print("\n\n      CLEANING & TRANSFORMATION      \n")

#CLEAN OF DIM PRODUCT DATASET
print("\nCLEANING DIM PRODUCT...")

#REPLACING "NULL" STRINGS WITH ACTUAL NULL VALUES
dim_product_clean = dim_product.select([
    when(col(c) == "NULL", None).otherwise(col(c)).alias(c)
    for c in dim_product.columns
])

#REMOVE INVALID PRODUCT SUBCATEGORY KEYS
dim_product_clean = dim_product_clean.filter(col("ProductSubcategoryKey") != 0)

#FILL MISSING VALUES IN NUMERIC COLUMNS WITH 0
num_cols_prod = ["StandardCost", "ListPrice", "Weight", "DealerPrice", 
                 "SafetyStockLevel", "ReorderPoint"]
for nc in num_cols_prod:
    if nc in dim_product_clean.columns:
        dim_product_clean = dim_product_clean.fillna(0, subset=[nc])

#CLEAN OF DIM PRODUCT SUBCATEGORY DATASET
print("\nCLEANING DIM PRODUCT SUBCATEGORY...")

dim_product_subcat_clean = dim_product_sub.filter(
    col("ProductSubcategoryKey") != 0
)

#CLEAN OF DIM PRODUCT CATEGORY DATASET
print("\nCLEANING DIM PRODUCT CATEGORY...")

dim_product_category_clean = dim_product_cat.filter(
    col("ProductCategoryKey") != 0
)

# CLEAN OF DIM CUSTOMER DATASET
print("\nCLEANING DIM CUSTOMER...")

#REPLACE "NULL" STRINGS WITH ACTUAL NULL VALUES
dim_customer_clean = dim_customer.select([
    when(col(c) == "NULL", None).otherwise(col(c)).alias(c)
    for c in dim_customer.columns
])

#REMOVE INVALID CUSTOMER KEYS
dim_customer_clean = dim_customer_clean.filter(col("CustomerKey") != 0)

#CONVERT DATE STRINGS TO DATE TYPE
dim_customer_clean = dim_customer_clean.withColumn(
    "BirthDate", to_date(col("BirthDate"))
).withColumn(
    "DateFirstPurchase", to_date(col("DateFirstPurchase"))
)

#STANDARDIZE GENDER VALUES OF CUSTOMERS
dim_customer_clean = dim_customer_clean.withColumn(
    "Gender",
    when(col("Gender") == "M", "Male")
    .when(col("Gender") == "F", "Female")
    .otherwise(col("Gender"))
)

#CLEAN OF DIM DATE DATASET
print("\nCLEANING DIM DATE...")

dim_date_clean = dim_date.filter(col("DateKey") != 0)

#CLEAN OF DIM GEOGRAPHY DATASET
print("\nCLEANING DIM GEOGRAPHY...")

#REPLACE "0" STRINGS WITH ACTUAL NULL VALUES
dim_geo_clean = dim_geo.select([
    when(col(c) == "0", None).otherwise(col(c)).alias(c)
    for c in dim_geo.columns
])

#REMOVE INVALID GEOGRAPHY KEYS
dim_geo_clean = dim_geo_clean.filter(col("GeographyKey") != 0)


#CLEAN OF FACT INTERNET SALES DATASET
print("\nCLEANING FACT INTERNET SALES...")

#REMOVAL OF FACT ROWS THAT HAVE INVALID PRODUCTKEY
fact_clean = fact_int_sales.join(
    dim_product_clean.select("ProductKey"),
    "ProductKey",
    "left"
)

#REMOVAL OF FACT ROWS THAT HAVE INVALID CUSTOMERKEY
fact_clean = fact_clean.join(
    dim_customer_clean.select("CustomerKey"),
    "CustomerKey",
    "left"
)

#REMOVAL OF FACT ROWS THAT HAVE INVALID ORDERDATEKEY
fact_clean = fact_clean.join(
    dim_date_clean.select(col("DateKey").alias("OrderDateKey")),
    "OrderDateKey",
    "left"
)

#REMOVE UNUSED COLUMNS FROM FACT TABLE
drop_cols_fact = ["CustomerPONumber", "CarrierTrackingNumber"]
for dc in drop_cols_fact:
    if dc in fact_clean.columns:
        fact_clean = fact_clean.drop(dc)

#PRINT SAMPLES OF CLEANED DATASETS
print("\nDIM PRODUCT CLEANED SAMPLE:")
dim_product_clean.show(5)

print("\nDIM PRODUCT SUBCATEGORY CLEANED SAMPLE:")
dim_product_subcat_clean.show(5)

print("\nDIM PRODUCT CATEGORY CLEANED SAMPLE:")
dim_product_category_clean.show(5)

print("\nDIM CUSTOMER CLEANED SAMPLE:")
dim_customer_clean.show(5)

print("\nDIM DATE CLEANED SAMPLE:")
dim_date_clean.show(5)

print("\nDIM GEOGRAPHY CLEANED SAMPLE:")
dim_geo_clean.show(5)

print("\nFACT INTERNET SALES CLEANED SAMPLE:")
fact_clean.show(5)

#BUILT OF FINAL STAR-SCHEMA FACT TABLE
print("\n\n      BUILDING STAR SCHEMA      \n")

#JOIN PRODUCT HIERARCHY: PRODUCT - SUBCATEGORY - CATEGORY
product_hier = dim_product_clean \
    .join(dim_product_subcat_clean, "ProductSubcategoryKey", "left") \
    .join(dim_product_category_clean, "ProductCategoryKey", "left")

#JOIN CUSTOMER - GEOGRAPHY
customer_geo = dim_customer_clean \
    .join(dim_geo_clean, "GeographyKey", "left_outer")  #ENSURING ALL ROWS FROM CUSTOMER ARE KEPT

#BUILT OF FINAL STAR-SCHEMA FACT TABLE
fact_final = fact_clean \
    .join(product_hier, "ProductKey", "left") \
    .join(customer_geo, "CustomerKey", "left") \
    .join(dim_date_clean, fact_clean["OrderDateKey"] == dim_date_clean["DateKey"], "left") \
    .drop("DateKey")  # Remove duplicate date key

#CHECK FOR MISSING GEOGRAPHY INFORMATION ON THE FINAL FACT TABLE
missing_geo = fact_final.filter(col("EnglishCountryRegionName").isNull()).count()
print(f"\nRows with missing geography info: {missing_geo}")

#REMOVAL OF DUPLICATE / AMBIGUOUS COLUMN NAMES
print("\nREMOVING DUPLICATE / AMBIGUOUS COLUMN NAMES...")
cols = fact_final.columns
duplicates = set([c for c in cols if cols.count(c) > 1])

if duplicates:
    print(f"Duplicate columns found: {duplicates}")
else:
    print("No duplicate columns found.")

#REMOVE DUPLICATE COLUMNS (BUT KEEP FIRST OCCURRENCE)
for col_name in duplicates:
    occurrences = [c for c in cols if c == col_name]
    for dup_col in occurrences[1:]:  #SKIP FIRST OCCURRENCE
        fact_final = fact_final.drop(dup_col)

print("Duplicate column cleanup complete.\n")

#EXPORT FINAL FACT TABLE SAMPLE FOR REPORTING
fact_final.select(
    "SalesOrderNumber",
    "OrderDate",
    "CustomerKey",
    "ProductKey",
    "EnglishProductName",
    "EnglishProductCategoryName",
    "EnglishCountryRegionName",
    "SalesAmount"
).toPandas().to_csv(
    os.path.join(BASE_DIR, "final_fact_table.csv"),
    index=False)

print("\nFinal Fact Table has been created and exported to 'final_fact_table.csv'.")

#PRINT OF FINAL FACT TABLE DIMENSIONS
print(f"Rows: {fact_final.count()}")
print(f"Columns: {len(fact_final.columns)}")

#DATA PREPARATION FOR MACHINE LEARNING

ml_fact = fact_final #CREATION OF A COPY FOR ML PREPARATION

#DROP OF COLIMNS WITH HIGH-CARDINALITY TEXT
print("\nREMOVING HIGH-CARDINALITY TEXT COLUMNS...")

drop_text_cols = [
    "EnglishProductName",          
    "FrenchProductName",
    "SpanishProductName",
    "FirstName",
    "LastName",
    "AddressLine1",
    "AddressLine2",
    "Phone",
    "EmailAddress",
    "MiddleName"
]

for d in drop_text_cols:
    if d in ml_fact.columns:
        ml_fact = ml_fact.drop(d)

print("Dropped high-cardinality text fields for ML")

#CONVERSION OF DATE COLUMNS TO NUMERIC FEATURES
print("\nCONVERTING DATE COLUMNS INTO NUMERIC FEATURES...")
#EXTRACT YEAR, MONTH, DAY FROM ORDER DATE
if "OrderDate" in ml_fact.columns:
    ml_fact = ml_fact.withColumn("OrderYear", year(col("OrderDate"))) \
                     .withColumn("OrderMonth", month(col("OrderDate"))) \
                     .withColumn("OrderDay", dayofmonth(col("OrderDate")))
    
    
#ENCODING THE CATEGORICAL FEATURES FOR THE ML MODELS TO PROCESS CORRECTLY
print("\nENCODING CATEGORICAL FEATURES...")

categorical_cols = [
    "EnglishProductSubcategoryName",
    "EnglishProductCategoryName",
    "City",
    "StateProvinceName",
    "CountryRegionCode",
    "Gender",
    "EnglishEducation",   
    "EnglishOccupation",
    "MaritalStatus",
]

categorical_cols = [c for c in categorical_cols if c in ml_fact.columns] #CHECK FOR COLUMN EXISTENCE TO AVOID ERRORS IN TRANSFORMATION

indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in categorical_cols] #HANDLE INVALIDS TO AVOID ERRORS 
encoders = [OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_vec") for c in categorical_cols] #CREATION OF VECTORS FOR CATEGORICAL FEATURES SO THEY CAN BE USED IN ML MODELS
#CONSTRUCTION OF PIPELINE WITH BOTH INDEXERS AND ENCODERS
pipeline = Pipeline(stages=indexers + encoders)
model = pipeline.fit(ml_fact)
fact_encoded = model.transform(ml_fact)

#FILL THE MISSING VALUES IN THE FINAL FACT TABLE
numeric_cols = [c for c, t in fact_encoded.dtypes if t in ("int", "double", "float", "bigint")]
string_cols = [c for c, t in fact_encoded.dtypes if t == "string"]
#ENSURE NO MISSING VALUES EXIST IN NUMERIC COLUMNS
fact_encoded = fact_encoded.fillna(0, subset=numeric_cols)
fact_encoded = fact_encoded.fillna("Unknown", subset=string_cols)

print("\nML dataset is now numeric-friendly and clean.\n")
fact_encoded.printSchema()

#MACHINE LEARNING TARGET CREATION

#TARGET VARIABLE ARE HIGH-VALUE CUSTOMERS 
#CUSTOMERS IN THE TOP 30% OF TOTAL REVENUE ARE LABELED AS 1
customer_revenue = (
    fact_encoded
    .groupBy("CustomerKey")
    .agg(spark_sum("SalesAmount").alias("total_customer_revenue"))
)

#CALCULATION IN THE 70TH PERCENTILE TO SEPARATE HIGH-VALUE CUSTOMERS
revenue_threshold = customer_revenue.approxQuantile(
    "total_customer_revenue", [0.7], 0.01
)[0]

#LABEL ASSIGNMENT BASED ON REVENUE THRESHOLD 1 FOR HIGH-VALUE CUSTOMERS, 0 OTHERWISE
customer_revenue = customer_revenue.withColumn(
    "label",
    when(col("total_customer_revenue") >= revenue_threshold, 1).otherwise(0)
)

#JOIN CUSTOMER LABELS BACK TO THE MAIN FACT TABLE SO ROW-LEVEL TRANSACTIONS CAN BE USED FOR MODELING
fact_encoded = fact_encoded.join(
    customer_revenue.select("CustomerKey", "label"),
    on="CustomerKey",
    how="inner"
)

#DATA VISUALIZATION AND PLOTTING
#CONVERSION OF FINAL FACT TABLE TO PANDAS FOR VISUALIZATION
fact_final_pd = fact_final.toPandas() 

#CREATION OF MAPPINGS FOR REGIONS BASED ON COUNTRIES
#CREATION OF DICTIONARY TO MAP COUNTRIES TO REGIONS
country_to_region = {
    'United Kingdom': 'Europe',
    'France': 'Europe',
    'Germany': 'Europe',  
    'United States': 'Americas',
    'Canada': 'Americas',
    'Australia': 'Oceania'
}

#REGION COLUMN FOR MAPPING COUNTRIES TO REGIONS
fact_final_pd['Region'] = fact_final_pd['EnglishCountryRegionName'].map(country_to_region)

#CHECK IF ANY COUNTRIES WERE NOT MAPPED
print("\nMissing values in 'Region':")
print(fact_final_pd['Region'].isnull().sum())  

#GROUP BY 'YEARLYINCOME' AND 'REGION' AND SUM SALES AMOUNT
sales_binc_reg = fact_final_pd.groupby(['YearlyIncome', 'Region'])['SalesAmount'].sum().reset_index()

#SORT DATA BY SALES AMOUNT FOR BETTER VISUALIZATION
sales_binc_reg = sales_binc_reg.sort_values(by='SalesAmount', ascending=False)

#PLOT SALES BY YEARLY INCOME AND REGION
print("\nGenerating Sales by Yearly Income and Customer Region Plot...")
plt.figure(figsize=(14, 8))
sns.barplot(x='SalesAmount', y='Region', hue='YearlyIncome', data=sales_binc_reg)
plt.title('Total Sales by Yearly Income and Region')
plt.xlabel('Total Sales ($)')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

#SECOND VISUALIZATION OF TOP PRODUCTS BY SALES IN EACH REGION
#CALCULATE TOTAL SALES BY PRODUCT IN EACH REGION
sales_bprod_reg = fact_final_pd.groupby(
    ['EnglishProductName', 'Region']
)['SalesAmount'].sum().reset_index()

#GET THE TOP PRODUCT BY SALES IN EACH REGION
top_prod_breg = sales_bprod_reg.loc[
    sales_bprod_reg.groupby('Region')['SalesAmount'].idxmax()
]

#SORT DATA BY SALES AMOUNT FOR BETTER VISUALIZATION
top_prod_breg = top_prod_breg.sort_values(by='SalesAmount', ascending=False)

#PLOT TOP PRODUCT BY SALES IN EACH REGION
print("\nGenerating Sales by Product in each Region Plot (Top Product by Region)...")
plt.figure(figsize=(14, 10))
sns.barplot(
    x='SalesAmount', 
    y='EnglishProductName', 
    hue='Region',  
    data=top_prod_breg
)
plt.title('Top Product by Sales in Each Region')
plt.xlabel('Total Sales ($)')
plt.ylabel('Product Name')
plt.tight_layout()
plt.show()

#MACHINE LEARNING MODELING FOR PREDICTING CUSTOMERS THAT ARE MOST LIKELY TO GENERATE A HIGH SALES VOLUME
from pyspark.ml.feature import VectorAssembler, StandardScaler #FOR FEATURE ASSEMBLY AND SCALING
from pyspark.ml.classification import LogisticRegression, GBTClassifier #FOR LOGISTIC REGRESSION AND GRADIENT BOOSTED TREES CLASSIFIERS
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator #FOR MODEL EVALUATION
from sklearn.metrics import auc, confusion_matrix #FOR AUC AND CONFUSION MATRIX CALCULATION
from pyspark.sql.functions import rand #FOR CONDITIONAL OPERATIONS AND RANDOM SAMPLING
import matplotlib.pyplot as plt #FOR PLOTTING
import seaborn as sns 

#FEATURE VECTOR ASSEMBLY EXCLUDING TARGET AND LEAKAGE VARIABLES
exclude_cols = [
    "label",
    "OrderQuantity",
    "SalesAmount",
    "OrderDate"
]
#SELECTION OF NUMERIC AND ENCODED FEATURES FOR MODEL INPUT
feature_cols = [
    c for c in fact_encoded.columns
    if (
        c.endswith("_vec") or
        (c not in exclude_cols and fact_encoded.schema[c].dataType.simpleString()
         in ["int", "double", "bigint", "float"])
    )
]
#ASSEMBLER TO COMBINE FEATURES INTO A SINGLE VECTOR
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

ml_data = assembler.transform(fact_encoded)

#CREATION OF CLASS WEIGHTS TO HANDLE CLASS IMBALANCE
class_counts = (
    ml_data.groupBy("label")
    .count()
    .collect()
)

total = sum(row["count"] for row in class_counts)

weights = {     #CREATION OF A DICTIONARY WITH WEIGHTS FOR EACH CLASS SO THAT THEY ARE INVERSELY PROPORTIONAL TO THEIR FREQUENCY
    row["label"]: total / row["count"]
    for row in class_counts
}

#ADD WEIGHT COLUMN TO THE DATAFRAME
ml_data = ml_data.withColumn(
    "weight",
    when(col("label") == 1, weights.get(1, 1.0))
    .otherwise(weights.get(0, 1.0))
)

#FEATURE SCALING FOR LOGISTIC REGRESSION PERFORMANCE
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(ml_data)  #FITTING THE STANDARD SCALER MODEL TO THE DATA FOR THE MEAN AND STANDARD DEVIATION
ml_data = scaler_model.transform(ml_data)

#THE LIST OF ALL COLUMNS IN THE DATAFRAME
all_cols = set(ml_data.columns)

#SELECTION OF NUMERIC FEATURES
numeric_feature_cols = [
    c for c, t in ml_data.dtypes
    if t in ("int", "double", "bigint", "float")
    and c not in exclude_cols
    and c != "weight"
]

#SELECTION OF COLUMNS THAT ARE INDEXED CATEGORICAL FEATURES
indexed_categorical_cols = [
    c for c in ml_data.columns
    if c.endswith("_idx")
]

#SELECTION OF NUMERIC AND ENCODED FEATURES FOR GBT MODEL INPUT
gbt_feature_cols = [
    c for c in ml_data.columns
    if c.endswith("_vec") or c in numeric_feature_cols
]
#ASSEMBLER FOR GBT FEATURES BY COMBINING SELECTED FEATURES INTO A VECTOR
gbt_assembler = VectorAssembler(
    inputCols=gbt_feature_cols,
    outputCol="gbt_features"
)

ml_data = gbt_assembler.transform(ml_data) #TRANSFORMATION OF THE DATAFRAME TO INCLUDE GBT FEATURES

#TESTING AND TRAINING SPLIT BASED ON CUSTOMERS TO AVOID DATA LEAKAGE
#RANDOM SHUFFLING OF CUSTOMER IDS
customer_ids = (
    ml_data.select("CustomerKey")
    .distinct()
    .orderBy(rand())
)
#SELECT 80% OF CUSTOMERS FOR TRAINING, 20% FOR TESTING
train_customers = customer_ids.limit(
    int(customer_ids.count() * 0.8)
)

test_customers = customer_ids.subtract(train_customers) #REMAINING CUSTOMERS FOR TESTING
#JOIN THE TRAINING SET WITH THE ORIGINAL DATAFRAME
train_df = ml_data.join(train_customers, on="CustomerKey", how="inner")
test_df = ml_data.join(test_customers, on="CustomerKey", how="inner")

#DEFINITION OF THE TWO MODELS BEFORE TRAINING
models = {
    "Logistic Regression": LogisticRegression(
        featuresCol="scaled_features",
        labelCol="label",
        maxIter=100,
        regParam=0.05,
        elasticNetParam=0.0
    ),

    "Gradient Boosted Trees": GBTClassifier(
        featuresCol="gbt_features",
        labelCol="label",
        maxIter=50,
        maxDepth=5,
        stepSize=0.1
    )
}

#MODEL TRAINING
trained_models = {
    name: model.fit(train_df)
    for name, model in models.items()
}

predictions = {} #DICTIONARY TO HOLD PREDICTIONS FROM EACH MODEL

#ITERATION OVER EACH TRAINED MODEL TO GENERATE PREDICTIONS
for name, model in trained_models.items():
    if name == "Logistic Regression":
        model.setThreshold(0.45)
    predictions[name] = model.transform(test_df)


#MODEL EVALUATION SECTION (AUC, PRECISION, RECALL)
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)
#MODEL PERFORMANCE OUTPUT
print("\nMODEL PERFORMANCE SUMMARY:")
for name, preds in predictions.items():
    auc_score = evaluator_auc.evaluate(preds)
    precision = evaluator_precision.evaluate(preds)
    recall = evaluator_recall.evaluate(preds)
    print(f"{name:20s} | AUC: {auc_score:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

#FUNCTIONS FOR  PLOTTING THE CONFUSION MATRICES
def plot_coma(preds, model_name, ax):
    pdf = preds.select("label", "prediction").toPandas() #CONVERSION TO PANDAS FOR USE WITH SEABORN
    cm = confusion_matrix(pdf["label"], pdf["prediction"]) #GENERATION OF CONFUSION MATRIX

    sns.heatmap(
        cm, annot=True, fmt="g", cmap="Blues",
        xticklabels=[0, 1], yticklabels=[0, 1], ax=ax
    )
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

#PLOTTING OF CONFUSION MATRICES
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

for i, (name, preds) in enumerate(predictions.items()):
    plot_coma(preds, name, ax1 if i == 0 else ax2)

plt.tight_layout()
plt.show()

