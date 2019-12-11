import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import findspark
findspark.init()
from pyspark.sql import functions as f
#from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.window import Window
from pyspark.ml.feature import RFormula, PCA

path = './CMAPSSData/'

def clean_project_directory():
    print(os.getcwd())
    if os.path.isdir(os.getcwd() + '/metastore_db'):
        shutil.rmtree(os.getcwd() + '/metastore_db', ignore_errors = True)
    if os.path.exists(os.getcwd() + '/derby.log'):
        os.remove(os.get_cwd() + '/derby.log')

clean_project_directory()

def create_spark_session(app_name, exe_memory):
    return SparkSession.builder.master('local[*]')\
        .appName(app_name)\
        .config('spark.executor.memory', exe_memory)\
        .getOrCreate()
spark = create_spark_session('Predictive Maintenance', '1gb')
sc = spark.sparkContext
sc.getConf().getAll()

def read_csv_as_rdd(sc, path, files_list, sep):
    rdd = sc.union([sc.textFile(path + f) for f in files_list])
    rdd = rdd.map(lambda x: x.split(sep))
    return rdd

train_files = sorted([filename for filename in os.listdir(path) if filename.startswith('train') and filename.endswith('.txt')])
test_files = sorted([filename for filename in os.listdir(path) if filename.startswith('test') and filename.endswith('.txt')])
print(train_files, test_files)

train_rdd = read_csv_as_rdd(sc, path, [train_files[0]], ' ')
test_rdd = read_csv_as_rdd(sc, path, [test_files[0]], ' ')

print('列数为', len(train_rdd.collect()[0]))

def convert_rdd_to_df(rdd, header):
    return rdd.toDF(header)
header = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 
        's15', 's16', 's17','s18', 's19', 's20', 's21', 's22', 's23']
train_df = convert_rdd_to_df(train_rdd, header)
test_df = convert_rdd_to_df(test_rdd, header)

##pd.read_csv(path + train_files[0], sep = " ", header = None).isnull().sum()

def remove_na_rows(df):
    return df.na.drop()
train_df = remove_na_rows(train_df)
test_df = remove_na_rows(test_df)

def set_df_data_types(df, int_list = [], dob_list = []):
    if len(int_list) > 0:
        for f in int_list:
            df = df.withColumn(f, df[f].cast(IntegerType()))

    if len(dob_list) > 0:
        for f in dob_list:
            df = df.withColumn(f, df[f].cast(DoubleType()))
    return df

int_list = ['id', 'cycle']
dob_list = list(set(train_df.columns) - set(int_list))
train_df = set_df_data_types(train_df, int_list, dob_list)
test_df = set_df_data_types(test_df, int_list, dob_list)

#显示每一类空值的数量
for col in train_df.columns:
    print(col, train_df.filter(isnan(col)).count(), 
        train_df.filter(isnull(col)).count())

train_df.groupBy('id').count().sort(f.col('count')).show(1)
train_df.groupBy('id').count().sort(f.col('count').desc()).show(1)
test_df.groupBy('id').count().sort(f.col('count')).show()
test_df.groupBy('id').count().sort(f.col('count').desc()).show(1)

def add_rul_labeling(df, idx, time):
    rul = df.groupBy([idx]).agg({time: 'max'}).sort((idx))
    rul = rul.toDF(idx, 'max')
    df = df.join(rul, on = [idx], how = 'left')
    df = df.withColumn('rul', df['max'] - df[time])
    df = df.drop('max')
    return df

train_df = add_rul_labeling(train_df, 'id', 'cycle')
train_df = train_df.drop('s22').drop('s23')
test_df = test_df.drop('s22').drop('s23')

