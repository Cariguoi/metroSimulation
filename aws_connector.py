import sys
import boto3
import os
import pymysql

ENDPOINT = "datageo.cphx1sr3zx0n.us-east-1.rds.amazonaws.com"
PORT = "3306"
USER = "admin"
REGION = "us-east-1"
DBNAME = "db_geo"
MDP = "123456789Gh!"
os.environ['LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN'] = '1'


class RDS:
    def __init__(self):
        self.rds = pymysql.connect(
            host=ENDPOINT,
            user=USER,
            passwd=MDP,
            port=3306, database=DBNAME
        )

    def get_data(self):
        try:
            cur = self.rds.cursor()
            cur.execute("""SELECT DISTINCT nom_station, latitude_station, logitude_station FROM stations WHERE commune_station like '%Paris%'""")
            query_results = cur.fetchall()
            return list(query_results)
        except Exception as e:
            print("Database connection failed due to {}".format(e))
