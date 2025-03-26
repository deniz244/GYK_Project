import pandas as pd
from sqlalchemy import create_engine
import psycopg2

engine = create_engine("postgresql://postgres:12345@localhost/Gyk1Northwind")

#query = """ SELECT * FROM orders """



query = """
SELECT 
    o.order_id,
    od.product_id,
    
FROM orders o

JOIN  order_details od ON od.order_id = o.order_id 
"""

df = pd.read_sql(query, engine)

print(df.head())