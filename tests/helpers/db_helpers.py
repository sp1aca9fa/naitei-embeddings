from src.db import get_connection
from psycopg2 import sql

def check_row_count(table_name: str) -> int:
    conn = get_connection()
    query = sql.SQL(
        "SELECT count(*) FROM {table}"
    ).format(
        table=sql.Identifier(table_name)
    )
    try:
        with conn.cursor() as cur:
            cur.execute(query,)
            return cur.fetchone()[0]
    finally:
        conn.close()
