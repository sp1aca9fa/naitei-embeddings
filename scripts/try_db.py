import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers import get_provider
from src.db import get_connection, get_or_create_embedding

import numpy as np
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


provider = get_provider()
sentence = "Experienced web developer"

get_or_create_embedding("job_embeddings", sentence, provider)
print(check_row_count("job_embeddings"))

get_or_create_embedding("job_embeddings", sentence, provider)
print(check_row_count("job_embeddings"))

provider_mock = get_provider("mock")
get_or_create_embedding("job_embeddings", sentence, provider_mock)
print(check_row_count("job_embeddings"))
