import logging

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

import config

logger = logging.getLogger(__name__)

def get_connection() -> psycopg2.extensions.connection:
    """Open a Postgres connection with pgvector type support registered."""
    conn = psycopg2.connect(config.database_url)
    register_vector(conn)
    return conn

def insert_job_embedding(job_id: str, source_text: str, embedding: np.ndarray, model_name: str) -> None:
    """Store one job-requirement embedding."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO job_embeddings (source_text, embedding, model_name, job_id) VALUES (%s, %s, %s, %s)",
            (source_text, vector, model_name, job_id),
        )
    conn.commit()
    conn.close()

def get_job_embedding(id: str) -> tuple[str, np.ndarray]:
    """Fetch the content and embedding for one row by id."""
    try:
        with conn.cursor() as cur:
        cur.execute(
            "SELECT %s FROM job_embeddings",
            id,
        )
        return (cur.fetchone().source_text, cur.fetchone().embedding)
    finally:
        conn.close()
