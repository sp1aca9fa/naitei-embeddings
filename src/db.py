import logging

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from src import config

logger = logging.getLogger(__name__)

def get_connection() -> psycopg2.extensions.connection:
    """Open a Postgres connection with pgvector type support registered."""
    conn = psycopg2.connect(config.database_url)
    register_vector(conn)
    logger.debug("connected to database")
    return conn

def insert_job_embedding(job_id: str, source_text: str, embedding: np.ndarray, model_name: str) -> str:
    """Store one job-requirement embedding."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO job_embeddings (source_text, embedding, model_name, job_id) VALUES (%s, %s, %s, %s) RETURNING id",
                (source_text, embedding, model_name, job_id),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        logger.info("inserted row for job %s", job_id)
        return new_id
    finally:
        conn.close()
        logger.debug("connection closed")


def get_job_embedding(id: str) -> tuple[str, np.ndarray] | None:
    """Fetch the content and embedding for one row by id."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_text, embedding FROM job_embeddings WHERE id = %s",
                (id,),
            )
            if row := cur.fetchone():
                logger.info("successfully fetched job")
                return row[0], row[1]
            logger.info("no job found with id %s", id)
            return None
    finally:
        conn.close()
        logger.debug("connection closed")
