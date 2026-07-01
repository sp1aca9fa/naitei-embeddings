import logging

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2 import sql

from src import config

logger = logging.getLogger(__name__)

ID_COLUMNS = {
    "job_embeddings": "job_id",
    "cv_bullet_embeddings": "cv_version_id",
    "cv_sentence_embeddings": "parent_bullet_id",
}

def get_connection() -> psycopg2.extensions.connection:
    """Open a Postgres connection with pgvector type support registered."""
    conn = psycopg2.connect(config.database_url)
    register_vector(conn)
    logger.debug("connected to database")
    return conn


def insert_embedding(table_name: str, ref_id: str, source_text: str, embedding: np.ndarray, model_name: str) -> str:
    """
    Store an embedding to the specified table.
    ref_id is job_id on job_embeddings table, cv_version_id on cv_bullet_embeddings table and parent_bullet_id on cv_sentence_embeddings table
    """
    if table_name not in ID_COLUMNS:
        raise ValueError(f"Unknown table {table_name}")
    conn = get_connection()
    query = sql.SQL(
        "INSERT INTO {table} (source_text, embedding, model_name, {id_col}) "
        "VALUES (%s, %s, %s, %s) RETURNING id"
    ).format(
        table=sql.Identifier(table_name),
        id_col=sql.Identifier(ID_COLUMNS[table_name])
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                query, (source_text, embedding, model_name, ref_id),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        logger.info("inserted row for %s %s", table_name, ref_id)
        return new_id
    finally:
        conn.close()
        logger.debug("connection closed")


def get_embedding(table_name: str, row_id: str) -> tuple[str, np.ndarray] | None:
    """Fetch the content and embedding for one row by id."""
    if table_name not in ID_COLUMNS:
        raise ValueError(f"Unknown table {table_name}")
    conn = get_connection()
    query = sql.SQL(
        "SELECT source_text, embedding FROM {table} WHERE id = %s"
    ).format(
        table=sql.Identifier(table_name)
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (row_id,),
            )
            if row := cur.fetchone():
                logger.info("successfully fetched data")
                return row[0], row[1]
            logger.info("no data found on %s with id %s", table_name, row_id)
            return None
    finally:
        conn.close()
        logger.debug("connection closed")


def search_embedding(table_name: str, query_vec: np.ndarray, n: int) -> list[tuple]:
    """Search the database for a query vector and return a list of rows in descending order of text similarity"""
    if table_name not in ID_COLUMNS:
        raise ValueError(f"Unknown table {table_name}")
    conn = get_connection()
    query = sql.SQL(
        """
        SELECT source_text, model_name, 1 - (embedding <=> %s) AS similarity
        FROM {table}
        ORDER BY embedding <=> %s
        LIMIT %s
        """
    ).format(
        table=sql.Identifier(table_name)
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (query_vec, query_vec, n)
            )
            results = cur.fetchall()
            return results
    finally:
        conn.close()
        logger.debug("connection closed")
