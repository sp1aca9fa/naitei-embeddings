import logging
import hashlib
import uuid

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2 import sql

from src import config
from src.providers.base import EmbeddingProvider

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

def text_to_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def insert_embedding(table_name: str, ref_id: str, source_text: str, embedding: np.ndarray, model_name: str, content_hash: str) -> None:
    """
    Store an embedding to the specified table.
    ref_id is job_id on job_embeddings table, cv_version_id on cv_bullet_embeddings table and parent_bullet_id on cv_sentence_embeddings table
    """
    if table_name not in ID_COLUMNS:
        raise ValueError(f"Unknown table {table_name}")
    query = sql.SQL(
        "INSERT INTO {table} (source_text, embedding, model_name, content_hash, {id_col}) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING embedding"
    ).format(
        table=sql.Identifier(table_name),
        id_col=sql.Identifier(ID_COLUMNS[table_name])
    )
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                query, (source_text, embedding, model_name, content_hash, ref_id),
            )
            embedding = cur.fetchone()
        if embedding is not None:
            conn.commit()
            logger.info("inserted row to %s, hash %s", table_name, content_hash)
    finally:
        conn.close()
        logger.debug("connection closed")


def get_embedding(table_name: str, content_hash: str, model_name: str) -> np.ndarray | None:
    """Fetch the content and embedding for one row by text hash and model name."""
    if table_name not in ID_COLUMNS:
        raise ValueError(f"Unknown table {table_name}")
    query = sql.SQL(
        "SELECT embedding FROM {table} WHERE content_hash = %s AND model_name = %s"
    ).format(
        table=sql.Identifier(table_name)
    )
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (content_hash, model_name),
            )
            if row := cur.fetchone():
                logger.info("successfully fetched data")
                return row[0]
            logger.info("no data found on %s with hash %s", table_name, content_hash)
            return None
    finally:
        conn.close()
        logger.debug("connection closed")

def get_or_create_embedding(table_name: str, text: str, provider: EmbeddingProvider) -> np.ndarray:
    """Fetches the vector from the DB or inserts a new entry and returns the vector from the new entry."""
    if table_name not in ID_COLUMNS:
        raise ValueError(f"Unknown table {table_name}")
    text = text.lower().strip()
    content_hash = text_to_hash(text)
    existing = get_embedding(table_name, content_hash, provider.name)
    if existing is not None:
        return existing
    vec = provider.embed([text])
    insert_embedding(table_name, str(uuid.uuid5(uuid.NAMESPACE_DNS, content_hash)), text, vec[0], provider.name, content_hash)
    return vec[0]


def search_embedding(table_name: str, query_vec: np.ndarray, n: int) -> list[tuple]:
    """Search the database for a query vector and return a list of rows in descending order of text similarity"""
    if table_name not in ID_COLUMNS:
        raise ValueError(f"Unknown table {table_name}")
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
    conn = get_connection()
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


def delete_embedding_by_hash(table_name: str, content_hash: str, model_name: str) -> None:
    """Searches for hash and model in the specified table and deletes the row."""
    if table_name not in ID_COLUMNS:
        raise ValueError(f"Unknown table {table_name}")
    query = sql.SQL(
        "DELETE FROM {table} WHERE content_hash = %s AND model_name = %s"
    ).format(
        table=sql.Identifier(table_name)
    )
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (content_hash, model_name)
            )
            conn.commit()
            if cur.rowcount:
                logger.info("successfully deleted row")
    finally:
        conn.close()
        logger.debug("connection closed")
