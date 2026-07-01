CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE job_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid NOT NULL,
    source_text text NOT NULL,
    embedding vector(768) NOT NULL,
    model_name varchar(255) NOT NULL,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE cv_bullet_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cv_version_id uuid NOT NULL,
    source_text text NOT NULL,
    embedding vector(768) NOT NULL,
    model_name varchar(255) NOT NULL,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE cv_sentence_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_bullet_id uuid NOT NULL REFERENCES cv_bullet_embeddings(id),
    source_text text NOT NULL,
    embedding vector(768) NOT NULL,
    model_name varchar(255) NOT NULL,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX ON job_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON cv_bullet_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON cv_sentence_embeddings USING hnsw (embedding vector_cosine_ops);
