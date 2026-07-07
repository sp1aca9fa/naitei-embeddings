ALTER TABLE job_embeddings ADD COLUMN content_hash text NOT NULL;
ALTER TABLE cv_bullet_embeddings ADD COLUMN content_hash text NOT NULL;

CREATE UNIQUE INDEX ON job_embeddings (content_hash, model_name);
CREATE UNIQUE INDEX ON cv_bullet_embeddings (content_hash, model_name);
