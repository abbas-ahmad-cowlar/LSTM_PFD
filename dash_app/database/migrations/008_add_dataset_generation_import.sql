-- Migration: Add dataset_generations and dataset_imports tables
-- Description: Tables for tracking Phase 0 data generation and MAT file import jobs
-- Date: 2024-11-22

-- Create dataset_generations table
CREATE TABLE IF NOT EXISTS dataset_generations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',

    -- Generation parameters
    num_signals INTEGER,
    num_faults INTEGER,
    output_path VARCHAR(500),

    -- Progress tracking
    progress INTEGER DEFAULT 0,
    celery_task_id VARCHAR(255),

    -- Performance metrics
    duration_seconds FLOAT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    CONSTRAINT dataset_generations_name_idx UNIQUE (name)
);

CREATE INDEX IF NOT EXISTS idx_dataset_generations_status ON dataset_generations(status);
CREATE INDEX IF NOT EXISTS idx_dataset_generations_celery_task ON dataset_generations(celery_task_id);
CREATE INDEX IF NOT EXISTS idx_dataset_generations_created_at ON dataset_generations(created_at DESC);

-- Create dataset_imports table
CREATE TABLE IF NOT EXISTS dataset_imports (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',

    -- Import parameters
    num_files INTEGER,
    num_signals INTEGER,
    output_path VARCHAR(500),

    -- Progress tracking
    progress INTEGER DEFAULT 0,
    celery_task_id VARCHAR(255),

    -- Performance metrics
    duration_seconds FLOAT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    CONSTRAINT dataset_imports_name_idx UNIQUE (name)
);

CREATE INDEX IF NOT EXISTS idx_dataset_imports_status ON dataset_imports(status);
CREATE INDEX IF NOT EXISTS idx_dataset_imports_celery_task ON dataset_imports(celery_task_id);
CREATE INDEX IF NOT EXISTS idx_dataset_imports_created_at ON dataset_imports(created_at DESC);

-- Add trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_dataset_generations_updated_at BEFORE UPDATE ON dataset_generations
FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_dataset_imports_updated_at BEFORE UPDATE ON dataset_imports
FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
