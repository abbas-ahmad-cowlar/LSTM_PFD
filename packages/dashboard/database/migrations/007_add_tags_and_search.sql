-- Migration 007: Add Tags and Search Functionality
-- Feature #5: Experiment Tags & Search

-- =============================================
-- PART 1: TAGS SYSTEM
-- =============================================

-- Table 1: Tags (master list of all tags)
CREATE TABLE IF NOT EXISTS tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,          -- Lowercase, normalized
    slug VARCHAR(50) NOT NULL UNIQUE,          -- URL-safe version
    color VARCHAR(7),                          -- Hex color for UI (e.g., "#3498db")

    -- Metadata
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    usage_count INTEGER DEFAULT 0,             -- How many experiments use this tag

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_tags_name ON tags(name);
CREATE INDEX idx_tags_usage ON tags(usage_count DESC);  -- For "popular tags"
CREATE INDEX idx_tags_slug ON tags(slug);

-- Table 2: Experiment-Tag relationship (many-to-many)
CREATE TABLE IF NOT EXISTS experiment_tags (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,

    -- Who added this tag? (for audit trail)
    added_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,

    -- Prevent duplicate tags on same experiment
    UNIQUE(experiment_id, tag_id)
);

CREATE INDEX idx_exp_tags_experiment ON experiment_tags(experiment_id);
CREATE INDEX idx_exp_tags_tag ON experiment_tags(tag_id);
CREATE INDEX idx_exp_tags_added_by ON experiment_tags(added_by);

-- Table 3: Saved searches (bookmarked queries)
CREATE TABLE IF NOT EXISTS saved_searches (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    name VARCHAR(200) NOT NULL,                -- User-provided name
    query TEXT NOT NULL,                       -- The search query

    -- Metadata
    is_pinned BOOLEAN DEFAULT FALSE,           -- Show at top of saved searches
    usage_count INTEGER DEFAULT 0,             -- Track how often used

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    last_used_at TIMESTAMP,

    -- User can't have duplicate saved search names
    UNIQUE(user_id, name)
);

CREATE INDEX idx_saved_searches_user ON saved_searches(user_id);
CREATE INDEX idx_saved_searches_pinned ON saved_searches(user_id, is_pinned) WHERE is_pinned = TRUE;

-- =============================================
-- PART 2: FULL-TEXT SEARCH
-- =============================================

-- Add search_vector column to experiments table for full-text search
ALTER TABLE experiments
    ADD COLUMN IF NOT EXISTS search_vector tsvector;

-- Add notes column if it doesn't exist (for searchable text beyond name)
ALTER TABLE experiments
    ADD COLUMN IF NOT EXISTS notes TEXT;

-- Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_experiments_search
    ON experiments USING GIN(search_vector);

-- Create trigger function to auto-update search_vector
CREATE OR REPLACE FUNCTION experiments_search_vector_update()
RETURNS TRIGGER AS $$
BEGIN
    -- Weight: A (highest) for name, B for notes, C for model_type
    NEW.search_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.notes, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.model_type, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to call the function
DROP TRIGGER IF EXISTS experiments_search_vector_trigger ON experiments;
CREATE TRIGGER experiments_search_vector_trigger
    BEFORE INSERT OR UPDATE ON experiments
    FOR EACH ROW
    EXECUTE FUNCTION experiments_search_vector_update();

-- Update existing experiments to populate search_vector
UPDATE experiments
SET search_vector =
    setweight(to_tsvector('english', COALESCE(name, '')), 'A') ||
    setweight(to_tsvector('english', COALESCE(notes, '')), 'B') ||
    setweight(to_tsvector('english', COALESCE(model_type, '')), 'C')
WHERE search_vector IS NULL;

-- =============================================
-- PART 3: SEED DATA (Initial Tags)
-- =============================================

-- Insert common tags
INSERT INTO tags (name, slug, color, usage_count) VALUES
    ('baseline', 'baseline', '#3498db', 0),
    ('production', 'production', '#27ae60', 0),
    ('research', 'research', '#9b59b6', 0),
    ('high-accuracy', 'high-accuracy', '#e74c3c', 0),
    ('fast-training', 'fast-training', '#f39c12', 0),
    ('experimental', 'experimental', '#95a5a6', 0),
    ('archived', 'archived', '#7f8c8d', 0),
    ('review-needed', 'review-needed', '#e67e22', 0),
    ('bug-fix', 'bug-fix', '#c0392b', 0),
    ('optimization', 'optimization', '#16a085', 0)
ON CONFLICT (name) DO NOTHING;

-- =============================================
-- PART 4: HELPER VIEWS
-- =============================================

-- View: Experiments with their tags
CREATE OR REPLACE VIEW experiments_with_tags AS
SELECT
    e.id,
    e.name,
    e.model_type,
    e.status,
    e.created_at,
    COALESCE(
        ARRAY_AGG(
            json_build_object(
                'id', t.id,
                'name', t.name,
                'slug', t.slug,
                'color', t.color
            )
        ) FILTER (WHERE t.id IS NOT NULL),
        ARRAY[]::json[]
    ) AS tags
FROM experiments e
LEFT JOIN experiment_tags et ON e.id = et.experiment_id
LEFT JOIN tags t ON et.tag_id = t.id
GROUP BY e.id, e.name, e.model_type, e.status, e.created_at;

COMMENT ON TABLE tags IS 'Master list of experiment tags';
COMMENT ON TABLE experiment_tags IS 'Many-to-many relationship between experiments and tags';
COMMENT ON TABLE saved_searches IS 'User-saved search queries for quick access';
COMMENT ON COLUMN experiments.search_vector IS 'Full-text search index (auto-updated)';
COMMENT ON COLUMN experiments.notes IS 'User notes for the experiment (searchable)';
