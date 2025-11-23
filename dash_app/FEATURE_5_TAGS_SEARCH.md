# Feature #5: Experiment Tags & Search - Implementation Guide

## Overview

This document describes the implementation of the **Experiment Tags & Search** feature, which enables users to organize experiments with tags and perform advanced searches with filters.

## Status: Backend Complete ✅

### Completed Components

#### 1. Database Schema ✅
- **Location:** `database/migrations/007_add_tags_and_search.sql`
- **Tables Created:**
  - `tags` - Master list of all tags
  - `experiment_tags` - Many-to-many relationship between experiments and tags
  - `saved_searches` - User-saved search queries
- **Enhancements:**
  - Added `notes` column to `experiments` table (searchable text)
  - Added `search_vector` column for PostgreSQL full-text search
  - Created GIN index for fast full-text search
  - Auto-update trigger for search_vector

#### 2. Data Models ✅
- **Location:** `models/`
- **Models:**
  - `Tag` (`models/tag.py`) - Tag entity with name, slug, color, usage_count
  - `ExperimentTag` (`models/tag.py`) - Junction table for experiment-tag relationships
  - `SavedSearch` (`models/saved_search.py`) - Saved search queries
  - Enhanced `Experiment` model with `notes` and `search_vector` fields

#### 3. Service Layer ✅
- **TagService** (`services/tag_service.py`)
  - `create_or_get_tag()` - Create or retrieve existing tag
  - `add_tag_to_experiment()` - Add tag to experiment
  - `remove_tag_from_experiment()` - Remove tag from experiment
  - `get_popular_tags()` - Get most-used tags
  - `suggest_tags()` - Autocomplete suggestions
  - `bulk_add_tags()` / `bulk_remove_tags()` - Bulk operations
  - `get_tag_statistics()` - Usage analytics

- **SearchService** (`services/search_service.py`)
  - `search()` - Main search with advanced filtering
  - `_parse_query()` - Parse search syntax
  - `_build_sql_query()` - Build SQLAlchemy query
  - `_rank_results()` - Relevance ranking
  - `save_search()` / `get_saved_searches()` - Saved searches
  - `use_saved_search()` / `delete_saved_search()` - Manage saved searches

#### 4. REST API Endpoints ✅
- **Tags API** (`api/tags.py`)
  - `GET /api/tags/popular` - Get popular tags
  - `GET /api/tags/suggest?q=<query>` - Tag autocomplete
  - `POST /api/tags/create` - Create new tag
  - `GET /api/tags/experiment/<id>` - Get experiment tags
  - `POST /api/tags/experiment/<id>/add` - Add tag to experiment
  - `DELETE /api/tags/experiment/<id>/remove/<tag_id>` - Remove tag
  - `POST /api/tags/bulk/add` - Bulk add tags
  - `POST /api/tags/bulk/remove` - Bulk remove tags
  - `GET /api/tags/statistics` - Tag usage statistics

- **Search API** (`api/search.py`)
  - `GET /api/search/?q=<query>` - Search experiments
  - `GET /api/search/saved` - Get saved searches
  - `POST /api/search/saved` - Save search query
  - `GET /api/search/saved/<id>` - Execute saved search
  - `DELETE /api/search/saved/<id>` - Delete saved search
  - `PUT /api/search/saved/<id>/pin` - Pin/unpin saved search
  - `GET /api/search/help` - Search syntax help

#### 5. Configuration ✅
- **Location:** `config.py`, `.env.example`
- **Feature Flags:**
  - `FEATURE_TAGS_ENABLED` - Enable/disable tags
  - `FEATURE_SEARCH_ENABLED` - Enable/disable search
  - `FEATURE_SAVED_SEARCHES_ENABLED` - Enable/disable saved searches
  - `FEATURE_TAG_SUGGESTIONS_ENABLED` - Enable/disable autocomplete
- **Configuration Options:**
  - `SEARCH_ENGINE` - postgres/elasticsearch
  - `SEARCH_MAX_RESULTS` - Max search results
  - `TAGS_MAX_PER_EXPERIMENT` - Tag limit per experiment
  - `SEARCH_DEBOUNCE_MS` - Search debounce delay
  - `TAG_AUTOCOMPLETE_MIN_CHARS` - Min chars for autocomplete
  - `SEARCH_CACHE_TTL_SECONDS` - Cache TTL

## Search Query Syntax

### Tag Filters
```
tag:baseline               → Filter by tag
tag:baseline,production    → OR logic (any tag matches)
tag:baseline tag:production → AND logic (all tags must match)
```

### Accuracy Filters
```
accuracy:>0.95    → Greater than 95%
accuracy:<0.90    → Less than 90%
accuracy:=0.968   → Exactly 96.8%
accuracy:>=0.95   → Greater than or equal to 95%
```

### Date Filters
```
created:>2025-01-01    → After Jan 1, 2025
created:<2025-03-01    → Before Mar 1, 2025
```

### Other Filters
```
model:resnet       → Model type contains "resnet"
status:completed   → Experiment status (pending/running/completed/failed/cancelled)
```

### Keywords (Full-Text Search)
```
resnet learning_rate    → Search for "resnet" AND "learning_rate" in name/notes/model
```

### Combined Examples
```
tag:baseline accuracy:>0.95 resnet
→ Baseline experiments with >95% accuracy containing "resnet"

model:cnn status:completed created:>2025-01-01
→ Completed CNN experiments created after Jan 1, 2025

tag:baseline,production accuracy:>0.96
→ Baseline OR production experiments with >96% accuracy
```

## API Usage Examples

### Get Popular Tags
```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8050/api/tags/popular?limit=10
```

### Tag Autocomplete
```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8050/api/tags/suggest?q=base"
```

### Add Tag to Experiment
```bash
curl -X POST \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"tag_name":"baseline","color":"#3498db"}' \
  http://localhost:8050/api/tags/experiment/123/add
```

### Search Experiments
```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8050/api/search/?q=tag:baseline%20accuracy:>0.95"
```

### Save Search
```bash
curl -X POST \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"name":"Best baseline models","query":"tag:baseline accuracy:>0.95","is_pinned":true}' \
  http://localhost:8050/api/search/saved
```

## Database Migration

### Run Migration
```bash
cd dash_app
python database/run_migration.py
```

The migration will:
1. Create `tags`, `experiment_tags`, `saved_searches` tables
2. Add `notes` and `search_vector` columns to `experiments`
3. Create indexes for performance
4. Set up auto-update trigger for full-text search
5. Seed initial tags (baseline, production, research, etc.)

## Performance Considerations

### Indexing
- GIN index on `experiments.search_vector` for fast full-text search
- B-tree indexes on foreign keys and frequently queried columns
- Composite indexes on `experiment_tags(experiment_id, tag_id)`

### Caching (Redis)
- Search results cached for 60 seconds (configurable)
- Cache key format: `search:{user_id}:{query}:{limit}`
- Cache invalidation on experiment updates

### Query Optimization
- Uses PostgreSQL full-text search (ts_vector)
- Eager loading of relationships to avoid N+1 queries
- Query timeout: 5 seconds max
- Result limit: 100 default, 500 max

### Scalability
- Current: PostgreSQL full-text search (<100k experiments)
- Future: Elasticsearch for larger scale (>100k experiments)
- Feature flag `SEARCH_ENGINE` allows easy switching

## Pending Implementation (UI)

### Frontend Components (TODO)
1. **Enhanced Experiment History Page**
   - Advanced search bar with query builder
   - Tag filter pills (click to filter)
   - Saved searches dropdown
   - Search help modal

2. **Tag Management UI**
   - Tag badges on experiment cards
   - Tag input with autocomplete
   - Bulk tag operations
   - Tag cloud/analytics view

3. **Callbacks (Dash)**
   - Search query handling
   - Tag add/remove interactions
   - Saved search management
   - Real-time tag suggestions

## Testing

### Unit Tests (TODO)
```bash
# Test TagService
pytest tests/test_tag_service.py

# Test SearchService
pytest tests/test_search_service.py

# Test API endpoints
pytest tests/test_tags_api.py
pytest tests/test_search_api.py
```

### Integration Tests (TODO)
```bash
# Test full search flow
pytest tests/integration/test_search_flow.py

# Test tag operations
pytest tests/integration/test_tag_operations.py
```

### Load Testing
```bash
# Test search performance with 10,000 experiments
python scripts/benchmark_search.py
```

## Security

### Authentication
- All API endpoints require API key (`@require_api_key` decorator)
- Rate limiting via middleware

### Authorization
- Users can only search their own experiments
- Saved searches are user-scoped

### Input Validation
- Tag names: max 50 chars, sanitized
- Query strings: validated and escaped
- SQL injection prevention via SQLAlchemy ORM

## Future Enhancements

### Phase 2 (Optional)
1. **Elasticsearch Integration**
   - Fuzzy search
   - Advanced relevance scoring
   - Better performance at scale

2. **Advanced Features**
   - Tag hierarchies (parent/child tags)
   - Tag permissions (private/shared tags)
   - Smart search suggestions (ML-based)
   - Search analytics dashboard

3. **Collaboration**
   - Share saved searches with team
   - Tag-based access control
   - Collaborative tagging

## Troubleshooting

### Search Not Working
1. Check if feature flag `FEATURE_SEARCH_ENABLED` is true
2. Verify `search_vector` column exists and is populated
3. Check GIN index: `EXPLAIN ANALYZE SELECT * FROM experiments WHERE search_vector @@ to_tsquery('...')`

### Tags Not Appearing
1. Check if feature flag `FEATURE_TAGS_ENABLED` is true
2. Verify relationships: `SELECT * FROM experiment_tags WHERE experiment_id = X`
3. Check tag usage count: `SELECT * FROM tags ORDER BY usage_count DESC`

### Slow Search Performance
1. Check if GIN index is being used: `EXPLAIN ANALYZE`
2. Verify Redis caching is enabled
3. Reduce `SEARCH_MAX_RESULTS` limit
4. Consider pagination for large result sets

## Support

For questions or issues:
- GitHub Issues: https://github.com/your-org/LSTM_PFD/issues
- Documentation: See `feature_5.md` for full specification
- API Reference: `GET /api/search/help`

## Contributors

- Backend Implementation: Syed Abbas Ahmad
- Feature Specification: Feature #5 Document
- Database Design: Based on industry best practices

---

**Last Updated:** 2025-11-21
**Version:** 1.0.0
**Status:** Backend Complete, UI Pending
