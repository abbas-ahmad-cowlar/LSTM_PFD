# FEATURE #5: EXPERIMENT TAGS & SEARCH

**Duration:** 1-2 weeks (7-10 days)  
**Priority:** P1 (Medium-High - Organizational requirement)  
**Assigned To:** Full-Stack Developer

---

## 5.1 OBJECTIVES

### Primary Objective
Implement a comprehensive tagging and search system that enables users to organize, categorize, and quickly find experiments among hundreds or thousands of historical runs, improving experiment management and reducing time spent looking for past work.

### Success Criteria
- Users can add/remove tags on experiments (UI + API)
- Full-text search across experiment names, tags, and notes
- Search returns results in <500ms for 10,000+ experiments
- Tag autocomplete suggests existing tags (prevents duplicates)
- Filter experiments by multiple tags (AND/OR logic)
- Search supports advanced syntax: `tag:baseline accuracy:>0.95`
- Saved searches (bookmarked queries)
- Tag cloud/analytics showing most-used tags
- Feature can be disabled globally via config flag
- Mobile-responsive search interface

### Business Value
- **Time Savings:** Find "that ResNet from 2 months ago" in 5 seconds instead of 10 minutes
- **Organization:** Group experiments logically (baseline, production, research, customer-demo)
- **Collaboration:** Tag experiments for team review (#review-needed, #discuss-with-john)
- **Reproducibility:** Tag experiments with business context (#q4-report, #customer-acme)
- **Knowledge Transfer:** New team members can search by tag to learn

---

## 5.2 ARCHITECTURAL DESIGN (MODULAR)

### Feature Toggle System

```yaml
# config/features.yaml

# GLOBAL FEATURE FLAGS
FEATURE_TAGS_ENABLED: true                    # ← Master toggle for tags
FEATURE_SEARCH_ENABLED: true                  # ← Master toggle for search
FEATURE_SAVED_SEARCHES_ENABLED: true          # ← Saved searches (optional)
FEATURE_TAG_SUGGESTIONS_ENABLED: true         # ← Autocomplete (optional)

# SEARCH ENGINE CONFIGURATION
SEARCH_ENGINE: 'postgres'                     # Options: 'postgres', 'elasticsearch'
SEARCH_POSTGRES_FULL_TEXT: true               # Use PostgreSQL full-text search
SEARCH_ELASTICSEARCH_ENABLED: false           # Use Elasticsearch (future upgrade)
SEARCH_MAX_RESULTS: 100                       # Limit search results

# TAG SYSTEM CONFIGURATION
TAGS_MAX_PER_EXPERIMENT: 10                   # Limit tags per experiment
TAGS_MAX_LENGTH: 50                           # Max characters per tag
TAGS_CASE_SENSITIVE: false                    # "Baseline" = "baseline"
TAGS_ALLOW_SPACES: false                      # "my tag" → "my-tag" (slugify)
TAGS_RESERVED_WORDS: ['all', 'none', 'system']  # Prevent reserved tags

# PERFORMANCE TUNING
SEARCH_DEBOUNCE_MS: 300                       # Delay before search (reduce load)
TAG_AUTOCOMPLETE_MIN_CHARS: 2                 # Start suggesting after 2 chars
TAG_AUTOCOMPLETE_MAX_RESULTS: 10              # Limit autocomplete results
SEARCH_CACHE_TTL_SECONDS: 60                  # Cache search results (Redis)
```

### System Architecture

```
┌──────────────────────────────────────────────────────────┐
│  USER INTERFACE (Experiment History Page)               │
│  ┌────────────────────────────────────────────────────┐ │
│  │  [🔍 Search: __________________________] [Filter]  │ │
│  │  Popular tags: [baseline] [production] [research]  │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────────┬──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────┐
│  SEARCH SERVICE (services/search_service.py)            │
│  ┌────────────────────────────────────────────────────┐ │
│  │  1. Parse query (extract filters, keywords)        │ │
│  │  2. Check cache (Redis) - return if hit            │ │
│  │  3. Route to search engine (Postgres/Elasticsearch)│ │
│  │  4. Apply filters (tags, date range, accuracy)     │ │
│  │  5. Rank results (relevance score)                 │ │
│  │  6. Cache results (TTL: 60 seconds)                │ │
│  │  7. Return to UI                                    │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────────┬──────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌─────────────────┐    ┌────────────────────┐
│  PostgreSQL     │    │  Elasticsearch     │
│  (Default)      │    │  (Optional upgrade)│
│  ┌───────────┐  │    │  ┌──────────────┐  │
│  │ GIN Index │  │    │  │ Inverted     │  │
│  │ (Full-text│  │    │  │ Index        │  │
│  │  search)  │  │    │  │ (Fuzzy match)│  │
│  └───────────┘  │    │  └──────────────┘  │
└─────────────────┘    └────────────────────┘

MODULAR DESIGN:
- Default: PostgreSQL (no additional infrastructure)
- Upgrade path: Elasticsearch (for large scale, fuzzy search)
- Feature flags control which engine is used
- Easy to swap engines without code changes
```

### Search Query Parser Design

```
Query Syntax Examples:

1. Simple keyword search:
   "resnet accuracy"
   → Search name/notes for "resnet" AND "accuracy"

2. Tag filter:
   "tag:baseline tag:production"
   → Show experiments with BOTH tags

3. Accuracy filter:
   "accuracy:>0.95"
   → Show experiments with accuracy > 95%

4. Date range:
   "created:>2025-01-01 created:<2025-03-01"
   → Experiments created in Jan-Feb 2025

5. Model type filter:
   "model:resnet"
   → Show only ResNet experiments

6. Combined:
   "tag:baseline model:resnet accuracy:>0.96 convergence"
   → Baseline ResNet experiments >96% accuracy with "convergence" in notes

7. OR logic (tags):
   "tag:baseline,production"
   → Experiments with baseline OR production tag

Parser Architecture:
  Input: "tag:baseline accuracy:>0.95 resnet"
  ↓
  Tokenizer: ["tag:baseline", "accuracy:>0.95", "resnet"]
  ↓
  Parser: {
    filters: {
      tags: ["baseline"],
      accuracy: {operator: ">", value: 0.95}
    },
    keywords: ["resnet"]
  }
  ↓
  SQL Builder: SELECT * FROM experiments WHERE ...
```

---

## 5.3 DATABASE SCHEMA

### Tag Storage (Many-to-Many)

```sql
-- Table 1: Tags (master list of all tags)
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,  -- Lowercase, normalized
    slug VARCHAR(50) NOT NULL UNIQUE,  -- URL-safe version (e.g., "my-tag")
    color VARCHAR(7),  -- Hex color for UI (e.g., "#3498db")
    
    -- Metadata
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    usage_count INTEGER DEFAULT 0,  -- How many experiments use this tag
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_tags_name ON tags(name);
CREATE INDEX idx_tags_usage ON tags(usage_count DESC);  -- For "popular tags"

-- Table 2: Experiment-Tag relationship (many-to-many)
CREATE TABLE experiment_tags (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    
    -- Who added this tag? (for audit trail)
    added_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    added_at TIMESTAMP DEFAULT NOW(),
    
    -- Prevent duplicate tags on same experiment
    UNIQUE(experiment_id, tag_id)
);

CREATE INDEX idx_exp_tags_experiment ON experiment_tags(experiment_id);
CREATE INDEX idx_exp_tags_tag ON experiment_tags(tag_id);

-- Table 3: Saved searches (bookmarked queries)
CREATE TABLE saved_searches (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    name VARCHAR(200) NOT NULL,  -- User-provided name (e.g., "Best baseline models")
    query TEXT NOT NULL,  -- The search query (e.g., "tag:baseline accuracy:>0.95")
    
    -- Metadata
    is_pinned BOOLEAN DEFAULT FALSE,  -- Show at top of saved searches
    usage_count INTEGER DEFAULT 0,  -- Track how often used
    
    created_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP,
    
    UNIQUE(user_id, name)  -- User can't have duplicate saved search names
);

CREATE INDEX idx_saved_searches_user ON saved_searches(user_id);
CREATE INDEX idx_saved_searches_pinned ON saved_searches(is_pinned) WHERE is_pinned = TRUE;

-- Enhance experiments table for full-text search
ALTER TABLE experiments 
    ADD COLUMN search_vector tsvector;  -- PostgreSQL full-text search column

-- Create GIN index for fast full-text search
CREATE INDEX idx_experiments_search ON experiments USING GIN(search_vector);

-- Trigger to auto-update search_vector when experiment changes
CREATE OR REPLACE FUNCTION experiments_search_vector_update() 
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.notes, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.model_type, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER experiments_search_vector_trigger
    BEFORE INSERT OR UPDATE ON experiments
    FOR EACH ROW
    EXECUTE FUNCTION experiments_search_vector_update();
```

### Example Data

```sql
-- Insert tags
INSERT INTO tags (name, slug, color, created_by, usage_count) VALUES
    ('baseline', 'baseline', '#3498db', 1, 45),
    ('production', 'production', '#27ae60', 1, 23),
    ('research', 'research', '#9b59b6', 1, 67),
    ('high-accuracy', 'high-accuracy', '#e74c3c', 1, 12),
    ('fast-training', 'fast-training', '#f39c12', 1, 8),
    ('customer-demo', 'customer-demo', '#1abc9c', 1, 5);

-- Tag experiments
INSERT INTO experiment_tags (experiment_id, tag_id, added_by) VALUES
    (1234, 1, 1),  -- Experiment #1234 tagged "baseline"
    (1234, 4, 1),  -- Experiment #1234 tagged "high-accuracy"
    (1567, 2, 1),  -- Experiment #1567 tagged "production"
    (1890, 3, 1);  -- Experiment #1890 tagged "research"

-- Saved search
INSERT INTO saved_searches (user_id, name, query, is_pinned) VALUES
    (1, 'Best baseline models', 'tag:baseline accuracy:>0.95', true),
    (1, 'Failed experiments last week', 'status:failed created:>2025-06-08', false);
```

---

## 5.4 SEARCH SERVICE IMPLEMENTATION

### Service Architecture

```python
# services/search_service.py

from typing import List, Dict, Optional
from sqlalchemy import or_, and_
from models.experiment import Experiment
from models.tag import Tag, ExperimentTag
from database.connection import get_db_session
import re

class SearchService:
    """
    Centralized search service for experiments.
    
    Features:
    - Full-text search (name, notes, model type)
    - Tag filtering (AND/OR logic)
    - Advanced filters (accuracy, date, status)
    - Result ranking by relevance
    - Caching (Redis)
    """
    
    @staticmethod
    def search(query: str, user_id: int, limit: int = 100) -> Dict:
        """
        Main search entry point.
        
        Args:
            query: Search query (e.g., "tag:baseline accuracy:>0.95 resnet")
            user_id: User performing search (for authorization)
            limit: Max results to return
            
        Returns:
            {
                'results': [list of experiments],
                'total': int (total matches),
                'query_info': {parsed query details},
                'suggestions': [alternative queries]
            }
        """
        
        # 1. Check feature flag
        if not Config.FEATURE_SEARCH_ENABLED:
            return {'results': [], 'total': 0, 'error': 'Search is disabled'}
        
        # 2. Check cache (Redis)
        cache_key = f"search:{user_id}:{query}:{limit}"
        cached_results = redis_client.get(cache_key)
        if cached_results:
            return json.loads(cached_results)
        
        # 3. Parse query
        parsed_query = SearchService._parse_query(query)
        
        # 4. Build SQL query
        sql_query = SearchService._build_sql_query(parsed_query, user_id)
        
        # 5. Execute search
        session = get_db_session()
        results = session.query(Experiment).filter(sql_query).limit(limit).all()
        
        # 6. Rank results by relevance
        ranked_results = SearchService._rank_results(results, parsed_query)
        
        # 7. Build response
        response = {
            'results': [SearchService._serialize_experiment(exp) for exp in ranked_results],
            'total': len(ranked_results),
            'query_info': parsed_query,
            'suggestions': SearchService._get_suggestions(parsed_query)
        }
        
        # 8. Cache results (TTL: 60 seconds)
        redis_client.setex(cache_key, Config.SEARCH_CACHE_TTL_SECONDS, json.dumps(response))
        
        return response
    
    @staticmethod
    def _parse_query(query: str) -> Dict:
        """
        Parse search query into structured format.
        
        Syntax:
            tag:baseline               → Filter by tag
            tag:baseline,production    → Filter by tag (OR logic)
            accuracy:>0.95             → Accuracy filter
            created:>2025-01-01        → Date filter
            model:resnet               → Model type filter
            status:completed           → Status filter
            "exact phrase"             → Exact match (future)
            keyword1 keyword2          → Full-text search
        
        Returns:
            {
                'tags': ['baseline', 'production'],
                'tag_logic': 'OR',  # 'AND' or 'OR'
                'accuracy': {'operator': '>', 'value': 0.95},
                'created_after': '2025-01-01',
                'model_type': 'resnet',
                'status': 'completed',
                'keywords': ['keyword1', 'keyword2']
            }
        """
        
        parsed = {
            'tags': [],
            'tag_logic': 'AND',
            'accuracy': None,
            'created_after': None,
            'created_before': None,
            'model_type': None,
            'status': None,
            'keywords': []
        }
        
        # Tokenize query
        tokens = query.split()
        
        for token in tokens:
            # Tag filter: tag:baseline or tag:baseline,production
            if token.startswith('tag:'):
                tag_value = token[4:]  # Remove "tag:" prefix
                if ',' in tag_value:
                    # Multiple tags with OR logic
                    parsed['tags'].extend(tag_value.split(','))
                    parsed['tag_logic'] = 'OR'
                else:
                    parsed['tags'].append(tag_value)
            
            # Accuracy filter: accuracy:>0.95, accuracy:=0.968, accuracy:<0.90
            elif token.startswith('accuracy:'):
                accuracy_value = token[9:]  # Remove "accuracy:" prefix
                operator = re.match(r'([><=]+)', accuracy_value).group(1)
                value = float(accuracy_value.lstrip('><='))
                parsed['accuracy'] = {'operator': operator, 'value': value}
            
            # Date filter: created:>2025-01-01, created:<2025-03-01
            elif token.startswith('created:'):
                date_value = token[8:]
                operator = re.match(r'([><])', date_value).group(1)
                date = date_value.lstrip('><')
                if operator == '>':
                    parsed['created_after'] = date
                elif operator == '<':
                    parsed['created_before'] = date
            
            # Model type filter: model:resnet, model:transformer
            elif token.startswith('model:'):
                parsed['model_type'] = token[6:].lower()
            
            # Status filter: status:completed, status:failed
            elif token.startswith('status:'):
                parsed['status'] = token[7:].lower()
            
            # Keyword (full-text search)
            else:
                parsed['keywords'].append(token)
        
        return parsed
    
    @staticmethod
    def _build_sql_query(parsed_query: Dict, user_id: int):
        """
        Build SQLAlchemy query from parsed query.
        
        Returns:
            SQLAlchemy filter expression
        """
        
        filters = []
        
        # Authorization: User can only search own experiments
        filters.append(Experiment.user_id == user_id)
        
        # Tag filter
        if parsed_query['tags']:
            if parsed_query['tag_logic'] == 'AND':
                # All tags must be present (AND logic)
                for tag_name in parsed_query['tags']:
                    tag = db.session.query(Tag).filter_by(name=tag_name.lower()).first()
                    if tag:
                        filters.append(
                            Experiment.id.in_(
                                db.session.query(ExperimentTag.experiment_id)
                                .filter(ExperimentTag.tag_id == tag.id)
                            )
                        )
            else:
                # Any tag can be present (OR logic)
                tag_names = [t.lower() for t in parsed_query['tags']]
                tags = db.session.query(Tag).filter(Tag.name.in_(tag_names)).all()
                tag_ids = [t.id for t in tags]
                filters.append(
                    Experiment.id.in_(
                        db.session.query(ExperimentTag.experiment_id)
                        .filter(ExperimentTag.tag_id.in_(tag_ids))
                    )
                )
        
        # Accuracy filter
        if parsed_query['accuracy']:
            operator = parsed_query['accuracy']['operator']
            value = parsed_query['accuracy']['value']
            
            if operator == '>':
                filters.append(Experiment.accuracy > value)
            elif operator == '>=':
                filters.append(Experiment.accuracy >= value)
            elif operator == '<':
                filters.append(Experiment.accuracy < value)
            elif operator == '<=':
                filters.append(Experiment.accuracy <= value)
            elif operator == '=':
                filters.append(Experiment.accuracy == value)
        
        # Date filters
        if parsed_query['created_after']:
            filters.append(Experiment.created_at >= parsed_query['created_after'])
        if parsed_query['created_before']:
            filters.append(Experiment.created_at <= parsed_query['created_before'])
        
        # Model type filter
        if parsed_query['model_type']:
            filters.append(Experiment.model_type.ilike(f"%{parsed_query['model_type']}%"))
        
        # Status filter
        if parsed_query['status']:
            filters.append(Experiment.status == parsed_query['status'])
        
        # Keyword search (PostgreSQL full-text search)
        if parsed_query['keywords']:
            keyword_query = ' & '.join(parsed_query['keywords'])  # AND logic for keywords
            filters.append(
                Experiment.search_vector.match(keyword_query)
            )
        
        # Combine all filters with AND
        return and_(*filters)
    
    @staticmethod
    def _rank_results(results: List[Experiment], parsed_query: Dict) -> List[Experiment]:
        """
        Rank results by relevance.
        
        Ranking factors:
        1. Exact name match (highest priority)
        2. Multiple keyword matches
        3. Recent experiments (created in last 30 days)
        4. Higher accuracy
        """
        
        scored_results = []
        
        for exp in results:
            score = 0
            
            # Exact name match
            if parsed_query['keywords']:
                for keyword in parsed_query['keywords']:
                    if keyword.lower() in exp.name.lower():
                        score += 10
            
            # Recent (created in last 30 days)
            if exp.created_at > datetime.now() - timedelta(days=30):
                score += 5
            
            # High accuracy (bonus for >95%)
            if exp.accuracy and exp.accuracy > 0.95:
                score += 3
            
            scored_results.append((exp, score))
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [exp for exp, score in scored_results]
    
    @staticmethod
    def _get_suggestions(parsed_query: Dict) -> List[str]:
        """
        Generate search suggestions based on query.
        
        Example: User searches "tag:basline" (typo) → Suggest "tag:baseline"
        """
        
        suggestions = []
        
        # Suggest correcting tag typos (fuzzy match)
        if parsed_query['tags']:
            for tag in parsed_query['tags']:
                similar_tags = SearchService._find_similar_tags(tag)
                if similar_tags:
                    suggestions.append(f"Did you mean: tag:{similar_tags[0]}?")
        
        # Suggest adding accuracy filter if not present
        if not parsed_query['accuracy'] and len(parsed_query['keywords']) > 0:
            suggestions.append("Try adding: accuracy:>0.95")
        
        return suggestions[:3]  # Max 3 suggestions
```

---

## 5.5 IMPLEMENTATION PLAN (DAY-BY-DAY)

### Day 1-2: Database Schema & Models

**Day 1 Morning: Database Schema**
- Write migrations for `tags`, `experiment_tags`, `saved_searches` tables
- Write migration to add `search_vector` column to `experiments`
- Create trigger for auto-updating `search_vector`
- Run migrations on dev database
- Test full-text search (INSERT experiment, verify search_vector populated)

**Day 1 Afternoon: SQLAlchemy Models**
- Create `models/tag.py` (Tag, ExperimentTag models)
- Create `models/saved_search.py`
- Update `models/experiment.py` (add tags relationship)
- Write unit tests for models (relationships work correctly)

**Day 2 Morning: Tag Service**
- Create `services/tag_service.py`
- Implement `create_tag()` (create or return existing)
- Implement `add_tag_to_experiment()` (many-to-many insert)
- Implement `remove_tag_from_experiment()`
- Implement `get_popular_tags()` (sorted by usage_count)
- Implement `suggest_tags()` (autocomplete, fuzzy match)

**Day 2 Afternoon: Search Service (Core)**
- Create `services/search_service.py`
- Implement `_parse_query()` (tokenize and parse filters)
- Write unit tests for parser (test all query syntaxes)
- Implement `_build_sql_query()` (convert parsed query to SQLAlchemy)
- Test SQL generation (ensure queries are valid)

**Testing Criteria:**
- ✅ Migrations run without errors
- ✅ Tag can be created and added to experiment
- ✅ Full-text search finds experiments by keyword
- ✅ Query parser correctly extracts filters
- ✅ SQL query builder generates valid SQL

**Deliverable:** Database schema ready, core services implemented.

---

### Day 3-4: Search Implementation & Optimization

**Day 3 Morning: Search Execution**
- Implement `search()` method (main entry point)
- Implement `_rank_results()` (relevance scoring)
- Implement `_get_suggestions()` (typo correction, suggestions)
- Test search with various queries

**Day 3 Afternoon: Caching Layer**
- Add Redis caching to `search()` method
- Set TTL: 60 seconds (configurable via `SEARCH_CACHE_TTL_SECONDS`)
- Implement cache invalidation (when experiment updated/deleted)
- Test cache hit/miss rates

**Day 4 Morning: Performance Optimization**
- Add EXPLAIN ANALYZE to search queries (identify slow queries)
- Optimize indexes (ensure GIN index used for full-text)
- Add query timeout (5 seconds max)
- Load test: Search with 10,000 experiments (measure latency)

**Day 4 Afternoon: Feature Flags**
- Add global feature toggles (`FEATURE_SEARCH_ENABLED`, `FEATURE_TAGS_ENABLED`)
- Test disabling features (ensure graceful degradation)
- Document configuration options

**Testing Criteria:**
- ✅ Search returns results in <500ms (10,000 experiments)
- ✅ Cache hit: <10ms response time
- ✅ Cache invalidated when experiment changes
- ✅ Feature flags work (search disabled → returns empty results)
- ✅ Query timeout prevents long-running queries

**Deliverable:** Performant search service with caching.

---

### Day 5-6: UI Implementation

**Day 5 Morning: Search Bar Component**
- Enhance `layouts/experiment_history.py`
- Add search bar at top of page
- Implement debounced search (300ms delay before query)
- Display search results in table (replace experiment list)
- Show "No results" message if no matches

**Search Bar Design:**

```
Experiment History Page

┌────────────────────────────────────────────────────────┐
│  🔍 [Search experiments...___________________________] │
│  [🏷️ Tags] [📅 Date] [🎯 Accuracy] [💾 Saved Searches]│
│                                                        │
│  Popular tags: [baseline] [production] [research]     │
│  (Click tag to filter)                                 │
└────────────────────────────────────────────────────────┘

Results (42 found):
┌──────────┬─────────────────┬──────────┬──────────┬────────┐
│   Date   │      Name       │ Accuracy │  Status  │  Tags  │
├──────────┼─────────────────┼──────────┼──────────┼────────┤
│ 06/15    │ ResNet_Baseline │  96.8%   │Complete✅│[baseline]│
│          │                 │          │          │[high-acc]│
├──────────┼─────────────────┼──────────┼──────────┼────────┤
│ 06/12    │ Transformer_v2  │  97.1%   │Complete✅│[research]│
│          │                 │          │          │          │
└──────────┴─────────────────┴──────────┴──────────┴────────┘
```

**Day 5 Afternoon: Tag Autocomplete**
- Implement tag input with autocomplete
- Query `TagService.suggest_tags()` as user types
- Show dropdown with suggestions
- Allow creating new tags (if not in suggestions)

**Tag Input Design:**

```
Add tags to experiment:

┌──────────────────────────────────────────────┐
│  Tags: [baseline] [high-accuracy] [+ Add tag]│
│                                              │
│  Type to add tag: [resea_____________]       │
│  ┌──────────────────────────────────────┐   │
│  │ 📌 research (used 67 times)          │   │
│  │ 📌 researcher-johns-experiments      │   │
│  │ ➕ Create new tag: "resea"           │   │
│  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

**Day 6 Morning: Advanced Filters UI**
- Create filter sidebar (collapsible)
- Add date range picker
- Add accuracy slider (min/max)
- Add model type dropdown
- Add status checkboxes
- Wire filters to search query builder

**Filter Sidebar Design:**

```
┌─────────────────────────────┐
│  FILTERS                    │
├─────────────────────────────┤
│  📅 Date Range              │
│  From: [2025-01-01___]      │
│  To:   [2025-06-15___]      │
│                             │
│  🎯 Accuracy                │
│  [======●========] 95-100%  │
│                             │
│  🤖 Model Type              │
│  [☑] ResNet                 │
│  [☐] Transformer            │
│  [☐] PINN                   │
│  [☑] CNN                    │
│                             │
│  ✅ Status                  │
│  [☑] Completed              │
│  [☐] Failed                 │
│  [☐] Running                │
│                             │
│  [Apply Filters] [Clear]    │
└─────────────────────────────┘
```

**Day 6 Afternoon: Saved Searches UI**
- Add "Save Search" button (appears after search)
- Modal to name saved search
- Display list of saved searches (sidebar or dropdown)
- Click saved search → Executes that query
- Delete/edit saved searches

**Testing Criteria:**
- ✅ Type in search bar → Results update after 300ms
- ✅ Type 2 characters → Autocomplete suggestions appear
- ✅ Click popular tag → Filters by that tag
- ✅ Apply filters → Search query updated with filters
- ✅ Save search → Appears in saved searches list
- ✅ Click saved search → Executes query

**Deliverable:** Fully functional search UI.

---

### Day 7: Experiment Detail Page Integration

**Morning: Tag Management on Experiment Page**
- Add tag section to `layouts/experiment_results.py`
- Display current tags (with remove buttons)
- Add tag input (with autocomplete)
- Update tags in real-time (Ajax, no page reload)

**Design:**

```
Experiment Results Page

┌────────────────────────────────────────────────────┐
│  Experiment: ResNet34_Standard                     │
│  Status: Completed ✅  |  Accuracy: 96.8%          │
│                                                    │
│  Tags: [baseline 🗙] [high-accuracy 🗙]            │
│        [+ Add tag_________]                        │
│                                                    │
│  (Click 🗙 to remove tag, type to add new tag)    │
└────────────────────────────────────────────────────┘
```

**Afternoon: Bulk Tag Operations**
- Add bulk tag feature to experiment history
- Select multiple experiments (checkboxes)
- "Bulk Actions" dropdown → "Add tags", "Remove tags"
- Modal to add/remove tags from selected experiments

**Testing Criteria:**
- ✅ Add tag on experiment page → Tag appears immediately
- ✅ Remove tag → Tag disappears immediately
- ✅ Select 5 experiments → Bulk add "baseline" tag → All 5 updated
- ✅ Tag autocomplete works on experiment page

**Deliverable:** Tag management on experiment pages.

---

### Day 8-9: Tag Analytics & Saved Searches

**Day 8 Morning: Tag Cloud Page**
- Create `/tags` page (dedicated tag analytics)
- Display all tags as cloud (size = usage count)
- Click tag → Search experiments with that tag
- Display tag statistics (created by, usage trend)

**Tag Cloud Design:**

```
┌────────────────────────────────────────────────────┐
│  TAG CLOUD (78 tags)                               │
├────────────────────────────────────────────────────┤
│                                                    │
│    baseline (45)     production (23)              │
│                                                    │
│        research (67)                              │
│                                                    │
│  high-accuracy (12)  fast-training (8)            │
│                                                    │
│    customer-demo (5)        debug (3)             │
│                                                    │
│  (Font size proportional to usage count)          │
└────────────────────────────────────────────────────┘
```

**Day 8 Afternoon: Tag Management Admin Page**
- Create `/admin/tags` page (admin-only)
- List all tags with usage stats
- Rename tags (updates all experiments)
- Merge tags (combine two tags into one)
- Delete tags (removes from all experiments)

**Day 9 Morning: Saved Search Management**
- Implement saved search CRUD operations
- Pin/unpin saved searches (show pinned at top)
- Track usage count (most-used saved searches)
- Share saved searches with team (future: export query URL)

**Day 9 Afternoon: Search Analytics**
- Log all searches to database (anonymized)
- Track popular queries (what users search for)
- Create `/admin/search-analytics` page
- Display top queries, zero-result queries (for improvement)

**Testing Criteria:**
- ✅ Tag cloud displays correctly (sizes proportional)
- ✅ Click tag in cloud → Searches for that tag
- ✅ Admin can rename tag → All experiments updated
- ✅ Merge tags → Experiments get merged tag
- ✅ Saved search usage count increments
- ✅ Search analytics page shows popular queries

**Deliverable:** Tag analytics and advanced management.

---

### Day 10: Testing, Documentation & Polish

**Morning: End-to-End Testing**
- Test all search syntaxes (see test scenarios below)
- Test with large dataset (10,000+ experiments)
- Test mobile responsiveness (search bar, filters)
- Test edge cases (empty query, special characters, SQL injection)

**Afternoon: Documentation**
- Write user guide: "Searching & Organizing Experiments"
- Write admin guide: "Tag Management Best Practices"
- Create video tutorial (3 minutes: "Using Tags & Search")
- Document query syntax (cheat sheet)

**Polish:**
- Add keyboard shortcuts (Ctrl+K → Focus search bar)
- Add recent searches (local storage, last 5 queries)
- Add search tips (tooltip: "Try tag:baseline accuracy:>0.95")
- Add loading indicators (show "Searching..." during query)

**Testing Criteria:**
- ✅ All test scenarios pass
- ✅ Search works on mobile (UI responsive)
- ✅ Documentation complete and clear
- ✅ No SQL injection vulnerabilities (parameterized queries)

**Deliverable:** Production-ready search & tagging system.

---

## 5.6 QUERY SYNTAX EXAMPLES (USER DOCUMENTATION)

```markdown
# Search Query Syntax

## Basic Search
Type keywords to search experiment names, notes, and model types:
```
resnet baseline
```
Finds experiments with "resnet" AND "baseline" in name/notes.

## Tag Filters

### Single tag:
```
tag:baseline
```

### Multiple tags (AND logic):
```
tag:baseline tag:production
```
Experiments must have BOTH tags.

### Multiple tags (OR logic):
```
tag:baseline,production
```
Experiments can have EITHER tag.

## Accuracy Filters

### Greater than:
```
accuracy:>0.95
```

### Range:
```
accuracy:>0.90 accuracy:<0.98
```

## Date Filters

### After date:
```
created:>2025-01-01
```

### Between dates:
```
created:>2025-01-01 created:<2025-03-01
```

## Model Type Filter
```
model:resnet
```

## Status Filter
```
status:completed
status:failed
status:running
```

## Combined Queries
```
tag:baseline model:resnet accuracy:>0.96 convergence
```
Finds baseline ResNet experiments >96% accuracy with "convergence" in notes.

## Saved Searches
Save frequently used queries:
1. Run search
2. Click "Save Search"
3. Name it (e.g., "Best baseline models")
4. Access from saved searches dropdown
```

---

## 5.7 DO'S AND DON'TS

### ✅ DO's

1. **DO normalize tag names (lowercase, trim spaces)**
   - Reason: "Baseline" = "baseline" = "BASELINE" (prevent duplicates)
   - Implementation: `tag.lower().strip()`

2. **DO use PostgreSQL full-text search for MVP**
   - Reason: No additional infrastructure, fast enough for 10k experiments
   - Upgrade path: Elasticsearch for 100k+ experiments

3. **DO cache search results (Redis, 60 seconds TTL)**
   - Reason: Reduces database load, improves response time
   - Invalidate cache when experiment changes

4. **DO debounce search input (300ms delay)**
   - Reason: Prevents query on every keystroke, reduces load
   - User experience: Feels instant, not sluggish

5. **DO limit tags per experiment (max 10)**
   - Reason: Prevents tag spam, keeps UI clean
   - Enforced in service layer

6. **DO provide autocomplete for tags**
   - Reason: Prevents typos, suggests existing tags
   - Reduces duplicate tags ("basline" vs "baseline")

7. **DO slugify tags (convert to URL-safe format)**
   - Reason: "My Tag" → "my-tag" (consistent, searchable)
   - Store both: `name` (display) and `slug` (URL/search)

8. **DO track tag usage count**
   - Reason: Show popular tags, prioritize in autocomplete
   - Increment/decrement when tag added/removed

9. **DO implement saved searches**
   - Reason: Power users repeatedly search for same criteria
   - Saves time, improves workflow

10. **DO provide query syntax help (tooltip/cheat sheet)**
    - Reason: Advanced syntax not intuitive, users need guidance
    - Contextual help: Show examples as user types

### ❌ DON'Ts

1. **DON'T allow unlimited tags per experiment**
   - Reason: Tag spam, cluttered UI
   - Enforce limit: 10 tags per experiment

2. **DON'T use LIKE '%keyword%' without index**
   - Reason: Full table scan, extremely slow on large datasets
   - Use: Full-text search (GIN index) or Elasticsearch

3. **DON'T search synchronously (blocking UI)**
   - Reason: Slow queries freeze interface
   - Use: Debounced input + async search

4. **DON'T forget to escape special characters in SQL**
   - Reason: SQL injection vulnerability
   - Use: Parameterized queries (SQLAlchemy handles this)

5. **DON'T allow case-sensitive tags**
   - Reason: Creates duplicates ("Baseline", "baseline")
   - Normalize: Always lowercase

6. **DON'T allow tags with only numbers (e.g., "123")**
   - Reason: Confusing, non-descriptive
   - Validation: Require at least one letter

7. **DON'T forget to update search_vector when experiment changes**
   - Reason: Stale search results (experiment renamed but not found)
   - Solution: PostgreSQL trigger auto-updates

8. **DON'T return all results (no pagination)**
   - Reason: Performance, memory issues with 10k+ results
   - Limit: Default 100 results, allow pagination

9. **DON'T ignore empty queries**
   - Reason: Empty query = "SELECT * FROM experiments" (slow)
   - Validation: Require at least one keyword or filter

10. **DON'T skip authorization checks**
    - Reason: Security - users shouldn't see other users' experiments
    - Enforce: Filter by `user_id` in all queries

---

## 5.8 TESTING CHECKLIST

### Unit Tests

- [ ] `TagService.create_tag()` creates tag (or returns existing)
- [ ] `TagService.add_tag_to_experiment()` creates relationship
- [ ] `TagService.suggest_tags()` returns fuzzy matches
- [ ] `SearchService._parse_query()` correctly extracts all filters
- [ ] `SearchService._parse_query()` handles malformed queries
- [ ] `SearchService._build_sql_query()` generates valid SQL
- [ ] `SearchService._rank_results()` ranks correctly (exact match first)
- [ ] Tag normalization: "Baseline" → "baseline"
- [ ] Tag slugification: "My Tag" → "my-tag"
- [ ] Max tags enforced: 11th tag rejected

### Integration Tests

- [ ] Search by keyword → Returns matching experiments
- [ ] Search by tag → Returns experiments with that tag
- [ ] Search by accuracy → Filters correctly
- [ ] Search by date range → Filters correctly
- [ ] Combined search (tags + accuracy + keywords) → Returns correct results
- [ ] Empty query → Returns error or all experiments (depending on config)
- [ ] Cache hit → Returns results in <10ms
- [ ] Cache miss → Queries database
- [ ] Add tag to experiment → Search finds experiment by new tag
- [ ] Remove tag → Search no longer finds experiment by removed tag

### Manual QA

- [ ] Type in search bar → Results update after 300ms
- [ ] Type 2 characters in tag input → Autocomplete appears
- [ ] Select autocomplete suggestion → Tag added
- [ ] Click popular tag → Searches for that tag
- [ ] Apply filters → Search query updated
- [ ] Clear filters → Search resets
- [ ] Save search → Appears in saved searches list
- [ ] Click saved search → Executes query
- [ ] Add tag on experiment page → Tag appears immediately
- [ ] Remove tag → Tag disappears immediately
- [ ] Bulk add tags (5 experiments) → All updated
- [ ] Tag cloud displays correctly (sizes proportional)
- [ ] Mobile: Search bar responsive, filters collapsible
- [ ] Keyboard shortcut (Ctrl+K) → Focuses search bar

---

## 5.9 SUCCESS METRICS

### Quantitative
- Search response time: <500ms (p95) for 10,000 experiments
- Cache hit rate: >70% (most searches are repeated)
- Tag adoption: 50%+ of experiments have at least one tag
- Autocomplete usage: 60%+ of tags added via autocomplete (not manual typing)
- Saved searches: 20%+ of users create at least one saved search
- Zero SQL injection vulnerabilities

### Qualitative
- Users can find experiments in <10 seconds (vs 2-5 minutes before)
- Positive feedback: "Search is fast and intuitive"
- No complaints about duplicate tags (normalization works)
- Users discover tags via autocomplete (prevents reinventing tags)

---

## 5.10 FUTURE ENHANCEMENTS (POST-MVP)

### Phase 2 Features (if successful):

1. **Elasticsearch Integration**
   - For scale (100k+ experiments)
   - Fuzzy search ("basline" finds "baseline")
   - Highlighting (show matching keywords in results)

2. **Tag Hierarchies**
   - Parent-child relationships (e.g., "production" → "production-v1", "production-v2")
   - Search for parent → Returns all children

3. **Smart Tag Suggestions (ML-based)**
   - Analyze experiment config/results
   - Suggest tags: "High accuracy → Suggest 'high-accuracy' tag"

4. **Team Tag Sharing**
   - Organization-wide tags (vs per-user)
   - Tag permissions (who can add "production" tag?)

5. **Search Filters Presets**
   - "Show me last week's experiments"
   - "Show me best models by accuracy"
   - One-click filters

6. **Export Search Results**
   - CSV export of search results
   - API endpoint: `/api/search/export?query=...`

---

**END OF FEATURE #5 PLAN**

---

This completes the comprehensive planning document for **Feature #5: Experiment Tags & Search**.

**Key Modularity Features:**
✅ Global feature toggles (disable tags/search independently)  
✅ Pluggable search backend (PostgreSQL → Elasticsearch upgrade path)  
✅ Caching layer (Redis, configurable TTL)  
✅ Debounced input (configurable delay)  
✅ Extensible query syntax (easy to add new filters)  
✅ Service layer abstraction (search logic decoupled from UI)  

**Ready for your team to implement as a modular, scalable system.**