# Enhanced Academic Reference System - Documentation

## Overview

The academic reference system has been significantly enhanced to provide comprehensive article search, retrieval, and curation capabilities with full transparency and audit trails.

## New Features

### 1. Multi-Platform Article Search

The system now searches across multiple major scientific platforms:

- **PubMed**: Biomedical literature database (MEDLINE)
- **arXiv**: Preprint server for latest ML/AI research
- **Semantic Scholar**: AI-powered search with citation metrics
- **Google Scholar**: Comprehensive academic search (when available)
- **bioRxiv/medRxiv**: Biology and medicine preprints (framework in place)

### 2. Comprehensive Metadata

Each reference now includes extensive metadata:

#### Publication Information
- Title, authors, year
- Journal/venue
- Abstract (up to 500 characters)

#### Identifiers
- DOI (Digital Object Identifier)
- PMID (PubMed ID)
- arXiv ID
- Other platform-specific identifiers

#### Access Links
- Main article URL
- PDF download URL (when available)
- Alternative access links

#### Citation Metrics
- Citation count
- Relevance score (High/Medium/Low)

#### Audit Trail
- Retrieval timestamp
- Search rank
- Source database
- Citation count at time of retrieval

### 3. Search Reasoning and Transparency

The system provides complete transparency about its search methodology:

- **Search Strategy**: Multi-query approach explained
- **Platforms Searched**: Which databases were queried
- **Queries Used**: Exact search terms
- **Results by Platform**: Number of results from each source
- **Filtering Applied**: How duplicates were removed

### 4. Enhanced CrewAI Integration

The Literature Agent in the multi-agent system now:
- Automatically fetches relevant articles
- Integrates findings into diagnostic reports
- Provides reasoning about article selection
- Includes top references in agent analysis

### 5. Forensic Curation Capability

Each reference includes audit information for forensic review:
- When it was retrieved
- From which database
- Its ranking in search results
- Citation metrics at retrieval time
- Relevance assessment

## Usage

### Basic Usage (Backward Compatible)

```python
from academic_references import AcademicReferenceFetcher

fetcher = AcademicReferenceFetcher()

# Old style - returns just references list
references = fetcher.get_references_for_classification(
    class_name="melanoma",
    domain="skin lesion classification",
    max_per_source=3
)

# Display references
from academic_references import format_references_for_display
formatted = format_references_for_display(references)
print(formatted)
```

### Advanced Usage (With Reasoning)

```python
# New style - returns references and reasoning
references, reasoning = fetcher.get_references_for_classification(
    class_name="melanoma",
    domain="skin lesion classification",
    max_per_source=3,
    include_reasoning=True  # Enable reasoning metadata
)

# Display with full reasoning and audit trail
formatted = format_references_for_display(references, reasoning)
print(formatted)
```

### Multi-Agent System Integration

```python
from multi_agent_system import ManagerAgent

# Create manager with CrewAI enabled
manager = ManagerAgent(use_crewai=True)

# Context will be automatically populated with references
context = {
    'gradcam_description': 'High activation in lesion border',
    'model': 'ResNet50'
}

# Run analysis - references will be fetched automatically
report = manager.coordinate_analysis(
    predicted_class="melanoma",
    confidence=0.92,
    context=context
)

# Report includes literature section with full citations
print(report)
```

## Reference Data Structure

Each reference dictionary contains:

```python
{
    'title': str,                    # Article title
    'authors': str,                  # Formatted author list
    'year': str,                     # Publication year
    'journal': str,                  # Journal/venue name
    'abstract': str,                 # Abstract preview
    'platform': str,                 # Source platform
    'source': str,                   # Detailed source info
    'url': str,                      # Main article URL
    'download_url': str,             # PDF download URL
    'doi': str,                      # DOI identifier
    'pmid': str,                     # PubMed ID
    'arxiv_id': str,                 # arXiv ID
    'citation_count': int,           # Number of citations
    'relevance_score': str,          # 'High', 'Medium', 'Low'
    'audit_info': {
        'retrieved_date': str,       # ISO timestamp
        'search_rank': int,          # Position in search results
        'database': str,             # Source database name
        'citation_count': int        # Citations at retrieval
    }
}
```

## Reasoning Data Structure

```python
{
    'timestamp': str,                           # ISO timestamp
    'search_strategy': {
        'approach': str,                        # Overall strategy
        'PubMed': str,                         # Platform-specific strategy
        'arXiv': str,
        'Semantic Scholar': str
    },
    'platforms_searched': List[str],            # List of platforms
    'query_used': List[str],                    # Search queries
    'total_results_found': int,                 # Total unique results
    'filtering_applied': str,                   # Deduplication method
    'results_by_platform': Dict[str, int]       # Results per platform
}
```

## Display Format

The formatted output includes:

1. **Search Methodology Section**
   - Timestamp
   - Search strategy explanation
   - Platforms searched with results count
   - Total unique results
   - Filtering methods

2. **Articles and Citations Section**
   - Each article with full bibliographic data
   - Authors, year, journal
   - Abstract preview
   - Citation metrics
   - Identifiers (DOI, PMID, arXiv ID)
   - Access links (view, download, search)
   - Audit trail information

3. **Citation Guidance Footer**
   - How to use the references
   - Citation best practices
   - Access information

## Integration with Streamlit Apps

In `app4.py` and `app5.py`, the system is used as follows:

```python
from academic_references import AcademicReferenceFetcher, format_references_for_display

# Create fetcher
ref_fetcher = AcademicReferenceFetcher()

# Fetch references
references = ref_fetcher.get_references_for_classification(
    class_name=predicted_class,
    domain="image classification",
    max_per_source=3
)

# Display in expander
if references:
    with st.expander("📚 Referências Acadêmicas Encontradas"):
        st.markdown(format_references_for_display(references))
```

For enhanced display with reasoning:

```python
# Fetch with reasoning
references, reasoning = ref_fetcher.get_references_for_classification(
    class_name=predicted_class,
    domain="image classification",
    max_per_source=3,
    include_reasoning=True
)

# Display with full transparency
if references:
    with st.expander("📚 Referências Acadêmicas com Metodologia de Busca"):
        st.markdown(format_references_for_display(references, reasoning))
```

## API Rate Limiting and Best Practices

### PubMed
- No API key required for basic usage
- Rate limit: 3 requests/second without key
- Consider NCBI API key for production

### arXiv
- No authentication required
- Rate limit: ~1 request/second
- Bulk downloads available via OAI-PMH

### Semantic Scholar
- No API key required for basic usage
- Rate limit: 100 requests/5 minutes
- Production use may require API key

### Google Scholar
- Uses `scholarly` library
- Rate limiting via delays
- May be blocked by Google (use sparingly)

### Best Practices
1. Cache results when possible
2. Implement exponential backoff on errors
3. Respect rate limits
4. Use specific search queries
5. Filter results by relevance

## Error Handling

The system gracefully handles:
- Network errors (DNS, timeouts)
- API rate limiting
- Invalid responses
- Missing data fields
- Platform unavailability

All errors are logged and do not crash the application.

## Future Enhancements

Potential future additions:
1. bioRxiv/medRxiv full API integration
2. CrossRef API for DOI lookup
3. PubMed Central full-text access
4. Citation network analysis
5. Relevance scoring with ML
6. Custom ranking algorithms
7. Save/export functionality
8. Reference management integration

## Credits

Enhanced Academic Reference System
- Multiple platform integration
- Comprehensive metadata collection
- Audit trail and forensic curation
- Search reasoning transparency

Part of the DiagnostiCAI platform
Version: 5.1
Date: December 2024
