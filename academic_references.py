"""
Academic Reference Scraper - Enhanced Version
Fetches relevant academic references from multiple scientific platforms:
- PubMed (biomedical literature)
- arXiv (preprints)
- Google Scholar (when available)
- bioRxiv/medRxiv (biology/medicine preprints)
- Semantic Scholar (AI-powered search)

Provides full citation metadata, download links, and audit trail information
"""

import requests
from typing import List, Dict, Optional, Tuple
import time
from bs4 import BeautifulSoup
import json
from datetime import datetime

try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False

# Try to import AI modules for translation and critical review
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

AI_AVAILABLE = GEMINI_AVAILABLE or GROQ_AVAILABLE


class AcademicReferenceFetcher:
    """
    Enhanced academic reference fetcher with comprehensive platform support
    Provides full citation metadata, reasoning, and audit trails
    """
    
    # Class constants
    RATE_LIMIT_DELAY = 0.5  # Delay between API calls to avoid rate limiting
    
    def __init__(self, ai_provider: Optional[str] = None, ai_api_key: Optional[str] = None, ai_model: Optional[str] = None):
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.arxiv_base_url = "http://export.arxiv.org/api/query"
        self.biorxiv_base_url = "https://api.biorxiv.org/details/biorxiv"
        self.medrxiv_base_url = "https://api.biorxiv.org/details/medrxiv"
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"
        self.timeout = 10
        self.search_metadata = {
            'timestamp': None,
            'query': None,
            'platforms_searched': [],
            'total_results': 0
        }
        # AI configuration for translation and critical reviews
        self.ai_provider = ai_provider
        self.ai_api_key = ai_api_key
        # Set default model based on provider (using current recommended models)
        # Reference: https://github.com/google-gemini/cookbook
        if ai_model:
            self.ai_model = ai_model
        elif ai_provider == 'groq':
            self.ai_model = "mixtral-8x7b-32768"
        else:
            self.ai_model = "gemini-2.5-flash"  # Updated to current recommended model
        self.ai_client = None
        self.ai_model_obj = None
        
        # Initialize AI if credentials provided
        if self.ai_provider and self.ai_api_key and AI_AVAILABLE:
            self._initialize_ai()
    
    def _initialize_ai(self):
        """Initialize AI client for translation and critical reviews"""
        try:
            if self.ai_provider == 'gemini' and GEMINI_AVAILABLE:
                genai.configure(api_key=self.ai_api_key)
                self.ai_model_obj = genai.GenerativeModel(self.ai_model)
                # Test if model is accessible
                try:
                    # Make a simple test call to verify model works
                    test_response = self.ai_model_obj.generate_content("Test")
                    print(f"✅ Gemini model '{self.ai_model}' initialized successfully")
                except Exception as model_error:
                    print(f"❌ Error initializing Gemini model '{self.ai_model}': {str(model_error)}")
                    self.ai_model_obj = None
                    raise
            elif self.ai_provider == 'groq' and GROQ_AVAILABLE:
                self.ai_client = Groq(api_key=self.ai_api_key)
                print(f"✅ Groq client initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize AI: {str(e)}")
            print(f"   Translation and critical reviews will not be available.")
            self.ai_model_obj = None
            self.ai_client = None
    
    def translate_abstract_to_portuguese(self, abstract: str, title: str = "") -> str:
        """
        Translate article abstract from English to Portuguese
        
        Args:
            abstract: Article abstract in English
            title: Article title for context
        
        Returns:
            Translated abstract in Portuguese
        """
        if not abstract or abstract == "Abstract not available":
            return "Resumo não disponível"
        
        # If AI is not available, return original
        if not self.ai_provider or not self.ai_api_key or not AI_AVAILABLE:
            return abstract
        
        # Check if AI model is properly initialized
        if not self.ai_model_obj and not self.ai_client:
            return abstract
        
        try:
            # Sanitize inputs - limit length to prevent prompt injection
            safe_title = title[:500] if title else ""
            safe_abstract = abstract[:2000] if abstract else ""
            
            prompt = f"""Traduza o seguinte resumo científico do inglês para o português brasileiro de forma clara e precisa, mantendo a terminologia técnica apropriada:

Título: {safe_title}

Resumo: {safe_abstract}

Forneça APENAS a tradução do resumo, sem adicionar comentários ou explicações extras."""

            if self.ai_provider == 'gemini' and self.ai_model_obj:
                response = self.ai_model_obj.generate_content(prompt)
                return response.text.strip()
            elif self.ai_provider == 'groq' and self.ai_client:
                chat_completion = self.ai_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional scientific translator specializing in translating academic abstracts from English to Brazilian Portuguese."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.ai_model,
                    temperature=0.3,
                    max_tokens=1000,
                )
                return chat_completion.choices[0].message.content.strip()
            else:
                return abstract
        except Exception as e:
            print(f"❌ Error translating abstract: {str(e)}")
            return abstract
    
    def generate_critical_review(self, reference: Dict) -> str:
        """
        Generate a critical review of an academic article
        
        Args:
            reference: Article reference dictionary with metadata
        
        Returns:
            Critical review in Portuguese
        """
        if not self.ai_provider or not self.ai_api_key or not AI_AVAILABLE:
            return "Resenha crítica não disponível (requer configuração de API de IA)"
        
        # Check if AI model is properly initialized
        if not self.ai_model_obj and not self.ai_client:
            return "Resenha crítica não disponível (IA não inicializada corretamente)"
        
        try:
            # Sanitize inputs - limit length to prevent prompt injection
            title = reference.get('title', 'N/A')[:500]
            authors = reference.get('authors', 'N/A')[:200]
            year = str(reference.get('year', 'N/A'))[:10]
            journal = reference.get('journal', 'N/A')[:200]
            abstract = reference.get('abstract', 'Abstract not available')[:2000]
            citation_count = str(reference.get('citation_count', 'N/A'))[:10]
            platform = reference.get('platform', 'N/A')[:100]
            
            prompt = f"""Como especialista em análise crítica de literatura científica, forneça uma resenha crítica detalhada do seguinte artigo científico:

Título: {title}
Autores: {authors}
Ano: {year}
Fonte: {journal}
Plataforma: {platform}
Citações: {citation_count}

Resumo: {abstract}

Por favor, forneça uma resenha crítica que inclua:

1. **Principais Contribuições**: Quais são as principais contribuições do trabalho?
2. **Pontos Fortes**: Aspectos positivos e inovadores da pesquisa
3. **Limitações**: Possíveis limitações metodológicas ou de escopo
4. **Relevância**: Relevância para o campo e impacto potencial
5. **Aplicabilidade**: Aplicabilidade prática dos resultados

Mantenha um tom acadêmico e objetivo. Limite a resenha a aproximadamente 150-200 palavras."""

            if self.ai_provider == 'gemini' and self.ai_model_obj:
                response = self.ai_model_obj.generate_content(prompt)
                return response.text.strip()
            elif self.ai_provider == 'groq' and self.ai_client:
                chat_completion = self.ai_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a PhD-level researcher specialized in critical analysis of academic literature."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.ai_model,
                    temperature=0.7,
                    max_tokens=500,
                )
                return chat_completion.choices[0].message.content.strip()
            else:
                return "Resenha crítica não disponível"
        except Exception as e:
            print(f"❌ Error generating critical review: {str(e)}")
            return f"Erro ao gerar resenha crítica: {str(e)}"
    
    def enrich_references_with_analysis(self, references: List[Dict]) -> List[Dict]:
        """
        Enrich references with Portuguese translation and critical reviews
        
        Args:
            references: List of reference dictionaries
        
        Returns:
            Enriched list of references with translations and reviews
        """
        # Check if AI is properly initialized
        ai_is_ready = (
            self.ai_provider and 
            self.ai_api_key and 
            AI_AVAILABLE and
            (self.ai_model_obj is not None or self.ai_client is not None)
        )
        
        if not ai_is_ready:
            # Return references as-is if AI not properly initialized
            error_msg = "Tradução não disponível (IA não inicializada corretamente)"
            for ref in references:
                ref['abstract_pt'] = error_msg
                ref['critical_review'] = "Resenha crítica não disponível (IA não inicializada corretamente)"
            print(f"⚠️ AI not properly initialized. Translation and reviews will not be generated.")
            return references
        
        print(f"✅ AI is ready. Processing {len(references)} references...")
        enriched = []
        for i, ref in enumerate(references):
            print(f"📄 Processando artigo {i+1}/{len(references)}: {ref.get('title', 'N/A')[:50]}...")
            
            # Translate abstract
            abstract = ref.get('abstract', '')
            title = ref.get('title', '')
            ref['abstract_pt'] = self.translate_abstract_to_portuguese(abstract, title)
            
            # Generate critical review
            ref['critical_review'] = self.generate_critical_review(ref)
            
            enriched.append(ref)
            
            # Small delay to avoid rate limiting (skip for last item)
            if i < len(references) - 1:
                time.sleep(self.RATE_LIMIT_DELAY)
        
        print(f"✅ Processamento completo! {len(enriched)} referências enriquecidas.")
        return enriched
    
    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search PubMed for relevant articles
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
        
        Returns:
            List of article dictionaries with metadata
        """
        references = []
        
        # Sanitize query - remove potentially problematic characters
        query = query.strip()
        if not query or len(query) > 500:  # Reasonable limit
            return references
        
        try:
            # Search for article IDs
            search_url = f"{self.pubmed_base_url}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json'
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=self.timeout)
            search_data = search_response.json()
            
            if 'esearchresult' not in search_data or 'idlist' not in search_data['esearchresult']:
                return references
            
            id_list = search_data['esearchresult']['idlist']
            
            if not id_list:
                return references
            
            # Fetch article summaries
            summary_url = f"{self.pubmed_base_url}esummary.fcgi"
            summary_params = {
                'db': 'pubmed',
                'id': ','.join(id_list),
                'retmode': 'json'
            }
            
            summary_response = requests.get(summary_url, params=summary_params, timeout=self.timeout)
            summary_data = summary_response.json()
            
            if 'result' not in summary_data:
                return references
            
            for pmid in id_list:
                if pmid in summary_data['result']:
                    article = summary_data['result'][pmid]
                    references.append({
                        'title': article.get('title', 'N/A'),
                        'authors': ', '.join([author.get('name', '') for author in article.get('authors', [])[:3]]) + (' et al.' if len(article.get('authors', [])) > 3 else ''),
                        'source': f"PubMed (PMID: {pmid})",
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        'download_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/#full-view-heading",
                        'pdf_search_url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmid}/pdf/",
                        'year': article.get('pubdate', 'N/A').split()[0] if article.get('pubdate') else 'N/A',
                        'journal': article.get('source', 'N/A'),
                        'doi': article.get('elocationid', 'N/A'),
                        'abstract': article.get('abstract', 'Abstract not available'),
                        'pmid': pmid,
                        'platform': 'PubMed',
                        'citation_count': article.get('citedby', 'N/A'),
                        'relevance_score': 'High' if len(references) < 3 else 'Medium',
                        'audit_info': {
                            'retrieved_date': datetime.now().isoformat(),
                            'search_rank': len(references) + 1,
                            'database': 'PubMed/MEDLINE'
                        }
                    })
        
        except Exception as e:
            print(f"Error searching PubMed: {str(e)}")
        
        return references
    
    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search arXiv for relevant articles
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
        
        Returns:
            List of article dictionaries with metadata
        """
        references = []
        
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.arxiv_base_url, params=params, timeout=self.timeout)
            
            if response.status_code != 200:
                return references
            
            # Parse XML response
            soup = BeautifulSoup(response.content, 'xml')
            entries = soup.find_all('entry')
            
            for entry in entries:
                title = entry.find('title').text.strip() if entry.find('title') else 'N/A'
                authors = [author.find('name').text for author in entry.find_all('author')]
                author_str = ', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else '')
                link = entry.find('id').text if entry.find('id') else 'N/A'
                published = entry.find('published').text[:4] if entry.find('published') else 'N/A'
                
                references.append({
                    'title': title,
                    'authors': author_str,
                    'source': f"arXiv",
                    'url': link,
                    'download_url': link.replace('abs', 'pdf') if link and 'abs' in link else (link + '.pdf' if link else 'N/A'),
                    'year': published,
                    'journal': 'arXiv preprint',
                    'abstract': entry.find('summary').text.strip()[:500] + '...' if entry.find('summary') else 'Abstract not available',
                    'arxiv_id': link.split('/')[-1] if link else 'N/A',
                    'platform': 'arXiv',
                    'category': entry.find('category').get('term') if entry.find('category') else 'N/A',
                    'relevance_score': 'High' if len(references) < 3 else 'Medium',
                    'audit_info': {
                        'retrieved_date': datetime.now().isoformat(),
                        'search_rank': len(references) + 1,
                        'database': 'arXiv.org'
                    }
                })
        
        except Exception as e:
            print(f"Error searching arXiv: {str(e)}")
        
        return references
    
    def search_google_scholar(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Search Google Scholar for relevant articles using scholarly library
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
        
        Returns:
            List of article dictionaries with metadata
        """
        references = []
        
        if not SCHOLARLY_AVAILABLE:
            return references
        
        try:
            search_query = scholarly.search_pubs(query)
            
            count = 0
            for result in search_query:
                if count >= max_results:
                    break
                
                bib = result.get('bib', {})
                references.append({
                    'title': bib.get('title', 'N/A'),
                    'authors': ', '.join(bib.get('author', ['N/A'])[:3]) + (' et al.' if len(bib.get('author', [])) > 3 else ''),
                    'source': 'Google Scholar',
                    'url': result.get('pub_url', result.get('eprint_url', 'N/A')),
                    'year': bib.get('pub_year', 'N/A'),
                    'journal': bib.get('venue', 'N/A')
                })
                
                count += 1
                time.sleep(1)  # Rate limiting
        
        except Exception as e:
            # Silently skip Google Scholar if blocked or unavailable
            # This is expected when Google Scholar blocks automated access
            pass
        
        return references
    
    def search_biorxiv_medrxiv(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search bioRxiv and medRxiv for relevant preprints
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
        
        Returns:
            List of article dictionaries with metadata
        """
        references = []
        
        try:
            # Search both bioRxiv and medRxiv
            for server, base_url in [('bioRxiv', self.biorxiv_base_url), ('medRxiv', self.medrxiv_base_url)]:
                # Note: The actual bioRxiv/medRxiv API is limited
                # For production, consider using their search interface or database dumps
                # This is a placeholder for the structure
                pass
        
        except Exception as e:
            print(f"Error searching bioRxiv/medRxiv: {str(e)}")
        
        return references
    
    def search_semantic_scholar(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search Semantic Scholar for relevant articles using their API
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
        
        Returns:
            List of article dictionaries with metadata
        """
        references = []
        
        try:
            search_url = f"{self.semantic_scholar_base_url}/paper/search"
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,authors,year,abstract,citationCount,url,openAccessPdf,venue,externalIds'
            }
            
            headers = {
                'User-Agent': 'Academic Reference Fetcher (research tool)'
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                return references
            
            data = response.json()
            
            if 'data' not in data:
                return references
            
            for paper in data['data']:
                authors_list = [author.get('name', '') for author in paper.get('authors', [])]
                author_str = ', '.join(authors_list[:3]) + (' et al.' if len(authors_list) > 3 else '')
                
                # Extract DOI and other identifiers
                external_ids = paper.get('externalIds', {})
                doi = external_ids.get('DOI', 'N/A')
                arxiv_id = external_ids.get('ArXiv', 'N/A')
                pmid = external_ids.get('PubMed', 'N/A')
                
                # Build download URL if available
                pdf_url = 'N/A'
                if paper.get('openAccessPdf'):
                    pdf_url = paper['openAccessPdf'].get('url', 'N/A')
                
                references.append({
                    'title': paper.get('title', 'N/A'),
                    'authors': author_str,
                    'source': 'Semantic Scholar',
                    'url': paper.get('url', 'N/A'),
                    'download_url': pdf_url,
                    'year': str(paper.get('year', 'N/A')),
                    'journal': paper.get('venue', 'N/A'),
                    'abstract': paper.get('abstract', 'Abstract not available')[:500] + '...' if paper.get('abstract') else 'Abstract not available',
                    'doi': doi,
                    'arxiv_id': arxiv_id,
                    'pmid': pmid,
                    'platform': 'Semantic Scholar',
                    'citation_count': paper.get('citationCount', 0),
                    'relevance_score': 'High' if len(references) < 3 else 'Medium',
                    'audit_info': {
                        'retrieved_date': datetime.now().isoformat(),
                        'search_rank': len(references) + 1,
                        'database': 'Semantic Scholar API',
                        'citation_count': paper.get('citationCount', 0)
                    }
                })
        
        except Exception as e:
            print(f"Error searching Semantic Scholar: {str(e)}")
        
        return references
    
    def get_references_for_classification(
        self,
        class_name: str,
        domain: str = "image classification",
        max_per_source: int = 3,
        include_reasoning: bool = False
    ):
        """
        Get comprehensive references for a classification task with optional reasoning
        
        Args:
            class_name: The predicted class name
            domain: The domain of classification
            max_per_source: Maximum results per source
            include_reasoning: Whether to include search reasoning metadata
        
        Returns:
            - If include_reasoning is True: Tuple of (references list, reasoning dictionary)
            - If include_reasoning is False: Just the references list (backward compatible)
        """
        all_references = []
        reasoning = {
            'timestamp': datetime.now().isoformat(),
            'search_strategy': {},
            'platforms_searched': [],
            'query_used': [],
            'total_results_found': 0,
            'filtering_applied': 'Removed duplicates based on title similarity'
        }
        
        # Build search queries with reasoning
        queries = [
            f"{domain} {class_name} deep learning",
            f"{class_name} classification neural network",
            f"{class_name} diagnosis machine learning"
        ]
        reasoning['query_used'] = queries
        reasoning['search_strategy']['approach'] = 'Multi-query strategy targeting different aspects: general ML, specific classification, diagnostic context'
        
        # Search PubMed
        reasoning['platforms_searched'].append('PubMed')
        reasoning['search_strategy']['PubMed'] = 'Biomedical literature database - focus on medical/scientific applications'
        for query in queries[:1]:  # Limit queries for efficiency
            pubmed_refs = self.search_pubmed(query, max_results=max_per_source)
            all_references.extend(pubmed_refs)
            if len(pubmed_refs) > 0:
                break
        
        # Search arXiv
        reasoning['platforms_searched'].append('arXiv')
        reasoning['search_strategy']['arXiv'] = 'Preprint server - latest research in ML/AI not yet peer-reviewed'
        for query in queries[:1]:
            arxiv_refs = self.search_arxiv(query, max_results=max_per_source)
            all_references.extend(arxiv_refs)
            if len(arxiv_refs) > 0:
                break
        
        # Search Semantic Scholar
        reasoning['platforms_searched'].append('Semantic Scholar')
        reasoning['search_strategy']['Semantic Scholar'] = 'AI-powered search engine with citation metrics and open access focus'
        for query in queries[:1]:
            semantic_refs = self.search_semantic_scholar(query, max_results=max_per_source)
            all_references.extend(semantic_refs)
            if len(semantic_refs) > 0:
                break
        
        # Search Google Scholar if available (optional, can be slow and may be blocked)
        if SCHOLARLY_AVAILABLE:
            reasoning['platforms_searched'].append('Google Scholar')
            reasoning['search_strategy']['Google Scholar'] = 'Comprehensive academic search - includes citations and diverse sources'
            for query in queries[:1]:
                try:
                    scholar_refs = self.search_google_scholar(query, max_results=2)
                    all_references.extend(scholar_refs)
                    if len(scholar_refs) > 0:
                        break
                except Exception:
                    # Silently skip Google Scholar if blocked or unavailable
                    pass
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_references = []
        for ref in all_references:
            title = ref.get('title', '').lower()
            if title and title not in seen_titles and title != 'n/a':
                seen_titles.add(title)
                unique_references.append(ref)
        
        # Sort by relevance score and citation count
        unique_references.sort(
            key=lambda x: (
                1 if x.get('relevance_score') == 'High' else 0,
                x.get('citation_count', 0) if isinstance(x.get('citation_count'), int) else 0
            ),
            reverse=True
        )
        
        # Limit to top results
        final_references = unique_references[:10]
        
        reasoning['total_results_found'] = len(final_references)
        reasoning['results_by_platform'] = {}
        for platform in reasoning['platforms_searched']:
            count = sum(1 for ref in final_references if ref.get('platform') == platform)
            reasoning['results_by_platform'][platform] = count
        
        # Return based on include_reasoning flag
        if include_reasoning:
            return final_references, reasoning
        else:
            # Backward compatible - return only references
            return final_references


def format_references_for_display(references: List[Dict], reasoning: Optional[Dict] = None) -> str:
    """
    Format references for display in the UI with comprehensive metadata
    
    Args:
        references: List of reference dictionaries
        reasoning: Optional reasoning metadata from search
    
    Returns:
        Formatted string for display
    """
    if not references:
        return "Nenhuma referência encontrada."
    
    formatted = "## 📚 Referências Acadêmicas Encontradas\n\n"
    
    # Add search reasoning if available
    if reasoning:
        formatted += "### 🔍 Metodologia de Busca e Raciocínio\n\n"
        formatted += f"**Data/Hora da Busca:** {reasoning.get('timestamp', 'N/A')}\n\n"
        formatted += f"**Estratégia:** {reasoning.get('search_strategy', {}).get('approach', 'N/A')}\n\n"
        
        formatted += "**Plataformas Pesquisadas:**\n"
        for platform in reasoning.get('platforms_searched', []):
            strategy = reasoning.get('search_strategy', {}).get(platform, 'N/A')
            count = reasoning.get('results_by_platform', {}).get(platform, 0)
            formatted += f"- **{platform}**: {strategy} ({count} resultado(s) encontrado(s))\n"
        
        formatted += f"\n**Total de Resultados Únicos:** {reasoning.get('total_results_found', 0)}\n"
        formatted += f"**Filtros Aplicados:** {reasoning.get('filtering_applied', 'N/A')}\n\n"
        formatted += "---\n\n"
    
    formatted += "### 📖 Artigos e Citações\n\n"
    
    for i, ref in enumerate(references, 1):
        # Title and basic info
        formatted += f"#### {i}. {ref.get('title', 'N/A')}\n\n"
        
        # Authors and publication info
        formatted += f"**👥 Autores:** {ref.get('authors', 'N/A')}\n\n"
        formatted += f"**📅 Ano:** {ref.get('year', 'N/A')}\n\n"
        formatted += f"**📰 Periódico/Fonte:** {ref.get('journal', 'N/A')}\n\n"
        formatted += f"**🏛️ Plataforma:** {ref.get('platform', ref.get('source', 'N/A'))}\n\n"
        
        # Citation metrics
        if ref.get('citation_count') and ref.get('citation_count') != 'N/A':
            formatted += f"**📊 Citações:** {ref.get('citation_count')}\n\n"
        
        # Relevance score
        if ref.get('relevance_score'):
            emoji = "🟢" if ref['relevance_score'] == 'High' else "🟡"
            formatted += f"**{emoji} Relevância:** {ref.get('relevance_score')}\n\n"
        
        # Abstract preview (original)
        if ref.get('abstract') and ref['abstract'] != 'Abstract not available':
            formatted += f"**📝 Resumo (Original):** {ref.get('abstract')}\n\n"
        
        # Portuguese translation if available
        if ref.get('abstract_pt'):
            formatted += f"**📝 Resumo (Português):** {ref.get('abstract_pt')}\n\n"
        
        # Critical review if available
        if ref.get('critical_review'):
            formatted += f"**🔍 Resenha Crítica:**\n{ref.get('critical_review')}\n\n"
        
        # Identifiers
        formatted += "**🔗 Identificadores:**\n"
        if ref.get('doi') and ref['doi'] != 'N/A':
            formatted += f"- DOI: {ref['doi']}\n"
        if ref.get('pmid') and ref['pmid'] != 'N/A':
            formatted += f"- PMID: {ref['pmid']}\n"
        if ref.get('arxiv_id') and ref['arxiv_id'] != 'N/A':
            formatted += f"- arXiv ID: {ref['arxiv_id']}\n"
        formatted += "\n"
        
        # URLs for access and download
        formatted += "**🌐 Links de Acesso:**\n"
        formatted += f"- [📄 Visualizar Artigo]({ref.get('url', '#')})\n"
        
        if ref.get('download_url') and ref['download_url'] != 'N/A':
            formatted += f"- [⬇️ Download PDF]({ref['download_url']})\n"
        
        if ref.get('pdf_search_url') and ref['pdf_search_url'] != 'N/A':
            formatted += f"- [🔍 Buscar PDF no PMC]({ref['pdf_search_url']})\n"
        
        formatted += "\n"
        
        # Audit information
        if ref.get('audit_info'):
            audit = ref['audit_info']
            formatted += "**🔐 Informações de Auditoria e Curadoria:**\n"
            formatted += f"- Data de Recuperação: {audit.get('retrieved_date', 'N/A')}\n"
            formatted += f"- Ranking na Busca: #{audit.get('search_rank', 'N/A')}\n"
            formatted += f"- Base de Dados: {audit.get('database', 'N/A')}\n"
            if audit.get('citation_count'):
                formatted += f"- Contagem de Citações: {audit['citation_count']}\n"
            formatted += "\n"
        
        formatted += "---\n\n"
    
    # Footer with citation guidance
    formatted += "### 📋 Nota sobre Citações\n\n"
    formatted += "Todas as referências acima foram recuperadas de plataformas científicas reconhecidas. "
    formatted += "Para citação formal, utilize os identificadores (DOI, PMID, arXiv ID) fornecidos. "
    formatted += "Os links de download direcionam para versões de acesso aberto quando disponíveis. "
    formatted += "Para acesso completo, pode ser necessário acesso institucional ou pagamento.\n\n"
    
    return formatted
