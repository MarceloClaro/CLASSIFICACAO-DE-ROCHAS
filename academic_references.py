"""
Academic Reference Scraper
Fetches relevant academic references from PubMed, arXiv, and other sources
"""

import requests
from typing import List, Dict, Optional
import time
from bs4 import BeautifulSoup
import json

try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False


class AcademicReferenceFetcher:
    """
    Fetches academic references related to image classification and analysis
    """
    
    def __init__(self):
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.arxiv_base_url = "http://export.arxiv.org/api/query"
        self.timeout = 10
    
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
                        'year': article.get('pubdate', 'N/A').split()[0] if article.get('pubdate') else 'N/A',
                        'journal': article.get('source', 'N/A')
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
                    'year': published,
                    'journal': 'arXiv preprint'
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
            print(f"Error searching Google Scholar: {str(e)}")
        
        return references
    
    def get_references_for_classification(
        self,
        class_name: str,
        domain: str = "image classification",
        max_per_source: int = 3
    ) -> List[Dict]:
        """
        Get comprehensive references for a classification task
        
        Args:
            class_name: The predicted class name
            domain: The domain of classification
            max_per_source: Maximum results per source
        
        Returns:
            Combined list of references from multiple sources
        """
        all_references = []
        
        # Build search queries
        queries = [
            f"{domain} {class_name} deep learning",
            f"{class_name} classification neural network",
            f"{class_name} diagnosis machine learning"
        ]
        
        # Search PubMed
        for query in queries[:1]:  # Limit queries for efficiency
            pubmed_refs = self.search_pubmed(query, max_results=max_per_source)
            all_references.extend(pubmed_refs)
            if len(pubmed_refs) > 0:
                break
        
        # Search arXiv
        for query in queries[:1]:
            arxiv_refs = self.search_arxiv(query, max_results=max_per_source)
            all_references.extend(arxiv_refs)
            if len(arxiv_refs) > 0:
                break
        
        # Search Google Scholar if available (optional, can be slow)
        # Uncomment if needed
        # for query in queries[:1]:
        #     scholar_refs = self.search_google_scholar(query, max_results=2)
        #     all_references.extend(scholar_refs)
        #     if len(scholar_refs) > 0:
        #         break
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_references = []
        for ref in all_references:
            title = ref.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_references.append(ref)
        
        return unique_references[:10]  # Limit to top 10


def format_references_for_display(references: List[Dict]) -> str:
    """
    Format references for display in the UI
    
    Args:
        references: List of reference dictionaries
    
    Returns:
        Formatted string for display
    """
    if not references:
        return "Nenhuma referência encontrada."
    
    formatted = "## 📚 Referências Acadêmicas\n\n"
    
    for i, ref in enumerate(references, 1):
        formatted += f"**{i}. {ref.get('title', 'N/A')}**\n"
        formatted += f"- **Autores:** {ref.get('authors', 'N/A')}\n"
        formatted += f"- **Fonte:** {ref.get('source', 'N/A')}\n"
        formatted += f"- **Ano:** {ref.get('year', 'N/A')}\n"
        formatted += f"- **Periódico:** {ref.get('journal', 'N/A')}\n"
        formatted += f"- **URL:** [{ref.get('url', 'N/A')}]({ref.get('url', '#')})\n\n"
    
    return formatted
