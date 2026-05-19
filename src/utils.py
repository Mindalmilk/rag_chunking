import json
from langchain_community.document_loaders import WebBaseLoader
from typing import List
from langchain_core.documents import Document
from pathlib import Path

class CorpusLoader:
    """Выгружает все данные из корпуса документов в нужном формате"""
    def __init__(self, corpus_path: Path = Path('data/corpus.json')):
        self.corpus_path = corpus_path

    def load_content(self) -> List[Document]:
        try:
            with open(self.corpus_path, 'r') as f:
                corpus_data = json.load(f)
            documents = []
            for doc_info in corpus_data:
                doc = Document(
                    page_content=doc_info['body'],
                    metadata={
                        'title': doc_info['title'],
                        'author': doc_info['author'],
                        'source': doc_info['source'],
                        'category': doc_info['category'],
                        'published_at': doc_info['published_at'],
                        'url': doc_info['url']
                    }
                )
                documents.append(doc)
            print('Successfully loaded content from corpus')
            return documents
        except Exception as e:
            print(f"Error loading content: {str(e)}")
            return []

# Для теста
class WebContentLoader:
    """Парсит данные по ссылке. Нужен для тестов"""
    def __init__(self, urls: List[str]):
        self.urls = urls

    def load_content(self) -> List[Document]:
        loader = WebBaseLoader(self.urls)
        try:
            documents = loader.load()
            print(f"Successfully loaded content from {len(self.urls)} URLs")
            return documents
        except Exception as e:
            print(f"Error loading content: {str(e)}")
            return []