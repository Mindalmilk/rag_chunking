from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStore:
    """
    Инициализирует Chroma с выбранной эмбеддинг-моделью HuggingFace.
    Позволяет добавлять документы, загружать существующую коллекцию
    и получать retriever.
    """
    def __init__(
        self,
        collection_name: str = "my_docs",
        persist_directory: str = "./data/chroma_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs: Optional[dict] = None,
        embedding_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            collection_name: Имя коллекции в Chroma.
            persist_directory: Путь для сохранения базы на диске.
            model_name: Название модели с HuggingFace Hub.
            model_kwargs: Доп. аргументы для модели (напр. 'device': 'cuda').
            embedding_kwargs: Доп. аргументы для энкодера (напр. 'normalize_embeddings': True).
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Настройки по умолчанию
        if model_kwargs is None:
            model_kwargs = {"device": "cpu"}
        if embedding_kwargs is None:
            embedding_kwargs = {"normalize_embeddings": True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=embedding_kwargs,
        )
        self.vectorstore: Optional[Chroma] = None

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """
        Создаёт новую векторную БД из списка документов.
        """
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        return self.vectorstore

    def load_existing(self) -> Chroma:
        """
        Загружает существующую коллекцию из persist_directory.
        """
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        return self.vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """Добавляет документы в уже существующую коллекцию."""
        if self.vectorstore is None:
            raise ValueError("Сначала создайте или загрузите векторную БД.")
        self.vectorstore.add_documents(documents)

    def as_retriever(self, search_kwargs: Optional[dict] = None) -> VectorStoreRetriever:
        """
        Возвращает retriever для поиска.
        По умолчанию k=4 и метрика similarity.
        """
        if self.vectorstore is None:
            raise ValueError("Векторная БД не инициализирована.")
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)