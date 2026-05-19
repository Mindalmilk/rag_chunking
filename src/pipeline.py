from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.utils import CorpusLoader
from src.chunkers import RecursiveChunker
from src.vector_store import VectorStore
from src.generator import AnswerGenerator

class RAGPipeline:
    """
    Полноценный RAG-пайплайн, объединяющий загрузку, чанкирование,
    индексацию и генерацию ответов.
    """

    def __init__(
        self,
        document_loader: CorpusLoader(),
        chunker: RecursiveChunker(),
        vector_store: VectorStore(),
        api_key: None
    ):
        """
        Args:
            document_loader: объект с методом load() -> List[Document]
            chunker: объект с методом split_documents(List[Document]) -> List[Document]
            vector_store: объект с методами add_documents(docs) и similarity_search(query, k)
            answer_generator: объект с методом generate(query, context_chunks) -> str
        """
        self.document_loader = document_loader
        self.chunker = chunker
        self.vector_store = vector_store
        self.answer_generator = None
        self._is_built = False
        self.api_key = api_key

    def build_from_document(self, search_kwargs=None) -> None:
        """
        Выполняет полную индексацию: загружает документы, разбивает на чанки,
        добавляет чанки в векторное хранилище.
        """
        print("1. Загрузка документов...")
        documents = self.document_loader.load_content()
        print(f"   Загружено документов: {len(documents)}")

        print("2. Разбиение на чанки...")
        chunks = self.chunker.split_documents(documents)
        print(f"   Создано чанков: {len(chunks)}")

        print("3. Индексация чанков в векторном хранилище...")
        self.vector_store.create_from_documents(chunks)
        self.answer_generator = AnswerGenerator(retriever=self.vector_store.as_retriever(search_kwargs or {"k": 4}), api_key=self.api_key)
        self._is_built = True
        print("   Индексация завершена.")

    def build_from_existing(self, search_kwargs=None):
        """Загружает существующий индекс из персистентного хранилища."""
        print("  Загрузка векторного хранилища")
        self.vector_store.load_existing()
        self.answer_generator = AnswerGenerator(retriever=self.vector_store.as_retriever(search_kwargs or {"k": 4}), api_key=self.api_key)
        self._is_built = True
        print("   Индексация завершена.")

    def query(self, query: str, search_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Выполняет поиск по индексированным чанкам и генерирует ответ.

        Args:
            query: текст запроса пользователя
            search_kwargs: параметры ретривера
        Returns:
            Словарь с ключами:
                - 'query': исходный запрос
                - 'retrieved_chunks': список найденных чанков (как Document)
                - 'answer': сгенерированный ответ
        """
        if not self._is_built:
            raise RuntimeError("Пайплайн не построен. Сначала выполните метод build().")
        retriever = self.vector_store.as_retriever(search_kwargs or {"k": 4})
        docs = retriever.invoke(query)
        context = [d.page_content for d in docs]

        prompt = self.create_prompt(question=query)
        answer = self.answer_generator.generate_response(prompt)

        return {
            "query": query,
            "retrieved_chunks": context,
            "answer": answer,
        }
    

    def create_prompt(self, question: str) -> str:
        """
        Формирует полный промпт для модели:
        извлекает документы через retriever, склеивает их и
        подставляет в шаблон вместе с вопросом.

        Returns:
            Строка с готовым промптом.
        """
        docs = self.answer_generator.retriever.invoke(question)
        context = self.answer_generator._format_docs(docs)
        prompt = self.answer_generator.prompt_template.format(context=context, question=question)
        return prompt