from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


class AnswerGenerator:
    """
    Генератор ответов на основе Groq LLM и извлечённого контекста.
    Позволяет отдельно формировать промпт (с контекстом из retriever)
    и отдельно генерировать ответ по готовому промпту.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **model_kwargs,
    ):
        """
        Args:
            retriever: LangChain retriever для поиска релевантных документов.
            model_name: название модели Groq.
            temperature: креативность (0 = детерминированно).
            max_tokens: максимум токенов в ответе.
            api_key: API-ключ Groq. Если не задан, используется GROQ_API_KEY.
            prompt_template: шаблон промпта с плейсхолдерами {context} и {question}.
                             Если None, используется стандартный RAG-шаблон.
            model_kwargs: дополнительные параметры для ChatGroq.
        """
        self.retriever = retriever
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **model_kwargs,
        )
        self.output_parser = StrOutputParser()

        # Стандартный шаблон, если не передан
        if prompt_template is None:
            prompt_template = (
                "Below is a question followed by some context from different sources. "
                "Please answer the question based on the context. "
                "The answer to the question is a word or entity. "
                "If the provided information is insufficient to answer the question, "
                "respond 'Insufficient Information'. Answer directly without explanation.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        self.prompt_template = prompt_template

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Объединяет содержимое документов в единый контекст."""
        return "\n\n".join(doc.page_content for doc in docs)

    def create_prompt(self, question: str) -> str:
        """
        Формирует полный промпт для модели:
        извлекает документы через retriever, склеивает их и
        подставляет в шаблон вместе с вопросом.

        Returns:
            Строка с готовым промптом.
        """
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)
        prompt = self.prompt_template.format(context=context, question=question)
        return prompt

    def generate_response(self, prompt: str) -> str:
        """
        Отправляет готовый промпт в LLM и возвращает текстовый ответ.

        Args:
            prompt: полный текст промпта (можно получить из create_prompt).

        Returns:
            Строка ответа модели.
        """
        # Преобразуем строку в HumanMessage и инвоцируем LLM
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return self.output_parser.invoke(response)

    def generate(self, question: str) -> str:
        """
        Сквозной метод: получает вопрос, строит промпт и возвращает ответ.
        Эквивалентен последовательному вызову create_prompt → generate_response.
        """
        prompt = self.create_prompt(question)
        return self.generate_response(prompt)