from langchain_text_splitters import (
    TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter
)
from typing import List
from nltk.tokenize import sent_tokenize
import nltk


class FixedChunker(TextSplitter):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._splitter = CharacterTextSplitter(
            separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


class NltkSemanticChunker(TextSplitter):
    def __init__(self, chunk_size=512, chunk_overlap=0, language='russian', **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        self.language = language
        # Внутренний RecursiveSplitter будет склеивать предложения в чанки
        self._merger = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_text(self, text: str) -> List[str]:
        sentences = sent_tokenize(text, language=self.language)
        # Склеиваем предложения через \n и передаём мержеру
        return self._merger.split_text("\n".join(sentences))

class RecursiveChunker(TextSplitter):
    def __init__(self, chunk_size=512, chunk_overlap=0, separators=None, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)