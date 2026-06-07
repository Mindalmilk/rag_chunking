from typing import List

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import (
    TextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


class FixedChunker(TextSplitter):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


class SentenceChunker(TextSplitter):
    def __init__(self, chunk_size=512, chunk_overlap=0, language="english", **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                nltk.download(pkg)

        self.language = language
        self._merger = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_text(self, text: str) -> List[str]:
        sentences = sent_tokenize(text, language=self.language)
        return self._merger.split_text("\n".join(sentences))


class SemanticChunker(TextSplitter):
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        language: str = "english",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        breakpoint_percentile_threshold: int = 80,
        distance_threshold: float | None = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                nltk.download(pkg)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.distance_threshold = distance_threshold
        self.model = SentenceTransformer(model_name, device=device)

        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split_text(self, text: str) -> List[str]:
        text = text.strip()

        if not text:
            return []

        sentences = [
            sentence.strip()
            for sentence in sent_tokenize(text, language=self.language)
            if sentence.strip()
        ]

        if len(sentences) <= 1:
            return self._fallback_splitter.split_text(text)

        groups = self._semantic_groups(sentences)

        chunks = []
        for group in groups:
            chunks.extend(self._split_group_by_size(group))

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _semantic_groups(self, sentences: List[str]) -> List[List[str]]:
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        distances = 1 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)

        if self.distance_threshold is None:
            threshold = np.percentile(
                distances,
                self.breakpoint_percentile_threshold,
            )
        else:
            threshold = self.distance_threshold

        groups = []
        current = [sentences[0]]

        for i, distance in enumerate(distances):
            if distance > threshold:
                groups.append(current)
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])

        if current:
            groups.append(current)

        return groups

    def _split_group_by_size(self, sentences: List[str]) -> List[str]:
        chunks = []
        current = []

        for sentence in sentences:
            candidate = " ".join(current + [sentence])

            if current and len(candidate) > self.chunk_size:
                chunk = " ".join(current)

                if len(chunk) <= self.chunk_size:
                    chunks.append(chunk)
                else:
                    chunks.extend(self._fallback_splitter.split_text(chunk))

                current = self._get_overlap_sentences(current)
                current.append(sentence)
            else:
                current.append(sentence)

        if current:
            chunk = " ".join(current)

            if len(chunk) <= self.chunk_size:
                chunks.append(chunk)
            else:
                chunks.extend(self._fallback_splitter.split_text(chunk))

        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        if self.chunk_overlap <= 0:
            return []

        result = []

        for sentence in reversed(sentences):
            candidate = [sentence] + result

            if len(" ".join(candidate)) > self.chunk_overlap:
                break

            result = candidate

        return result


class RecursiveChunker(TextSplitter):
    def __init__(self, chunk_size=512, chunk_overlap=0, separators=None, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)