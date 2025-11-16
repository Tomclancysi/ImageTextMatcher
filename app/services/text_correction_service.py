import re
from typing import List, Tuple, Optional
from autocorrect import Speller
from spellchecker import SpellChecker
from symspellpy import SymSpell, Verbosity


class TextCorrectionService:
    def __init__(self, language: str = "en"):
        self.language = language
        self.speller = Speller(lang=language)
        self.spell_checker = SpellChecker(language=language)
        
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        try:
            dictionary_path = "frequency_dictionary_en_82_765.txt"
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        except FileNotFoundError:
            self.sym_spell = None

    def correct_text(self, text: str, max_suggestions: int = 3) -> Tuple[str, List[str]]:
        """
        Correct text and return corrected text with suggestions.
        
        Args:
            text: Input text to correct
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            Tuple of (corrected_text, suggestions_list)
        """
        if not text or not text.strip():
            return text, []
        
        words = self._tokenize(text)
        corrected_words = []
        suggestions = []
        
        for word in words:
            if self._is_valid_word(word):
                corrected_word, word_suggestions = self._correct_word(word, max_suggestions)
                corrected_words.append(corrected_word)
                if word_suggestions:
                    suggestions.extend(word_suggestions)
            else:
                corrected_words.append(word)
        
        corrected_text = self._reconstruct_text(text, words, corrected_words)
        return corrected_text, suggestions

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text while preserving punctuation."""
        return re.findall(r'\b\w+\b|[^\w\s]', text)

    def _is_valid_word(self, word: str) -> bool:
        """Check if word is valid for correction."""
        return (len(word) > 1 and
                word.isalpha() and
                not word.isupper() and
                len(word) > 2)

    def _correct_word(self, word: str, max_suggestions: int) -> Tuple[str, List[str]]:
        """Correct a single word and return suggestions."""
        word_lower = word.lower()
        
        if word_lower in self.spell_checker:
            return word, []
        
        suggestions = []
        
        if self.sym_spell:
            sym_suggestions = self.sym_spell.lookup(word_lower, Verbosity.CLOSEST, max_edit_distance=2)
            if sym_suggestions:
                suggestions.extend([s.term for s in sym_suggestions[:max_suggestions]])
        
        if not suggestions:
            spell_suggestions = self.spell_checker.candidates(word_lower)
            if spell_suggestions:
                suggestions.extend(list(spell_suggestions)[:max_suggestions])
        
        if not suggestions:
            try:
                corrected = self.speller(word_lower)
                if corrected != word_lower:
                    suggestions.append(corrected)
            except:
                pass
        
        if suggestions:
            return suggestions[0], suggestions[:max_suggestions]
        else:
            return word, []

    def _reconstruct_text(self, original_text: str, original_words: List[str], corrected_words: List[str]) -> str:
        """Reconstruct text with corrected words while preserving original formatting."""
        result = original_text
        for orig, corr in zip(original_words, corrected_words):
            if orig != corr:
                pattern = r'\b' + re.escape(orig) + r'\b'
                result = re.sub(pattern, corr, result, count=1)
        return result

    def get_suggestions(self, text: str, max_suggestions: int = 5) -> List[str]:
        """Get spelling suggestions for text without correcting it."""
        _, suggestions = self.correct_text(text, max_suggestions)
        return suggestions
