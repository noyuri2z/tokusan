"""
Text classifier explanation using LIME.

This module provides the main functionality for explaining text classification
models using LIME (Local Interpretable Model-Agnostic Explanations). It extends
the original LIME library with robust support for Japanese text processing.

Key Features:
    - Automatic Japanese tokenization using SudachiPy
    - Plain language explanation generation in both English and Japanese
    - Word-level and character-level explanations
    - Bag-of-words and positional explanations

The main class is TextExplainer, which wraps any text classifier and provides
human-interpretable explanations of individual predictions.

Example:
    >>> from tokusan import TextExplainer
    >>> explainer = TextExplainer(class_names=['negative', 'positive'], lang='jp')
    >>> exp = explainer.explain_instance(text, classifier.predict_proba)
    >>> print(exp.as_list(label=1))
    [('良い', 0.15), ('素晴らしい', 0.12), ...]
"""

import itertools
import json
import re
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy as sp
import sklearn.metrics.pairwise
from sklearn.utils import check_random_state

from . import explanation
from .base import LimeBase


class TextDomainMapper(explanation.DomainMapper):
    """
    Maps feature IDs to words for text explanations.

    In LIME's internal representation, each unique word is assigned a
    numeric ID. This class provides the mapping between those IDs and
    the actual words, enabling human-readable explanations.

    Attributes:
        indexed_string: IndexedString instance containing the vocabulary.
    """

    def __init__(self, indexed_string: 'IndexedString'):
        """
        Initialize the text domain mapper.

        Args:
            indexed_string: An IndexedString object containing the original
                          text and its word-to-ID mapping.
        """
        self.indexed_string = indexed_string

    def map_exp_ids(
        self,
        exp: List[Tuple[int, float]],
        positions: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Convert feature IDs to word strings.

        Args:
            exp: List of (feature_id, weight) tuples from the explanation.
            positions: If True, include position information in word names.
                      Useful for non-bag-of-words models where the same
                      word at different positions has different meanings.

        Returns:
            List of (word, weight) tuples. If positions=True, word format
            is 'word_pos1-pos2-...' showing all occurrence positions.

        Example:
            >>> mapper.map_exp_ids([(0, 0.5), (1, -0.3)])
            [('great', 0.5), ('bad', -0.3)]
        """
        if positions:
            # Include position information for each word
            result = []
            for feature_id, weight in exp:
                word = self.indexed_string.word(feature_id)
                positions_list = self.indexed_string.string_position(feature_id)
                position_str = '-'.join(map(str, positions_list))
                result.append((f'{word}_{position_str}', weight))
            return result
        else:
            # Simple word mapping
            return [(self.indexed_string.word(x[0]), x[1]) for x in exp]

    def visualize_instance_html(
        self,
        exp: List[Tuple[int, float]],
        label: int,
        div_name: str,
        exp_object_name: str,
        text: bool = True,
        opacity: bool = True
    ) -> str:
        """
        Generate JavaScript for highlighting words in the original text.

        Creates JavaScript code that highlights words in the original text
        based on their importance weights. Positive weights are typically
        shown in one color (e.g., green) and negative in another (e.g., red).

        Args:
            exp: List of (feature_id, weight) tuples.
            label: The class label being explained.
            div_name: Name of the HTML div for rendering.
            exp_object_name: Name of the JavaScript explanation object.
            text: If False, return empty string.
            opacity: If True, vary color opacity based on weight magnitude.

        Returns:
            str: JavaScript code for text visualization.
        """
        if not text:
            return ''

        # Get the original text, escaped for HTML
        raw_text = self.indexed_string.raw_string()
        text_escaped = raw_text.encode('utf-8', 'xmlcharrefreplace').decode('utf-8')
        text_escaped = re.sub(r'[<>&]', '|', text_escaped)

        # Build list of (word, positions, weight) for highlighting
        word_data = [
            (
                self.indexed_string.word(x[0]),
                self.indexed_string.string_position(x[0]),
                x[1]
            )
            for x in exp
        ]

        # Flatten to individual occurrences
        all_occurrences = list(itertools.chain.from_iterable([
            itertools.product([w[0]], w[1], [w[2]])
            for w in word_data
        ]))
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]

        # Generate JavaScript
        return f'''
            {exp_object_name}.show_raw_text({json.dumps(all_occurrences)}, {label}, {json.dumps(text_escaped)}, {div_name}, {json.dumps(opacity)});
        '''


class IndexedString:
    """
    A string with word-level indexing for efficient perturbation.

    This class provides the core data structure for text explanation.
    It tokenizes text and maintains mappings between:
    - Word positions in the original string
    - Unique word vocabulary
    - Feature IDs for the LIME algorithm

    The class supports both bag-of-words mode (where all occurrences of
    a word are treated identically) and positional mode (where each
    occurrence is treated separately).

    Attributes:
        raw: The original input string.
        as_list: The string split into tokens and separators.
        as_np: NumPy array version of as_list.
        inverse_vocab: List mapping feature IDs to words.
        positions: Mapping from feature IDs to positions in as_list.
        bow: Whether bag-of-words mode is active.
    """

    def __init__(
        self,
        raw_string: str,
        split_expression: Union[str, Callable] = r'\W+',
        bow: bool = True,
        mask_string: Optional[str] = None
    ):
        """
        Initialize an IndexedString.

        Args:
            raw_string: The text to index.
            split_expression: Either a regex pattern for splitting, or a
                            callable that takes text and returns tokens.
                            For Japanese, pass the sudachi splitter function.
            bow: If True (bag of words), all occurrences of a word share
                the same feature ID. If False, each position is separate.
            mask_string: String to use when masking words in non-bow mode.
                        Defaults to 'UNKWORDZ'.

        Example:
            >>> indexed = IndexedString("The quick brown fox")
            >>> indexed.num_words()
            4
            >>> indexed.word(0)
            'The'
        """
        self.raw = raw_string
        self.mask_string = 'UNKWORDZ' if mask_string is None else mask_string

        # Handle callable tokenizers (e.g., Japanese tokenizer)
        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            tokens_set = set(tokens)

            def non_word(string):
                return string not in tokens_set
        else:
            # Use regex splitting
            # Non-capturing group to keep separators in the split
            splitter = re.compile(r'(%s)|$' % split_expression)
            self.as_list = [s for s in splitter.split(self.raw) if s]
            non_word = splitter.match

        self.as_np = np.array(self.as_list)

        # Calculate starting position of each token
        self.string_start = np.hstack((
            [0],
            np.cumsum([len(x) for x in self.as_np[:-1]])
        ))

        # Build vocabulary and position mappings
        vocab = {}
        self.inverse_vocab: List[str] = []
        self.positions: Union[List[List[int]], np.ndarray] = []
        self.bow = bow
        non_vocab = set()

        for i, word in enumerate(self.as_np):
            # Skip words already marked as non-vocabulary
            if word in non_vocab:
                continue

            # Skip separator tokens
            if non_word(word):
                non_vocab.add(word)
                continue

            if bow:
                # Bag of words: group all occurrences
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                # Positional: each occurrence is separate
                self.inverse_vocab.append(word)
                self.positions.append(i)

        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self) -> str:
        """Return the original raw string."""
        return self.raw

    def num_words(self) -> int:
        """Return the number of unique words (features) in the vocabulary."""
        return len(self.inverse_vocab)

    def word(self, id_: int) -> str:
        """
        Get the word corresponding to a feature ID.

        Args:
            id_: Feature ID (index into vocabulary).

        Returns:
            str: The word string.
        """
        return self.inverse_vocab[id_]

    def string_position(self, id_: int) -> np.ndarray:
        """
        Get the character positions where a word occurs.

        Args:
            id_: Feature ID of the word.

        Returns:
            np.ndarray: Array of starting character positions.
        """
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove: List[int]) -> str:
        """
        Create a version of the string with specified words removed.

        This is the core perturbation operation for LIME. By removing
        words and observing how predictions change, we can determine
        each word's importance.

        Args:
            words_to_remove: List of feature IDs to remove.

        Returns:
            str: The string with specified words removed or masked.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self._get_idxs(words_to_remove)] = False

        if not self.bow:
            # Non-bow: replace with mask string instead of removing
            return ''.join([
                self.as_list[i] if mask[i] else self.mask_string
                for i in range(mask.shape[0])
            ])

        # Bow mode: actually remove the words
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    @staticmethod
    def _segment_with_tokens(text: str, tokens: List[str]) -> List[str]:
        """
        Segment text around tokens from an external tokenizer.

        This method reconstructs the original string as a list of tokens
        and inter-token strings, preserving whitespace and punctuation.

        Args:
            text: Original text string.
            tokens: List of tokens from external tokenizer.

        Returns:
            List[str]: Alternating tokens and separators.

        Raises:
            ValueError: If tokens don't match the original text.
        """
        list_form = []
        text_ptr = 0

        for token in tokens:
            inter_token = []
            # Collect characters between tokens
            while not text[text_ptr:].startswith(token):
                inter_token.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError(
                        "Tokenization produced tokens that don't match string!"
                    )
            text_ptr += len(token)

            if inter_token:
                list_form.append(''.join(inter_token))
            list_form.append(token)

        # Append any remaining text
        if text_ptr < len(text):
            list_form.append(text[text_ptr:])

        return list_form

    def _get_idxs(self, words: List[int]) -> List[int]:
        """Get indices in as_list for the given feature IDs."""
        if self.bow:
            return list(itertools.chain.from_iterable([
                self.positions[z] for z in words
            ]))
        else:
            return self.positions[words]


class IndexedCharacters:
    """
    Character-level indexing for text explanation.

    Similar to IndexedString but treats each character as a separate
    feature. This is useful for languages without clear word boundaries
    or for character-level models.

    This class is particularly relevant as a fallback for Japanese text
    when SudachiPy is not available.
    """

    def __init__(
        self,
        raw_string: str,
        bow: bool = True,
        mask_string: Optional[str] = None
    ):
        """
        Initialize character-level indexing.

        Args:
            raw_string: The text to index.
            bow: If True, same characters share feature IDs.
            mask_string: Character to use for masking. Defaults to chr(0).
        """
        self.raw = raw_string
        self.as_list = list(self.raw)
        self.as_np = np.array(self.as_list)
        self.mask_string = chr(0) if mask_string is None else mask_string
        self.string_start = np.arange(len(self.raw))

        vocab = {}
        self.inverse_vocab: List[str] = []
        self.positions: Union[List[List[int]], np.ndarray] = []
        self.bow = bow
        non_vocab = set()

        for i, char in enumerate(self.as_np):
            if char in non_vocab:
                continue

            if bow:
                if char not in vocab:
                    vocab[char] = len(vocab)
                    self.inverse_vocab.append(char)
                    self.positions.append([])
                idx_char = vocab[char]
                self.positions[idx_char].append(i)
            else:
                self.inverse_vocab.append(char)
                self.positions.append(i)

        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self) -> str:
        """Return the original raw string."""
        return self.raw

    def num_words(self) -> int:
        """Return the number of unique characters."""
        return len(self.inverse_vocab)

    def word(self, id_: int) -> str:
        """Get the character for a feature ID."""
        return self.inverse_vocab[id_]

    def string_position(self, id_: int) -> np.ndarray:
        """Get positions where a character occurs."""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove: List[int]) -> str:
        """Create string with specified characters removed/masked."""
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self._get_idxs(words_to_remove)] = False

        if not self.bow:
            return ''.join([
                self.as_list[i] if mask[i] else self.mask_string
                for i in range(mask.shape[0])
            ])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    def _get_idxs(self, words: List[int]) -> List[int]:
        """Get indices for the given feature IDs."""
        if self.bow:
            return list(itertools.chain.from_iterable([
                self.positions[z] for z in words
            ]))
        else:
            return self.positions[words]


class TextExplainer:
    """
    Explains text classification predictions using LIME.

    This is the main class for explaining text classifiers. It works by:
    1. Tokenizing the input text into words (or characters)
    2. Creating perturbations by randomly removing words
    3. Getting predictions for the perturbed texts
    4. Fitting a local linear model to understand feature importance

    The class supports both English and Japanese text, with automatic
    SudachiPy integration for Japanese tokenization.

    Attributes:
        class_names: Names of the classification classes.
        lang: Language code ('en' or 'jp').
        bow: Whether to use bag-of-words mode.

    Example:
        >>> # English text
        >>> explainer = TextExplainer(class_names=['negative', 'positive'])
        >>> exp = explainer.explain_instance(
        ...     "This movie is great!",
        ...     classifier.predict_proba
        ... )

        >>> # Japanese text
        >>> explainer = TextExplainer(class_names=['フェイク', '本物'], lang='jp')
        >>> exp = explainer.explain_instance(
        ...     "このニュースは信頼できる内容です",
        ...     classifier.predict_proba
        ... )
    """

    def __init__(
        self,
        kernel_width: float = 25,
        kernel: Optional[Callable] = None,
        verbose: bool = False,
        class_names: Optional[List[str]] = None,
        feature_selection: str = 'auto',
        split_expression: Union[str, Callable] = r'\W+',
        bow: bool = True,
        mask_string: Optional[str] = None,
        random_state=None,
        char_level: bool = False,
        lang: str = "en"
    ):
        """
        Initialize the text explainer.

        Args:
            kernel_width: Width parameter for the exponential kernel.
                         Larger values give more weight to distant samples.
            kernel: Custom kernel function. If None, uses exponential kernel
                   on cosine distance.
            verbose: If True, print debugging information.
            class_names: List of class names corresponding to classifier
                        output indices. If None, uses '0', '1', etc.
            feature_selection: Method for selecting features in explanation.
                              Options: 'auto', 'forward_selection',
                              'highest_weights', 'lasso_path', 'none'.
            split_expression: Regex pattern or callable for tokenization.
                            Ignored when lang='jp' (uses SudachiPy).
            bow: If True, use bag-of-words (word occurrences are grouped).
                If False, each word position is treated separately.
            mask_string: String to replace removed words in non-bow mode.
            random_state: Random state for reproducibility.
            char_level: If True, treat each character as a feature.
            lang: Language code. 'jp' enables Japanese tokenization.
        """
        # Set up the kernel function for weighting samples
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level
        self.lang = lang

        # Set up Japanese tokenizer if language is Japanese
        if self.lang == "jp" and not char_level:
            from .japanese import splitter, active_japanese_tokenizer
            self.split_expression = splitter

    def explain_instance(
        self,
        text_instance: str,
        classifier_fn: Callable[[List[str]], np.ndarray],
        labels: Tuple[int, ...] = (1,),
        top_labels: Optional[int] = None,
        num_features: int = 10,
        num_samples: int = 5000,
        distance_metric: str = 'cosine',
        model_regressor=None
    ) -> explanation.Explanation:
        """
        Generate an explanation for a text classification prediction.

        This method perturbs the input text by removing words, gets
        predictions for the perturbed texts, and fits a local linear
        model to determine which words are most important.

        Args:
            text_instance: The text string to explain.
            classifier_fn: Function that takes a list of strings and returns
                          prediction probabilities. Shape: (n_samples, n_classes).
                          For sklearn classifiers, use classifier.predict_proba.
            labels: Tuple of label indices to explain. Default is (1,).
            top_labels: If set, ignore labels and explain the K classes
                       with highest prediction probabilities.
            num_features: Maximum number of words in the explanation.
            num_samples: Number of perturbed samples to generate.
                        More samples = more accurate but slower.
            distance_metric: Metric for computing sample distances.
                           Default is 'cosine'.
            model_regressor: Custom sklearn regressor for the local model.
                           Defaults to Ridge regression.

        Returns:
            Explanation: Object containing the explanation data.
                        Use .as_list() to get (word, weight) pairs.

        Example:
            >>> exp = explainer.explain_instance(
            ...     "This product is amazing!",
            ...     classifier.predict_proba,
            ...     num_features=5
            ... )
            >>> for word, weight in exp.as_list(label=1):
            ...     print(f"{word}: {weight:.3f}")
        """
        # Create indexed representation of the text
        if self.char_level:
            indexed_string = IndexedCharacters(
                text_instance,
                bow=self.bow,
                mask_string=self.mask_string
            )
        else:
            indexed_string = IndexedString(
                text_instance,
                bow=self.bow,
                split_expression=self.split_expression,
                mask_string=self.mask_string
            )

        domain_mapper = TextDomainMapper(indexed_string)

        # Generate perturbed samples and get predictions
        data, yss, distances = self._data_labels_distances(
            indexed_string,
            classifier_fn,
            num_samples,
            distance_metric=distance_metric
        )

        # Set default class names if not provided
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]

        # Create the explanation object
        ret_exp = explanation.Explanation(
            domain_mapper=domain_mapper,
            class_names=self.class_names,
            random_state=self.random_state
        )
        ret_exp.predict_proba = yss[0]

        # Determine which labels to explain
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()

        # Generate explanation for each label
        for label in labels:
            (
                ret_exp.intercept[label],
                ret_exp.local_exp[label],
                ret_exp.score[label],
                ret_exp.local_pred[label]
            ) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection
            )

        return ret_exp

    def explain_instance_plain_text(
        self,
        exp: explanation.Explanation,
        label: Optional[Union[int, str]] = None,
        n_words: int = 3
    ) -> str:
        """
        Generate a plain English summary of the explanation.

        Converts the numeric explanation into a human-readable sentence
        describing which words influenced the prediction.

        Args:
            exp: Explanation object from explain_instance().
            label: Label to summarize. If None, uses the top predicted label.
            n_words: Number of top words to include in the summary.

        Returns:
            str: A sentence summarizing the explanation.

        Example:
            >>> summary = explainer.explain_instance_plain_text(exp, n_words=3)
            >>> print(summary)
            'In this text, the model positive is characterized by words such as "great, amazing, excellent".'
        """
        # Determine label index
        if label is None:
            if hasattr(exp, 'top_labels') and exp.top_labels:
                label_idx = exp.top_labels[0]
            else:
                label_idx = 0
        elif isinstance(label, int):
            label_idx = label
        else:
            # Try to find label by name
            label_idx = 0
            if exp.class_names:
                for i, name in enumerate(exp.class_names):
                    if name == label:
                        label_idx = i
                        break

        # Get features for the label
        features = exp.local_exp.get(label_idx, [])

        # Sort by absolute weight and take top n
        if features:
            features_sorted = sorted(
                features,
                key=lambda x: -abs(x[1])
            )[:n_words]
        else:
            features_sorted = []

        # Map feature IDs to words
        words = []
        if features_sorted:
            mapped = exp.domain_mapper.map_exp_ids(features_sorted, positions=False)
            words = [w for w, _ in mapped]

        label_name = exp.class_names[label_idx] if exp.class_names else str(label_idx)

        if len(words) == 0:
            return f"In this text, the model {label_name} did not return any explanatory words."

        quoted = ', '.join(words)
        return (
            f'In this text, the overall probability we can see that the model '
            f'{label_name} is characterized by the words such as "{quoted}".'
        )

    def _data_labels_distances(
        self,
        indexed_string: Union[IndexedString, IndexedCharacters],
        classifier_fn: Callable,
        num_samples: int,
        distance_metric: str = 'cosine'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate perturbed samples and compute their distances.

        This internal method creates the neighborhood data used by LIME:
        1. Generates random binary masks (which words to keep)
        2. Creates perturbed text versions by removing words
        3. Gets predictions from the classifier
        4. Computes distances from the original input

        Args:
            indexed_string: The indexed text to perturb.
            classifier_fn: The classifier's prediction function.
            num_samples: Number of perturbed samples to generate.
            distance_metric: Metric for computing distances.

        Returns:
            Tuple of (data, labels, distances):
                - data: Binary matrix (num_samples, num_features).
                       Row 0 is all 1s (original instance).
                - labels: Predictions (num_samples, num_classes).
                - distances: Distance from each sample to original.
        """
        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric
            ).ravel() * 100

        doc_size = indexed_string.num_words()

        # Generate random number of words to remove for each sample
        sample = self.random_state.randint(1, doc_size + 1, num_samples - 1)

        # Initialize binary feature matrix (1 = word present)
        data = np.ones((num_samples, doc_size))
        data[0] = np.ones(doc_size)  # Original instance has all words

        features_range = range(doc_size)
        inverse_data = [indexed_string.raw_string()]  # Start with original

        # Generate perturbed samples
        for i, size in enumerate(sample, start=1):
            # Randomly select words to remove
            inactive = self.random_state.choice(
                features_range, size, replace=False
            )
            data[i, inactive] = 0
            # Create the perturbed text string
            inverse_data.append(indexed_string.inverse_removing(inactive))

        # Get predictions for all samples
        labels = classifier_fn(inverse_data)
        labels = np.asarray(labels)

        # Compute distances using sparse representation
        distances = distance_fn(sp.sparse.csr_matrix(data))

        return data, labels, distances


# =============================================================================
# Plain Language Explanation Functions (English)
# =============================================================================

def generate_sentence_for_feature(
    word: str,
    weight: float,
    class_name: str
) -> str:
    """
    Generate an English sentence explaining a single word's influence.

    Converts a LIME word-weight pair into a natural language description
    of how that word affects the prediction probability.

    Args:
        word: The word being explained.
        weight: LIME weight (positive = increases probability).
        class_name: Name of the class being explained.

    Returns:
        str: A sentence describing the word's influence.

    Example:
        >>> generate_sentence_for_feature("excellent", 0.15, "positive")
        'The word "excellent" strongly increased the predicted probability of positive (weight = 0.150).'
    """
    direction = "increased" if weight > 0 else "decreased"
    weight_abs = abs(weight)

    # Categorize strength based on absolute weight
    if weight_abs > 0.10:
        strength = "strongly"
    elif weight_abs > 0.05:
        strength = "moderately"
    else:
        strength = "slightly"

    return (
        f'The word "{word}" {strength} {direction} '
        f'the predicted probability of {class_name} (weight = {weight:.3f}).'
    )


def summarize_lime_explanation(
    explanation_obj: explanation.Explanation,
    class_idx: int = 1
) -> List[str]:
    """
    Generate a list of English sentences summarizing the explanation.

    Creates a comprehensive summary starting with an overview of the
    most influential word, followed by individual explanations for
    each word in the explanation.

    Args:
        explanation_obj: Explanation object from TextExplainer.
        class_idx: Index of the class to explain.

    Returns:
        List[str]: Sentences describing the explanation.

    Example:
        >>> sentences = summarize_lime_explanation(exp, class_idx=1)
        >>> for s in sentences:
        ...     print(s)
    """
    # Get explanation list
    local_exp = explanation_obj.local_exp
    if isinstance(local_exp, dict):
        local = local_exp.get(class_idx, [])
    else:
        local = local_exp[class_idx]

    # Map IDs to words
    mapped = explanation_obj.domain_mapper.map_exp_ids(local, positions=False)
    exp_list = mapped

    sentences = []
    class_names = getattr(explanation_obj, 'class_names', None)
    class_name = class_names[class_idx] if class_names else str(class_idx)

    # Generate sentence for each feature
    for word, weight in exp_list:
        sentences.append(generate_sentence_for_feature(word, weight, class_name))

    if not exp_list:
        return sentences

    # Add overview sentence about most influential word
    highest_word, highest_weight = max(exp_list, key=lambda x: abs(x[1]))
    overview = (
        f'Overall, "{highest_word}" had the largest impact on the prediction '
        f'with a weight of {highest_weight:.3f}, making it the most influential term.'
    )

    return [overview] + sentences


def print_lime_narrative(
    explanation_obj: explanation.Explanation,
    class_idx: int = 1
):
    """
    Print a formatted English explanation.

    Args:
        explanation_obj: Explanation object from TextExplainer.
        class_idx: Index of the class to explain.
    """
    narrative = summarize_lime_explanation(explanation_obj, class_idx=class_idx)

    print("\nNatural-Language Explanation of LIME Output")
    print("--------------------------------------------------")
    for sent in narrative:
        print("• " + sent)


# =============================================================================
# Plain Language Explanation Functions (Japanese)
# 日本語のプレーン文生成機能
# =============================================================================

def _format_weight_jp(weight: float) -> str:
    """
    Format a weight value for Japanese display.

    Includes sign, absolute value, and strength label (強/中/弱).

    Args:
        weight: The LIME weight value.

    Returns:
        str: Formatted string like "重み=+0.150（強）"
    """
    sign = '+' if weight > 0 else ('-' if weight < 0 else '')
    abs_w = abs(weight)

    if abs_w > 0.10:
        strength = '強'
    elif abs_w > 0.05:
        strength = '中'
    else:
        strength = '弱'

    return f"重み={sign}{abs_w:.3f}（{strength}）"


def generate_sentence_for_feature_jp(
    word: str,
    weight: float,
    class_name: str
) -> str:
    """
    Generate a Japanese sentence explaining a single word's influence.

    日本語の1特徴語と重みから、自然言語の文を生成します。

    Args:
        word: The word being explained (単語).
        weight: LIME weight (重み).
        class_name: Name of the class (クラス名).

    Returns:
        str: Japanese sentence describing the influence.

    Example:
        >>> generate_sentence_for_feature_jp("地震", 0.15, "フェイク")
        '単語「地震」は大きく上げました クラス「フェイク」の予測確率（重み = 0.150）。'
    """
    direction = "上げました" if weight > 0 else "下げました"
    weight_abs = abs(weight)

    if weight_abs > 0.10:
        strength = "大きく"
    elif weight_abs > 0.05:
        strength = "中程度に"
    else:
        strength = "わずかに"

    return (
        f'単語「{word}」は{strength}{direction} '
        f'クラス「{class_name}」の予測確率（重み = {weight:.3f}）。'
    )


def summarize_lime_explanation_jp(
    explanation_obj: explanation.Explanation,
    class_idx: int = 1
) -> List[str]:
    """
    Generate a Japanese summary of the LIME explanation.

    LIMEの説明オブジェクトから日本語のプレーン文を生成します。
    This function creates a two-sentence summary:
    1. Classification result and top 3 contributing words
    2. Additional contributing words and words for the alternate class

    Args:
        explanation_obj: Explanation object from TextExplainer.
        class_idx: Index of the class to explain.

    Returns:
        List[str]: Two Japanese sentences summarizing the explanation.
    """
    probs = getattr(explanation_obj, 'predict_proba', None)
    class_names = getattr(explanation_obj, 'class_names', None)

    if probs is None:
        return ["予測確率が取得できませんでした。"]

    # Ensure probabilities are a 1-D numpy array
    probs = np.asarray(probs).ravel()

    # Find primary and secondary predicted classes
    class_1_idx = int(np.argmax(probs))

    if probs.size > 1:
        order = np.argsort(probs)[::-1]
        class_2_idx = int(order[1]) if order.size > 1 else (1 - class_1_idx)
    else:
        class_2_idx = 1 - class_1_idx

    class_1 = class_names[class_1_idx] if class_names else str(class_1_idx)
    class_2 = class_names[class_2_idx] if class_names else str(class_2_idx)

    p0 = float(probs[class_1_idx])
    p1 = float(probs[class_2_idx])

    # Helper to get features for a class
    def _get_feats(idx: int):
        local_exp = explanation_obj.local_exp
        feats = local_exp.get(idx, []) if isinstance(local_exp, dict) else local_exp[idx]

        if feats:
            return feats

        # Fallback for binary classification: invert weights from other class
        if class_names and len(class_names) == 2:
            available_keys = list(local_exp.keys())
            if len(available_keys) == 1:
                other_idx = available_keys[0]
                if other_idx != idx:
                    return [(fid, -weight) for fid, weight in local_exp[other_idx]]
        return []

    feats_1 = _get_feats(class_1_idx)
    feats_2 = _get_feats(class_2_idx)

    mapper = getattr(explanation_obj, 'domain_mapper', None)

    def _map(feats):
        if not feats:
            return []
        return mapper.map_exp_ids(feats, positions=False)

    mapped_1 = _map(feats_1)
    mapped_2 = _map(feats_2)

    # Select top features with preference for positive weights
    def _select_features(mapped_feats, n: int = 3, exclude_words=None):
        if not mapped_feats:
            return []
        if exclude_words is None:
            exclude_words = set()

        candidates = [(w, wt) for w, wt in mapped_feats if w not in exclude_words]

        # Prefer positive weight words (increase probability)
        positives = sorted(
            [(w, wt) for w, wt in candidates if wt > 0],
            key=lambda x: -abs(x[1])
        )

        if len(positives) >= n:
            return positives[:n]

        # Fill remaining with highest absolute weight
        all_sorted = sorted(candidates, key=lambda x: -abs(x[1]))
        result = list(positives)
        seen = {w for w, _ in result}

        for w, wt in all_sorted:
            if len(result) >= n:
                break
            if w not in seen:
                result.append((w, wt))
                seen.add(w)

        return result

    # Select features for both sentences
    top3_1 = _select_features(mapped_1, n=3)
    exclude_for_next = {w for w, _ in top3_1}
    next3_1 = _select_features(mapped_1, n=3, exclude_words=exclude_for_next)
    top3_2 = _select_features(mapped_2, n=3)

    # Pad lists if needed
    def _pad_list(items, n: int = 3):
        padded = list(items)
        while len(padded) < n:
            padded.append(("-", 0.0))
        return padded

    top3_1 = _pad_list(top3_1, 3)
    next3_1 = _pad_list(next3_1, 3)
    top3_2 = _pad_list(top3_2, 3)

    # Format weights
    t1w0 = _format_weight_jp(top3_1[0][1])
    t1w1 = _format_weight_jp(top3_1[1][1])
    t1w2 = _format_weight_jp(top3_1[2][1])

    n1w0 = _format_weight_jp(next3_1[0][1])
    n1w1 = _format_weight_jp(next3_1[1][1])
    n1w2 = _format_weight_jp(next3_1[2][1])

    t2w0 = _format_weight_jp(top3_2[0][1])
    t2w1 = _format_weight_jp(top3_2[1][1])
    t2w2 = _format_weight_jp(top3_2[2][1])

    # Build sentences
    sent1 = (
        f"このインスタンスは{p0:.3f}対{p1:.3f}で{class_1}と分類されました。"
        f"{class_1}への分類に最も強い影響を与えた言葉は{top3_1[0][0]}, {top3_1[1][0]}, {top3_1[2][0]}で、"
        f"それぞれの{t1w0}, {t1w1}, {t1w2}となっています。"
    )

    sent2 = (
        f"他に{class_1}への分類の確率を上げた言葉として{next3_1[0][0]} ({n1w0})、"
        f"{next3_1[1][0]} ({n1w1})、{next3_1[2][0]} ({n1w2})などが挙げられます。"
        f"{class_2}への分類への確率を上げた言葉として、{top3_2[0][0]} ({t2w0})、"
        f"{top3_2[1][0]} ({t2w1})、{top3_2[2][0]} ({t2w2})などが挙げられます。"
    )

    return [sent1, sent2]


def print_lime_narrative_jp(
    explanation_obj: explanation.Explanation,
    class_idx: int = 1
):
    """
    Print a formatted Japanese explanation.

    日本語の説明ブロックを整形して出力します。

    Args:
        explanation_obj: Explanation object from TextExplainer.
        class_idx: Index of the class to explain.
    """
    narrative = summarize_lime_explanation_jp(explanation_obj, class_idx=class_idx)

    print("\nLIME出力の自然言語による説明")
    print("--------------------------------------------------")
    for sent in narrative:
        print("・ " + sent)
