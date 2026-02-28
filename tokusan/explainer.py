"""Text classifier explanation using LIME with Japanese text support."""

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
from .exceptions import ExplanationError


class TextDomainMapper(explanation.DomainMapper):
    """Maps feature IDs to words for text explanations."""

    def __init__(self, indexed_string: 'IndexedString'):
        """Initialize with an IndexedString containing the vocabulary."""
        self.indexed_string = indexed_string

    def map_exp_ids(
        self,
        exp: List[Tuple[int, float]],
        positions: bool = False
    ) -> List[Tuple[str, float]]:
        """Convert feature IDs to word strings, optionally including positions."""
        if positions:
            result = []
            for feature_id, weight in exp:
                word = self.indexed_string.word(feature_id)
                positions_list = self.indexed_string.string_position(feature_id)
                position_str = '-'.join(map(str, positions_list))
                result.append((f'{word}_{position_str}', weight))
            return result
        else:
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
        """Generate JavaScript for highlighting words by importance weight."""
        if not text:
            return ''

        raw_text = self.indexed_string.raw_string()
        text_escaped = raw_text.encode('utf-8', 'xmlcharrefreplace').decode('utf-8')
        text_escaped = re.sub(r'[<>&]', '|', text_escaped)

        word_data = [
            (
                self.indexed_string.word(x[0]),
                self.indexed_string.string_position(x[0]),
                x[1]
            )
            for x in exp
        ]

        all_occurrences = list(itertools.chain.from_iterable([
            itertools.product([w[0]], w[1], [w[2]])
            for w in word_data
        ]))
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]

        return f'''
            {exp_object_name}.show_raw_text({json.dumps(all_occurrences)}, {label}, {json.dumps(text_escaped)}, {div_name}, {json.dumps(opacity)});
        '''


class IndexedString:
    """A string with word-level indexing for efficient LIME perturbation."""

    def __init__(
        self,
        raw_string: str,
        split_expression: Union[str, Callable] = r'\W+',
        bow: bool = True,
        mask_string: Optional[str] = None,
        stopwords: Optional[set] = None
    ):
        """Index the string by tokenizing and building vocabulary mappings."""
        self.raw = raw_string
        self.mask_string = 'UNKWORDZ' if mask_string is None else mask_string

        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            tokens_set = set(tokens)

            def non_word(string):
                return string not in tokens_set
        else:
            splitter = re.compile(r'(%s)|$' % split_expression)
            self.as_list = [s for s in splitter.split(self.raw) if s]
            non_word = splitter.match

        self.as_np = np.array(self.as_list)

        self.string_start = np.hstack((
            [0],
            np.cumsum([len(x) for x in self.as_np[:-1]])
        ))

        vocab = {}
        self.inverse_vocab: List[str] = []
        self.positions: Union[List[List[int]], np.ndarray] = []
        self.bow = bow
        non_vocab = set()

        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if stopwords and word in stopwords:
                non_vocab.add(word)
                continue

            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)

        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self) -> str:
        """Return the original raw string."""
        return self.raw

    def num_words(self) -> int:
        """Return the number of unique words in the vocabulary."""
        return len(self.inverse_vocab)

    def word(self, id_: int) -> str:
        """Get the word for a feature ID."""
        return self.inverse_vocab[id_]

    def string_position(self, id_: int) -> np.ndarray:
        """Get character positions where a word occurs."""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove: List[int]) -> str:
        """Create a version of the string with specified words removed or masked."""
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self._get_idxs(words_to_remove)] = False

        if not self.bow:
            return ''.join([
                self.as_list[i] if mask[i] else self.mask_string
                for i in range(mask.shape[0])
            ])

        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    @staticmethod
    def _segment_with_tokens(text: str, tokens: List[str]) -> List[str]:
        """Reconstruct text as alternating tokens and separators."""
        list_form = []
        text_ptr = 0

        for token in tokens:
            inter_token = []
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

        if text_ptr < len(text):
            list_form.append(text[text_ptr:])

        return list_form

    def _get_idxs(self, words: List[int]) -> List[int]:
        """Get as_list indices for the given feature IDs."""
        if self.bow:
            return list(itertools.chain.from_iterable([
                self.positions[z] for z in words
            ]))
        else:
            return self.positions[words]


class IndexedCharacters:
    """Character-level indexing for text explanation (fallback when no tokenizer)."""

    def __init__(
        self,
        raw_string: str,
        bow: bool = True,
        mask_string: Optional[str] = None
    ):
        """Index each character as a separate feature."""
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
        """Create string with specified characters removed or masked."""
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
    """Explains text classification predictions using LIME perturbation."""

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
        lang: str = "en",
        stopwords: Optional[set] = None
    ):
        """Initialize with kernel, tokenizer, and language settings."""
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

        # Use SudachiPy for Japanese
        if self.lang == "jp" and not char_level:
            from .japanese import splitter, active_japanese_tokenizer
            self.split_expression = splitter

        # Auto-load Japanese stopwords if none provided
        if stopwords is not None:
            self.stopwords = stopwords
        elif self.lang == "jp":
            from .japanese import JAPANESE_STOPWORDS
            self.stopwords = JAPANESE_STOPWORDS
        else:
            self.stopwords = None

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
        """Generate a LIME explanation by perturbing the text and fitting a local model."""
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
                mask_string=self.mask_string,
                stopwords=self.stopwords
            )

        domain_mapper = TextDomainMapper(indexed_string)

        data, yss, distances = self._data_labels_distances(
            indexed_string,
            classifier_fn,
            num_samples,
            distance_metric=distance_metric
        )

        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]

        ret_exp = explanation.Explanation(
            domain_mapper=domain_mapper,
            class_names=self.class_names,
            random_state=self.random_state
        )
        ret_exp.predict_proba = yss[0]

        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()

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
        """Generate a plain English sentence summarizing which words influenced the prediction."""
        if label is None:
            if hasattr(exp, 'top_labels') and exp.top_labels:
                label_idx = exp.top_labels[0]
            else:
                label_idx = 0
        elif isinstance(label, int):
            label_idx = label
        else:
            label_idx = 0
            if exp.class_names:
                for i, name in enumerate(exp.class_names):
                    if name == label:
                        label_idx = i
                        break

        features = exp.local_exp.get(label_idx, [])

        if features:
            features_sorted = sorted(
                features,
                key=lambda x: -abs(x[1])
            )[:n_words]
        else:
            features_sorted = []

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
        """Generate perturbed samples, get predictions, and compute distances."""
        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric
            ).ravel() * 100

        doc_size = indexed_string.num_words()

        if doc_size == 0:
            raise ExplanationError(
                "テキストに分析可能な単語が見つかりませんでした。"
                "ストップワードや記号のみで構成されるテキストは分析できません。"
                "意味のある単語を含む、より長いテキストを入力してください。"
            )

        sample = self.random_state.randint(1, doc_size + 1, num_samples - 1)

        data = np.ones((num_samples, doc_size))
        data[0] = np.ones(doc_size)

        features_range = range(doc_size)
        inverse_data = [indexed_string.raw_string()]

        for i, size in enumerate(sample, start=1):
            inactive = self.random_state.choice(
                features_range, size, replace=False
            )
            data[i, inactive] = 0
            inverse_data.append(indexed_string.inverse_removing(inactive))

        labels = classifier_fn(inverse_data)
        labels = np.asarray(labels)

        distances = distance_fn(sp.sparse.csr_matrix(data))

        return data, labels, distances



def generate_sentence_for_feature(
    word: str,
    weight: float,
    class_name: str
) -> str:
    """Generate an English sentence explaining a single word's influence on the prediction."""
    direction = "increased" if weight > 0 else "decreased"
    weight_abs = abs(weight)

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
    class_idx: int = 1,
    stopwords: Optional[set] = None
) -> List[str]:
    """Generate English sentences summarizing the LIME explanation for a class."""
    local_exp = explanation_obj.local_exp
    if isinstance(local_exp, dict):
        local = local_exp.get(class_idx, [])
    else:
        local = local_exp[class_idx]

    mapped = explanation_obj.domain_mapper.map_exp_ids(local, positions=False)

    if stopwords:
        mapped = [(w, wt) for w, wt in mapped if w not in stopwords]

    exp_list = mapped

    sentences = []
    class_names = getattr(explanation_obj, 'class_names', None)
    class_name = class_names[class_idx] if class_names else str(class_idx)

    for word, weight in exp_list:
        sentences.append(generate_sentence_for_feature(word, weight, class_name))

    if not exp_list:
        return sentences

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
    """Print a formatted English LIME explanation."""
    narrative = summarize_lime_explanation(explanation_obj, class_idx=class_idx)

    print("\nNatural-Language Explanation of LIME Output")
    print("--------------------------------------------------")
    for sent in narrative:
        print("• " + sent)



def _format_weight_jp(weight: float) -> str:
    """Format a weight value with sign and strength label (強/中/弱)."""
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
    """Generate a Japanese sentence explaining a single word's influence on the prediction."""
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
    class_idx: int = 1,
    stopwords: Optional[set] = None
) -> List[str]:
    """Generate a variable-length Japanese summary of the LIME explanation."""
    probs = getattr(explanation_obj, 'predict_proba', None)
    class_names = getattr(explanation_obj, 'class_names', None)

    if probs is None:
        return ["予測確率が取得できませんでした。"]

    probs = np.asarray(probs).ravel()

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

    def _get_feats(idx: int):
        local_exp = explanation_obj.local_exp
        feats = local_exp.get(idx, []) if isinstance(local_exp, dict) else local_exp[idx]

        if feats:
            return feats

        # Binary classification fallback: invert weights from other class
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

    if stopwords:
        mapped_1 = [(w, wt) for w, wt in mapped_1 if w not in stopwords]
        mapped_2 = [(w, wt) for w, wt in mapped_2 if w not in stopwords]

    def _select_features(mapped_feats, n: int = 3, exclude_words=None):
        """Select top features, preferring positive weights."""
        if not mapped_feats:
            return []
        if exclude_words is None:
            exclude_words = set()

        candidates = [(w, wt) for w, wt in mapped_feats if w not in exclude_words]

        if stopwords:
            candidates = [(w, wt) for w, wt in candidates if w not in stopwords]

        positives = sorted(
            [(w, wt) for w, wt in candidates if wt > 0],
            key=lambda x: -abs(x[1])
        )

        if len(positives) >= n:
            return positives[:n]

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

    top3_1 = _select_features(mapped_1, n=3)
    exclude_for_next = {w for w, _ in top3_1}
    next3_1 = _select_features(mapped_1, n=3, exclude_words=exclude_for_next)
    top3_2 = _select_features(mapped_2, n=3)

    def _format_word_list(items):
        """Join (word, weight) pairs as 'word1 (weight1)、word2 (weight2)'."""
        if not items:
            return ""
        parts = [f"{word} ({_format_weight_jp(weight)})" for word, weight in items]
        return "、".join(parts)

    if top3_1:
        words_part = "、".join(word for word, _ in top3_1)
        weights_part = "、".join(_format_weight_jp(weight) for _, weight in top3_1)
        sent1 = (
            f"このインスタンスは{p0:.3f}対{p1:.3f}で{class_1}と分類されました。"
            f"{class_1}への分類に最も強い影響を与えた言葉は{words_part}で、"
            f"それぞれの{weights_part}となっています。"
        )
    else:
        sent1 = f"このインスタンスは{p0:.3f}対{p1:.3f}で{class_1}と分類されました。"

    sentences = [sent1]

    sent2_parts = []
    if next3_1:
        next_list = _format_word_list(next3_1)
        sent2_parts.append(f"他に{class_1}への分類の確率を上げた言葉として{next_list}などが挙げられます。")
    if top3_2:
        top2_list = _format_word_list(top3_2)
        sent2_parts.append(f"{class_2}への分類への確率を上げた言葉として、{top2_list}などが挙げられます。")

    if sent2_parts:
        sentences.append("".join(sent2_parts))

    return sentences


def print_lime_narrative_jp(
    explanation_obj: explanation.Explanation,
    class_idx: int = 1
):
    """Print a formatted Japanese LIME explanation."""
    narrative = summarize_lime_explanation_jp(explanation_obj, class_idx=class_idx)

    print("\nLIME出力の自然言語による説明")
    print("--------------------------------------------------")
    for sent in narrative:
        print("・ " + sent)
