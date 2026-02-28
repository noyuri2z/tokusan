"""Explanation class and domain mapping utilities for tokusan."""

import json
import string
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.utils import check_random_state

from .exceptions import ExplanationError


def _generate_random_id(size: int = 15, random_state=None) -> str:
    """Generate a random alphanumeric ID string for HTML div elements."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))


class DomainMapper:
    """Abstract base class for mapping feature IDs to domain-specific names."""

    def __init__(self):
        pass

    def map_exp_ids(
        self,
        exp: List[Tuple[int, float]],
        **kwargs
    ) -> List[Tuple[Any, float]]:
        """Map feature IDs to human-readable names (default: identity)."""
        return exp

    def visualize_instance_html(
        self,
        exp: List[Tuple[int, float]],
        label: int,
        div_name: str,
        exp_object_name: str,
        **kwargs
    ) -> str:
        """Generate HTML/JavaScript for visualizing the explained instance."""
        return ''


class Explanation:
    """Container for LIME explanation results with visualization support."""

    def __init__(
        self,
        domain_mapper: DomainMapper,
        mode: str = 'classification',
        class_names: Optional[List[str]] = None,
        random_state=None
    ):
        """Initialize an Explanation object."""
        self.random_state = random_state
        self.mode = mode
        self.domain_mapper = domain_mapper

        self.local_exp: Dict[int, List[Tuple[int, float]]] = {}
        self.intercept: Dict[int, float] = {}
        self.score: Dict[int, float] = {}
        self.local_pred: Dict[int, float] = {}

        if mode == 'classification':
            self.class_names = class_names
            self.top_labels: Optional[List[int]] = None
            self.predict_proba: Optional[np.ndarray] = None
        elif mode == 'regression':
            self.class_names = ['negative', 'positive']
            self.predicted_value: Optional[float] = None
            self.min_value: float = 0.0
            self.max_value: float = 1.0
            self.dummy_label: int = 1
        else:
            raise ExplanationError(
                f'Invalid explanation mode "{mode}". '
                'Should be either "classification" or "regression".'
            )

    def available_labels(self) -> List[int]:
        """Get label indices for which explanations are available."""
        if self.mode != "classification":
            raise NotImplementedError(
                'Not supported for regression explanations.'
            )

        if self.top_labels:
            return list(self.top_labels)
        return list(self.local_exp.keys())

    def as_list(
        self,
        label: int = 1,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Get the explanation as a list of (feature_name, weight) tuples."""
        label_to_use = label if self.mode == "classification" else self.dummy_label
        mapped_exp = self.domain_mapper.map_exp_ids(
            self.local_exp[label_to_use], **kwargs
        )
        return [(x[0], float(x[1])) for x in mapped_exp]

    def as_map(self) -> Dict[int, List[Tuple[int, float]]]:
        """Get the raw explanation map of label -> [(feature_id, weight)]."""
        return self.local_exp

    def as_pyplot_figure(
        self,
        label: int = 1,
        figsize: Tuple[int, int] = (4, 4),
        **kwargs
    ):
        """Create a matplotlib bar chart of feature importances."""
        import matplotlib.pyplot as plt

        exp = self.as_list(label=label, **kwargs)
        fig = plt.figure(figsize=figsize)

        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()

        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + 0.5

        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)

        if self.mode == "classification":
            title = f'Local explanation for class {self.class_names[label]}'
        else:
            title = 'Local explanation'
        plt.title(title)

        return fig

    def show_in_notebook(
        self,
        labels: Optional[List[int]] = None,
        predict_proba: bool = True,
        show_predicted_value: bool = True,
        **kwargs
    ):
        """Display the explanation in a Jupyter notebook."""
        from IPython.core.display import display, HTML
        display(HTML(self.as_html(
            labels=labels,
            predict_proba=predict_proba,
            show_predicted_value=show_predicted_value,
            **kwargs
        )))

    def save_to_file(
        self,
        file_path: str,
        labels: Optional[List[int]] = None,
        predict_proba: bool = True,
        show_predicted_value: bool = True,
        **kwargs
    ):
        """Save the explanation as an HTML file."""
        with open(file_path, 'w', encoding='utf8') as file:
            file.write(self.as_html(
                labels=labels,
                predict_proba=predict_proba,
                show_predicted_value=show_predicted_value,
                **kwargs
            ))

    def as_html(
        self,
        labels: Optional[List[int]] = None,
        predict_proba: bool = True,
        show_predicted_value: bool = True,
        **kwargs
    ) -> str:
        """Generate a self-contained HTML page with explanation visualizations."""
        def jsonize(x):
            return json.dumps(x, ensure_ascii=False)

        if labels is None and self.mode == "classification":
            labels = self.available_labels()

        random_state = check_random_state(self.random_state)
        random_id = _generate_random_id(size=15, random_state=random_state)

        out = '''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head></head><body>'''
        out += f'''
        <div class="tokusan top_div" id="top_div{random_id}"></div>
        '''

        predict_proba_js = ''
        if self.mode == "classification" and predict_proba and self.predict_proba is not None:
            class_names_json = jsonize([str(x) for x in self.class_names])
            proba_json = jsonize(list(self.predict_proba.astype(float)))
            predict_proba_js = f'''
            var pp_div = top_div.append('div').classed('tokusan predict_proba', true);
            var pp_text = pp_div.append('p').text('Prediction probabilities: {proba_json}');
            '''

        exp_js = f'''var exp_div;'''

        if self.mode == "classification":
            for label in labels:
                exp_data = jsonize(self.as_list(label))
                exp_js += f'''
                exp_div = top_div.append('div').classed('tokusan explanation', true);
                exp_div.append('h3').text('Class: {self.class_names[label]}');
                exp_div.append('pre').text({exp_data});
                '''
        else:
            exp_data = jsonize(self.as_list())
            exp_js += f'''
            exp_div = top_div.append('div').classed('tokusan explanation', true);
            exp_div.append('pre').text({exp_data});
            '''

        out += f'''
        <script>
        var top_div = document.getElementById('top_div{random_id}');
        top_div.innerHTML = '';
        {predict_proba_js}
        {exp_js}
        </script>
        '''
        out += '</body></html>'

        return out
