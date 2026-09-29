"""Configuration and quantile conventions for adaptive post-processing."""
from copy import deepcopy
from numbers import Integral, Real

import numpy as np

PERCENTILE_CATALOGUE = [1.] + np.arange(2.5, 100., 2.5).tolist() + [99., 100.]
OPERATION_SELECTORS = {'grouping_distance': 'grouping_percentiles', 'closing_radius': 'closing_percentiles',
                       'hole_volume': 'filling_percentiles', 'min_volume': 'size_percentiles',
                       'max_count': 'count_percentiles'}
DEFAULTS = {'hyperparameter_search': 'greedy', 'grouping_percentiles': [10, 25, 50],
            'closing_percentiles': [25, 50, 90], 'filling_percentiles': [25, 50, 90],
            'size_percentiles': [1, 5, 10], 'count_percentiles': [90, 95, 100],
            'hierarchy_repair': 'auto', 'connectivity': 26, 'aggregation': 'case'}
QUANTILE_CONVENTIONS = {'instance_continuous': 'numpy linear',
                        'case_continuous': 'linear interpolation of normalized midpoint cumulative weights',
                        'counts': 'inverse empirical CDF; one observation per positive case'}


def resolve_percentiles(value):
    if value is None:
        return []
    if isinstance(value, bool):
        raise ValueError('Percentile selectors cannot be booleans.')
    if isinstance(value, Integral):
        if value == 0 or abs(value) > len(PERCENTILE_CATALOGUE):
            raise ValueError('Integer percentile selectors must be nonzero and fit the percentile catalogue.')
        return PERCENTILE_CATALOGUE[:abs(value)] if value < 0 else PERCENTILE_CATALOGUE[-value:]
    if isinstance(value, (str, bytes, dict)):
        raise ValueError('Percentile selectors must be null, a nonzero integer, or a numeric iterable.')
    try:
        values = list(value)
    except TypeError as error:
        raise ValueError('Percentile selectors must be null, a nonzero integer, or a numeric iterable.') from error
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or not 0 <= v <= 100
           for v in values):
        raise ValueError('Percentile levels must be finite numbers within [0, 100].')
    return sorted(set(map(float, values)))


def resolve_configuration(configuration=None):
    if configuration is not None and not isinstance(configuration, dict):
        raise ValueError('post_processing must be a dictionary.')
    configuration = {} if configuration is None else configuration
    unknown = set(configuration) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'Unknown post_processing settings: {sorted(unknown)}')
    result = deepcopy(DEFAULTS)
    # Selectors are normalized below; do not deepcopy one-shot numeric iterables.
    result.update(configuration)
    for key, allowed in [('hyperparameter_search', ('greedy', 'exhaustive')),
                         ('hierarchy_repair', ('auto', 'parent', 'child')), ('aggregation', ('case', 'instance'))]:
        if result[key] not in allowed:
            raise ValueError(f'Invalid post_processing.{key}: expected one of {allowed}.')
    if (isinstance(result['connectivity'], bool) or not isinstance(result['connectivity'], Integral)
        or result['connectivity'] not in (6, 8, 18, 26)):
        raise ValueError('post_processing.connectivity must be 6, 8, 18, or 26.')
    result['connectivity'] = int(result['connectivity'])
    for key in OPERATION_SELECTORS.values():
        if result[key] is not None:
            result[key] = resolve_percentiles(result[key])
    return result


def distribution_quantiles(per_case, percentiles, aggregation='instance', discrete=False):
    """Return quantiles of raw observations, never averages of case quantiles."""
    eligible = [np.asarray(v, dtype=float) for v in per_case if len(v)]
    if not eligible or not len(percentiles):
        return np.empty(0)
    values = np.concatenate(eligible)
    if aggregation == 'instance':
        return np.percentile(values, percentiles, method='inverted_cdf' if discrete else 'linear')
    if aggregation != 'case':
        raise ValueError(f'Unknown aggregation: {aggregation}')
    weights = np.concatenate([np.full(len(v), 1. / len(v)) for v in eligible])
    order = np.argsort(values, kind='stable')
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    probabilities = np.asarray(percentiles, dtype=float) / 100.
    if discrete:
        indices = np.searchsorted(cumulative / cumulative[-1], probabilities, side='left')
        return values[np.minimum(indices, len(values) - 1)]
    if len(values) == 1:
        return np.full(len(probabilities), values[0])
    midpoints = cumulative - weights / 2.
    positions = (midpoints - midpoints[0]) / (midpoints[-1] - midpoints[0])
    return np.interp(probabilities, positions, values)


def percentile_table(per_case, aggregation='instance', discrete=False):
    values = distribution_quantiles(per_case, PERCENTILE_CATALOGUE, aggregation, discrete)
    return {str(p): int(v) if discrete else float(v) for p, v in zip(PERCENTILE_CATALOGUE, values)}


def distribution_candidates(per_case, percentiles, aggregation='instance', discrete=False, scale=1.):
    values = distribution_quantiles(per_case, percentiles, aggregation, discrete) * scale
    return [None] + sorted(set(int(v) if discrete else float(v) for v in values))
