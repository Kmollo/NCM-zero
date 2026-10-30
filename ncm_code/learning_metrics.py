"""Pure, strict reconstruction measurements for bounded codec diagnostics."""
from .data import EOS, PAD, VOCAB, encode
from .evaluate import score_ids


def _edit_distance(expected, observed):
    """Levenshtein distance over byte/token IDs, retaining misplaced PAD."""
    previous = list(range(len(observed) + 1))
    for i, expected_id in enumerate(expected, 1):
        current = [i]
        for j, observed_id in enumerate(observed, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (expected_id != observed_id)))
        previous = current
    return previous[-1]


def score_reconstructions(generated, rows, output_length):
    """Score fixed-width predictions without counting padding as correct bytes.

    ``byte_accuracy`` covers the expected UTF-8 byte positions only; early EOS
    and PAD are mismatches at those positions. ``active_accuracy`` also covers
    the expected EOS. Both are None when their denominator is zero. Strict
    exactness, UTF-8 validity and parsing come from the standard artifact scorer.
    ``padding_valid`` measures the region after the expected EOS independently
    from whether the EOS itself was correct.

    Counts, widths and token domains are validated before scoring any case.
    Content edit distance retains every token before the first EOS, including
    misplaced PAD, so malformed output cannot appear correct after filtering.
    This is an in-memory measurement only: no model execution or file writes.
    """
    if type(output_length) is not int or output_length < 1:
        raise ValueError('output_length must be a positive integer')
    if not isinstance(generated, list) or not isinstance(rows, list):
        raise ValueError('generated and rows must be lists')
    if len(generated) != len(rows):
        raise ValueError('prediction and row counts must match')
    expected_sequences = []
    for index, (ids, row) in enumerate(zip(generated, rows)):
        if not isinstance(ids, list) or len(ids) != output_length:
            raise ValueError(f'case {index}: prediction must have exactly {output_length} IDs')
        if any(type(token) is not int or not 0 <= token < VOCAB for token in ids):
            raise ValueError(f'case {index}: token IDs must be exact integers in [0, {VOCAB})')
        if not isinstance(row, dict) or not isinstance(row.get('completion'), str):
            raise ValueError(f'case {index}: completion must be a string')
        expected_sequences.append(encode(row['completion'], output_length))

    cases = []
    for ids, row, expected in zip(generated, rows, expected_sequences):
        expected_bytes = list(row['completion'].encode('utf-8'))
        byte_count = len(expected_bytes)
        first_eos = ids.index(EOS) if EOS in ids else None
        content_ids = ids[:first_eos] if first_eos is not None else ids[:]
        case = score_ids(ids, row)
        case.update(
            predicted_ids=ids[:], expected_ids=expected, expected_bytes=expected_bytes,
            expected_byte_count=byte_count,
            predicted_byte_count=sum(token < 256 for token in content_ids),
            content_token_count=len(content_ids),
            content_special_token_count=sum(token >= 256 for token in content_ids),
            content_token_edit_distance=_edit_distance(expected_bytes, content_ids),
            first_eos_index=first_eos,
            eos_correct=first_eos == byte_count and PAD not in ids[:byte_count],
            padding_valid=all(token == PAD for token in ids[byte_count + 1:]),
            byte_correct=sum(ids[i] == token for i, token in enumerate(expected_bytes)),
            byte_total=byte_count,
            active_correct=sum(ids[i] == expected[i] for i in range(byte_count + 1)),
            active_total=byte_count + 1,
        )
        case['byte_accuracy'] = case['byte_correct'] / byte_count if byte_count else None
        case['active_accuracy'] = case['active_correct'] / case['active_total']
        cases.append(case)

    report = {
        'count': len(cases),
        'exact': sum(case['exact'] for case in cases),
        'valid_outputs': sum(case['valid_output'] for case in cases),
        'parses': sum(case['parses'] for case in cases),
        'eos_correct': sum(case['eos_correct'] for case in cases),
        'padding_valid': sum(case['padding_valid'] for case in cases),
        'cases': cases,
    }
    for field in ('byte_correct', 'byte_total', 'active_correct', 'active_total'):
        report[field] = sum(case[field] for case in cases)
    for prefix in ('byte', 'active'):
        total = report[f'{prefix}_total']
        report[f'{prefix}_accuracy'] = report[f'{prefix}_correct'] / total if total else None
    return report
