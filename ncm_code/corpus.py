"""Small project-authored mechanism corpus, not general coding pretraining data.

Each recipe is a family. All variants stay together; normalized AST/JSON
fingerprints additionally join near-duplicate families before splitting.
"""
import ast
import json
import random
from pathlib import Path
from .artifacts import atomic_json, digest, file_hash
from .data import encode

PROVENANCE = {
    'kind': 'project-authored-synthetic',
    'origin': 'Written for Zero in this workspace; no benchmark solutions or third-party corpus imported.',
    'rights_basis': 'Project-authored material for this internal experiment; no third-party license claim.',
    'redistribution_license': 'Unassigned; decide before public dataset redistribution.',
    'limitations': 'Short synthetic programs and mock tool envelopes; not evidence of general coding or agent competence.'
}


def recipes():
    # (family, expression, literal input expression, expected-value factory).
    return [
        ('affine', 'x * {n} + 3', '7', lambda n: 7*n+3),
        ('quadratic', 'x*x + {n}*x - 2', '4', lambda n: 14+4*n),
        ('bit_mask', 'x & ((1 << {n}) - 1)', '12345', lambda n: 12345 & ((1 << n)-1)),
        ('bit_rotate', '((x << {n}) | (x >> (24-{n}))) & 16777215', '12345', lambda n: ((12345 << n)|(12345 >> (24-n))) & 16777215),
        ('divmod', 'divmod(x, {n})', '47', lambda n: divmod(47,n)),
        ('power', 'x ** {n}', '2', lambda n: 2**n),
        ('slice_stride', 'x[::{n}]', '"abcdefghijk"', lambda n: 'abcdefghijk'[::n]),
        ('reverse_suffix', 'x[::-1] + "tag{n}"', '"zero"', lambda n: 'oreztag'+str(n)),
        ('unicode_marker', 'x + "\\u03bb{n}\\U0001f680"', '"go"', lambda n: 'go\u03bb'+str(n)+'\U0001f680'),
        ('count_threshold', 'sum(v > {n} for v in x)', '[0, 5, 21]', lambda n: sum(v>n for v in [0,5,21])),
        ('map_shift', '[v + {n} for v in x]', '[1, 4, 9]', lambda n: [1+n,4+n,9+n]),
        ('pair_products', '[a*b + {n} for a,b in x]', '[(2, 3), (4, 5)]', lambda n: [6+n,20+n]),
        ('enumerate_dict', '{str(i): v for i,v in enumerate(x, {n})}', '"ab"', lambda n: {str(n):'a',str(n+1):'b'}),
        ('join_fields', '"sep{n}".join(x)', '["a", "b", "c"]', lambda n: ('sep'+str(n)).join(['a','b','c'])),
        ('strip_edges', 'x.strip("x{n}")', '"xxxhelloxxx"', lambda n: 'hello'),
        ('prefix_flag', 'x.startswith("item{n}:")', '"item1:ok"', lambda n: n==1),
        ('sum_squares', 'sum(v*v for v in x) + {n}', '[2, 3]', lambda n: 13+n),
        ('nested_lookup', 'x.get("key{n}", {}).get("value", -1)', '{}', lambda n: -1),
    ]


def authored_rows(variants=20):
    rows = []
    for family, expression, argument, expected in recipes():
        for n in range(1, variants+1):
            # Replace only the numeric placeholder; Python dictionary braces stay literal.
            expr = expression.replace('{n}', str(n))
            completion = f'def solve(x):\n    return {expr}\n'
            rows.append({'id': f'{family}-{n:02}', 'family': family, 'kind': 'python',
                         'prompt': f'Write Python solve(x) returning {expr}. Output only code.',
                         'completion': completion,
                         'verification': f'assert solve({argument}) == {expected(n)!r}'})
    tools = {
        'read_file': lambda n: {'path': f'src/unit_{n}.py'},
        'search_docs': lambda n: {'query': f'widget {n} usage', 'limit': n},
        'run_tests': lambda n: {'suite': f'unit_{n}', 'timeout_seconds': 30},
        'save_note': lambda n: {'key': f'checkpoint_{n}', 'text': f'verified phase {n}; next: inspect'},
        'inspect_issue': lambda n: {'project': 'demo', 'number': n},
        'list_artifacts': lambda n: {'cursor': f'page_{n}', 'include_archived': False},
    }
    for name, arguments in tools.items():
        for n in range(1, variants+1):
            payload = {'tool': name, 'arguments': arguments(n)}
            completion = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
            rows.append({'id': f'{name}-{n:02}', 'family': name, 'kind': 'tool_json',
                         'prompt': f'Emit only the mock tool JSON for {name} with arguments '+json.dumps(arguments(n), separators=(',', ':')),
                         'completion': completion, 'verification': payload})
    return rows


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id='_', ctx=node.ctx), node)

    def visit_arg(self, node):
        node.arg = '_'
        return node

    def visit_FunctionDef(self, node):
        node.name = '_'
        return self.generic_visit(node)

    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)


def fingerprint(row):
    if row['kind'] == 'python':
        return digest(ast.dump(Normalize().visit(ast.parse(row['completion'])), include_attributes=False))
    value = json.loads(row['completion'])
    # Preserve tool and field names; abstract changing leaf values.
    def shape(x):
        if isinstance(x, dict): return {k: shape(v) for k,v in x.items()}
        if isinstance(x, list): return [shape(v) for v in x]
        return type(x).__name__
    return digest({'tool': value['tool'], 'arguments': shape(value['arguments'])})


def prepare(output, prefix_length=256, output_length=256, seed=17):
    output = Path(output)
    if output.exists():
        raise ValueError('Dataset destination already exists; use a new version')
    rows = authored_rows()
    groups = {}
    for row in rows:
        encode(row['prompt'], prefix_length)
        encode(row['completion'], output_length)
        fp = fingerprint(row)
        groups.setdefault(fp, set()).add(row['family'])
    # Connected components join any recipes with overlapping normalized shapes.
    families = {r['family'] for r in rows}
    parent = {f:f for f in families}
    def find(f):
        while parent[f] != f:
            f = parent[f]
        return f
    for family_set in groups.values():
        first = min(family_set)
        for f in family_set:
            parent[find(f)] = find(first)
    components = sorted({find(f) for f in families})
    random.Random(seed).shuffle(components)
    held = max(1, len(components)//6)
    if len(components) < 6:
        raise ValueError('Insufficient independent groups')
    assigned = {c: ('test' if i < held else 'development' if i < 2*held else 'train')
                for i,c in enumerate(components)}
    splits = {name: [r for r in rows if assigned[find(r['family'])] == name]
              for name in ('train','development','test')}
    output.mkdir(parents=True)
    (output/'authoring_recipe.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(output/'provenance.json', PROVENANCE)
    manifest = {'schema_version': 2, 'seed': seed, 'prefix_length': prefix_length,
                'output_length': output_length, 'authoring_recipe_sha256': file_hash(__file__),
                'provenance_sha256': file_hash(output/'provenance.json'), 'splits': {},
                'benchmark_tasks_imported': False, 'training_started': False,
                'grouping': 'recipe family plus normalized AST / tool-schema connected components'}
    for name, records in splits.items():
        path = output/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8', newline='\n')
        manifest['splits'][name] = {'records': len(records), 'sha256': file_hash(path),
                                  'families': sorted({r['family'] for r in records}),
                                  'max_prompt_bytes_with_eos': max(len(r['prompt'].encode())+1 for r in records),
                                  'max_completion_bytes_with_eos': max(len(r['completion'].encode())+1 for r in records)}
    atomic_json(output/'manifest.json', manifest)
    load_bundle(output)
    return manifest


def load_bundle(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 2 or set(manifest['splits']) != {'train','development','test'}:
        raise ValueError('Expected version-2 train/development/test manifest')
    if file_hash(path/'provenance.json') != manifest['provenance_sha256']:
        raise ValueError('Provenance hash mismatch')
    if file_hash(path/'authoring_recipe.py') != manifest['authoring_recipe_sha256']:
        raise ValueError('Authoring recipe archive hash mismatch')
    seen_ids, families, fingerprints, exact = set(), {}, {}, {}
    splits = {}
    for name, spec in manifest['splits'].items():
        source = path/f'{name}.jsonl'
        if file_hash(source) != spec['sha256']:
            raise ValueError(f'{name} data hash mismatch')
        rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line]
        if len(rows) != spec['records'] or not rows:
            raise ValueError('Empty or inconsistent split')
        if sorted({r['family'] for r in rows}) != spec['families']:
            raise ValueError('Family manifest mismatch')
        for row in rows:
            if row['id'] in seen_ids or row['kind'] not in ('python', 'tool_json'):
                raise ValueError('Duplicate ID or unsupported format')
            seen_ids.add(row['id'])
            encode(row['prompt'], manifest['prefix_length'])
            encode(row['completion'], manifest['output_length'])
            for table, key in ((families,row['family']), (fingerprints,fingerprint(row)),
                               (exact,digest(row['completion']))):
                if key in table and table[key] != name:
                    raise ValueError('Family or duplicate crosses data splits')
                table[key] = name
        splits[name] = rows
    return manifest, splits


def verify_bundle(path):
    """Verify authored fixtures in WASI. Does not execute generated code on host."""
    from zero_baseline.sandbox import run_python
    manifest, splits = load_bundle(path)
    rows = sum(splits.values(), [])
    python_rows = [r for r in rows if r['kind']=='python']
    for row in rows:
        if row['kind']=='tool_json' and json.loads(row['completion']) != row['verification']:
            raise ValueError('Tool fixture verification failed')
    probe = ('import json\nrows=json.loads('+repr(json.dumps(python_rows))+')\n'
             'for row in rows:\n    scope={}\n    exec(row["completion"],scope)\n'
             '    exec(row["verification"],scope)\nprint("ZERO_CORPUS_VERIFIED",len(rows))\n')
    result = run_python(probe, {}, timeout=30)
    passed = result['exit_code']==0 and f'ZERO_CORPUS_VERIFIED {len(python_rows)}' in result['stdout']
    report = {'passed': passed, 'manifest_sha256': file_hash(Path(path)/'manifest.json'),
              'python_fixtures': len(python_rows), 'tool_json_fixtures':len(rows)-len(python_rows),
              'sandbox': result, 'training': False}
    atomic_json(Path(path)/'verification.json', report)
    if not passed:
        raise ValueError(f'Corpus verification failed: {result}')
    return report
