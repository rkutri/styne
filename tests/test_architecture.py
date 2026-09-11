from pathlib import Path
import re


SOURCE_ROOT = Path(__file__).parents[1] / 'src' / 'styne'
BACKEND_ROOT = SOURCE_ROOT / 'backend'


def source_files():
    return [
        path for path in SOURCE_ROOT.rglob('*.py')
        if BACKEND_ROOT not in path.parents
    ]


def test_optional_backends_are_isolated():
    pattern = re.compile(r'^(?:from|import)\s+(?:torch|jax)\b', re.MULTILINE)

    offenders = [
        path.relative_to(SOURCE_ROOT) for path in source_files()
        if pattern.search(path.read_text(encoding='utf-8'))
    ]

    assert not offenders


def test_graph_paths_do_not_extract_backend_scalars():
    forbidden = ('.numpy(', '.item(', '.detach(')

    offenders = [
        path.relative_to(SOURCE_ROOT) for path in source_files()
        if any(token in path.read_text(encoding='utf-8') for token in forbidden)
    ]

    assert not offenders
