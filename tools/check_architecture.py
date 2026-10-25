"""Static checks for ArrayLab's canonical module boundaries.

Includes imports inside functions and literal importlib.import_module calls.
Tests and verification code are excluded: their job is to exercise composition.
Dynamic module names supplied by the application registry are reported, not
guessed. This checker does not execute application or numerical code.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ('arraylab.framework', 'arraylab.shared', 'arraylab.features', 'arraylab.application')


def module_name(root, path):
    parts = path.relative_to(root).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def excluded(module):
    return bool({'tests', 'verification'} & set(module.split('.')))


def _package(module, path):
    return module if path.name == '__init__.py' else module.rpartition('.')[0]


def _relative(name, level, package):
    if not level:
        return name or ''
    parts = package.split('.')
    prefix = parts[:len(parts)-level+1]
    return '.'.join(prefix + ([name] if name else []))


def dependencies(tree, module, path, modules):
    """Resolve module targets without importing them or invoking __getattr__."""
    package = _package(module, path)
    rows, unresolved = [], []
    importlib_names = {'importlib'}
    import_module_names = set()
    module_bindings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'importlib': importlib_names.add(alias.asname or alias.name)
                module_bindings[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module == 'importlib' and not node.level:
            import_module_names.update(alias.asname or alias.name for alias in node.names if alias.name == 'import_module')
        if isinstance(node, ast.ImportFrom):
            base = _relative(node.module, node.level, package)
            for alias in node.names:
                candidate = base + '.' + alias.name
                if candidate in modules: module_bindings[alias.asname or alias.name] = candidate

    def qualified(node):
        if isinstance(node, ast.Name): return module_bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            parent = qualified(node.value)
            if parent: return parent + '.' + node.attr
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            rows.extend((alias.name, node.lineno, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _relative(node.module, node.level, package)
            for alias in node.names:
                candidate = base + '.' + alias.name
                target = candidate if candidate in modules else base
                rows.append((target, node.lineno, alias.name))
        elif isinstance(node, ast.Call):
            function = node.func
            is_import = (isinstance(function, ast.Name) and function.id in import_module_names) or (
                isinstance(function, ast.Attribute) and function.attr == 'import_module' and
                isinstance(function.value, ast.Name) and function.value.id in importlib_names)
            if not is_import:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                unresolved.append(node.lineno)
                continue
            name = node.args[0].value
            if name.startswith('.'):
                explicit = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == 'package'), None)
                if isinstance(explicit, ast.Constant) and isinstance(explicit.value, str):
                    parent = explicit.value
                elif isinstance(explicit, ast.Name) and explicit.id == '__name__':
                    parent = module
                else:
                    parent = package
                count = len(name)-len(name.lstrip('.'))
                name = _relative(name[count:], count, parent)
            rows.append((name, node.lineno, None))
        elif isinstance(node, ast.Attribute):
            name = qualified(node)
            if not name or not name.startswith('arraylab.'):
                continue
            parts = name.split('.')
            for end in range(len(parts), 0, -1):
                target = '.'.join(parts[:end])
                if target in modules:
                    rows.append((target, node.lineno, parts[end] if end < len(parts) else None))
                    break
    return sorted(set(rows), key=lambda row: (row[0], row[1], row[2] or '')), sorted(set(unresolved))


def _feature(module):
    parts = module.split('.')
    return parts[2] if len(parts) >= 3 and parts[:2] == ['arraylab', 'features'] else None


def _within(module, prefixes):
    return any(module == prefix or module.startswith(prefix + '.') for prefix in prefixes)


def boundary_rule(source, target, symbol=None):
    if not target.startswith('arraylab.'):
        return None
    canonical_source = _within(source, CANONICAL)
    if canonical_source and not _within(target, CANONICAL) and target != 'arraylab.__init__':
        return 'legacy_import'
    if _within(source, ('arraylab.framework',)) and _within(target, ('arraylab.features', 'arraylab.application', 'arraylab.shared')):
        return 'framework_depends_on_higher_layer'
    if _within(source, ('arraylab.shared',)) and _within(target, ('arraylab.features', 'arraylab.application')):
        return 'shared_depends_on_business'
    feature = _feature(source)
    if feature:
        if _within(target, ('arraylab.application',)):
            return 'business_depends_on_application'
        other = _feature(target)
        if other and other != feature:
            if target != 'arraylab.features.' + other:
                return 'cross_feature_internal_import'
            if symbol is not None and symbol.startswith('_'):
                return 'cross_feature_private_symbol'
    return None


def strongly_connected(graph):
    index = 0; indices = {}; low = {}; stack = []; on_stack = set(); cycles = []
    def visit(node):
        nonlocal index
        indices[node] = low[node] = index; index += 1
        stack.append(node); on_stack.add(node)
        for target in sorted(graph[node]):
            if target not in indices:
                visit(target); low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            component = []
            while True:
                target = stack.pop(); on_stack.remove(target); component.append(target)
                if target == node: break
            if len(component) > 1 or node in graph[node]: cycles.append(sorted(component))
    for node in sorted(graph):
        if node not in indices: visit(node)
    return sorted(cycles)


def inspect_project(root=ROOT):
    root = Path(root).resolve()
    files = {module_name(root, p): p for p in (root/'arraylab').rglob('*.py')}
    files = {name: path for name, path in files.items() if not excluded(name)}
    modules = set(files)
    graph = {name: set() for name in files}
    violations, dynamic = [], []
    for name, path in sorted(files.items()):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        imports, unresolved = dependencies(tree, name, path, modules)
        for line in unresolved:
            dynamic.append({'module': name, 'path': str(path.relative_to(root)), 'line': line})
        for target, line, symbol in imports:
            if target in modules and target != name:
                graph[name].add(target)
            rule = boundary_rule(name, target, symbol)
            if rule:
                violations.append({'rule': rule, 'module': name, 'dependency': target,
                                   'symbol': symbol, 'path': str(path.relative_to(root)), 'line': line})
    cycles = strongly_connected(graph)
    return {'success': not violations and not cycles, 'checked_modules': len(files),
            'import_edges': sum(map(len, graph.values())), 'violations': violations,
            'cycles': cycles, 'unresolved_dynamic_imports': dynamic,
            'scope': 'Static runtime imports including delayed and literal dynamic imports; tests/verification excluded'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--report', type=Path)
    p.add_argument('--json', action='store_true')
    args = p.parse_args(argv); result = inspect_project(args.root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Checked {result['checked_modules']} modules / {result['import_edges']} import edges")
        for issue in result['violations']:
            print(f"{issue['path']}:{issue['line']} [{issue['rule']}] {issue['module']} -> {issue['dependency']}")
        for cycle in result['cycles']:
            print('[cycle] ' + ' -> '.join(cycle))
        print('Architecture boundaries passed' if result['success'] else 'Architecture boundaries failed')
    return 0 if result['success'] else 1


if __name__ == '__main__': raise SystemExit(main())
