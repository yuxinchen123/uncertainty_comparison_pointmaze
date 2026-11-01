"""Inspect fixed CPU wheels and default package dependencies without importing numerical code."""

from email import policy
from email.parser import BytesParser
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile


VERSIONS = {'torch': '2.6.0+cpu', 'filelock': '4.2.4', 'networkx': '3.7', 'jinja2': '3.1.6',
    'sympy': '1.13.1', 'mpmath': '1.3.0', 'markupsafe': '3.0.3'}


def normalized(name):
    """Use the standard package-name equivalence without importing the package itself."""
    return re.sub(r'[-_.]+', '-', name).lower()


def wheel_metadata(row):
    """Bind exact downloaded wheel bytes, safe paths, package metadata and declared ABI tags."""
    path = Path(row['path'])
    if not path.is_absolute() or path != path.resolve(strict=True) or not path.is_file():
        raise ValueError('a wheel must be its exact ordinary absolute file path')
    hasher, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
            size += len(chunk)
    if hasher.hexdigest() != row['sha256'] or size != row['bytes']:
        raise ValueError('downloaded wheel bytes differ from their official lock')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError('a wheel contains duplicate archive paths')
        for entry in entries:
            name = PurePosixPath(entry.filename.rstrip('/'))
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if name.is_absolute() or '..' in name.parts or str(name) != entry.filename.rstrip('/') or kind == stat.S_IFLNK:
                raise ValueError('a wheel contains an unsafe or linked archive path')
            if name.suffix == '.pth' or name.name in ('sitecustomize.py', 'usercustomize.py'):
                raise ValueError('a wheel contains a Python startup hook')
        metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
        wheel_names = [name for name in names if name.endswith('.dist-info/WHEEL')]
        if len(metadata_names) != 1 or len(wheel_names) != 1 or metadata_names[0].rsplit('/', 1)[0] != wheel_names[0].rsplit('/', 1)[0]:
            raise ValueError('a wheel needs its one exact paired metadata directory')
        raw = archive.read(metadata_names[0])
        message = BytesParser(policy=policy.default).parsebytes(raw)
        wheel = BytesParser(policy=policy.default).parsebytes(archive.read(wheel_names[0]))
        name = normalized(message['Name'])
        if name not in VERSIONS or message['Version'] != VERSIONS[name] or normalized(row['project']) != name:
            raise ValueError('a wheel changed its exact reference package/version')
        expected_tags = ({'cp312-cp312-linux_x86_64'} if name == 'torch' else
            {'cp312-cp312-manylinux_2_17_x86_64', 'cp312-cp312-manylinux2014_x86_64', 'cp312-cp312-manylinux_2_28_x86_64'}
            if name == 'markupsafe' else {'py3-none-any'})
        tags = wheel.get_all('Tag', [])
        if set(tags) != expected_tags or len(tags) != len(expected_tags):
            raise ValueError('a reference wheel has a different exact CP312/Linux or pure-Python tag set')
        if name == 'torch' and any(Path(name).name.startswith(('libtorch_cuda', 'libcudart', 'libcudnn', 'libcublas')) for name in names):
            raise ValueError('the CPU reference wheel unexpectedly contains CUDA libraries')
        return {'name': name, 'version': message['Version'], 'requires_dist': message.get_all('Requires-Dist', []),
            'requires_python': message.get('Requires-Python'), 'tags': wheel.get_all('Tag', []),
            'metadata_sha256': hashlib.sha256(raw).hexdigest(), 'wheel_sha256': row['sha256'], 'filename': path.name,
            'wheel_file_count': len(entries), 'startup_hooks': [], 'symlinks': []}


def dependency_check(distributions, environment):
    """Validate every active default dependency using PEP508 markers and PEP440 versions."""
    from packaging.requirements import Requirement
    from packaging.version import Version
    rows = {}
    for row in distributions:
        name = normalized(row['name'])
        if name in rows:
            raise ValueError('effective distribution names must be unique: ' + name)
        rows[name] = row
    requested = {name: set() for name in rows}
    parsed = {name: [(text, Requirement(text)) for text in row.get('requires_dist', [])] for name, row in rows.items()}
    # Before: orbax requests etils[epath,epy]; after: etils' five required extra dependencies are active too.
    changed = True
    while changed:
        changed = False
        for name in sorted(rows):
            contexts = ['', *sorted(requested[name])]
            for text, requirement in parsed[name]:
                if requirement.marker is not None and not any(requirement.marker.evaluate({**environment, 'extra': extra}) for extra in contexts):
                    continue
                if requirement.url is not None:
                    raise ValueError('a URL dependency requires an additional exact source binding')
                provider = normalized(requirement.name)
                if provider in requested:
                    additions = {normalized(extra) for extra in requirement.extras} - requested[provider]
                    if additions:
                        requested[provider].update(additions)
                        changed = True
    failures, checked = [], []
    for name in sorted(rows):
        contexts = ['', *sorted(requested[name])]
        for text, requirement in parsed[name]:
            if requirement.marker is not None and not any(requirement.marker.evaluate({**environment, 'extra': extra}) for extra in contexts):
                continue
            dependency = normalized(requirement.name)
            actual = rows.get(dependency, {}).get('version')
            accepted = actual is not None and (not requirement.specifier or Version(actual) in requirement.specifier)
            record = {'package': name, 'requirement': text, 'provider': dependency, 'version': actual, 'passed': accepted}
            checked.append(record)
            if not accepted:
                failures.append(record)
    if failures:
        raise ValueError('the complete proposed default dependency closure is invalid: ' + json.dumps(failures))
    if any(name in sys.modules for name in ('jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax')):
        raise ValueError('metadata dependency validation imported a numerical module')
    return {'passed': True, 'distribution_count': len(rows), 'active_dependency_count': len(checked), 'requested_extras': {name: sorted(extras) for name, extras in requested.items() if extras}, 'dependencies': checked}


def proposed_closure(base, wheels, environment):
    """Replace only the seven approved names in the observed base metadata, then check all requirements."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    rows = {normalized(row['name']): row for row in base}
    if len(rows) != len(base):
        raise ValueError('observed base distribution names must be unique after normalization')
    if {row['name'] for row in wheels} != set(VERSIONS) or len(wheels) != 7:
        raise ValueError('the seven exact CPU/reference wheels are required')
    for row in wheels:
        if row['requires_python'] and Version(environment['python_full_version']) not in SpecifierSet(row['requires_python']):
            raise ValueError('a reference wheel excludes the actual target Python version')
        rows[row['name']] = row
    return dependency_check(list(rows.values()), environment)


def snapshot(hash_shared=False):
    """Capture actual interpreter/providers and optionally all shared numerical package bytes without imports."""
    from importlib import metadata
    import platform
    import sysconfig
    from packaging.markers import default_environment
    from packaging.tags import sys_tags
    if not sys.dont_write_bytecode:
        raise ValueError('metadata probes must use -B to preserve base bytecode files')
    distributions, effective, metadata_files = [], {}, {}
    for distribution in metadata.distributions():
        name = normalized(distribution.metadata['Name'])
        path = Path(distribution._path).resolve(strict=True)
        row = {'name': name, 'version': distribution.version, 'requires_dist': distribution.requires or [],
            'metadata_path': str(path)}
        distributions.append(row)
        if name not in effective:
            effective[name] = row
        for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if file.is_file() and file.suffix != '.pyc' and '__pycache__' not in file.parts:
                metadata_files[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    shared = {}
    if hash_shared:
        packages = ('jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt', 'nvidia-cudnn-cu12',
            'nvidia-cublas-cu12', 'nvidia-cuda-runtime-cu12', 'jaxtari', 'flax', 'optax', 'numpy', 'packaging', 'pip')
        for name in packages:
            distribution = metadata.distribution(name)
            for relative in distribution.files or []:
                if Path(relative).suffix == '.pyc' or '__pycache__' in Path(relative).parts:
                    continue
                path = Path(distribution.locate_file(relative)).resolve(strict=True)
                if not path.is_file() or not path.is_relative_to(Path(sys.base_prefix).resolve()):
                    raise ValueError('a shared package file escapes the exact base prefix')
                hasher, size = hashlib.sha256(), 0
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        hasher.update(chunk)
                        size += len(chunk)
                shared[str(path)] = {'sha256': hasher.hexdigest(), 'bytes': size}
    if any(name in sys.modules for name in ('jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax')):
        raise ValueError('a metadata snapshot imported a numerical module')
    return {'python': sys.executable, 'version': platform.python_version(), 'prefix': sys.prefix,
        'base_prefix': sys.base_prefix, 'purelib': sysconfig.get_path('purelib'), 'sys_path': sys.path,
        'dont_write_bytecode': sys.dont_write_bytecode, 'environment': default_environment(),
        'supported_tags': [str(tag) for tag in sys_tags()], 'distributions': sorted(distributions, key=lambda row: (row['name'], row['metadata_path'])),
        'effective': effective, 'metadata_files': metadata_files, 'shared_files': shared, 'numerical_imports': []}


def installed_wheel_files(row, purelib):
    """Rehash installed original wheel entries, excluding only the installer's rewritten RECORD."""
    root = Path(purelib).resolve(strict=True)
    results = {}
    with zipfile.ZipFile(row['path']) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename.endswith('.dist-info/RECORD'):
                continue
            if '.data/' in info.filename:
                scheme = info.filename.split('.data/', 1)[1]
                if scheme.startswith('data/'):
                    target = root.parents[2] / scheme[len('data/'):]
                elif scheme.startswith(('purelib/', 'platlib/')):
                    target = root / scheme.split('/', 1)[1]
                else:
                    raise ValueError('an exact wheel uses an unsupported private installation scheme')
            else:
                target = root / info.filename
            if not target.is_relative_to(root.parents[2]):
                raise ValueError('an installed wheel data entry escapes the private prefix')
            if not target.is_file() or target.is_symlink() or target.resolve(strict=True) != target:
                raise ValueError('an installed wheel entry is not its exact ordinary private file')
            first, second = hashlib.sha256(), hashlib.sha256()
            with archive.open(info) as source, target.open('rb') as actual:
                for stream, hasher in ((source, first), (actual, second)):
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        hasher.update(chunk)
            if first.digest() != second.digest():
                raise ValueError('an installed private wheel file differs from its locked bytes')
            results[str(target)] = second.hexdigest()
    return {'wheel_sha256': row['sha256'], 'verified_files': results}
