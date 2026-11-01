"""Read installed Slurm metadata without executing a daemon or loading a plugin.

This collector is deliberately separate from every qualification profile. Its
caller must first produce the reviewed, actual zero-GPU allocation observation.
No command in this module executes slurmd, slurmstepd, a package tool, ldd, or a
management library. Missing, changed and unsupported files remain unresolved.
"""

import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import os
from pathlib import Path
import re
import shlex
import stat
import struct
import subprocess
import zlib

import uva_active_configuration_observation as ownership


KIND = 'passive_installed_slurm_metadata_observation'
CAPABILITY = 'bounded_passive_installed_slurm_metadata_version_1'
DEPENDENCIES = ('uva_active_configuration_observation.py',)
SETTINGS = ('SLURM_CONF', 'SlurmdSpoolDir', 'TaskPlugin', 'ProctrackType', 'GresTypes',
    'PluginDir', 'SlurmUser', 'SlurmdUser', 'SlurmdLogFile', 'SlurmdPidFile',
    'CgroupPlugin', 'CgroupMountpoint', 'ConstrainDevices', 'ConstrainCores',
    'ConstrainRAMSpace', 'EnableControllers', 'IgnoreSystemd', 'IgnoreSystemdOnFailure')
FILE_SETTINGS = (*SETTINGS, 'Include', 'NodeName', 'Gres', 'AutoDetect', 'Name', 'Type', 'File', 'Count')
PROGRAMS = ('slurmd', 'slurmstepd', 'scontrol')
SYSTEMD_ROOTS = ('/etc/systemd/system', '/run/systemd/system', '/usr/lib/systemd/system', '/lib/systemd/system')
SYSTEMD_UNITS = ('slurmd.service', 'slurm-slurmd.service')
PROGRAM_ROOTS = ('/usr', '/opt/slurm', '/bin', '/sbin', '/lib', '/lib64')
PLUGIN_NAMES = ('gres_gpu.so', 'gpu_nvml.so', 'gpu_generic.so', 'cgroup_v1.so', 'cgroup_v2.so',
    'cgroup_none.so', 'auth_munge.so', 'auth_slurm.so', 'hash_k12.so')
LIMITS = {'symlink_hops': 16, 'include_files': 64, 'configuration_bytes_per_file': 1024 * 1024,
    'configuration_total_bytes': 64 * 1024 * 1024, 'unit_bytes': 64 * 1024, 'dropin_files': 32,
    'path_directories': 32, 'program_candidates': 32, 'binary_bytes': 32 * 1024 * 1024,
    'plugin_directories': 8, 'manual_bytes': 2 * 1024 * 1024, 'package_database_bytes': 16 * 1024 * 1024,
    'passwd_bytes': 1024 * 1024, 'query_bytes': 1024 * 1024, 'total_read_bytes': 192 * 1024 * 1024}
ZERO_CREDIT = {**ownership.ZERO_CREDIT, 'physical_gres_mapping_verified': False,
    'configured_isolation_qualified': False, 'installed_binary_matches_upstream_verified': False}


def require(condition, message):
    """Stop an unsupported operation without silently changing its scope."""
    if not condition:
        raise ValueError(message)


def digest(raw):
    """Hash exact bytes before any filtering or parsing."""
    return hashlib.sha256(raw).hexdigest()


def lexical_path(value):
    """Accept only an ordinary absolute spelling without traversal or control characters."""
    require(isinstance(value, (str, Path)), 'a literal absolute metadata path is required')
    value = str(value)
    path = Path(value)
    require(len(value) <= 4096 and len(path.parts) <= 64 and path.is_absolute()
        and str(path) == value and '..' not in path.parts
        and not any(ord(character) < 32 for character in value), 'unsupported metadata path spelling')
    return path


def within(path, roots):
    """Keep a path in its specifically declared installed-file namespace."""
    return any(path == Path(root) or Path(root) in path.parents for root in roots)


def resolve_links(path, roots):
    """Record a bounded symlink chain before pinning its final ordinary inode."""
    path = lexical_path(path)
    require(within(path, roots), 'metadata path leaves its allowed namespace')
    links, seen = [], set()
    for unused in range(LIMITS['symlink_hops'] + 1):
        # Example: /sbin/slurmd -> /usr/sbin/slurmd; every parent link is retained too.
        changed = False
        for candidate in reversed((path, *path.parents)):
            value = candidate.lstat()
            if not stat.S_ISLNK(value.st_mode):
                continue
            require(len(links) < LIMITS['symlink_hops'] and candidate not in seen, 'symlink cycle or hop limit')
            target = os.readlink(candidate)
            next_path = Path(os.path.normpath(str(candidate.parent / target)))
            next_path = lexical_path(next_path)
            links.append({'path': str(candidate), 'target': target, 'owner_uid': value.st_uid,
                'mode': stat.S_IMODE(value.st_mode), 'inode': value.st_ino, 'device': value.st_dev})
            seen.add(candidate)
            path = next_path / path.relative_to(candidate)
            require(within(path, roots), 'resolved installed-file path leaves its allowed namespace')
            changed = True
            break
        if not changed:
            return path, links
    raise ValueError('installed-file symlink resolution did not finish')


def observed_links_after(links):
    """Retain each original alias's later identity rather than assuming it stayed unchanged."""
    records = []
    for link in links:
        row = {'path': link['path']}
        try:
            path = Path(link['path'])
            value = path.lstat()
            row.update(target=os.readlink(path), owner_uid=value.st_uid, mode=stat.S_IMODE(value.st_mode),
                inode=value.st_ino, device=value.st_dev)
            row['unchanged'] = all(row[name] == expected for name, expected in link.items())
        except OSError as error:
            row.update(unchanged=False, error_type=type(error).__name__, error=str(error))
        records.append(row)
    return records


class PassiveReader:
    """Bound every read and keep ownership tied to the inode actually read."""

    def __init__(self):
        """Start one collection budget without reading any installed file."""
        self.bytes_read = 0

    def read(self, path, roots, limit, retain=False):
        """Read only a pinned regular inode and retain exact failure or change facts."""
        row = {'requested_path': str(path), 'read_status': 'unavailable'}
        raw = None
        try:
            # An O_PATH handle cannot open a substituted character device for reading.
            path, links = resolve_links(path, roots)
            row.update(path=str(path), symlink_chain=links)
            require(self.bytes_read < LIMITS['total_read_bytes'], 'total passive-read byte limit reached')
            limit = min(limit, LIMITS['total_read_bytes'] - self.bytes_read)
            trace = row['read_trace'] = {}
            value = ownership.pinned_configuration_bytes(path, trace, limit)
            self.bytes_read += trace.get('bytes_read', 0)
            row['symlink_chain_after'] = observed_links_after(links)
            row.update(trace.get('descriptor_before', {}))
            row.update(directory_chain=trace.get('pinned_directory_chain', []), sha256=digest(value))
            if not trace['complete_file']:
                row.update(read_status='size_limit_exceeded', captured_prefix_byte_count=len(value))
            elif (trace['inode_changed_during_read'] or trace['path_replaced_during_read']
                    or any(link['unchanged'] is not True for link in row['symlink_chain_after'])):
                row['read_status'] = 'changed_during_read'
            else:
                row.update(read_status='read', captured_byte_count=len(value))
                raw = value
                if retain:
                    row['raw_base64'] = base64.b64encode(value).decode('ascii')
        except (OSError, ValueError, RuntimeError) as error:
            row.update(error_type=type(error).__name__, error=str(error))
        return row, raw


def setting_rows(raw, names):
    """Retain only non-secret assignments, preserving spelling, order and duplicates."""
    allowed = {name.lower() for name in names}
    rows = []
    for number, line in enumerate(raw.decode('utf-8', errors='strict').splitlines(), 1):
        # Before: 'TaskPlugin = task/cgroup # note'; after: ordered key/value plus source line.
        for match in re.finditer(r'(?:^|\s)([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^\s#]+)', line.split('#', 1)[0]):
            if match[1].lower() in allowed:
                rows.append({'line': number, 'name': match[1], 'value': match[2]})
    return rows


def controller_query(command, allowed_names):
    """Keep only allowed fields from the exact read-only controller request."""
    require(command in (['scontrol', 'show', 'config'], ['scontrol', 'show', 'node', 'jaguar03', '-o']),
            'unsupported controller command')
    result = subprocess.run(command, capture_output=True, timeout=20, check=False)
    record = {'command': command, 'returncode': result.returncode,
        'stdout_sha256': digest(result.stdout), 'stdout_byte_count': len(result.stdout),
        'stderr_sha256': digest(result.stderr), 'stderr_byte_count': len(result.stderr)}
    require(len(result.stdout) <= LIMITS['query_bytes'] and len(result.stderr) <= LIMITS['query_bytes'],
            'controller response exceeded its observation limit')
    try:
        record.update(fields=setting_rows(result.stdout, allowed_names), filter_status='parsed')
    except UnicodeError as error:
        record.update(fields=[], filter_status='unsupported_text_encoding', error_type=type(error).__name__)
    record['authority'] = 'controller_reported_values_not_local_file_or_binary_attestation'
    return record


def unique_setting(rows, name):
    """Return one nonempty setting only when the controller supplied it unambiguously."""
    values = [row['value'] for row in rows if row['name'].lower() == name.lower()]
    return values[0] if len(values) == 1 else None


def configuration_tree(reader, first, roots):
    """Read a bounded literal Include tree while preserving order and unresolved syntax."""
    records, includes, visited, stack, unresolved = [], [], set(), set(), []
    total = 0

    def visit(path, origin=None):
        """Visit one explicit include path without expanding unknown shell or glob syntax."""
        nonlocal total
        if len(records) >= LIMITS['include_files'] or total >= LIMITS['configuration_total_bytes']:
            unresolved.append({'path': str(path), 'reason': 'configuration_tree_limit'})
            return
        row, raw = reader.read(path, roots, min(LIMITS['configuration_bytes_per_file'],
            LIMITS['configuration_total_bytes'] - total))
        row.update(origin=origin, assignments=[], includes=[])
        records.append(row)
        if raw is None:
            unresolved.append({'path': str(path), 'reason': row['read_status']})
            return
        total += len(raw)
        canonical = row['path']
        if canonical in stack:
            unresolved.append({'path': str(path), 'reason': 'include_cycle'})
            return
        if canonical in visited:
            unresolved.append({'path': str(path), 'reason': 'repeated_include_not_collapsed'})
            return
        visited.add(canonical)
        stack.add(canonical)
        try:
            row['assignments'] = setting_rows(raw, FILE_SETTINGS)
            lines = raw.decode('utf-8').splitlines()
        except UnicodeError as error:
            row.update(parse_status='unsupported_text_encoding', error_type=type(error).__name__)
            unresolved.append({'path': canonical, 'reason': row['parse_status']})
            stack.remove(canonical)
            return
        for number, line in enumerate(lines, 1):
            match = re.match(r'(?i)^\s*include(?:\s+|\s*=\s*)(.*?)\s*(?:#.*)?$', line)
            if not match:
                continue
            token = match[1].strip()
            include = {'source_path': canonical, 'line': number, 'token': token}
            row['includes'].append(include)
            includes.append(include)
            # Examples: 'Include /etc/slurm/nodes.conf' is read; '*.conf' stays unresolved.
            if (not re.fullmatch(r'/[A-Za-z0-9_./-]+', token) or '..' in Path(token).parts
                    or not within(Path(token), roots)):
                include['status'] = 'unsupported_literal_or_namespace'
                unresolved.append({'path': canonical, 'line': number, 'reason': include['status']})
                continue
            include['status'] = 'literal_child_observed'
            visit(Path(token), {'path': canonical, 'line': number})
        stack.remove(canonical)

    # Read the active main file and its explicit include order without assuming root ownership.
    visit(first)
    return {'files': records, 'include_order': includes, 'unresolved': unresolved,
        'complete': not unresolved, 'captured_input_bytes': total, 'qualification_granted': False,
        'filtered_excerpts_not_effective_configuration': True}


def directory_entries(path, roots, pattern, limit):
    """List one ordinary installed directory without recursively scanning its descendants."""
    row = {'requested_path': str(path), 'entries': []}
    try:
        path, links = resolve_links(path, roots)
        row.update(path=str(path), symlink_chain=links)
        descriptors = [os.open('/', os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)]
        try:
            # Reject a parent replacement rather than following its new symlink during listing.
            for component in path.parts[1:]:
                descriptors.append(os.open(component, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=descriptors[-1]))
            descriptor = descriptors[-1]
            before = ownership.descriptor_metadata(descriptor, path)
            # The descriptor alias keeps listing attached to this verified directory inode.
            names = []
            with os.scandir('/proc/self/fd/' + str(descriptor)) as entries:
                for number, entry in enumerate(entries, 1):
                    require(number <= 512, 'installed-directory examined-entry limit exceeded')
                    if re.fullmatch(pattern, entry.name):
                        names.append(entry.name)
                        if len(names) > limit:
                            raise ValueError('installed-directory entry limit exceeded')
            row.update(status='listed', descriptor=before, entries=sorted(names))
            require(before == ownership.descriptor_metadata(descriptor, path), 'installed directory changed while listed')
            after = path.lstat()
            require((after.st_dev, after.st_ino) == (before['device'], before['inode']), 'installed directory path replaced')
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
    except (OSError, ValueError, RuntimeError) as error:
        row.update(status='unavailable', error_type=type(error).__name__, error=str(error), entries=[])
    return row


def service_tokens(raw):
    """Read safe unit execution metadata without evaluating substitutions or exposing environment secrets."""
    rows, section = [], ''
    try:
        lines = raw.decode('utf-8').splitlines()
    except UnicodeError as error:
        return [{'status': 'unsupported_text_encoding', 'error_type': type(error).__name__}]
    for number, line in enumerate(lines, 1):
        if re.fullmatch(r'\s*\[[A-Za-z]+\]\s*', line):
            section = line.strip()
            continue
        match = re.match(r'^\s*(ExecStart|User|Group|EnvironmentFile)\s*=\s*(.*?)\s*$', line)
        if section != '[Service]' or not match:
            continue
        name, value = match.groups()
        record = {'line': number, 'name': name, 'value_sha256': digest(value.encode())}
        if name in {'User', 'Group'} and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*|[0-9]+', value):
            record['value'] = value
        elif name == 'EnvironmentFile':
            record['status'] = 'not_read_environment_may_contain_secrets'
        elif name == 'ExecStart':
            try:
                tokens = shlex.split(value, posix=True)
                require(tokens and not any(character in value for character in ('$', '%', '\\', '\x00')),
                        'unit substitution or continuation remains unresolved')
                program = tokens[0].lstrip('-:+!@')
                require(Path(program).is_absolute() and Path(program).name in PROGRAMS, 'unsupported unit executable')
                record.update(program=program, token_count=len(tokens), safe_tokens=[tokens[0]])
                # Preserve exact simple flags and file arguments; unknown arguments are hashed only.
                pending_path = False
                for token in tokens[1:]:
                    if pending_path:
                        require(token.startswith('/') and re.fullmatch(r'/[A-Za-z0-9_./-]+', token),
                                'unsupported unit path argument')
                        record['safe_tokens'].append(token)
                        pending_path = False
                    elif token in {'-f', '-L'}:
                        record['safe_tokens'].append(token)
                        pending_path = True
                    elif re.fullmatch(r'-(?:D|i|s|v+)', token):
                        record['safe_tokens'].append(token)
                    else:
                        raise ValueError('unreviewed unit argument retained only by hash')
                require(not pending_path, 'unit file option has no literal argument')
                record['status'] = 'safe_tokens_recorded_not_executed'
            except ValueError as error:
                record['status'] = 'unresolved_execution_tokens'
                record['reason'] = str(error)
                record.pop('safe_tokens', None)
        rows.append(record)
    return rows


def service_observations(reader):
    """Observe fixed unit names and bounded drop-ins without systemctl or process inspection."""
    records, listings, dropin_count = [], [], 0
    for root in SYSTEMD_ROOTS:
        for name in SYSTEMD_UNITS:
            path = Path(root) / name
            row, raw = reader.read(path, SYSTEMD_ROOTS, LIMITS['unit_bytes'])
            if raw is not None:
                row['service_fields'] = service_tokens(raw)
            records.append(row)
            listing = directory_entries(Path(root) / (name + '.d'), SYSTEMD_ROOTS,
                r'[A-Za-z0-9_.-]+\.conf', LIMITS['dropin_files'] - dropin_count)
            listings.append(listing)
            for item in listing['entries']:
                dropin_count += 1
                row, raw = reader.read(Path(listing['path']) / item, SYSTEMD_ROOTS, LIMITS['unit_bytes'])
                if raw is not None:
                    row['service_fields'] = service_tokens(raw)
                records.append(row)
    return {'units_and_dropins': records, 'directory_observations': listings,
        'effective_systemd_execution_resolved': False, 'no_environment_file_read': True}


def elf_metadata(raw):
    """Parse bounded ELF metadata directly, without running the image or loading its dependencies."""
    require(len(raw) >= 64 and raw[:4] == b'\x7fELF' and raw[4] == 2 and raw[5] == 1,
            'only complete little-endian ELF64 images are parsed')
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    offset, size, count = header[5], header[9], header[10]
    require(size == 56 and 0 < count <= 256 and offset + size * count <= len(raw), 'unsupported ELF program headers')
    segments = [struct.unpack_from('<IIQQQQQQ', raw, offset + size * index) for index in range(count)]
    for segment in segments:
        require(segment[2] + segment[5] <= len(raw), 'ELF segment leaves the captured image')
    build_ids, dynamic = [], []
    for segment in segments:
        kind, unused, start, virtual, unused2, filesz, memsz, align = segment
        if kind == 4:
            position, end = start, start + filesz
            while position < end:
                require(position + 12 <= end, 'truncated ELF note header')
                namesz, descsz, note_type = struct.unpack_from('<III', raw, position)
                position += 12
                require(namesz <= 4096 and descsz <= 4096, 'ELF note size is unsupported')
                name = raw[position:position + namesz]
                position += (namesz + 3) // 4 * 4
                require(position + descsz <= end, 'truncated ELF note value')
                value = raw[position:position + descsz]
                position += (descsz + 3) // 4 * 4
                if name.rstrip(b'\0') == b'GNU' and note_type == 3:
                    build_ids.append(value.hex())
        elif kind == 2:
            require(filesz % 16 == 0 and filesz // 16 <= 4096, 'unsupported ELF dynamic segment')
            for position in range(start, start + filesz, 16):
                tag, value = struct.unpack_from('<qQ', raw, position)
                if tag == 0:
                    break
                dynamic.append((tag, value))
    string_tables = [value for tag, value in dynamic if tag == 5]
    string_sizes = [value for tag, value in dynamic if tag == 10]
    names = {'needed': [], 'rpath': [], 'runpath': []}
    selected = [(tag, value) for tag, value in dynamic if tag in {1, 15, 29}]
    if selected:
        require(len(string_tables) == len(string_sizes) == 1 and string_sizes[0] <= len(raw), 'ambiguous ELF string table')
        maps = [segment for segment in segments if segment[0] == 1 and segment[3] <= string_tables[0]
            and string_tables[0] + string_sizes[0] <= segment[3] + segment[5]]
        require(len(maps) == 1, 'ELF string table is not in one captured load segment')
        base = maps[0][2] + string_tables[0] - maps[0][3]
        for tag, value in selected:
            require(value < string_sizes[0], 'ELF string offset leaves its table')
            end = raw.find(b'\0', base + value, base + string_sizes[0])
            require(end >= 0 and end - base - value <= 4096, 'unterminated or long ELF dependency string')
            names[{1: 'needed', 15: 'rpath', 29: 'runpath'}[tag]].append(raw[base + value:end].decode('utf-8'))
    require(len(build_ids) <= 1, 'multiple ELF build identifiers remain ambiguous')
    return {'class': 'ELF64', 'byte_order': 'little', 'machine': header[2], 'build_ids': build_ids,
        **names, 'image_executed': False, 'dependencies_loaded': False}


def installed_image(reader, path, role):
    """Record one ordinary installed binary or plugin without using it as executable code."""
    row, raw = reader.read(path, PROGRAM_ROOTS, LIMITS['binary_bytes'])
    row['role'] = role
    if raw is not None:
        try:
            row['elf'] = elf_metadata(raw)
        except ValueError as error:
            row.update(elf_status='unsupported', elf_error=str(error))
    return row


def program_candidates(services, path_value):
    """Keep fixed program names from safe observed unit paths and bounded absolute PATH entries."""
    candidates, unresolved = [], []
    for row in services['units_and_dropins']:
        for setting in row.get('service_fields', []):
            if setting.get('program'):
                candidates.append(setting['program'])
    entries = path_value.split(':')
    if len(entries) > LIMITS['path_directories']:
        unresolved.append('PATH_directory_limit')
        entries = entries[:LIMITS['path_directories']]
    for entry in entries:
        try:
            path = lexical_path(entry)
            require(within(path, PROGRAM_ROOTS), 'PATH entry is outside installed program roots')
            candidates.extend(str(path / name) for name in PROGRAMS)
        except ValueError:
            unresolved.append('unsupported_PATH_entry_sha256=' + digest(entry.encode()))
    unique = list(dict.fromkeys(candidates))
    if len(unique) > LIMITS['program_candidates']:
        unresolved.append('program_candidate_limit')
    return unique[:LIMITS['program_candidates']], unresolved


def plugin_observations(reader, settings):
    """Hash only named ordinary Slurm plugins from the observed PluginDir values."""
    value = unique_setting(settings, 'PluginDir')
    result = {'controller_plugin_dir': value, 'files': [], 'unresolved': []}
    if value is None:
        result['unresolved'].append('missing_or_ambiguous_PluginDir')
        return result
    directories = value.split(':')
    if len(directories) > LIMITS['plugin_directories']:
        result['unresolved'].append('PluginDir_count_limit')
        return result
    for directory in directories:
        try:
            path = lexical_path(directory)
            require(within(path, PROGRAM_ROOTS), 'PluginDir leaves installed namespace')
            result['files'].extend(installed_image(reader, path / name, 'slurm_plugin_not_loaded') for name in PLUGIN_NAMES)
        except ValueError as error:
            result['unresolved'].append(str(error))
    return result


def package_metadata(reader, programs):
    """Read selected Debian package labels and file rosters without executing a package utility."""
    roots = ('/var/lib/dpkg',)
    row, raw = reader.read(Path('/var/lib/dpkg/status'), roots, LIMITS['package_database_bytes'])
    result = {'database': row, 'packages': [], 'file_rosters': [], 'binary_ownership_verified': False}
    if raw is None:
        result['unsupported'] = 'dpkg_status_missing_changed_or_over_limit_no_other_package_tool_executed'
        return result
    try:
        paragraphs = raw.decode('utf-8').split('\n\n')
    except UnicodeError as error:
        result.update(parse_status='unsupported_text_encoding', error_type=type(error).__name__)
        return result
    for paragraph in paragraphs:
        fields = dict(re.findall(r'(?m)^([A-Za-z0-9-]+): (.*)$', paragraph))
        if not re.fullmatch(r'(?:slurm|libslurm)[a-zA-Z0-9.+-]*', fields.get('Package', '')):
            continue
        selected = {name: fields[name] for name in ('Package', 'Version', 'Architecture', 'Status', 'Source') if name in fields}
        result['packages'].append(selected)
        require(len(result['packages']) <= 32, 'Slurm package count limit reached')
        for name in (selected['Package'], selected['Package'] + ':' + selected.get('Architecture', '')):
            listing, contents = reader.read(Path('/var/lib/dpkg/info') / (name + '.list'), roots, LIMITS['manual_bytes'])
            if contents is not None:
                known = {program.get('path') for program in programs if program.get('path')}
                try:
                    listing['matching_observed_program_paths'] = [line for line in contents.decode('utf-8').splitlines() if line in known]
                except UnicodeError as error:
                    listing.update(parse_status='unsupported_text_encoding', error_type=type(error).__name__)
            result['file_rosters'].append(listing)
    result['package_label_is_not_upstream_binary_attestation'] = True
    return result


def installed_manuals(reader):
    """Retain exact bounded installed manual bytes and bounded decompression without executing man."""
    records = []
    for name, section in [('slurmd', 8), ('slurmstepd', 8), ('scontrol', 1), ('slurm.conf', 5), ('gres.conf', 5), ('cgroup.conf', 5)]:
        for suffix in ('', '.gz'):
            path = Path('/usr/share/man/man' + str(section)) / (name + '.' + str(section) + suffix)
            row, raw = reader.read(path, ('/usr/share/man',), LIMITS['manual_bytes'], retain=True)
            if raw is not None and suffix:
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
                        expanded = handle.read(LIMITS['manual_bytes'] + 1)
                    require(len(expanded) <= LIMITS['manual_bytes'], 'installed manual decompression exceeds limit')
                    row.update(decompressed_sha256=digest(expanded), decompressed_byte_count=len(expanded))
                except (OSError, EOFError, ValueError, zlib.error) as error:
                    row.update(decompression_status='unsupported', error=str(error))
            records.append(row)
    return records


def service_accounts(reader, settings):
    """Keep only public identity fields for UID64030 and the observed Slurm account names."""
    row, raw = reader.read(Path('/etc/passwd'), ('/etc',), LIMITS['passwd_bytes'])
    result = {'database': row, 'selected_accounts': [], 'password_fields_retained': False, 'gecos_retained': False}
    if raw is None:
        return result
    names = {re.sub(r'\([0-9]+\)$', '', value) for name in ('SlurmUser', 'SlurmdUser')
        if (value := unique_setting(settings, name)) is not None}
    try:
        lines = raw.decode('utf-8').splitlines()
    except UnicodeError as error:
        result.update(parse_status='unsupported_text_encoding', error_type=type(error).__name__)
        return result
    for line in lines:
        fields = line.split(':')
        if len(fields) != 7:
            continue
        if fields[0] in names or fields[2] == '64030':
            result['selected_accounts'].append({'name': fields[0], 'uid': fields[2], 'gid': fields[3],
                'home': fields[5], 'shell': fields[6]})
    result['account_metadata_does_not_establish_config_trust'] = True
    return result


def collect_owned_metadata(allocation):
    """Collect passive installed evidence only after exact actual zero-GPU ownership is rechecked."""
    ownership.validate_allocation(allocation)
    current = ownership.own_process()
    identity_fields = ('hostname', 'pid', 'uid', 'effective_uid', 'uid_fields', 'parent_pid',
        'process_start_time', 'cgroup', 'cpu_affinity')
    require(all(current[name] == allocation['process'][name] for name in identity_fields),
        'passive collection must remain in the actual observed process')
    require(allocation.get('allocation_binding_passed') is True, 'reviewed owned allocation binding required')
    reader = PassiveReader()
    observation = {'schema_version': 1, 'kind': KIND, 'capability': CAPABILITY,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'process': current,
        'job_id': allocation['job_id'], 'step_id': allocation['step_id'], 'limits': LIMITS, **ZERO_CREDIT}
    try:
        # Controller claims stay distinct from installed files and their untrusted ownership.
        controller = controller_query(['scontrol', 'show', 'config'], SETTINGS)
        node = controller_query(['scontrol', 'show', 'node', 'jaguar03', '-o'],
            ('NodeName', 'Version', 'SlurmdStartTime', 'BootTime', 'NodeHostName', 'Gres'))
        observation.update(controller_configuration=controller, controller_node=node)
        usable = (controller['returncode'] == 0 and controller.get('stderr_byte_count', 0) == 0
            and controller.get('filter_status', 'parsed') == 'parsed')
        settings = controller['fields'] if usable else []
        observation['controller_settings_usable_for_candidate_discovery'] = usable
        spool = unique_setting(settings, 'SlurmdSpoolDir')
        config_roots = ['/etc/slurm']
        if spool and re.fullmatch(r'/[A-Za-z0-9_./-]+', spool) and '..' not in Path(spool).parts:
            config_roots.append(str(Path(spool) / 'conf-cache'))
        observation['configuration_roots'] = config_roots
        observation['configuration_trees'] = [configuration_tree(reader, Path(root) / 'slurm.conf', config_roots) for root in config_roots]
        observation['gres_and_cgroup_files'] = []
        for root in config_roots:
            for name in ('gres.conf', 'cgroup.conf'):
                row, raw = reader.read(Path(root) / name, config_roots, LIMITS['configuration_bytes_per_file'])
                if raw is not None:
                    # Only allowlisted assignments are retained; private authentication values never enter artifacts.
                    try:
                        row['assignments'] = setting_rows(raw, FILE_SETTINGS)
                    except UnicodeError as error:
                        row.update(parse_status='unsupported_text_encoding', error_type=type(error).__name__)
                observation['gres_and_cgroup_files'].append(row)

        # Discover metadata candidates without running their program or reading another process.
        services = observation['service_units'] = service_observations(reader)
        candidates, unresolved = program_candidates(services, os.environ.get('PATH', ''))
        programs = observation['installed_programs'] = [installed_image(reader, Path(path), 'slurm_program_not_executed') for path in candidates]
        observation.update(program_discovery_unresolved=unresolved, plugins=plugin_observations(reader, settings),
            packages=package_metadata(reader, programs), installed_manuals=installed_manuals(reader),
            service_accounts=service_accounts(reader, settings))
        observation['nvidia_management_library'] = {'loaded': False, 'device_opened': False,
            'candidates': [installed_image(reader, Path(path), 'management_library_file_not_loaded') for path in
                ('/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1', '/lib/x86_64-linux-gnu/libnvidia-ml.so.1',
                 '/usr/lib64/libnvidia-ml.so.1', '/usr/lib/nvidia/libnvidia-ml.so.1')],
            'actual_process_loaded_library_not_observed': True}
        observation['startup_gres_records'] = {'status': 'not_read_no_exact_installed_startup_record_parser_yet',
            'controller_slurmd_start_time': unique_setting(node['fields'], 'SlurmdStartTime'),
            'controller_log_file': unique_setting(settings, 'SlurmdLogFile'),
            'other_job_or_process_records_read': False}
        observation['unresolved_fields'] = ['actual_slurmd_merged_GRES_File_order', 'physical_GRES_mapping',
            'kernel_device_enforcement', 'vendor_build_equivalence_to_upstream',
            'effective_systemd_environment_and_execution', 'startup_GRES_log_binding',
            'same_process_CUDA_UUID_and_PCI_not_observed_in_zero_GPU_scope']
        observation['collection_completed'] = True
    except Exception as error:
        observation.update(collection_completed=False, error_type=type(error).__name__, error=str(error))
    finally:
        observation.update(finished_utc=datetime.now(timezone.utc).isoformat(), total_bytes_read=reader.bytes_read)
    return observation
