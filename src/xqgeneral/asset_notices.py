"""Carry upstream declarations for exact inference assets without assigning a model license."""
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit

from .evidence import digest, file_signature


PROFILE = 'inference_asset_notices_v1'
SCOPE = 'upstream_declarations_for_exact_assets_only'
FIELDS = {'name', 'assets', 'source_url', 'license_id', 'license_file', 'license_sha256'}


def _relative(value):
    if not isinstance(value, str):
        raise ValueError('Component notice paths must be relative strings')
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or '..' in path.parts or '\\' in value or
            path.as_posix() != value or any(part.startswith('.') for part in path.parts)):
        raise ValueError('Component notice paths must stay inside their directory')
    return value


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Component notices require exact SHA256 identities')
    return value


def _records(value):
    if not isinstance(value, list) or not value:
        raise ValueError('Component notices require a nonempty component list')
    names = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError('Use the explicit component notice record schema')
        for field in ['name', 'source_url', 'license_id']:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError('Component notices require a name, source URL and license identifier')
        if row['name'] in names:
            raise ValueError('Component notice names must be unique')
        names.add(row['name'])
        source = urlsplit(row['source_url'])
        if (source.scheme != 'https' or not source.hostname or source.username or source.password or
                source.query or source.fragment):
            raise ValueError('Use a stable HTTPS source URL without credentials or query parameters')
        if not isinstance(row['assets'], dict) or not row['assets']:
            raise ValueError('Each component must bind its exact inference assets')
        for relative, checksum in row['assets'].items():
            _relative(relative)
            _hash(checksum)
        _relative(row['license_file'])
        _hash(row['license_sha256'])
    return value


def prepare_asset_notices(specification, asset_paths):
    """Resolve input documents before export; output metadata contains no source paths."""
    specification = Path(specification).resolve()
    signatures = {specification: file_signature(specification)}
    value = json.loads(specification.read_text())
    if (not isinstance(value, dict) or set(value) != {'profile', 'components'} or
            value['profile'] != PROFILE):
        raise ValueError('Use the inference asset-notices input schema')
    components, documents = [], {}
    for ordinal, row in enumerate(_records(value['components']), 1):
        for relative, expected in row['assets'].items():
            if relative not in asset_paths or digest(asset_paths[relative]) != expected:
                raise ValueError('Component notice belongs to a different inference asset')
        document = specification.parent
        for part in PurePosixPath(row['license_file']).parts:
            document = document / part
            if document.is_symlink():
                raise ValueError('Component license inputs must not use symbolic links')
        if not document.is_file():
            raise ValueError('Component license text is missing')
        signatures[document] = file_signature(document)
        if not document.read_text().strip() or digest(document) != row['license_sha256']:
            raise ValueError('Component license text differs from its declared identity')
        target = f'notices/component-{ordinal:04d}-LICENSE.txt'
        documents[target] = document
        components.append(dict(row, assets=dict(row['assets']), license_file=target))
    if any(file_signature(path) != expected for path, expected in signatures.items()):
        raise ValueError('Component notice inputs changed during preparation')
    return {'profile': PROFILE, 'scope': SCOPE, 'components': components}, documents, signatures


def validate_asset_notices(value, files):
    """Validate declarations against the already hashed, self-contained bundle files."""
    if (not isinstance(value, dict) or set(value) != {'profile', 'scope', 'components'} or
            value['profile'] != PROFILE or value['scope'] != SCOPE):
        raise ValueError('Use the explicit asset-bound component notice schema')
    documents = set()
    for row in _records(value['components']):
        for relative, expected in row['assets'].items():
            if (not (relative.startswith('decoder/') or relative == 'expert/px0.pb.gz') or
                    relative not in files or files[relative]['sha256'] != expected):
                raise ValueError('Component notice belongs to a different inference asset')
        document = row['license_file']
        if (not document.startswith('notices/') or document in documents or document not in files or
                files[document]['sha256'] != row['license_sha256']):
            raise ValueError('Component license is missing, duplicated or rebound to different bytes')
        documents.add(document)
    if {name for name in files if name.startswith('notices/')} != documents:
        raise ValueError('Inference bundle contains an unassigned component notice document')
