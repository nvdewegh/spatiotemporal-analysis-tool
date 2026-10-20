"""
Browser-side persistence of user settings and the uploaded dataset.

Settings are kept in the browser's localStorage and the uploaded files in IndexedDB,
through a small Streamlit component (browser_storage/index.html). Nothing is stored on
the server, so different users never share or overwrite each other's data.

Flow per page load:
1. The component reads the saved state once and returns it (a "snapshot").
2. restore_from_snapshot() puts the saved files and settings in session state and reruns.
3. The app feeds the saved files through its normal upload code (restored_files()) and
   applies the saved settings before its widgets are created (apply_pending_settings()).
4. sync() at the end of every run saves changed settings, and the dataset when it changed.
"""
import base64
import hashlib
import io
import json
import os

import streamlit as st
import streamlit.components.v1 as components

NAMESPACE = 'samtool-v1'      # bump when the saved format changes; old data is then ignored
MAX_DATASET_BYTES = 50 * 1024 * 1024

_component = components.declare_component(
    'samtool_browser_storage',
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'browser_storage'))


class StoredFile(io.BytesIO):
    """A saved file that behaves like a Streamlit UploadedFile for the app's loading code."""
    def __init__(self, name, data):
        super().__init__(data)
        self.name = name


def _state():
    s = st.session_state
    s.setdefault('_bs_restored', False)          # snapshot received and applied
    s.setdefault('_bs_files', None)              # [(name, bytes)] of the saved dataset
    s.setdefault('_bs_saved_hash', None)         # hash of the dataset saved in the browser
    s.setdefault('_bs_ops', [])                  # pending writes, removed when acknowledged
    s.setdefault('_bs_next_op', 1)
    s.setdefault('_bs_pending_settings', None)   # saved settings to apply on the next run
    s.setdefault('_bs_last_settings', {})        # last settings written to the browser
    s.setdefault('_bs_notice', None)
    return s


def _queue(op):
    s = _state()
    op['id'] = s._bs_next_op
    s._bs_next_op += 1
    s._bs_ops.append(op)


def files_hash(files):
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(name.encode('utf-8') + b'\0' + hashlib.sha256(data).digest())
    return digest.hexdigest()


def restored_files():
    """The saved dataset as file-like objects for the upload code, or None."""
    files = _state()._bs_files
    return [StoredFile(name, data) for name, data in files] if files else None


def has_saved_dataset():
    return bool(_state()._bs_files)


def remember_uploaded(files):
    """Keep the currently uploaded files; save them in the browser if they changed."""
    s = _state()
    contents = [(f.name, f.getvalue()) for f in files]
    digest = files_hash(contents)
    s._bs_files = contents
    if digest == s._bs_saved_hash:
        return
    size = sum(len(data) for _, data in contents)
    if size > MAX_DATASET_BYTES:
        s._bs_notice = f'Dataset not saved in the browser: larger than {MAX_DATASET_BYTES // 2**20} MB.'
        return
    _queue({'type': 'save_dataset', 'hash': digest,
            'files': [{'name': name, 'data': base64.b64encode(data).decode('ascii')} for name, data in contents]})
    s._bs_saved_hash = digest


def forget_dataset():
    """Remove the dataset from the session and from the browser."""
    s = _state()
    s._bs_files = None
    if s._bs_saved_hash is not None:
        _queue({'type': 'clear_dataset'})
    s._bs_saved_hash = None


def reset_settings(keys):
    """Remove saved settings and the current values, so widgets return to their defaults."""
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
    _state()._bs_last_settings = {}
    _queue({'type': 'clear_settings'})


def apply_pending_settings(validators):
    """
    Apply restored settings, once, before the widgets are created.
    validators: {key: function(value) -> cleaned value, or None to skip}
    """
    s = _state()
    pending = s._bs_pending_settings
    if not pending:
        return
    s._bs_pending_settings = None
    for key, validate in validators.items():
        if key not in pending:
            continue
        try:
            value = validate(pending[key])
        except Exception:
            value = None
        if value is not None:
            st.session_state[key] = value


def restore_from_snapshot(snapshot):
    """Turn the browser snapshot into session state; invalid parts are ignored."""
    s = _state()
    s._bs_restored = True
    if not isinstance(snapshot, dict):
        return
    dataset = snapshot.get('dataset')
    if isinstance(dataset, dict) and isinstance(dataset.get('files'), list) and dataset['files']:
        try:
            files = [(str(f['name']), base64.b64decode(f['data'], validate=True)) for f in dataset['files']]
            if s._bs_files is None:        # an upload in this session takes precedence
                s._bs_files = files
                s._bs_saved_hash = files_hash(files)
                s._bs_notice = 'Restored your previous dataset: ' + ', '.join(name for name, _ in files)
        except Exception:
            _queue({'type': 'clear_dataset'})
            s._bs_notice = 'The saved dataset could not be read and was removed.'
    settings = snapshot.get('settings')
    if isinstance(settings, dict):
        s._bs_pending_settings = settings
        s._bs_last_settings = dict(settings)
    if snapshot.get('error'):
        s._bs_notice = 'Saved data could not be fully restored; defaults are used where needed.'


def take_notice():
    s = _state()
    notice, s._bs_notice = s._bs_notice, None
    return notice


def _json_safe(value):
    try:
        return json.loads(json.dumps(value, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
    except Exception:
        return None


def sync(setting_keys):
    """Render the storage component at the end of a run: save settings, flush writes, restore once."""
    s = _state()
    settings = None
    if s._bs_restored:   # never save before the saved state has been read
        # Keep the last known value of settings whose widget is not shown in this run
        settings = dict(s._bs_last_settings)
        settings.update({k: _json_safe(st.session_state[k]) for k in setting_keys if k in st.session_state})
        s._bs_last_settings = settings
    value = _component(namespace=NAMESPACE, settings=settings, ops=list(s._bs_ops),
                       key='samtool_browser_storage', default=None)
    if isinstance(value, dict):
        done = value.get('done', 0) or 0
        s._bs_ops = [op for op in s._bs_ops if op['id'] > done]
        if not s._bs_restored and ('snapshot' in value or 'error' in value):
            restore_from_snapshot(value.get('snapshot'))   # None (storage unavailable) -> defaults
            st.rerun()
