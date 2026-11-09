"""
Streamlit views for PDP reference systems (G2, R2, R4, C4).

Choice and status of the reference system, the reference geometry view, comparison of one encounter across
systems with linked cell explanation, two encounters within one system, C4 mean versus median, dataset-level
comparison (distances, rankings, clusters) and exports. Calculations live in modules/pdp_reference_systems.py
and modules/pdp_analysis.py; this module only arranges them.
"""
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from modules import pdp_analysis as pa
from modules import pdp_reference_systems as rs

ROLE_COLOURS = {0: '#3675a9', 1: '#d47726'}
ROLE_NAMES = {0: 'Cyclist', 1: 'Vehicle'}
SPEC_OPTIONS = [('G2', None), ('R2', None), ('R4', None), ('C4', 'mean'), ('C4', 'median')]
AGGREGATION_POLICY = (
    "**Aggregation policy.** Raw distance = sum over the active descriptors and all windows of |difference| "
    "between inequality-matrix cells (G2: X + Y, exactly as before). Normalised distance = raw ÷ (2 × number of "
    "descriptors × comparable cells × windows): an *analytical upper bound*. Descriptors that depend on each "
    "other (two projections of the same positions, along and left of one path) cannot all reach opposite "
    "relations at once, so the jointly attainable maximum can be lower and 100% may be unreachable. "
    "The mean descriptor distance (raw ÷ descriptors) is shown separately and is not the legacy raw sum. "
    "No automatic weights or scalar standardisation are applied.")
SCOPE_NOTE = ("Descriptive analysis on the dataset the references were fitted on, not held-out validation. "
              "A higher silhouette or a larger departure from a mean path does not by itself mean danger; "
              "R4 or C4 need not cluster better than G2.")


def _s():
    return st.session_state


# =============================================================================
# ACTIVE REFERENCE
# =============================================================================

_PACKAGED = {}


def _packaged_reference():
    if 'ref' not in _PACKAGED:
        try:
            _PACKAGED['ref'] = (rs.load_reference(), None)
        except (OSError, rs.ReferenceError) as exc:
            _PACKAGED['ref'] = (None, str(exc))
    return _PACKAGED['ref']


def active_reference():
    """(reference, source label, error). An explicitly supplied JSON replaces the packaged A2 preset."""
    text = _s().get('pdp_reference_override')
    if text:
        try:
            return rs.load_reference(text=text), 'supplied reference JSON', None
        except rs.ReferenceError as exc:
            return None, 'supplied reference JSON', str(exc)
    ref, error = _packaged_reference()
    return ref, 'packaged A2 preset (modules/reference_data/A2_pdp_references_v02.json)', error


def spec_from_state():
    s = _s()
    system = s.get('pdp_reference_system', rs.DEFAULT_SYSTEM)
    estimator = s.get('pdp_c4_estimator', rs.DEFAULT_ESTIMATOR)
    return rs.system_spec(system if system in rs.SYSTEMS else rs.DEFAULT_SYSTEM,
                          estimator if estimator in rs.ESTIMATORS else rs.DEFAULT_ESTIMATOR)


def channels_for(df, spec, ref):
    """Descriptor columns for df under spec, computed once per (data, reference definition) and reused."""
    if spec[0] == 'G2':
        return df
    cache = _s().setdefault('_pdp_ref_channels', {})
    key = (pa._data_fingerprint(df), rs.spec_identity(ref, spec))
    if key not in cache:
        if len(cache) >= 6:
            cache.pop(next(iter(cache)))
        cache[key] = rs.add_descriptor_columns(df, spec, ref)
    return cache[key]


class ReferenceChoice:
    """The reference system chosen for the next computation and what it needs."""

    def __init__(self, df, spec, ref, source, error, compat):
        self.df, self.spec, self.ref, self.source, self.error, self.compat = df, spec, ref, source, error, compat
        self.system, self.estimator = spec
        self.descriptor_ids = rs.DESCRIPTOR_ORDER[self.system]
        self.is_g2 = self.system == 'G2'
        self.key = None if self.is_g2 or ref is None else rs.spec_identity(ref, spec)
        self.descriptors = None if self.is_g2 else rs.descriptor_columns(self.system)

    @property
    def usable(self):
        if self.is_g2:
            return True
        if self.ref is None or self.compat[0] == 'incompatible':
            return False
        return self.compat[0] == 'reference_dataset' or bool(_s().get('pdp_reference_ack'))

    def blockers(self, variant_keys, external_points):
        """Reasons the next computation is unavailable (never a silent fall-back to G2)."""
        if self.is_g2:
            return []
        out = []
        name = rs.spec_label(self.spec)
        if self.ref is None:
            out.append(f"**{name} unavailable:** the reference geometry is invalid ({self.error}). "
                       "Choose G2 or supply a valid reference JSON. G2 is not used in its place.")
        elif self.compat[0] == 'incompatible':
            out.append(f"**{name} unavailable for this dataset:** {self.compat[1]}")
        elif not self.usable:
            out.append(f"**{name} needs confirmation:** {self.compat[1]} Tick the confirmation under "
                       "**Reference System** to use these frozen references with this dataset.")
        if external_points and self.ref is not None and self.usable:
            try:
                self.external_values(external_points)
            except Exception as exc:               # never drop or replace the points silently
                out.append(f"**External points cannot be projected in {name}:** {exc}")
        return out

    # Buffer and Rough per descriptor: the counterparts of Buffer X/Y and Rough X/Y in G2
    def buffers(self):
        return tuple(float(_s().get(f'pdp_ref_buffer_{d}', DEFAULT_BUFFER)) for d in self.descriptor_ids)

    def roughs(self):
        return tuple(float(_s().get(f'pdp_ref_rough_{d}', DEFAULT_ROUGH)) for d in self.descriptor_ids)

    def variant_parameters(self, variant_key):
        """Arguments for compute_pdp_distance_matrix (R/C systems), used only by the matching variants."""
        uses_buffer, uses_rough = variant_key in pa.BUFFER_VARIANTS, variant_key in pa.ROUGH_VARIANTS
        n = len(self.descriptor_ids)
        return dict(descriptor_buffers=self.buffers() if uses_buffer else (0.0,) * n,
                    descriptor_roughs=self.roughs() if uses_rough else (0.0,) * n)

    def run_key(self, variant_keys):
        """Reference identity plus the per-descriptor Buffer/Rough values the selected variants use."""
        if self.key is None:
            return None
        key = (self.key,)
        if any(k in pa.BUFFER_VARIANTS for k in variant_keys):
            key += (('buffer',) + self.buffers(),)
        if any(k in pa.ROUGH_VARIANTS for k in variant_keys):
            key += (('rough',) + self.roughs(),)
        return key if len(key) > 1 else self.key

    def external_values(self, external_points):
        """Descriptor values of external points, with the same projection as the trajectories."""
        if not external_points:
            return None
        frame = external_frame(external_points, self.spec, self.ref)
        return tuple(tuple(float(frame[c].iloc[k]) for c in self.descriptors) for k in range(len(frame)))

    def channels(self, df):
        if self.is_g2:
            return df
        if not self.usable:
            raise rs.ReferenceError("Reference system not usable for this dataset.")
        return channels_for(df, self.spec, self.ref)

    def record(self):
        rec = {'system': self.system, 'estimator': self.estimator, 'label': rs.spec_label(self.spec),
               'descriptor_order': list(self.descriptor_ids), 'identity': self.key}
        if not self.is_g2:
            rec['descriptor_buffer'] = dict(zip(self.descriptor_ids, self.buffers()))
            rec['descriptor_rough'] = dict(zip(self.descriptor_ids, self.roughs()))
        if not self.is_g2 and self.ref is not None:
            rec.update({'reference_id': self.ref.get('reference_id', 'supplied'), 'source': self.source,
                        'reference_hash': self.ref['_identity']})
        return rec


DEFAULT_BUFFER, DEFAULT_ROUGH = 1.0, 0.5           # the G2 defaults for Buffer X/Y and Rough X/Y


def external_frame(external_points, spec, ref):
    """External points (name, x, y) with their descriptor values and projection metadata."""
    frame = pd.DataFrame({'config_source': 'external', 'tst': 0.0,
                          'obj': [f'EXT_{k}' for k in range(len(external_points))],
                          'name': [p[0] for p in external_points],
                          'x': [float(p[1]) for p in external_points], 'y': [float(p[2]) for p in external_points]})
    return rs.add_descriptor_columns(frame, spec, ref)


def descriptor_value_text(rec, kind):
    """'cyclist_road_along 1, vehicle_road_along 0.5' for a run record (kind 'buffer' or 'rough')."""
    values = (rec or {}).get(f'descriptor_{kind}')
    return ", ".join(f"{d} {v:g}" for d, v in values.items()) if values else None


def current_reference_record():
    return _s().get('_pdp_reference_record') or {'system': 'G2', 'estimator': None, 'label': 'G2',
                                                 'descriptor_order': list(rs.DESCRIPTOR_ORDER['G2']), 'identity': None}


def reference_record_text(rec):
    rec = rec or {'system': 'G2', 'label': 'G2', 'descriptor_order': ['global_x', 'global_y']}
    text = f"{rec['label']} ({', '.join(rec['descriptor_order'])})"
    if rec.get('reference_hash'):
        text += f" · reference {rec.get('reference_id')} #{rec['reference_hash'][:10]}"
    return text


def run_reference_record():
    return ((_s().get('pdp_run_settings') or {}).get('reference')) or current_reference_record()


def xy_view_available(view):
    """x/y-only views are shown for G2 results only; otherwise say why and where to look instead."""
    rec = run_reference_record()
    if rec.get('system', 'G2') == 'G2':
        return True
    st.info(f"The {view} works on global x/y and is available for G2 results only. These results use "
            f"**{rec['label']}**. Use **Reference systems: G2, R2, R4 and C4** below for descriptor-aware matrices, "
            "cell explanations and comparisons.")
    return False


# =============================================================================
# CONTROLS AND REFERENCE VIEW
# =============================================================================

def reference_controls(df):
    s = _s()
    st.subheader("Reference System", help=(
        "**G2**: global x and y (the existing baseline).  \n"
        "**R2**: position along each of the two road directions (independent projections; the roads are not "
        "perpendicular).  \n**R4**: R2 plus the signed distance to the left of each road.  \n"
        "**C4**: position along and to the left of the fixed cyclist and vehicle flow paths.  \n\n"
        "Every point, cyclist and vehicle, is described by the same complete descriptor set. "
        "R/C reference systems are separate from Buffer/Rough: **R does not mean Rough**."))
    if s.get('pdp_reference_system') not in rs.SYSTEMS:
        s.pdp_reference_system = rs.DEFAULT_SYSTEM
    if s.get('pdp_c4_estimator') not in rs.ESTIMATORS:
        s.pdp_c4_estimator = rs.DEFAULT_ESTIMATOR
    col_sys, col_est = st.columns([3, 2])
    with col_sys:
        system = st.selectbox("Reference system", list(rs.SYSTEMS), key='pdp_reference_system',
                              format_func=lambda k: rs.SYSTEM_LABELS[k])
    with col_est:
        st.radio("C4 flow-path estimator", list(rs.ESTIMATORS), key='pdp_c4_estimator', horizontal=True,
                 format_func=lambda k: 'Mean (default)' if k == 'mean' else 'Median (comparison)',
                 disabled=system != 'C4',
                 help="Used by C4 only. Mean: smoothed spatial mean reference path (default). Median: smoothed "
                      "spatial median reference path, at the same cross-sections, support and smoothing. "
                      "Changing it makes C4 results outdated.")
    spec = spec_from_state()
    ref, source, error = active_reference()
    compat = rs.data_compatibility(ref, df) if ref is not None else ('incompatible', error)
    st.caption(f"Descriptors, in order: {', '.join(f'`{d}`' for d in rs.DESCRIPTOR_ORDER[spec[0]])}"
               + ("" if spec[0] == 'G2' else
                  f" · reference: {source}" + (f" · flow paths: smoothed spatial {spec[1]}" if spec[0] == 'C4' else "")))
    if spec[0] != 'G2':
        if ref is None:
            st.error(f"Reference geometry invalid: {error}. {spec[0]} cannot be computed; G2 is not used instead.")
        elif compat[0] == 'reference_dataset':
            st.caption(f"✓ {compat[1]} References are frozen: changing encounters, time range or cluster count "
                       "never refits them.")
        elif compat[0] == 'incompatible':
            st.error(compat[1])
        else:
            st.warning(compat[1])
            st.checkbox("Use the frozen A2 references with this dataset (same local metre frame)",
                        key='pdp_reference_ack')
        descriptor_parameter_inputs(spec[0])
    choice = ReferenceChoice(df, spec, ref, source, error, compat)
    s._pdp_reference_record = choice.record()

    with st.expander("Reference geometry view and reference JSON", expanded=False):
        if ref is None:
            st.error(f"No valid reference loaded: {error}")
        else:
            show_reference_view(df, ref)
        supply_reference_json(ref)
    return choice


def descriptor_parameter_inputs(system):
    """Buffer and Rough per descriptor: what Buffer X/Y and Rough X/Y are for G2."""
    s = _s()
    variants = s.get('pdp_variants', ["Fundamental"])
    buffer_users = [v for v in variants if v in ("Buffer", "Buffer + Rough")]
    rough_users = [v for v in variants if v in ("Rough", "Buffer + Rough")]
    st.markdown(f"**Buffer and Rough per descriptor ({system})**", help=(
        "The descriptor-space counterparts of Buffer X/Y and Rough X/Y, in metres along or to the left of the "
        "road or flow path. **Rough d**: two values of descriptor d count as equal when they differ by at most "
        "Rough d. **Buffer d**: every point gets two extra points with descriptor d at value − Buffer d and "
        "value + Buffer d (other descriptors unchanged), exactly as G2 adds points at x ± Buffer X. Buffering is "
        "done after projection, in descriptor values. Buffer X/Y and Rough X/Y above are used by G2 only."))
    st.caption((f"Buffer: used by {', '.join(buffer_users)}." if buffer_users
                else "Buffer: not used (Buffer variants not selected).") + "  \n"
               + (f"Rough: used by {', '.join(rough_users)}." if rough_users
                  else "Rough: not used (Rough variants not selected)."))
    for descriptor in rs.DESCRIPTOR_ORDER[system]:
        for kind, default in (('buffer', DEFAULT_BUFFER), ('rough', DEFAULT_ROUGH)):
            key = f'pdp_ref_{kind}_{descriptor}'
            if key not in s:
                s[key] = default
        col_b, col_r = st.columns(2)
        with col_b:
            st.number_input(f"Buffer {descriptor} (m)", min_value=0.0, max_value=5.0, step=0.1, format="%.2f",
                            key=f'pdp_ref_buffer_{descriptor}')
        with col_r:
            st.number_input(f"Rough {descriptor} (m)", min_value=0.0, max_value=5.0, step=0.1, format="%.2f",
                            key=f'pdp_ref_rough_{descriptor}')


def supply_reference_json(ref):
    s = _s()
    st.markdown("**Replace the reference (optional).** A compatible JSON with the same structure "
                "(road_estimates, flows, median_flows_sensitivity_comparison) creates a new reference identity, "
                "so all reference-system results become outdated.")
    upload = st.file_uploader("Reference JSON (replaces the packaged A2 preset)", type=['json'],
                              key='pdp_reference_upload')
    col_a, col_b = st.columns(2)
    if upload is not None and col_a.button("Use this reference JSON", key='pdp_reference_use_upload'):
        text = upload.getvalue().decode('utf-8', errors='replace')
        try:
            rs.load_reference(text=text)
        except rs.ReferenceError as exc:
            st.error(f"Not used: {exc}")
        else:
            s.pdp_reference_override = text
            st.rerun()
    if s.get('pdp_reference_override') and col_b.button("Return to the packaged A2 preset", key='pdp_reference_reset'):
        s.pop('pdp_reference_override', None)
        st.rerun()
    if ref is not None:
        export = {k: v for k, v in ref.items() if not k.startswith('_')}
        st.download_button("Download active reference JSON", json.dumps(export, indent=2),
                           file_name=f"{ref.get('reference_id', 'reference')}.json", mime='application/json',
                           key='pdp_reference_download')


def _road_line(ref, role, half_length=40):
    road = ref['road_estimates'][rs.ROAD_KEYS[role]]
    o, u = np.array(road['origin_xy_m']), np.array(road['unit_direction'])
    return o, u, o + np.array([-half_length, half_length])[:, None] * u


def show_reference_view(df, ref):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("R2/R4: road centres (diagram-derived)",
                                        "C4: smoothed mean (solid) and median (dashed) flow paths"))
    for col in (1, 2):
        for obj in (0, 1):
            xs, ys = [], []
            for _, part in df[df['obj'] == obj].sort_values('tst').groupby('config_source'):
                xs += list(part['x']) + [None]
                ys += list(part['y']) + [None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(width=0.7, color=ROLE_COLOURS[obj]),
                                     opacity=0.25, name=f"{ROLE_NAMES[obj]} trajectories (observed)",
                                     showlegend=col == 1, hoverinfo='skip'), row=1, col=col)
    for role in ('cyclist', 'vehicle'):
        colour = ROLE_COLOURS[0 if role == 'cyclist' else 1]
        o, u, line = _road_line(ref, role)
        fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], mode='lines', line=dict(width=3, color=colour),
                                 name=f"{role} road centre ({'lower road' if role == 'cyclist' else 'vehicle approach'})"),
                      row=1, col=1)
        tip, left = o + 6 * u, o + 6 * u + 3 * np.array([-u[1], u[0]])
        fig.add_annotation(x=tip[0], y=tip[1], ax=o[0], ay=o[1], xref='x', yref='y', axref='x', ayref='y',
                           showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor=colour, text='')
        fig.add_annotation(x=left[0], y=left[1], ax=tip[0], ay=tip[1], xref='x', yref='y', axref='x', ayref='y',
                           showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=colour, text='+left',
                           font=dict(color=colour, size=11))
        for estimator, dash in (('mean', 'solid'), ('median', 'dash')):
            flow = rs.flows_for(ref, estimator)[role]
            poly = np.array(flow['directed_polyline_xy_m'])
            counts = flow.get('support_counts') or [None] * len(poly)
            fig.add_trace(go.Scatter(
                x=poly[:, 0], y=poly[:, 1], mode='lines', line=dict(width=3 if estimator == 'mean' else 2,
                                                                     color=colour, dash=dash),
                name=f"{role} flow: {flow['label']}", customdata=counts,
                hovertemplate=f"{role} {estimator} flow<br>x %{{x:.2f}} m, y %{{y:.2f}} m"
                              "<br>supported by %{customdata} encounters<extra></extra>"), row=1, col=2)
            if estimator == 'mean':
                low = np.array([c is not None and c < 10 for c in counts])
                fig.add_trace(go.Scatter(x=poly[low, 0], y=poly[low, 1], mode='markers',
                                         marker=dict(symbol='x', size=7, color=colour),
                                         name=f"{role} mean: fewer than 10 contributing encounters",
                                         customdata=np.array(counts)[low],
                                         hovertemplate="support %{customdata}<extra></extra>"), row=1, col=2)
                a, b = poly[-2], poly[-1]
                fig.add_annotation(x=b[0], y=b[1], ax=a[0] - 4 * (b[0] - a[0]), ay=a[1] - 4 * (b[1] - a[1]),
                                   xref='x2', yref='y2', axref='x2', ayref='y2', showarrow=True, arrowhead=3,
                                   arrowwidth=2, arrowcolor=colour, text='')
    fig.update_xaxes(range=[18, 67], title_text='x (m)')
    fig.update_yaxes(range=[1, 25], title_text='y (m)', row=1, col=1)
    fig.update_yaxes(scaleanchor='x', scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor='x2', scaleratio=1, row=1, col=2)
    fig.update_layout(height=520, legend=dict(orientation='h', y=-0.2), margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True, key='pdp_reference_view')
    geo = ref.get('geometry_source') or {}
    st.caption(
        "Units: metres in the original local x/y frame; no geographical CRS is inferred. Road centres are "
        "**diagram-derived with approximate metric registration** "
        f"(scale ≈ {geo.get('pixel_to_metre_scale', float('nan')):.5f} m/pixel from rounded labels; "
        f"{geo.get('source_part', '?')}, slides {geo.get('road_diagram_slide', '?')} and "
        f"{geo.get('metric_label_slide', '?')}), not surveyed geometry. Roads describe the road layout; flow paths "
        "describe observed movement: they are different reference choices. Arrows show each direction and its "
        "positive-left side. Crosses mark mean-path vertices supported by fewer than 10 encounters (minimum 3).")
    st.caption("Mean/median: fixed spatial cross-sections, one interpolated position per encounter inside its own "
               "coverage, equal weight per encounter, local linear smoothing over ≤5 stations. These are declared "
               "implementation choices, not a verified reproduction of CB's algorithm.")
    if os.path.exists(rs.PACKAGED_OVERLAY_PATH) and ref.get('reference_id') == 'A2-detailed-v02':
        st.image(rs.PACKAGED_OVERLAY_PATH, use_container_width=True,
                 caption="Preparation overlay (build_references_v02.py). The red dashed line is the rejected earlier "
                         "vehicle-road estimate, shown for history only; it is not used anywhere.")
    support = []
    for estimator in rs.ESTIMATORS:
        for role in ('cyclist', 'vehicle'):
            c = np.array(rs.flows_for(ref, estimator)[role]['support_counts'])
            support.append({'Flow path': f"{role} ({estimator})", 'Vertices': len(c), 'Min support': int(c.min()),
                            'Max support': int(c.max()), 'Vertices with < 10 encounters': int((c < 10).sum())})
    st.dataframe(pd.DataFrame(support), hide_index=True, use_container_width=True)


# =============================================================================
# EXPLORER
# =============================================================================

def _spec_choice(label, key, default):
    s = _s()
    labels = [rs.spec_label(sp) for sp in SPEC_OPTIONS]
    if s.get(key) not in labels:
        s[key] = rs.spec_label(default)
    chosen = st.selectbox(label, labels, key=key)
    return SPEC_OPTIONS[labels.index(chosen)]


def _run_context():
    run = _s().get('pdp_run_settings') or {}
    return run.get('configs') or [], run.get('objects') or [], run.get('start'), run.get('end'), run.get('window') or 1


def _window_selector(df, prefix, configs, objects, start, end, default_window):
    times = sorted(df[df['config_source'].isin(configs) & df['obj'].isin(objects) &
                      (df['tst'] >= start) & (df['tst'] <= end)]['tst'].unique())
    if not times:
        return None
    max_w = min(len(times), 12)
    col_w, col_t = st.columns(2)
    window = col_w.number_input("Window length (time steps)", 1, max_w, min(int(default_window), max_w),
                                key=f'{prefix}_window',
                                help="Matrices of one window: (objects × window) points. Up to 12 time steps.")
    starts = times[:len(times) - int(window) + 1]
    if _s().get(f'{prefix}_tst') not in starts:
        _s()[f'{prefix}_tst'] = starts[0]
    first = col_t.selectbox("Window starts at tst", starts, key=f'{prefix}_tst', format_func=lambda t: f"{t:g}")
    i = times.index(first)
    return times[i:i + int(window)]


def _labels(points):
    return [f"{'C' if int(o) == 0 else 'V' if int(o) == 1 else 'O' + str(o)}{int(p)} t{t:g}"
            for p, o, t in zip(points['point'], points['obj'], points['tst'])]


def _matrix_heatmap(fig, matrix, labels, row, col, kind='ineq', text=None):
    if kind == 'ineq':
        colours, zmax = pa.INEQ_COLORS, 2
        hover = [[f"row {labels[i]} vs column {labels[j]}: {rs.RELATION_WORDS[int(matrix[i, j])]}"
                  for j in range(len(labels))] for i in range(len(labels))]
    else:
        colours, zmax = ['#eef3f8', '#c62828'], 1
        hover = [[f"row {labels[i]} vs column {labels[j]}: " + ('changed' if matrix[i, j] else 'same')
                  for j in range(len(labels))] for i in range(len(labels))]
    scale = []
    for k, c in enumerate(colours):
        scale += [[k / len(colours), c], [(k + 1) / len(colours), c]]
    fig.add_trace(go.Heatmap(z=matrix, x=labels, y=labels, zmin=-0.5, zmax=zmax + 0.5, colorscale=scale,
                             showscale=False, text=hover, hoverinfo='text', xgap=1, ygap=1), row=row, col=col)
    fig.update_yaxes(autorange='reversed', row=row, col=col)


def _projection_trace(fig, ref, spec, descriptor, point, col, colour, name):
    """Draw the geometry and projection that produce one scalar for one point."""
    x, y = float(point['x']), float(point['y'])
    if descriptor in ('global_x', 'global_y'):
        if descriptor == 'global_x':
            fig.add_trace(go.Scatter(x=[x, x], y=[y, 1], mode='lines', line=dict(dash='dot', color=colour),
                                     name=f"{name}: x = {x:.2f} m", showlegend=True), row=1, col=col)
        else:
            fig.add_trace(go.Scatter(x=[x, 18], y=[y, y], mode='lines', line=dict(dash='dot', color=colour),
                                     name=f"{name}: y = {y:.2f} m", showlegend=True), row=1, col=col)
        return
    foot, text = rs.projection_of(point, descriptor, ref, spec)
    fig.add_trace(go.Scatter(x=[x, foot[0]], y=[y, foot[1]], mode='lines+markers',
                             line=dict(dash='dot', color=colour, width=2), marker=dict(size=[0, 9], color=colour),
                             name=f"{name}: {text}", hovertemplate=f"{name} projection<br>{text}<extra></extra>"),
                  row=1, col=col)


def _geometry_trace(fig, ref, spec, descriptor, col):
    if descriptor.startswith('global'):
        return
    role, kind, _ = descriptor.split('_')
    colour = ROLE_COLOURS[0 if role == 'cyclist' else 1]
    if kind == 'road':
        _, _, line = _road_line(ref, role)
        fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], mode='lines', line=dict(color=colour, width=2.5),
                                 name=f"{role} road centre (direction →, positive left)"), row=1, col=col)
    else:
        flow = rs.flows_for(ref, spec[1])[role]
        poly = np.array(flow['directed_polyline_xy_m'])
        fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], mode='lines+markers', marker=dict(size=3),
                                 line=dict(color=colour, width=2.5), name=f"{role} {flow['label']}"),
                      row=1, col=col)


def _cell_map(df, ref, config, objects, points_a, points_b, spec_a, spec_b, desc_a, desc_b, i, j, key):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=(f"A: {rs.spec_label(spec_a)} · {desc_a}",
                                        f"B: {rs.spec_label(spec_b)} · {desc_b}"))
    enc = df[(df['config_source'] == config) & df['obj'].isin(objects)].sort_values('tst')
    for col, (pts, spec, desc) in enumerate(((points_a, spec_a, desc_a), (points_b, spec_b, desc_b)), start=1):
        for obj, part in enc.groupby('obj'):
            fig.add_trace(go.Scatter(x=part['x'], y=part['y'], mode='lines',
                                     line=dict(color=ROLE_COLOURS.get(int(obj), 'grey'), width=1), opacity=0.5,
                                     name=f"{ROLE_NAMES.get(int(obj), obj)} trajectory", showlegend=col == 1,
                                     hoverinfo='skip'), row=1, col=col)
        _geometry_trace(fig, ref, spec, desc, col)
        for k, symbol, label in ((i, 'star', 'row point'), (j, 'diamond', 'column point')):
            p = pts.iloc[k]
            colour = ROLE_COLOURS.get(int(p['obj']), 'black')
            fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers+text', text=[label.split()[0]],
                                     textposition='top center',
                                     marker=dict(symbol=symbol, size=15, color=colour, line=dict(width=1.5, color='black')),
                                     name=f"{label}: {rs.point_label(p)}", showlegend=col == 1), row=1, col=col)
            _projection_trace(fig, ref, spec, desc, p, col, colour, label)
    xs = list(enc['x']) + [18, 67]
    fig.update_xaxes(range=[min(xs) - 2, max(xs) + 2], title_text='x (m)')
    fig.update_yaxes(range=[1, 25], title_text='y (m)', row=1, col=1)
    fig.update_yaxes(scaleanchor='x', scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor='x2', scaleratio=1, row=1, col=2)
    fig.update_layout(height=520, legend=dict(orientation='h', y=-0.25, font=dict(size=10)), margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True, key=key)


def cross_system_view(df, ref, prefix, fixed=None):
    """One encounter, two reference systems, explicitly mapped channels, changed cells and their explanation."""
    configs, objects, start, end, run_window = _run_context()
    c1, c2 = st.columns(2)
    config = c1.selectbox("Encounter", configs, key=f'{prefix}_config')
    objs = c2.multiselect("Objects", objects, default=objects, key=f'{prefix}_objects')
    if not objs:
        st.info("Select at least one object.")
        return
    times = _window_selector(df, prefix, [config], objs, start, end, run_window)
    if not times:
        st.info("No time steps in the completed run's time range.")
        return
    if fixed:
        spec_a, spec_b = fixed
        st.markdown(f"**A: {rs.spec_label(spec_a)}** versus **B: {rs.spec_label(spec_b)}**: identical input, "
                    "window and descriptor meanings; only the flow-path estimator differs.")
    else:
        c3, c4 = st.columns(2)
        with c3:
            spec_a = _spec_choice("System A", f'{prefix}_spec_a', ('G2', None))
        with c4:
            spec_b = _spec_choice("System B", f'{prefix}_spec_b', ('C4', 'mean'))
    if ref is None and any(sp[0] != 'G2' for sp in (spec_a, spec_b)):
        st.error("No valid reference geometry is loaded, so only G2 can be shown.")
        return
    da, db = rs.DESCRIPTOR_ORDER[spec_a[0]], rs.DESCRIPTOR_ORDER[spec_b[0]]
    c5, c6 = st.columns(2)
    desc_a = c5.selectbox("Channel of A", da, key=f'{prefix}_chan_a_{spec_a[0]}',
                          format_func=lambda d: f"{d} ({rs.DESCRIPTOR_LABELS[d]})")
    if fixed:
        desc_b = desc_a
        c6.markdown(f"Channel of B: `{desc_b}` (same descriptor meaning)")
    else:
        default_b = da.index(desc_a) if da.index(desc_a) < len(db) else 0
        if _s().get(f'{prefix}_chan_b_{spec_b[0]}') not in db:
            _s()[f'{prefix}_chan_b_{spec_b[0]}'] = db[default_b]
        desc_b = c6.selectbox("Channel of B (mapped explicitly)", db, key=f'{prefix}_chan_b_{spec_b[0]}',
                              format_func=lambda d: f"{d} ({rs.DESCRIPTOR_LABELS[d]})")
    st.caption(f"Mapping compared cell by cell: **A.{desc_a} ↔ B.{desc_b}**. Only these two channels are compared; "
               "systems with different descriptor counts are never subtracted as whole arrays.")
    pts_a = rs.window_points(channels_for(df, spec_a, ref), config, objs, times)
    pts_b = rs.window_points(channels_for(df, spec_b, ref), config, objs, times)
    col_a, col_b = rs.descriptor_column(desc_a), rs.descriptor_column(desc_b)
    ma, mb = rs.channel_matrix(pts_a, col_a), rs.channel_matrix(pts_b, col_b)
    changed = rs.changed_cells(ma, mb)
    labels = _labels(pts_a)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                        subplot_titles=(f"A: {rs.spec_label(spec_a)} · {desc_a}",
                                        f"B: {rs.spec_label(spec_b)} · {desc_b}", "Changed cells (red)"))
    _matrix_heatmap(fig, ma, labels, 1, 1)
    _matrix_heatmap(fig, mb, labels, 1, 2)
    _matrix_heatmap(fig, (ma != mb).astype(int), labels, 1, 3, kind='diff')
    fig.update_layout(height=430, margin=dict(t=40))
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True, key=f'{prefix}_matrices')
    n_off = len(ma) * (len(ma) - 1)
    st.caption(f"{config}, tst {times[0]:g}–{times[-1]:g}. Rows and columns: points in calculation order (time, then "
               "object); C = cyclist, V = vehicle. Cell (row i, column j): green 0 = column scalar greater (row < "
               f"column), yellow 1 = equal, red 2 = column scalar smaller. Changed: {len(changed)} of {n_off} "
               "off-diagonal cells.")
    if not changed:
        st.success("No relation changes between these two channels in this window.")
        return
    options = [f"row {labels[i]} · column {labels[j]}: A {pa.INEQ_SYMBOLS[a]} → B {pa.INEQ_SYMBOLS[b]}"
               for i, j, a, b in changed]
    pick = st.selectbox(f"Explain a changed cell ({len(changed)})", range(len(options)),
                        format_func=lambda k: options[k], key=f'{prefix}_cell')
    i, j, rel_a, rel_b = changed[min(pick, len(changed) - 1)]
    rows = rs.explain_cell(pts_a, pts_b, i, j, col_a, col_b)
    table = pd.DataFrame(rows)
    table.insert(1, 'Role in cell', ['row point i', 'column point j'])
    table = table.rename(columns={'A scalar': f"A: {desc_a} (m)", 'B scalar': f"B: {desc_b} (m)"})
    for k, sp, d, pts in (('A', spec_a, desc_a, pts_a), ('B', spec_b, desc_b, pts_b)):
        table[f"{k} projection"] = [rs.projection_of(pts.iloc[n], d, ref, sp)[1] for n in (i, j)]
    st.dataframe(table, hide_index=True, use_container_width=True)
    va, vb = rows[0]['A scalar'], rows[0]['B scalar']
    wa, wb = rows[1]['A scalar'], rows[1]['B scalar']
    st.markdown(
        f"- **A ({rs.spec_label(spec_a)}, {desc_a})**: column − row = {wa:.3f} − {va:.3f} = {wa - va:+.3f} m → "
        f"cell {rel_a} ({rs.RELATION_WORDS[rel_a]})\n"
        f"- **B ({rs.spec_label(spec_b)}, {desc_b})**: column − row = {wb:.3f} − {vb:.3f} = {wb - vb:+.3f} m → "
        f"cell {rel_b} ({rs.RELATION_WORDS[rel_b]})")
    _cell_map(df, ref, config, objs, pts_a, pts_b, spec_a, spec_b, desc_a, desc_b, i, j, key=f'{prefix}_map')
    st.caption("Dotted lines run from each point to the place where its scalar is read: its projection on the road "
               "line or flow path (along = arc length to that foot point; left = signed distance, positive left of "
               "the direction), or the x/y axis for G2. A line joining the two raw points alone would not explain "
               "a curved projection.")
    buf = io.StringIO()
    out = pts_a[['point', 'config_source', 'obj', 'tst', 'x', 'y']].copy()
    out[f'A_{desc_a}'] = pts_a[col_a].to_numpy()
    out[f'B_{desc_b}'] = pts_b[col_b].to_numpy()
    out.to_csv(buf, index=False)
    buf.write('\nmatrix_A_rows_points_cols_points\n')
    pd.DataFrame(ma, index=labels, columns=labels).to_csv(buf)
    buf.write('\nmatrix_B_rows_points_cols_points\n')
    pd.DataFrame(mb, index=labels, columns=labels).to_csv(buf)
    st.download_button("Download point mapping, scalars and both matrices (CSV)", buf.getvalue(),
                       file_name=f"pdp_cells_{config}_{rs.spec_label(spec_a)}_vs_{rs.spec_label(spec_b)}.csv",
                       mime='text/csv', key=f'{prefix}_download')


def two_encounter_view(df, ref):
    """Existing comparison of two encounters within one system, now per descriptor."""
    configs, objects, start, end, run_window = _run_context()
    if len(configs) < 2:
        st.info("The completed run has fewer than two encounters.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        spec = _spec_choice("Reference system", 'pdp_xe_spec', ('G2', None))
    a = c2.selectbox("Encounter A", configs, key='pdp_xe_a')
    b = c3.selectbox("Encounter B", [c for c in configs if c != a], key='pdp_xe_b')
    if spec[0] != 'G2' and ref is None:
        st.error("No valid reference geometry is loaded.")
        return
    data = channels_for(df, spec, ref)
    cols = rs.descriptor_columns(spec[0]) if spec[0] != 'G2' else ('x', 'y')
    sel = data[data['obj'].isin(objects) & (data['tst'] >= start) & (data['tst'] <= end)]
    total, info = pa.compute_pdp_distance_pair(sel[sel['config_source'] == a], sel[sel['config_source'] == b],
                                               run_window, descriptors=cols, details=True)
    n_desc = len(cols)
    max_d = pa.pdp_max_possible_distance(len(objects), run_window, info['n_windows_used'], n_descriptors=n_desc)
    rows = [{'Descriptor': d, 'Raw': info['per_descriptor'][c], 'Self precedence': info['self'][c],
             'Mutual precedence': info['mutual'][c]} for d, c in zip(rs.DESCRIPTOR_ORDER[spec[0]], cols)]
    rows.append({'Descriptor': 'Total', 'Raw': total, 'Self precedence': sum(info['self'].values()),
                 'Mutual precedence': sum(info['mutual'].values())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    pct = pa.normalised_pdp_percent(total, max_d)
    st.caption(f"{rs.spec_label(spec)}, completed-run settings (objects {', '.join(map(str, objects))}, tst "
               f"{start:g}–{end:g}, window {run_window}, {info['n_windows_used']} windows, Fundamental). "
               f"Raw {total:g}" + (f" · normalised {pct:.2f}% of the analytical bound" if pct is not None else "")
               + f" · mean descriptor distance {total / n_desc:.1f}. Self = same object at different (or equal) "
               "times; mutual = both cross-object blocks. Self + mutual = total.")
    times = _window_selector(df, 'pdp_xe', [a, b], objects, start, end, run_window)
    if not times:
        return
    pa_pts = rs.window_points(data, a, objects, times)
    pb_pts = rs.window_points(data, b, objects, times)
    if len(pa_pts) != len(pb_pts):
        st.warning("The two encounters have different points in this window; matrices are not comparable.")
        return
    labels = _labels(pa_pts)
    fig = make_subplots(rows=len(cols), cols=3, vertical_spacing=0.05 if len(cols) > 2 else 0.1,
                        subplot_titles=[t for d in rs.DESCRIPTOR_ORDER[spec[0]]
                                        for t in (f"{a} · {d}", f"{b} · {d}", "|difference|")])
    for r, c in enumerate(cols, start=1):
        m1, m2 = rs.channel_matrix(pa_pts, c), rs.channel_matrix(pb_pts, c)
        _matrix_heatmap(fig, m1, labels, r, 1)
        _matrix_heatmap(fig, m2, labels, r, 2)
        diff = np.abs(m1 - m2)
        scale = [[0, pa.DIFF_COLORS[0]], [1 / 3, pa.DIFF_COLORS[0]], [1 / 3, pa.DIFF_COLORS[1]],
                 [2 / 3, pa.DIFF_COLORS[1]], [2 / 3, pa.DIFF_COLORS[2]], [1, pa.DIFF_COLORS[2]]]
        fig.add_trace(go.Heatmap(z=diff, x=labels, y=labels, zmin=-0.5, zmax=2.5, colorscale=scale, showscale=False,
                                 xgap=1, ygap=1, hovertemplate='row %{y} · column %{x}: |Δ| %{z}<extra></extra>'),
                      row=r, col=3)
        fig.update_yaxes(autorange='reversed', row=r, col=3)
    fig.update_layout(height=330 * len(cols), margin=dict(t=40))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(tickfont=dict(size=7))
    st.plotly_chart(fig, use_container_width=True, key='pdp_xe_matrices')
    st.caption("One window shown; the distance above sums all windows of the completed run.")


def mean_median_view(df, ref):
    if ref is None:
        st.error("No valid reference geometry is loaded.")
        return
    rows = []
    for role in ('cyclist', 'vehicle'):
        m = np.array(ref['flows'][role]['directed_polyline_xy_m'])
        d = np.array(ref['median_flows_sensitivity_comparison'][role]['directed_polyline_xy_m'])
        n = min(len(m), len(d))
        sep = np.linalg.norm(m[:n] - d[:n], axis=1)
        rows.append({'Flow path': role, 'Vertices (mean/median)': f"{len(m)}/{len(d)}",
                     'Median separation at shared stations (m)': round(float(np.median(sep)), 3),
                     'Max separation (m)': round(float(sep.max()), 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Both paths use the same cross-sections, coverage, support threshold and smoothing; only the "
               "estimator differs. The median path is a coordinate-wise median at common cross-sections, not an "
               "observed (medoid) trajectory. Dataset-level effects: choose C4 (mean) and C4 (median) in the "
               "**Dataset level** tab.")
    cross_system_view(df, ref, 'pdp_mm', fixed=(('C4', 'mean'), ('C4', 'median')))


# ---------------------------------------------------------------- dataset level

def _dataset_matrices(df, ref, specs):
    s = _s()
    configs, objects, start, end, window = _run_context()
    store = s.setdefault('_pdp_ref_dataset', {})
    out = {}
    for spec in specs:
        key = (s.get('pdp_run_signature'), rs.spec_identity(ref, spec) if ref else spec)
        if key not in store:
            return None, spec
        out[spec] = store[key]
    return out, None


def _compute_dataset(df, ref, specs):
    s = _s()
    configs, objects, start, end, window = _run_context()
    store = s.setdefault('_pdp_ref_dataset', {})
    with st.status(f"Computing {len(specs)} reference system(s) on {len(configs)} encounters", expanded=True) as status:
        for spec in specs:
            key = (s.get('pdp_run_signature'), rs.spec_identity(ref, spec) if ref else spec)
            if key in store:
                continue
            bar = st.progress(0.0, text=f"{rs.spec_label(spec)}: preparing descriptors")
            data = channels_for(df, spec, ref)
            descriptors = None if spec[0] == 'G2' else rs.descriptor_columns(spec[0])
            matrix, ids, comps = pa.compute_pdp_distance_matrix(
                data, tuple(configs), tuple(objects), start, end, window_length=window,
                descriptors=descriptors, reference_key=None if spec[0] == 'G2' else rs.spec_identity(ref, spec),
                return_components=True,
                progress=lambda c, t, b=bar, sp=spec: b.progress(c / t, text=f"{rs.spec_label(sp)}: {c} of {t} pairs"))
            bar.progress(1.0, text=f"{rs.spec_label(spec)}: done")
            store[key] = {'matrix': matrix, 'ids': list(ids), 'components': comps}
            if len(store) > 12:
                store.pop(next(iter(store)))
        status.update(label="Reference-system comparison ready", state='complete', expanded=False)


def dataset_view(df, ref):
    s = _s()
    configs, objects, start, end, window = _run_context()
    if len(configs) < 3:
        st.info("Dataset-level comparison needs at least three encounters in the completed run.")
        return
    available = [sp for sp in SPEC_OPTIONS if sp[0] == 'G2' or ref is not None]
    labels = [rs.spec_label(sp) for sp in available]
    run_label = run_reference_record().get('label', 'G2')
    default = [l for l in dict.fromkeys(['G2', run_label, 'C4 (mean)']) if l in labels]
    if 'pdp_ds_specs' in s and not set(s.pdp_ds_specs).issubset(labels):
        del s['pdp_ds_specs']
    chosen = st.multiselect("Reference systems to compare", labels, default=default, key='pdp_ds_specs')
    specs = [available[labels.index(l)] for l in chosen]
    st.caption(f"All use the completed run's settings: {len(configs)} encounters, objects "
               f"{', '.join(map(str, objects))}, tst {start:g}–{end:g}, window {window}, Fundamental, no external "
               f"points. {len(configs) * (len(configs) - 1) // 2} unique encounter pairs.")
    if len(specs) < 1:
        return
    results, missing = _dataset_matrices(df, ref, specs)
    if results is None:
        if st.button(f"Compute {len(specs)} reference system(s)", key='pdp_ds_compute', type='primary'):
            _compute_dataset(df, ref, specs)
            st.rerun()
        st.caption("Stored results for identical settings are reused. Changing the completed run, reference or "
                   "estimator requires computing again.")
        return
    n_windows = s.get('pdp_norm_n_windows')
    summary = []
    for spec, res in results.items():
        n_desc = len(rs.DESCRIPTOR_ORDER[spec[0]])
        bound = pa.pdp_max_possible_distance(len(objects), window, n_windows, n_descriptors=n_desc)
        tri = res['matrix'][np.triu_indices(len(res['ids']), 1)]
        row = {'System': rs.spec_label(spec), 'Descriptors': n_desc, 'Analytical bound': bound,
               'Mean raw': round(float(tri.mean()), 1),
               'Mean normalised (%)': round(float(tri.mean() / bound * 100), 2) if bound else None,
               'Mean descriptor distance': round(float(tri.mean() / n_desc), 1)}
        for d, c in zip(rs.DESCRIPTOR_ORDER[spec[0]], (res['components'])):
            row[f"share {d}"] = f"{res['components'][c][np.triu_indices(len(res['ids']), 1)].sum() / max(tri.sum(), 1):.0%}"
        summary.append(row)
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
    st.caption(AGGREGATION_POLICY)

    if len(specs) >= 2:
        c1, c2 = st.columns(2)
        sx = c1.selectbox("Scatter: x axis", chosen, index=0, key='pdp_ds_x')
        sy = c2.selectbox("Scatter: y axis", chosen, index=1, key='pdp_ds_y')
        rx, ry = results[available[labels.index(sx)]], results[available[labels.index(sy)]]
        pairs = rs.unique_pairs(len(rx['ids']))
        nx = len(rs.DESCRIPTOR_ORDER[available[labels.index(sx)][0]])
        ny = len(rs.DESCRIPTOR_ORDER[available[labels.index(sy)][0]])
        bx = pa.pdp_max_possible_distance(len(objects), window, n_windows, n_descriptors=nx)
        by = pa.pdp_max_possible_distance(len(objects), window, n_windows, n_descriptors=ny)
        x = np.array([rx['matrix'][a, b] for a, b in pairs]) / bx * 100
        y = np.array([ry['matrix'][a, b] for a, b in pairs]) / by * 100
        names = [f"{rx['ids'][a]} – {rx['ids'][b]}" for a, b in pairs]
        fig = go.Figure(go.Scatter(x=x, y=y, mode='markers', text=names, marker=dict(size=6, opacity=0.7),
                                   hovertemplate='%{text}<br>x %{x:.2f}%<br>y %{y:.2f}%<extra></extra>'))
        fig.update_layout(height=420, xaxis_title=f"{sx}: normalised distance (%)",
                          yaxis_title=f"{sy}: normalised distance (%)",
                          title=f"{len(pairs)} unique encounter pairs · Spearman ρ = {rs.spearman(x, y):.3f}")
        st.plotly_chart(fig, use_container_width=True, key='pdp_ds_scatter')

    k = st.slider("Number of clusters (fixed k for all systems)", 2, min(10, len(configs) - 1),
                  min(max(2, int(s.get('pdp_chosen_n_clusters') or 3)), min(10, len(configs) - 1)), key='pdp_ds_k')
    from sklearn.metrics import silhouette_score
    member = pd.DataFrame({'Encounter': next(iter(results.values()))['ids']})
    labels_by = {}
    sil = {}
    for spec, res in results.items():
        lab, _ = pa.perform_hierarchical_clustering(res['matrix'], k)
        labels_by[rs.spec_label(spec)] = lab
        member[rs.spec_label(spec)] = lab
        sil[rs.spec_label(spec)] = (silhouette_score(res['matrix'], lab, metric='precomputed')
                                    if len(set(lab)) > 1 else float('nan'))
    names = list(labels_by)
    ari = pd.DataFrame([[rs.adjusted_rand(labels_by[a], labels_by[b]) for b in names] for a in names],
                       index=names, columns=names).round(3)
    c1, c2 = st.columns([3, 2])
    c1.dataframe(member, hide_index=True, use_container_width=True, height=300)
    c2.markdown("**Adjusted Rand index between memberships**")
    c2.dataframe(ari, use_container_width=True)
    c2.markdown("**Mean silhouette**  \n" + "  \n".join(f"{n}: {v:.3f}" for n, v in sil.items()))
    st.caption(f"Average-linkage clusters at k = {k}, cluster numbers are arbitrary per system (compare memberships, "
               "not numbers). " + SCOPE_NOTE)

    target = st.selectbox("Ranking: nearest encounters to", member['Encounter'], key='pdp_ds_target')
    rank_rows = {}
    n_rank = min(5, len(member) - 1)
    for spec, res in results.items():
        t = res['ids'].index(target)
        order = [i for i in np.argsort(res['matrix'][t], kind='mergesort') if i != t][:n_rank]
        rank_rows[rs.spec_label(spec)] = [f"{res['ids'][i]} ({res['matrix'][t, i]:g})" for i in order]
    st.dataframe(pd.DataFrame(rank_rows, index=[f"#{i + 1}" for i in range(n_rank)]), use_container_width=True)
    st.caption("Nearest encounters by raw distance (in brackets); ties keep encounter order.")

    export = {'format': 'samtool-pdp-reference-comparison', 'run': _run_export(), 'clusters_k': k,
              'linkage': 'average', 'systems': {}}
    for spec, res in results.items():
        export['systems'][rs.spec_label(spec)] = {
            'reference': rs.reference_metadata(ref, spec) if ref else {'system': spec[0]},
            'encounters': res['ids'], 'raw_distance': res['matrix'].tolist(),
            'components': {c: m.tolist() for c, m in res['components'].items()},
            'cluster_labels': [int(v) for v in labels_by[rs.spec_label(spec)]]}
    st.download_button("Download distances, components, clusters and reference metadata (JSON)",
                       json.dumps(export, default=_json_default), file_name='pdp_reference_systems_comparison.json',
                       mime='application/json', key='pdp_ds_download')


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _run_export():
    s = _s()
    run = dict(s.get('pdp_run_settings') or {})
    run['data_fingerprint'] = str((s.get('pdp_run_signature') or ('',))[0])
    return run


def exports_view(df, ref):
    s = _s()
    rec = run_reference_record()
    spec = (rec.get('system', 'G2'), rec.get('estimator'))
    configs, objects, start, end, window = _run_context()
    st.markdown(f"Exports for the **completed run** ({reference_record_text(rec)}).")
    if spec[0] != 'G2' and ref is None:
        st.error("The reference used by the completed run is not loaded.")
        return
    data = channels_for(df, spec, ref)
    scalars = rs.scalar_table(data, spec, configs, objects, start, end)
    st.download_button(f"Scalar values and projection flags ({len(scalars):,} points, CSV)",
                       scalars.to_csv(index=False), file_name=f"pdp_scalars_{rs.spec_label(spec)}.csv",
                       mime='text/csv', key='pdp_rx_scalars')
    summary = rs.projection_summary(data[data['config_source'].isin(configs) & data['obj'].isin(objects) &
                                         (data['tst'] >= start) & (data['tst'] <= end)], spec)
    if summary:
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
        st.caption("Tangent-ray extensions are not observed flow; ambiguous projections are equally close to two "
                   "places on the path; possible discontinuities mark jumps of the projected position larger than "
                   "3 × the observed movement + 0.5 m. Values are never smoothed.")
    variant = 'fundamental' if 'fundamental' in (s.get('pdp_results_all') or {}) else None
    if variant:
        external = tuple(map(tuple, (s.get('pdp_run_settings') or {}).get('external') or ())) or None
        descriptors = None if spec[0] == 'G2' else rs.descriptor_columns(spec[0])
        external_values = None
        if external and descriptors is not None:
            frame = external_frame(external, spec, ref)
            external_values = tuple(tuple(float(v) for v in frame[list(descriptors)].iloc[k])
                                    for k in range(len(frame)))
        matrix, ids, comps = pa.compute_pdp_distance_matrix(
            data, tuple(configs), tuple(objects), start, end, window_length=window,
            descriptors=descriptors, reference_key=rec.get('identity'), return_components=True,
            external_points=external, external_descriptors=external_values)
        bound = pa.pdp_max_possible_distance(**_norm())
        rows = []
        for a, b in rs.unique_pairs(len(ids)):
            row = {'Config A': ids[a], 'Config B': ids[b], 'Raw': matrix[a, b],
                   'Normalised (%)': pa.normalised_pdp_percent(matrix[a, b], bound),
                   'Mean descriptor distance': matrix[a, b] / len(comps)}
            row.update({f"raw {c}": m[a, b] for c, m in comps.items()})
            rows.append(row)
        st.download_button(f"Pairwise distances with descriptor contributions ({len(rows)} pairs, CSV)",
                           pd.DataFrame(rows).to_csv(index=False), file_name=f"pdp_pairs_{rs.spec_label(spec)}.csv",
                           mime='text/csv', key='pdp_rx_pairs')
    bundle = {'format': 'samtool-pdp-reference-run', 'input_file': s.get('filename'), 'run': _run_export(),
              'reference': rs.reference_metadata(ref, spec) if ref is not None else {'system': 'G2'},
              'source_content_sha256': rs.data_content_sha256(df) if {'config', 'obj'}.issubset(df.columns) else None,
              'normalisation': {**_norm(), 'policy': AGGREGATION_POLICY},
              'clustering': {'n_clusters': s.get('pdp_chosen_n_clusters'), 'linkage': 'average'}}
    st.download_button("Run settings, reference geometry and metadata (JSON)",
                       json.dumps(bundle, indent=2, default=_json_default),
                       file_name=f"pdp_run_{rs.spec_label(spec)}.json", mime='application/json', key='pdp_rx_bundle')


def _norm():
    s = _s()
    return dict(n_objects=s.get('pdp_norm_n_objects'), window_length=s.get('pdp_norm_window_length'),
                n_windows=s.get('pdp_norm_n_windows'), n_external_points=s.get('pdp_norm_n_external_points', 0),
                n_descriptors=s.get('pdp_norm_n_descriptors', 2))


def render_reference_explorer(df, choice):
    with st.expander("Reference systems: G2, R2, R4 and C4 (compare systems, explain cells, mean vs median)",
                     expanded=False):
        rec = run_reference_record()
        st.caption(f"Completed run: **{reference_record_text(rec)}**. All tabs use its encounters, objects, time "
                   "range and window; the Fundamental relations are compared (Rough tolerance 0).")
        ref, _, error = active_reference()
        if ref is None:
            st.warning(f"Reference geometry unavailable ({error}): only G2 can be shown.")
        tabs = st.tabs(["Same encounter across systems", "Two encounters, one system", "C4 mean vs median",
                        "Dataset level", "Exports"])
        with tabs[0]:
            cross_system_view(df, ref, 'pdp_xs')
        with tabs[1]:
            two_encounter_view(df, ref)
        with tabs[2]:
            mean_median_view(df, ref)
        with tabs[3]:
            dataset_view(df, ref)
        with tabs[4]:
            exports_view(df, ref)
