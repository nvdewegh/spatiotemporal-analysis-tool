"""
Moving between Sequence Analysis, Visual Exploration and PDP.

- Sequence Analysis can hand its supporting trajectories to Visual Exploration ("Inspect matching trajectories")
  and its supporting configurations to PDP ("Compare these configurations in PDP").
- A hand-off keeps the original analysis population, pattern and time window, so the receiving page can say
  exactly what it shows and offer a way back to the same sequence result.
- Button callbacks run before the next script run, which is the only moment Streamlit allows widget-backed keys
  such as analysis_method or visual_start to be set.
"""
import plotly.graph_objects as go
import streamlit as st

INSPECT_KEY = 'seq_inspect'          # Sequence Analysis -> Visual Exploration
PDP_HANDOFF_KEY = 'seq_pdp_handoff'  # Sequence Analysis -> PDP
LAST_SHARED_KEY = 'seq_last_shared_inputs'


def make_handoff(source, configs, objects, window, supporting, pattern, matching, settings, signature):
    """Everything the receiving page needs: population, window, supporting trajectories and how they were found."""
    supporting = list(supporting)
    return {
        'source': source,
        'population_configs': list(configs), 'population_objects': list(objects),
        'window': (float(window[0]), float(window[1])),
        'supporting': supporting,
        'supporting_configs': sorted({t.rsplit('_obj', 1)[0] for t in supporting}, key=_natural),
        'pattern': pattern, 'matching': matching, 'settings': settings, 'signature': signature,
    }


def _natural(s):
    import re
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', str(s))]


def go_inspect(handoff):
    """Callback: open Visual Exploration with the sequence time window and the supporting paths highlighted."""
    s = st.session_state
    s[INSPECT_KEY] = handoff
    s.visual_start, s.visual_end = handoff['window']
    s.analysis_method = 'Visual Exploration'


def go_pdp(handoff):
    """Callback: open PDP with the supporting configurations and the sequence time window."""
    s = st.session_state
    s[PDP_HANDOFF_KEY] = handoff
    s.pdp_start, s.pdp_end = handoff['window']
    s.analysis_method = 'PDP Analysis'


def go_back_to_sequences():
    """Callback: return to Sequence Analysis; its own settings are kept, so the same result is shown again."""
    st.session_state.analysis_method = 'Sequence Analysis'


def clear(key):
    st.session_state.pop(key, None)


def population_note(handoff, configs, objects):
    """Message when the sidebar population differs from the one the hand-off was made with (else None)."""
    changed = []
    if sorted(map(str, configs)) != sorted(map(str, handoff['population_configs'])):
        changed.append('configurations')
    if sorted(map(str, objects)) != sorted(map(str, handoff['population_objects'])):
        changed.append('objects')
    if not changed:
        return None
    return ("The sidebar " + " and ".join(changed) + " changed after this hand-off. The view below still uses the "
            "original analysis population; go back to Sequence Analysis to recompute with the current selection.")


def inspect_figure(base_fig, df, handoff, trajectory_label=lambda t: t):
    """
    All trajectories of the original population in the hand-off time window: supporting ones in colour with
    start/end markers, the others thin grey. Each supporting trajectory is exactly one coloured trace.
    """
    start, end = handoff['window']
    pop = df[df['config_source'].isin(handoff['population_configs']) & df['obj'].isin(handoff['population_objects'])
             & (df['tst'] >= start) & (df['tst'] <= end)]
    supporting = set(handoff['supporting'])
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22']
    fig = base_fig
    groups = sorted(pop.groupby(['config_source', 'obj']), key=lambda kv: (_natural(kv[0][0]), kv[0][1]))
    first_other = True
    for (config, obj), g in groups:
        tid = f"{config}_obj{obj}"
        if tid in supporting:
            continue
        g = g.sort_values('tst', kind='mergesort')
        fig.add_trace(go.Scatter(x=g['x'], y=g['y'], mode='lines', name='Other trajectories',
                                 legendgroup='other', showlegend=first_other,
                                 line=dict(color='rgba(150,150,150,0.45)', width=1),
                                 meta={'trajectory': tid, 'role': 'other'},
                                 hovertemplate=tid + '<extra></extra>'))
        first_other = False
    i = 0
    for (config, obj), g in groups:
        tid = f"{config}_obj{obj}"
        if tid not in supporting:
            continue
        g = g.sort_values('tst', kind='mergesort')
        colour = palette[i % len(palette)]
        i += 1
        fig.add_trace(go.Scatter(x=g['x'], y=g['y'], mode='lines+markers', name=trajectory_label(tid),
                                 line=dict(color=colour, width=3), marker=dict(size=6, color=colour),
                                 meta={'trajectory': tid, 'role': 'supporting',
                                       'tst': [float(v) for v in g['tst']]}, legendgroup=tid,
                                 customdata=g['tst'],
                                 hovertemplate=tid + '<br>tst %{customdata}<br>x %{x:.2f}, y %{y:.2f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=[g['x'].iloc[0], g['x'].iloc[-1]], y=[g['y'].iloc[0], g['y'].iloc[-1]],
                                 mode='markers', showlegend=False, legendgroup=tid, hoverinfo='skip',
                                 marker=dict(size=12, color=colour, symbol=['circle', 'square'],
                                             line=dict(color='white', width=2))))
    fig.update_layout(title=(f"{len(supporting)} of {len(groups)} trajectories match {handoff['pattern']} "
                             f"({handoff['matching']}) · tst {start:g} to {end:g}"),
                      height=600, legend=dict(title='Trajectory'))
    return fig


def shared_inputs_changed(current):
    """
    Shared inputs (dataset, configurations, objects) that changed since Sequence Analysis was last shown.
    Returns the list of changed names and stores the current values.
    """
    s = st.session_state
    old = s.get(LAST_SHARED_KEY)
    s[LAST_SHARED_KEY] = current
    if not old:
        return []
    return [k for k in current if old.get(k) != current[k]]
