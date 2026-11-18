"""
Browser check of the local A2 viewer inside SAMTool (Streamlit), with the real student ZIP extracted fresh.
Usage: viewer_check.py <app url> <extracted viewer folder> <incomplete folder> [password]
Prints PASS/FAIL lines and a network summary. Media are checked against PIL crops of the sprite sheets.
"""
import hashlib
import io
import json
import os
import statistics
import sys

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL, FOLDER, INCOMPLETE = sys.argv[1:4]
PASSWORD = sys.argv[4] if len(sys.argv) > 4 else None
results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok)))
    print(('PASS ' if ok else 'FAIL ') + name + (f' — {detail}' if detail else ''), flush=True)


def expected_cell(enc, sample):
    im = Image.open(os.path.join(FOLDER, 'assets', 'detail', f'{enc:02d}', f'{sample // 4:03d}.webp')).convert('RGB')
    w, h = im.width // 2, im.height // 2
    c = sample % 4
    return np.asarray(im.crop(((c % 2) * w, (c // 2) * h, (c % 2) * w + w, (c // 2) * h + h)), dtype=float)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1500, 'height': 1300})
    sent_http, ws_frames = [], []
    page.on('request', lambda r: sent_http.append((r.method, r.url, len(r.post_data_buffer or b''))))
    page.on('websocket', lambda ws: ws.on('framesent', lambda payload: ws_frames.append(
        payload if isinstance(payload, (bytes, bytearray)) else payload.encode())))

    page.goto(URL)
    if PASSWORD:
        page.get_by_role('textbox', name='Password').fill(PASSWORD)
        page.get_by_role('textbox', name='Password').press('Enter')
    page.get_by_text('A2 video and trajectory viewer').first.click()
    frame = page.frame_locator('iframe[title*="local_a2_viewer"]')
    frame.locator('#folder').wait_for(state='attached', timeout=60000)
    handle = page.locator('iframe[title*="local_a2_viewer"]').element_handle().content_frame()

    def js(expr):
        return handle.evaluate(expr)

    def wait_synced(sample=None, enc=None, timeout=20000):
        cond = ("() => { const c = document.getElementById('camera').dataset, t = document.getElementById('trajectory').dataset;"
                "return c.sample !== undefined && c.sample === t.sample && c.encounter === t.encounter"
                + (f" && t.sample === '{sample}'" if sample is not None else '')
                + (f" && t.encounter === '{enc}'" if enc is not None else '') + "; }")
        handle.wait_for_function(cond, timeout=timeout)
        return js("() => [document.getElementById('camera').dataset.sample, document.getElementById('trajectory').dataset.sample,"
                  " document.getElementById('trajectory').dataset.encounter]")

    def camera_pixels():
        url = js("() => document.getElementById('camera').toDataURL('image/png')")
        return np.asarray(Image.open(io.BytesIO(__import__('base64').b64decode(url.split(',')[1]))).convert('RGB'), dtype=float)

    def select(enc):
        js(f"() => {{ const s = document.getElementById('encounter'); s.value = '{enc}'; s.dispatchEvent(new Event('change')); }}")

    def go(sample):
        js(f"() => {{ const t = document.getElementById('timeline'); t.value = '{sample}'; t.dispatchEvent(new Event('input')); }}")

    # ---------------------------------------------------------------- wrong folder first
    http_before, ws_before = len(sent_http), len(ws_frames)
    frame.locator('#folder').set_input_files(os.path.join(FOLDER, 'assets'))
    handle.wait_for_function("() => /does not contain index.html/.test(document.getElementById('status').textContent)")
    check('wrong folder (assets) is explained', True, js("() => document.getElementById('status').textContent")[:110])

    # ---------------------------------------------------------------- incomplete folder
    frame.locator('#folder').set_input_files(INCOMPLETE)
    handle.wait_for_function("() => /missing for/.test(document.getElementById('status').textContent)")
    status = js("() => document.getElementById('status').textContent")
    check('incomplete folder lists encounters without frames', 'Config_3' in status and 'Config_25' in status
          and 'Config_0,' not in status and 'Config_1' in status, status[:140])
    select(1); wait_synced(enc=1)
    go(74)
    handle.wait_for_function("() => !document.getElementById('media-error').hidden")
    err = js("() => document.getElementById('media-error').textContent")
    handle.wait_for_function("() => document.getElementById('trajectory').dataset.sample === '74'")
    s = js("() => [document.getElementById('camera').dataset.sample, document.getElementById('trajectory').dataset.sample]")
    check('missing sheet: error with recovery, animation still at 74', 'extract the complete viewer ZIP again' in err
          and s[1] == '74' and s[0] == 'none', err[:120])
    go(80); s = wait_synced(80, 1)
    check('next available sheet recovers camera', s == ['80', '80', '1'] and js("() => document.getElementById('media-error').hidden"))

    # ---------------------------------------------------------------- complete student folder
    frame.locator('#folder').set_input_files(FOLDER)
    handle.wait_for_function("() => /all camera frames present/.test(document.getElementById('status').textContent)",
                             timeout=60000)
    opts = js("() => [...document.getElementById('encounter').options].map(o => o.textContent)")
    check('all 26 encounter IDs available', opts == [f'Config_{i}' for i in range(26)], f'{len(opts)} options')
    check('opens at analysis start, both views sample 2', wait_synced(2, 0)[:2] == ['2', '2'])

    for enc in (0, 25, 12):
        select(enc); wait_synced(enc=enc)
        js("() => document.getElementById('reference').click()")
        for sample in (74, 73, 75):
            if sample != 74:
                go(sample)
            s = wait_synced(sample, enc)
            diff = np.abs(camera_pixels() - expected_cell(enc, sample)).mean()
            check(f'Config_{enc} sample {sample}: camera and points both at {sample}, frame matches sheet cell',
                  s[:2] == [str(sample)] * 2 and diff < 3, f'mean |Δ| {diff:.2f}')
        a, b = expected_cell(enc, 73), expected_cell(enc, 74)
        check(f'Config_{enc}: adjacent camera frames differ (73 vs 74)', np.abs(a - b).mean() > 0.05,
              f'{np.abs(a - b).mean():.3f}')

    # ---------------------------------------------------------------- playback rate and sync
    for enc in (0, 25):
        select(enc); go(2); wait_synced(2, enc)
        js("() => { window.__a2viewer.drawn.length = 0; document.getElementById('play').click(); }")
        page.wait_for_timeout(4000)
        js("() => document.getElementById('play').click()")
        drawn = js("() => window.__a2viewer.drawn")
        times = [d[0] for d in drawn]
        samples = [d[2] for d in drawn]
        rate = (len(drawn) - 1) / ((times[-1] - times[0]) / 1000)
        gaps = np.diff(times)
        check(f'Config_{enc} playback ≈ 24 Hz', 22.5 <= rate <= 24.5,
              f'{rate:.2f} Hz over {len(drawn)} samples, median step {statistics.median(gaps):.1f} ms, max {gaps.max():.0f} ms')
        check(f'Config_{enc} playback: consecutive samples, camera = points for every draw',
              samples == list(range(samples[0], samples[0] + len(samples))) and all(d[3] == d[2] for d in drawn))
        s = wait_synced(enc=enc)
        check(f'Config_{enc} paused in sync', s[0] == s[1], str(s))

    # ---------------------------------------------------------------- rapid scrubbing
    rng = np.random.default_rng(1)
    targets = [int(v) for v in rng.integers(2, 123, 40)]
    js("() => { const t = document.getElementById('timeline'); for (const v of %s) { t.value = String(v); "
       "t.dispatchEvent(new Event('input')); } }" % json.dumps(targets))
    s = wait_synced(targets[-1], 25)
    diff = np.abs(camera_pixels() - expected_cell(25, targets[-1])).mean()
    check('rapid scrubbing (40 inputs) ends with both views on the last sample', s[:2] == [str(targets[-1])] * 2 and diff < 3,
          f'sample {targets[-1]}, mean |Δ| {diff:.2f}')

    # ---------------------------------------------------------------- switch encounter during playback
    select(0); go(30); wait_synced(30, 0)
    js("() => document.getElementById('play').click()")
    page.wait_for_timeout(700)
    select(25)
    page.wait_for_timeout(1200)
    playing = js("() => document.getElementById('play').textContent") == 'Pause'
    js("() => document.getElementById('play').click()")
    s = wait_synced(enc=25)
    decoded = js("() => window.__a2viewer")
    check('switching encounter during playback continues with the new encounter, in sync',
          playing and s[2] == '25' and s[0] == s[1] and int(s[0]) > 40, str(s))
    js("() => document.getElementById('interval').value = 'full'")
    js("() => document.getElementById('interval').dispatchEvent(new Event('change'))")
    go(0); check('full clip reaches sample 0', wait_synced(0, 25)[:2] == ['0', '0'])
    go(136); check('full clip reaches sample 136', wait_synced(136, 25)[:2] == ['136', '136'])

    # ---------------------------------------------------------------- network
    new_http = sent_http[http_before:]
    new_ws = ws_frames[ws_before:]
    big = [r for r in new_http if r[2] > 0]
    local = [r for r in new_http if 'webp' in r[1] or 'data.js' in r[1] or r[1].startswith('file:')]
    blob_or_path = [f for f in new_ws if b'LocalComparisonViewer' in f or b'scratchpad' in f or b'RIFF' in f
                    or b'WEBP' in f or b'ENCOUNTERS' in f]
    print(f'NETWORK during folder selection and playback: {len(new_http)} HTTP requests, '
          f'{sum(r[2] for r in new_http)} request body bytes, {len(new_ws)} websocket frames sent '
          f'({sum(len(f) for f in new_ws)} bytes)')
    for r in new_http[:15]:
        print('   ', r)
    check('no HTTP request bodies, no media or data.js requests to the server', not big and not local,
          f'{len(new_http)} requests')
    check('no websocket frame carries file contents, frame data or local paths', not blob_or_path,
          f'{len(new_ws)} frames, max {max((len(f) for f in new_ws), default=0)} bytes')
    browser.close()

print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed")
sys.exit(0 if all(ok for _, ok in results) else 1)
