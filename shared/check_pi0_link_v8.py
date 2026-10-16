"""Prove the tunnel -> A10 server -> checkpoint chain before any hardware is touched.

Sends one synthetic observation -- zero joint state and three flat grey 480x640
frames, which is exactly the request shape the trial sends -- and checks the
reply: ten finite (14,) rows, and a checkpoint path equal to the one expected.
The server echoes the SHA256 of the weights that answered, so a motion run's
report can record which checkpoint produced it.

Exits 0 only when the whole chain answers; anything else prints LINK_FAIL and
exits non-zero so the launcher stops before motors are enabled.
"""
import argparse
import base64
import json
import time
import urllib.request

import cv2
import numpy as np

CAMERAS = ('hand_left', 'hand_right', 'top_head')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18024/')
    parser.add_argument('--expected-checkpoint', required=True)
    args = parser.parse_args()

    grey = np.full((480, 640, 3), 128, np.uint8)
    ok, jpg = cv2.imencode('.jpg', grey, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise SystemExit('LINK_FAIL jpeg encode failed')
    encoded = base64.b64encode(jpg).decode()
    payload = {'state': [0.0] * 14, 'images': {name: encoded for name in CAMERAS}}
    request = urllib.request.Request(args.url, json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read())
    except Exception as error:
        raise SystemExit(f'LINK_FAIL {type(error).__name__}: {error}')
    if 'error' in result:
        raise SystemExit(f"LINK_FAIL server: {result['error']}")
    if result.get('checkpoint') != args.expected_checkpoint:
        raise SystemExit(f"LINK_FAIL checkpoint mismatch: {result.get('checkpoint')}")
    ensemble = result.get('ensemble')
    if not isinstance(ensemble, dict) or ensemble.get('enabled') is not False:
        raise SystemExit(f'LINK_FAIL v8 requires --no-ensemble, got {ensemble!r}; '
                         'existing server was left running')
    actions = np.asarray(result['actions'], dtype=np.float64)
    if actions.shape != (10, 14) or not np.isfinite(actions).all():
        raise SystemExit(f'LINK_FAIL bad actions {actions.shape}')
    print(json.dumps({'link': 'ok', 'ensemble_verified': ensemble,
                      'round_trip_s': round(time.monotonic() - started, 3),
                      'server_inference_s': result.get('inference_s'),
                      'checkpoint': result['checkpoint'],
                      'checkpoint_sha256': result.get('checkpoint_sha256'),
                      'first_action': [round(v, 4) for v in actions[0].tolist()]}, indent=2))


if __name__ == '__main__':
    main()
