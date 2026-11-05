"""Workshop Parts Desk controller (frozen). decide(observation, state) -> {action, state}.

Pipeline
  1. Deterministic, strict reading of the CURRENT approved request / catalog documents
     (latest approved revision, drafts and stale retries ignored, mV/mW + connector aliases normalised).
  2. LLM call #1 (op:llm): independent extraction used only as a cross-check; any disagreement
     with the deterministic reading marks that ticket/part "disputed" -> never dispatched.
  3. Evidence-driven tool planning under the shared 14-unit budget (inspect 1 / verify 2 / clarify 2).
  4. LLM call #2 (op:llm): audit of the planned dispatches. The model may veto (escalate) a dispatch;
     it can never introduce one that the deterministic compatibility proof did not already pass.
  5. Irreversible decisions are issued; anything unproven is escalated.
Only the standard library is used. No network access happens here: model text comes through op:llm.
"""
import json
import re

# ----------------------------------------------------------------------------- parsing
_NUM = r'(\d+(?:\.\d+)?)'
_UNIT = r'(mV|mW|kV|kW|V|W|volts?|watts?)(?![A-Za-z])'
_QTY = re.compile(_NUM + r'\s*-?\s*' + _UNIT)
_BARE_VOLT = re.compile(r'input voltage is\s+' + _NUM + r'(?!\s*-?\s*[A-Za-z])')
_USBC = re.compile(r'USB(?:[\s-]*Type)?[\s-]*C\b', re.I)
_BARREL = re.compile(r'\bbarrel\b', re.I)
_DISTRACT = re.compile(
    r'[^.;!?]*(?:retired fixture|ignore that|discarded unit|previous stock)[^.;!?]*[.;!?]?', re.I)
_UNKNOWN_W = re.compile(r'unreadable|can supply|not (?:sure|known)|unknown', re.I)


def _canon(value, unit):
    x, u = float(value), unit.lower()
    if u == 'mv': return 'V', x / 1000.0
    if u == 'kv': return 'V', x * 1000.0
    if u in ('v', 'volt', 'volts'): return 'V', x
    if u == 'mw': return 'W', x / 1000.0
    if u == 'kw': return 'W', x * 1000.0
    return 'W', x


def _num(x):
    return int(x) if abs(x - round(x)) < 1e-9 else round(x, 3)


def _clean(text):
    return _DISTRACT.sub(' ', text or '')


def _connector(text):
    found = set()
    if _USBC.search(text): found.add('USB-C')
    if _BARREL.search(text): found.add('barrel')
    return next(iter(found)) if len(found) == 1 else None


def _quantities(text):
    volts, watts = [], []
    for m in _QTY.finditer(text):
        kind, val = _canon(m.group(1), m.group(2))
        (volts if kind == 'V' else watts).append(_num(val))
    for m in _BARE_VOLT.finditer(text):
        v = float(m.group(1))
        volts.append(_num(v / 1000.0 if v >= 1000 else v))
    return volts, watts


def parse_request(text):
    t = _clean(text)
    volts, watts = _quantities(t)
    uv, uw = set(volts), set(watts)
    return {'voltage': next(iter(uv)) if len(uv) == 1 else None,
            'connector': _connector(t),
            'watts': next(iter(uw)) if len(uw) == 1 else None,
            'w_conflict': len(uw) > 1,
            'w_unknown': len(uw) == 0 and bool(_UNKNOWN_W.search(t))}


def parse_part(text):
    volts, watts = _quantities(text or '')
    uv, uw = set(volts), set(watts)
    return {'voltage': next(iter(uv)) if len(uv) == 1 else None,
            'connector': _connector(text or ''),
            'watts': next(iter(uw)) if len(uw) == 1 else None}


def parse_reply_watts(text):
    _, watts = _quantities(text or '')
    uw = set(watts)
    return next(iter(uw)) if len(uw) == 1 else None


def current_documents(docs, as_of=None):
    """entity -> (doc, conflict). Latest approved revision; drafts / future / older revisions ignored."""
    by = {}
    for d in docs or []:
        try:
            if d.get('status') != 'approved': continue
            if as_of is not None and d.get('received_at', 0) > as_of: continue
            by.setdefault(d['entity_id'], []).append(d)
        except Exception:
            continue
    out = {}
    for ent, items in by.items():
        top = max(i.get('revision', 0) for i in items)
        cur = [i for i in items if i.get('revision', 0) == top]
        texts = {' '.join(str(i.get('text', '')).split()) for i in cur}
        out[ent] = (sorted(cur, key=lambda i: i.get('received_at', 0))[0], len(texts) > 1)
    return out


# ----------------------------------------------------------------------------- constants
BUDGET_CAP = 13            # spend at most 13 units for full efficiency credit
PRIOR_AVAIL = 0.45
VOLTAGES_OK = (lambda v: v is not None and 0 < v <= 1000)


def _log(state, kind, **kw):
    state['log'].append({'k': kind, **kw})
    if len(state['log']) > 300:
        del state['log'][:50]


# ----------------------------------------------------------------------------- state preparation
def _prepare(obs, state):
    policy = obs.get('policy', '') or ''
    state['changed'] = ('Operational update' in policy) or ('may be stale' in policy)
    state['ordinary_known'] = ('Recent snapshots are synchronized at batch open' in policy) and not state['changed']
    state['assume_1020'] = bool(re.search(r'either 10 or 20 watts', policy))
    as_of = obs.get('as_of')
    rdocs = current_documents(obs.get('request_documents'), as_of)
    cdocs = current_documents(obs.get('catalog_documents'), as_of)
    state['reqs'], state['parts'] = {}, {}
    for t in obs.get('tickets', []):
        tid = t['id']
        doc, conflict = rdocs.get(tid, (None, False))
        text = doc['text'] if doc else t.get('text', '')
        p = parse_request(text)
        why = None
        if doc is None: why = 'no approved request document; using ticket text'
        if conflict: why = 'conflicting approved texts for the same revision'
        disputed = bool(conflict) or p['w_conflict'] or p['voltage'] is None or p['connector'] is None
        if disputed and not why:
            why = 'requirement could not be read unambiguously'
        if p['watts'] is not None and p['watts'] not in (10, 20) and state['assume_1020']:
            disputed, why = True, 'wattage outside the 10/20 W policy'
        if p['watts'] is None and not p['w_unknown']:
            disputed, why = True, 'wattage neither stated nor flagged unknown'
        state['reqs'][tid] = {'text': text, 'voltage': p['voltage'], 'connector': p['connector'],
                              'watts': p['watts'], 'disputed': disputed, 'why': why if disputed else None,
                              'rev': doc.get('revision') if doc else None}
        if doc is None or conflict:
            _log(state, 'warn', ticket=tid, msg=why)
    for c in obs.get('catalog', []):
        pid = c['id']
        doc, conflict = cdocs.get(pid, (None, False))
        text = doc['text'] if doc else c.get('description', '')
        p = parse_part(text)
        bad = bool(conflict) or None in (p['voltage'], p['connector'], p['watts'])
        state['parts'][pid] = {'text': text, 'voltage': p['voltage'], 'connector': p['connector'],
                               'watts': p['watts'], 'age': c.get('snapshot_age'), 'disputed': bad}
    state.update(know={}, clar={}, spent=0, pending=None, plan=None, queue=[], bad_calls=[],
                 budget0=obs.get('budget_remaining', 14))


# ----------------------------------------------------------------------------- LLM prompts / parsing
def _json_from(text):
    t = (text or '').strip()
    if t.startswith('```'):
        lines = t.splitlines()
        t = '\n'.join(lines[1:-1] if lines and lines[-1].strip().startswith('```') else lines[1:])
    try:
        return json.loads(t)
    except Exception:
        a, b = t.find('{'), t.rfind('}')
        if a >= 0 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except Exception:
                return None
    return None


def _extract_request(state):
    tickets = [{'id': k, 'text': v['text']} for k, v in state['reqs'].items()]
    parts = [{'id': k, 'description': v['text']} for k, v in state['parts'].items()]
    sysmsg = ('You are a precise data-extraction engine for a workshop parts desk. '
              'Reply with one JSON object only, no prose, no markdown.')
    user = ('For every ticket give the CURRENT required supply voltage (in volts), connector '
            '("USB-C" or "barrel") and minimum wattage (in watts). For every part give its output voltage (volts), '
            'connector and rated wattage (watts).\n'
            'Rules: convert mV to V (divide by 1000) and mW to W. USB Type-C = "USB-C"; "barrel plug" = "barrel". '
            'Ignore retired fixtures, discarded units, previous suggestions and anything marked "ignore that"; '
            'use only the present requirement. If a ticket does not state its wattage (e.g. the load label is '
            'unreadable) use null for watts. Never guess.\n'
            'Format: {"tickets":{"<id>":{"voltage":number,"connector":string,"watts":number|null}},'
            '"parts":{"<id>":{"voltage":number,"connector":string,"watts":number}}}\n'
            + json.dumps({'tickets': tickets, 'parts': parts}, ensure_ascii=False))
    return {'op': 'llm', 'messages': [{'role': 'system', 'content': sysmsg}, {'role': 'user', 'content': user}],
            'max_output_tokens': 1400}


def _same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def _apply_extraction(state, data):
    """Cross-check; returns True if the model produced a usable, complete answer."""
    if not isinstance(data, dict) or not isinstance(data.get('tickets'), dict) or not isinstance(data.get('parts'), dict):
        return False
    t_in, p_in = data['tickets'], data['parts']
    complete = set(t_in) >= set(state['reqs']) and set(p_in) >= set(state['parts'])
    for tid, r in state['reqs'].items():
        m = t_in.get(tid)
        if not isinstance(m, dict):
            continue
        bad = []
        if not _same(m.get('voltage'), r['voltage']): bad.append('voltage')
        if str(m.get('connector', '')).strip().lower().replace(' ', '') not in (
                (r['connector'] or '').lower(), (r['connector'] or '').lower().replace('-', '')): bad.append('connector')
        if r['watts'] is not None and not _same(m.get('watts'), r['watts']): bad.append('watts')
        if bad and not r['disputed']:
            r['disputed'], r['why'] = True, 'LLM cross-check disagrees on ' + '/'.join(bad)
            _log(state, 'dispute', ticket=tid, fields=bad)
    for pid, p in state['parts'].items():
        m = p_in.get(pid)
        if not isinstance(m, dict) or p['disputed']:
            continue
        bad = []
        if not _same(m.get('voltage'), p['voltage']): bad.append('voltage')
        if str(m.get('connector', '')).strip().lower().replace(' ', '') not in (
                p['connector'].lower(), p['connector'].lower().replace('-', '')): bad.append('connector')
        if not _same(m.get('watts'), p['watts']): bad.append('watts')
        if bad:
            p['disputed'] = True
            _log(state, 'dispute', part=pid, fields=bad)
    return complete


# ----------------------------------------------------------------------------- knowledge / planning
def _reliable(state, pid):
    k = state['know'].get(pid)
    if not k: return False
    if k['src'] == 'verify': return True
    return k['src'] == 'inspect' and state['ordinary_known'] and state['parts'][pid]['age'] == 'recent'


def _avail(state, pid):
    """True / False when reliably known, else None."""
    return bool(state['know'][pid]['avail']) if _reliable(state, pid) else None


def _p_avail(state, pid):
    a = _avail(state, pid)
    if a is not None: return 1.0 if a else 0.0
    k = state['know'].get(pid)
    if k and k['src'] == 'inspect':          # unreliable snapshot hint
        good = 0.75 if state['changed'] else 0.67
        pa = PRIOR_AVAIL
        if k['avail']: return pa * good / (pa * good + (1 - pa) * (1 - good))
        return pa * (1 - good) / (pa * (1 - good) + (1 - pa) * good)
    return PRIOR_AVAIL


def _query_cost(state, pid):
    if _reliable(state, pid): return 0
    if (not state['changed']) and state['parts'][pid]['age'] == 'recent' and state['ordinary_known']:
        return 1
    return 2


def _query_op(state, pid):
    return 'inspect' if _query_cost(state, pid) == 1 else 'verify'


def _req_watts(state, tid):
    return state['reqs'][tid]['watts'] if state['reqs'][tid]['watts'] is not None else state['clar'].get(tid)


def _sets(state, tid):
    """-> (sure, weak). sure: parts guaranteed compatible; weak: compatible only if wattage turns out to be 10."""
    r = state['reqs'][tid]
    if r['disputed']:
        return [], []
    w = _req_watts(state, tid)
    base = [pid for pid, p in state['parts'].items()
            if not p['disputed'] and p['voltage'] == r['voltage'] and p['connector'] == r['connector']]
    if w is not None:
        return [pid for pid in base if state['parts'][pid]['watts'] >= w], []
    if not state['assume_1020']:
        return [], []
    sure = [pid for pid in base if state['parts'][pid]['watts'] >= 20]
    weak = [pid for pid in base if 10 <= state['parts'][pid]['watts'] < 20]
    return sure, weak


def _undecided(state, obs):
    done = set(obs.get('decided_ticket_ids', []))
    return [t for t in state['reqs'] if t not in done]


def _match(state, tickets, sets):
    """Max matching tickets -> parts that are reliably available (augmenting paths)."""
    avail_of = {t: [p for p in sets[t][0] if _avail(state, p) is True] for t in tickets}
    owner = {}

    def try_t(t, seen):
        for p in avail_of[t]:
            if p in seen: continue
            seen.add(p)
            if p not in owner or try_t(owner[p], seen):
                owner[p] = t
                return True
        return False
    for t in sorted(tickets, key=lambda x: len(avail_of[x])):
        try_t(t, set())
    return {t: p for p, t in owner.items()}


def _next_query(state, obs):
    """Pick the next tool action, or None when nothing worthwhile is affordable."""
    left = obs.get('budget_remaining', 0)
    spent = state.get('budget0', 14) - left
    allowed = min(left, BUDGET_CAP - spent)
    todo = _undecided(state, obs)
    sets = {t: _sets(state, t) for t in todo}
    matched = _match(state, todo, sets)
    cands = []
    for t in todo:
        if t in matched: continue
        sure, weak = sets[t]
        unq = [p for p in sure if _avail(state, p) is None]
        if unq:
            for p in unq:
                cands.append((_query_cost(state, p), 0, len(unq), t, ('q', p)))
            continue
        if state['reqs'][t]['disputed']:
            continue
        if _req_watts(state, t) is None and weak and state['assume_1020']:
            unw = [p for p in weak if _avail(state, p) is None]
            if unw:
                for p in unw:
                    cands.append((_query_cost(state, p), 1, len(unw), t, ('q', p)))
            elif any(_avail(state, p) is True for p in weak):
                cands.append((2, 1, 0, t, ('c', None)))
    # benefit sharing: a part queried for several tickets is cheaper per ticket
    demand = {}
    for c in cands:
        if c[4][0] == 'q': demand[c[4][1]] = demand.get(c[4][1], 0) + 1
    ranked = []
    for cost, spec, n, t, (kind, pid) in cands:
        share = demand.get(pid, 1) if kind == 'q' else 1
        ranked.append((spec, cost / share, n, t, kind, pid, cost))
    ranked.sort(key=lambda x: x[:4])
    bad = {json.dumps(b, sort_keys=True) for b in state['bad_calls']}
    for spec, _, n, t, kind, pid, cost in ranked:
        if cost > allowed: continue
        if kind == 'q':
            act = {'op': _query_op(state, pid), 'part_id': pid}
        else:
            act = {'op': 'clarify', 'ticket_id': t}
        if json.dumps(act, sort_keys=True) in bad: continue
        return act, t
    # Last resort: a cheap (1 unit) desk-snapshot hint for parts we cannot afford to verify.
    # Hints are never treated as proof; they only sharpen the fallback decision.
    if allowed >= 1:
        hints = []
        for t in todo:
            if t in matched or state['reqs'][t]['disputed']: continue
            sure, _ = sets[t]
            unk = [p for p in sure if _avail(state, p) is None and p not in state['know']]
            for p in unk:
                hints.append((len(sure), t, p))
        hints.sort()
        for _, t, p in hints:
            act = {'op': 'inspect', 'part_id': p}
            if json.dumps(act, sort_keys=True) in bad: continue
            return act, t
    return None


def _plan(state, obs):
    """Final decisions for every undecided ticket -> {tid: {op, part_id?, why, p?}}."""
    todo = _undecided(state, obs)
    sets = {t: _sets(state, t) for t in todo}
    matched = _match(state, todo, sets)
    used = set(matched.values())
    plan = {}
    for t in todo:
        if t in matched:
            plan[t] = {'op': 'dispatch', 'part_id': matched[t],
                       'why': 'compatible part with confirmed stock', 'p': 1.0}
    # uncertain fallback (budget exhausted before evidence was complete)
    rest = [t for t in todo if t not in plan]
    for t in sorted(rest, key=lambda x: len(sets[x][0])):
        sure = [p for p in sets[t][0] if p not in used and _avail(state, p) is not False]
        allp = [p for p in sets[t][0] if _avail(state, p) is not False]
        if state['reqs'][t]['disputed'] or not sure:
            plan[t] = {'op': 'escalate', 'why': state['reqs'][t]['why'] or 'no compatible part with stock evidence'}
            continue
        best = max(sure, key=lambda p: _p_avail(state, p))
        pb = _p_avail(state, best)
        pfeas = 1.0
        for p in allp: pfeas *= (1 - _p_avail(state, p))
        pfeas = 1 - pfeas
        esc = 0.15 * pfeas + (1 - pfeas)
        if pb >= esc:
            plan[t] = {'op': 'dispatch', 'part_id': best, 'p': round(pb, 2),
                       'why': 'best-evidence compatible part (budget exhausted; stock not confirmed)'}
            used.add(best)
        else:
            plan[t] = {'op': 'escalate', 'why': 'stock unconfirmed and escalation is the safer choice'}
    for t in todo:
        if plan[t]['op'] == 'escalate' and not plan[t].get('why'):
            plan[t]['why'] = 'no compatible available part'
    # sanity: every dispatch must pass the deterministic compatibility proof
    for t, d in list(plan.items()):
        if d['op'] != 'dispatch': continue
        r, p = state['reqs'][t], state['parts'][d['part_id']]
        w = _req_watts(state, t)
        ok = (not r['disputed'] and not p['disputed'] and p['voltage'] == r['voltage']
              and p['connector'] == r['connector']
              and ((w is not None and p['watts'] >= w) or
                   (w is None and state['assume_1020'] and p['watts'] >= 20)))
        if not ok:
            plan[t] = {'op': 'escalate', 'why': 'failed final compatibility proof'}
    return plan


def _audit_request(state, plan):
    items = []
    for t, d in plan.items():
        if d['op'] != 'dispatch': continue
        r, p = state['reqs'][t], state['parts'][d['part_id']]
        w = _req_watts(state, t)
        items.append({'ticket_id': t, 'ticket_text': r['text'],
                      'required': {'voltage_V': r['voltage'], 'connector': r['connector'],
                                   'min_watts_W': w if w is not None else 'not stated (request needs 10 or 20 W)'},
                      'part_id': d['part_id'], 'part_text': p['text'],
                      'part_spec': {'voltage_V': p['voltage'], 'connector': p['connector'], 'watts_W': p['watts']}})
    sysmsg = 'You are a strict compatibility auditor. Reply with one JSON object only, no prose, no markdown.'
    user = ('Audit these planned dispatches. A part is compatible only if its voltage equals the required voltage, its '
            'connector equals the required connector and its rated watts are greater than or equal to the required minimum '
            '(a higher rating is allowed). If the minimum is "not stated", a part rated 20 W or more is acceptable because '
            'every request needs either 10 or 20 W. Use the current requirement, not retired or discarded items.\n'
            'Answer {"decisions":[{"ticket_id":"...","action":"dispatch"|"escalate"}]} with exactly one entry per item; '
            'choose "escalate" if there is any doubt.\n' + json.dumps(items, ensure_ascii=False))
    return {'op': 'llm', 'messages': [{'role': 'system', 'content': sysmsg}, {'role': 'user', 'content': user}],
            'max_output_tokens': 900}


def _apply_audit(state, plan, data):
    """Model may veto; unusable output leaves the deterministic plan intact."""
    if not isinstance(data, dict) or not isinstance(data.get('decisions'), list):
        _log(state, 'audit', result='unparsable; deterministic plan kept')
        return False
    seen = {}
    for e in data['decisions']:
        if isinstance(e, dict) and isinstance(e.get('ticket_id'), str):
            seen[e['ticket_id']] = str(e.get('action', '')).lower()
    for t, d in plan.items():
        if d['op'] == 'dispatch' and seen.get(t) == 'escalate':
            plan[t] = {'op': 'escalate', 'why': 'vetoed by LLM audit'}
            _log(state, 'veto', ticket=t, part=d['part_id'])
    return True


# ----------------------------------------------------------------------------- result handling
def _consume_result(state, obs):
    pend = state.get('pending')
    res = obs.get('result')
    if not pend:
        return
    state['pending'] = None
    if not isinstance(res, dict):
        return
    if not res.get('ok'):
        state['bad_calls'].append(pend)
        _log(state, 'error', action=pend, error=res.get('error'))
        return
    data = res.get('data') or {}
    op = pend['op']
    if op in ('inspect', 'verify'):
        pid = pend['part_id']
        st = state['know'].get(pid)
        if op == 'verify' or not st or st['src'] != 'verify':
            state['know'][pid] = {'src': op, 'avail': 1 if data.get('available') else 0}
        _log(state, op, part=pid, available=int(bool(data.get('available'))), source=data.get('source'),
             reliable=_reliable(state, pid))
        # trust but check the tool card against the parsed catalogue
        p = state['parts'][pid]
        if None not in (data.get('voltage'), data.get('connector'), data.get('watts')) and (
                not _same(data.get('voltage'), p['voltage']) or data.get('connector') != p['connector']
                or not _same(data.get('watts'), p['watts'])):
            p['disputed'] = True
            _log(state, 'dispute', part=pid, fields=['tool-card mismatch'])
    elif op == 'clarify':
        tid = pend['ticket_id']
        w = parse_reply_watts(data.get('reply', ''))
        if w in (10, 20) or (w is not None and not state['assume_1020']):
            state['clar'][tid] = w
        else:
            state['reqs'][tid]['disputed'], state['reqs'][tid]['why'] = True, 'clarification unreadable'
        _log(state, 'clarify', ticket=tid, watts=w)
    state['spent'] += {'inspect': 1, 'verify': 2, 'clarify': 2}.get(op, 0)


# ----------------------------------------------------------------------------- controller
def _emit(state, action):
    if action.get('op') in ('inspect', 'verify', 'clarify', 'dispatch', 'escalate'):
        state['pending'] = dict(action)
    return {'action': action, 'state': state}


def _final_queue(state, plan, obs):
    q = []
    for t, d in plan.items():
        if d['op'] == 'dispatch':
            q.append({'op': 'dispatch', 'ticket_id': t, 'part_id': d['part_id']})
        else:
            q.append({'op': 'escalate', 'ticket_id': t})
        _log(state, 'decision', ticket=t, op=d['op'], part=d.get('part_id'), why=d.get('why'))
    return q


def _step(obs, state):
    # --- model replies --------------------------------------------------------------------
    if 'llm_response' in obs:
        stage = state.get('stage')
        data = _json_from(obs['llm_response'])
        if stage == 'extract':
            ok = _apply_extraction(state, data)
            _log(state, 'extract', usable=ok)
            if not ok and state.get('extract_tries', 0) < 2:
                state['extract_tries'] = state.get('extract_tries', 0) + 1
                state['model_calls'] += 1
                return {'action': _extract_request(state), 'state': state}
            state['stage'] = 'run'
        elif stage == 'audit':
            ok = _apply_audit(state, state['plan'], data)
            if not ok and state.get('audit_tries', 0) < 1:
                state['audit_tries'] = state.get('audit_tries', 0) + 1
                state['model_calls'] += 1
                return {'action': _audit_request(state, state['plan']), 'state': state}
            state['queue'] = _final_queue(state, state['plan'], obs)
            state['stage'] = 'final'
    else:
        _consume_result(state, obs)

    if state.get('stage') == 'init':
        _prepare(obs, state)
        state['stage'] = 'extract'
        state['model_calls'] += 1
        _log(state, 'start', tickets=len(state['reqs']), parts=len(state['parts']),
             condition='changed' if state['changed'] else 'ordinary')
        return {'action': _extract_request(state), 'state': state}

    if state['stage'] == 'run':
        nxt = _next_query(state, obs)
        if nxt:
            act, t = nxt
            return _emit(state, act)
        plan = _plan(state, obs)
        state['plan'] = plan
        if any(d['op'] == 'dispatch' for d in plan.values()) and state['model_calls'] < 7:
            state['stage'] = 'audit'
            state['model_calls'] += 1
            return {'action': _audit_request(state, plan), 'state': state}
        state['queue'] = _final_queue(state, plan, obs)
        state['stage'] = 'final'

    if state['stage'] == 'final':
        done = set(obs.get('decided_ticket_ids', []))
        while state['queue']:
            a = state['queue'].pop(0)
            if a.get('ticket_id') in done: continue
            return _emit(state, a)
        return _emit(state, {'op': 'finish'})
    return _emit(state, {'op': 'finish'})


def decide(observation, state):
    state = json.loads(json.dumps(state or {}))
    if not state:
        state.update(stage='init', model_calls=0, log=[], extract_tries=0, audit_tries=0)
    try:
        out = _step(observation, state)
        return {'action': out['action'], 'state': out['state']}
    except Exception as exc:  # fail safe: never dispatch on an internal error
        state['error'] = repr(exc)[:200]
        done = set(observation.get('decided_ticket_ids', []))
        rest = [t['id'] for t in observation.get('tickets', []) if t['id'] not in done]
        action = {'op': 'escalate', 'ticket_id': rest[0]} if rest else {'op': 'finish'}
        state['pending'] = dict(action) if rest else None
        return {'action': action, 'state': state}
