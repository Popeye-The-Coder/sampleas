"""Public Agentic transition/scoring and bounded model protocol.
Selected verbatim from the frozen evaluator; no world generators or private cases.
Public worlds are development labels, never inputs to decide().
"""
from __future__ import annotations

from copy import deepcopy
import copy
import json
import math

PROTOCOL='domainpilot.v1'


REQUEST_CAP=64*1024*1024


class RunError(ValueError):
    def __init__(self,code,message): self.code=code; self.message=message; super().__init__(message)


class ParticipantError(RunError): pass


class InfrastructureError(RunError): pass


def decode(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d: raise ValueError('duplicate')
            d[k]=v
        return d
    try:
        obj=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite')))
        stack=[(obj,0)]; count=0
        while stack:
            v,depth=stack.pop(); count+=1
            if depth>40 or count>1500000: raise ValueError('complexity')
            if isinstance(v,float) and not math.isfinite(v): raise ValueError('nonfinite')
            if isinstance(v,dict): stack.extend((x,depth+1) for x in v.values())
            if isinstance(v,list): stack.extend((x,depth+1) for x in v)
        return obj
    except (ValueError,TypeError,UnicodeError,RecursionError):
        raise ParticipantError('invalid_json','Invalid JSON, duplicate keys, nonfinite numbers or excessive nesting') from None


def encode(obj,cap=REQUEST_CAP):
    try: raw=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()+b'\n'
    except (ValueError,TypeError,RecursionError): raise InfrastructureError('packaging_json','Organizer payload is not finite JSON') from None
    if len(raw)>cap: raise InfrastructureError('packaging_size','Organizer payload exceeds frozen frame cap; no history was truncated')
    return raw


def finite(value):
    json.dumps(value,allow_nan=False)
    return value


def validate_response(value):
    if not isinstance(value,dict) or set(value)!={'action','state'} or not all(isinstance(value[k],dict) for k in value):raise ValueError('Return exactly action and state objects')
    finite(value)
    if len(json.dumps(value).encode())>1024*1024:raise ValueError('Response too large')
    return value


FRAME_CAP = 1024 * 1024


def request_bytes(body):
    """One common body budget, including a conservative process envelope.

    Reserve a 32-character run ID and 20-digit sequence for both transports.
    Session also caps its actual serialized frame; HTTP uses the same compact
    UTF-8 JSON encoding. Oversized trusted observations must never cost a team
    eligibility, while state chosen by its adapter remains its responsibility.
    """
    envelope = {'protocol': PROTOCOL, 'run_id': '0' * 32,
                'seq': 10**20 - 1, 'kind': 'decide', 'payload': None}
    overhead = len(encode(envelope)) - len(b'null')
    empty_state = {'observation': body['observation'], 'state': {}}
    if len(encode(empty_state)) + overhead > FRAME_CAP:
        raise InfrastructureError('observation_size', 'Public observation exceeds request frame contract')
    raw = encode(body)
    if len(raw) + overhead > FRAME_CAP:
        raise ParticipantError('state_size', 'Participant state exceeds request frame contract')
    return raw


BODY_CAP = 128 * 1024


TEXT_CAP = 64 * 1024


REQUEST_CAP = 64 * 1024


CALL_TIMEOUT = 45


_MISSING = object()


def model_request(action):
    """Validate the frozen controller's bounded llm action and strip its op."""
    try:
        if not isinstance(action, dict) or set(action) - {'op', 'messages', 'max_output_tokens'} or action.get('op') != 'llm':
            raise ValueError
        messages = action.get('messages')
        maximum = action.get('max_output_tokens', 1024)
        if (not isinstance(messages, list) or not 1 <= len(messages) <= 24
                or type(maximum) is not int or not 16 <= maximum <= 2048):
            raise ValueError
        for item in messages:
            if (not isinstance(item, dict) or set(item) != {'role', 'content'}
                    or item['role'] not in ('system', 'user', 'assistant')
                    or not isinstance(item['content'], str)):
                raise ValueError
        if not any(m['role'] != 'system' for m in messages):
            raise ValueError
        if len(json.dumps(messages, ensure_ascii=False, allow_nan=False).encode()) > 48000:
            raise ValueError
        return {'messages': copy.deepcopy(messages), 'max_output_tokens': maximum}
    except Exception:
        raise ParticipantError('llm_request', 'Model request exceeds the published message or token bounds') from None


def model_reply(value):
    """Only model text and bounded, known numeric usage counters reach the relay."""
    try:
        if not isinstance(value, dict) or set(value) - {'text', 'usage'}:
            raise ValueError
        text, usage = value['text'], value.get('usage', {})
        if not isinstance(text, str) or not text or len(text.encode()) > TEXT_CAP or not isinstance(usage, dict):
            raise ValueError
        if len(json.dumps(usage, allow_nan=False).encode()) > 8192:
            raise ValueError
        safe = {}
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            if key in usage:
                if type(usage[key]) is not int or not 0 <= usage[key] <= 10**9:
                    raise ValueError
                safe[key] = usage[key]
        return {'text': text, 'usage': safe}
    except Exception:
        raise InfrastructureError('endpoint_response', 'Gateway returned an invalid model response; retry after checking the gateway') from None


class ModelTransport:
    """Trusted relay around an already-isolated Session; never runs source itself.

    Accepts decide(observation, state) or domainlab2's decide({observation,state}).
    Runtime must call finish_case before grading every case, including the last.
    start_case also finalizes an unfinished preceding case, preventing resets
    from bypassing the per-case minimum. Audit exposes aggregate counts only.
    """
    def __init__(self, session, provider, max_calls=8):
        if type(max_calls) is not int or not 1 <= max_calls <= 8:
            raise InfrastructureError('relay_config', 'Invalid relay model call limit')
        self.session, self.provider, self.max_calls = session, provider, max_calls
        self._active = False
        self._calls = 0
        self._cases = self._total = 0

    @property
    def audit(self):
        return {'cases': self._cases, 'model_calls': self._total}

    def start_case(self, case_id=None):
        if self._active:
            self.finish_case()
        self._active = True
        self._calls = 0

    def finish_case(self):
        if not self._active:
            raise InfrastructureError('relay_lifecycle', 'No active model relay case')
        self._active = False
        if self._calls < 1:
            raise ParticipantError('llm_required', 'At least one model call is required per case')
        self._cases += 1

    def decide(self, observation, state=_MISSING):
        if not self._active:
            raise InfrastructureError('relay_lifecycle', 'Start the model relay case before deciding')
        try:
            if state is _MISSING:
                if not isinstance(observation, dict) or set(observation) != {'observation', 'state'}:
                    raise ValueError
                body = copy.deepcopy(observation)
            else:
                body = copy.deepcopy({'observation': observation, 'state': state})
            if not isinstance(body['observation'], dict) or not isinstance(body['state'], dict):
                raise ValueError
            while True:
                request_bytes(body)
                kind, result = self.session.exchange('decide', body, cap=FRAME_CAP)
                if kind != 'result':
                    raise ParticipantError('solution_exception', 'Submitted adapter failed')
                response = validate_response(result)
                action = response['action']
                if action.get('op') != 'llm':
                    return response
                request = model_request(action)
                if self._calls >= self.max_calls:
                    raise ParticipantError('llm_limit', 'Model call limit exceeded')
                # Enforce participant state bounds before spending team inference.
                request_bytes({'observation': body['observation'], 'state': response['state']})
                try:
                    reply = model_reply(self.provider.call({'op': 'llm', **request}))
                except Exception:
                    raise InfrastructureError('endpoint_unavailable', 'Team model gateway failed; retry later') from None
                self._calls += 1
                self._total += 1
                body = {'observation': {**body['observation'], 'llm_response': reply['text'], 'llm_usage': reply['usage']},
                        'state': response['state']}
        except ParticipantError as exc:
            allowed = {'llm_request', 'llm_limit', 'state_size', 'solution_exception', 'invalid_json', 'protocol',
                       'timeout', 'unsolicited_output', 'output_limit', 'diagnostic_limit', 'worker_exit'}
            code = exc.code if exc.code in allowed else 'response_shape'
            raise ParticipantError(code, 'Submitted controller violated the runtime or model protocol') from None
        except InfrastructureError as exc:
            code = exc.code if exc.code in {'observation_size', 'endpoint_unavailable', 'runtime_unavailable'} else 'relay_unavailable'
            raise InfrastructureError(code, 'Trusted model relay unavailable; retry later') from None
        except Exception:
            raise ParticipantError('response_shape', 'Submitted controller returned an invalid response') from None


MAX_STEPS = 60


BUDGET = 14


PUBLIC_DIAGNOSTICS = ('fulfilled_fraction', 'useful_escalation_fraction', 'unnecessary_escalation_fraction',
                      'unsafe_dispatches', 'unavailable_dispatches', 'mean_tool_spend', 'invalid_actions')


TOOLS = {
    'inspect': {'cost': 1, 'arguments': ['part_id'], 'returns': 'Detailed part card and desk stock snapshot.'},
    'verify': {'cost': 2, 'arguments': ['part_id'], 'returns': 'Detailed part card and authoritative current warehouse availability.'},
    'clarify': {'cost': 2, 'arguments': ['ticket_id'], 'returns': 'Scripted customer reply giving required minimum wattage.'},
    'dispatch': {'cost': 0, 'arguments': ['ticket_id', 'part_id'], 'returns': 'Records an irreversible dispatch decision, without correctness feedback.'},
    'escalate': {'cost': 0, 'arguments': ['ticket_id'], 'returns': 'Records an irreversible manual handoff, without correctness feedback.'},
    'finish': {'cost': 0, 'arguments': [], 'returns': 'Closes the batch; undecided requests receive zero outcome credit.'},
}


POLICY = ('Dispatch one compatible available part per request, or escalate. Match voltage and connector exactly; '
          'output watts must meet or exceed the requested minimum. Every request needs either 10 or 20 watts. A higher wattage rating is permitted. '
          'Each dispatch consumes one unit. Each request has one irreversible decision. Dispatch responses '
          'acknowledge recording only; they do not reveal compatibility or fulfillment. All six requests share '
          '14 tool units. Recent snapshots are synchronized at batch open; aged snapshots can disagree with the warehouse. Invalid actions cost one unit when any remain. No outside technical assumptions are needed: '
          'these are fictional fixed-voltage training parts, not real repair instructions.')


def _compatible(part, req):
    return (part['voltage'] == req['voltage'] and part['connector'] == req['connector']
            and part['watts'] >= req['watts'])


def start(case):
    state = {'steps':0, 'spent':0, 'errors':0, 'decisions':{}, 'done':False,
             'tool_history':[], 'stock':dict(case['stock'])}
    return state, _observation(case, state)


def _observation(case, state, result=None):
    obs = {'case_id':case['case_id'], 'policy':(POLICY.replace('Recent snapshots are synchronized at batch open; aged snapshots can disagree with the warehouse.', 'Operational update: both recent and aged snapshots may be stale; only verify is authoritative.') if case['condition']=='changed' else POLICY), 'budget_remaining':BUDGET-state['spent'],
           'steps_remaining':MAX_STEPS-state['steps'], 'tools':deepcopy(TOOLS),
           'tickets':[{'id':t['id'],'text':t['text']} for t in case['tickets']],
           'catalog':[{'id':p['id'],'description':p['description'],'snapshot_age':p['snapshot_age']} for p in case['parts']],
           'decided_ticket_ids':list(state['decisions']), 'done':state['done']}
    if result is not None:
        obs['result'] = result
    return obs


def step(case, state, action):
    state = deepcopy(state)
    if state['done']:
        return state, _observation(case,state,{'ok':False,'error':'batch_closed'}), True
    state['steps'] += 1
    parts = {p['id']:p for p in case['parts']}
    tickets = {t['id']:t for t in case['tickets']}
    error = None
    op = action.get('op') if isinstance(action,dict) else None
    if not isinstance(op,str) or op not in TOOLS:
        error = 'invalid_operation'
    elif set(action) != {'op', *TOOLS[op]['arguments']}:
        error = 'invalid_arguments'
    elif 'part_id' in action and (not isinstance(action['part_id'],str) or action['part_id'] not in parts):
        error = 'unknown_part'
    elif 'ticket_id' in action and (not isinstance(action['ticket_id'],str) or action['ticket_id'] not in tickets):
        error = 'unknown_ticket'
    elif 'ticket_id' in action and action['ticket_id'] in state['decisions']:
        error = 'ticket_already_decided'
    elif state['spent'] + TOOLS[op]['cost'] > BUDGET:
        error = 'budget_exhausted'
    if error:
        state['errors'] += 1
        state['spent'] = min(BUDGET,state['spent']+1)
        result = {'ok':False,'error':error}
    else:
        state['spent'] += TOOLS[op]['cost']
        if op in ('inspect','verify'):
            part = parts[action['part_id']]
            data = {k:part[k] for k in ('id','voltage','connector','watts')}
            source = case['cache'] if op=='inspect' else state['stock']
            # Cached counts do not track writes. Decided IDs prevent retry exploitation.
            data.update({'available':source[part['id']], 'source':'desk_snapshot' if op=='inspect' else 'warehouse'})
            state['tool_history'].append({'op':op,'part_id':part['id']})
        elif op=='clarify':
            data = {'ticket_id':action['ticket_id'], 'reply':tickets[action['ticket_id']]['reply']}
            state['tool_history'].append({'op':op,'ticket_id':action['ticket_id']})
        elif op in ('dispatch','escalate'):
            decision = {'op':op}
            if op=='dispatch':
                pid = action['part_id']
                decision.update({'part_id':pid,'available_at_dispatch':state['stock'][pid] > 0})
                if state['stock'][pid] > 0:
                    state['stock'][pid] -= 1
            state['decisions'][action['ticket_id']] = decision
            data = {'recorded':True}
        else:
            state['done'] = True
            data = {'closed':True}
        result = {'ok':True,'data':data}
    if len(state['decisions'])==len(case['tickets']) or state['steps']>=MAX_STEPS:
        state['done'] = True
    return state, _observation(case,state,result), state['done']


def grade(case, state):
    n = len(case['tickets'])
    parts = {p['id']:p for p in case['parts']}
    fulfilled = useful = unnecessary = unsafe = unavailable = 0
    utility = 0.0
    for ticket in case['tickets']:
        decision = state['decisions'].get(ticket['id'])
        if not decision:
            continue
        feasible = any(_compatible(p,ticket['requirements']) and case['stock'][p['id']]>0 for p in case['parts'])
        if decision['op']=='escalate':
            if feasible:
                unnecessary += 1
                utility += .15
            else:
                useful += 1
                utility += 1
        else:
            part = parts[decision['part_id']]
            if not _compatible(part,ticket['requirements']):
                unsafe += 1
            elif not decision['available_at_dispatch']:
                unavailable += 1
            else:
                fulfilled += 1
                utility += 1
    correct = fulfilled + useful
    efficiency = max(0.0,min(1.0,BUDGET-state['spent']))
    earned = {'outcome':80*utility/n, 'integrity':15*len(state['decisions'])/n,
              'efficiency':5*correct/n*efficiency}
    if unsafe:
        earned = {name:0.0 for name in earned}
    components = {name:{'earned':round(value,6),'max':maximum}
                  for (name,value),maximum in zip(earned.items(),(80,15,5))}
    score = round(sum(v['earned'] for v in components.values()),6)
    return {'score':score, 'components':components,
            'diagnostics':{'fulfilled_fraction':fulfilled/n,'useful_escalation_fraction':useful/n,
                'unnecessary_escalation_fraction':unnecessary/n,'unsafe_dispatches':unsafe,
                'unavailable_dispatches':unavailable,'mean_tool_spend':state['spent'],'invalid_actions':state['errors']}}


def aggregate(results):
    if not results:
        return {'score':0.0,'components':{},'diagnostics':{key:0.0 for key in PUBLIC_DIAGNOSTICS}}
    return {'score':sum(r['score'] for r in results)/len(results),
            'components':{name:{'earned':sum(r['components'][name]['earned'] for r in results)/len(results),
                                'max':results[0]['components'][name]['max']}
                          for name in results[0]['components']},
            'diagnostics':feedback_diagnostics(results)}


def feedback_diagnostics(results):
    return {key:sum(r['diagnostics'][key] for r in results)/len(results) if results else 0.0
            for key in PUBLIC_DIAGNOSTICS}
