
"""WORKSHOP AI Gateway - Groq + Dashboard API."""

import hmac
import importlib.util
import json
import os
import time
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

GROQ_URL = os.environ.get(
    "GROQ_URL",
    "https://api.groq.com/openai/v1/chat/completions"
)

MODELS_URL = GROQ_URL.rsplit("/chat/completions", 1)[0] + "/models"

PREFERRED = [
    m for m in (
        os.environ.get("GROQ_MODEL"),
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b",
    ) if m
]

PROVIDER_TIMEOUT = 35
MAX_BODY = 65536

_model_cache = {
    "name": None,
    "at": 0.0,
    "available": []
}

app = FastAPI(
    title="Workshop AI Gateway",
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.environ.get(
            "ALLOWED_ORIGINS", "*"
        ).split(",")
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

stats = {
    "gateway_calls": 0,
    "gateway_errors": 0,
    "last_latency_ms": None,
    "last_error": None,
    "input_tokens": 0,
    "output_tokens": 0,
    "started": time.time()
}


def _token_ok(request: Request):
    expected = os.environ.get("TEAM_GATEWAY_TOKEN", "")
    got = request.headers.get("Authorization", "")

    if not expected or not hmac.compare_digest(
        got, "Bearer " + expected
    ):
        raise HTTPException(
            status_code=403,
            detail="forbidden"
        )


async def resolve_model(client: httpx.AsyncClient) -> str:

    if (
        _model_cache["name"]
        and time.time() - _model_cache["at"] < 600
    ):
        return _model_cache["name"]

    key = os.environ["GROQ_API_KEY"]

    try:
        response = await client.get(
            MODELS_URL,
            headers={"Authorization": "Bearer " + key},
            timeout=10
        )

        ids = (
            [m["id"] for m in response.json().get("data", [])]
            if response.status_code == 200
            else []
        )

    except Exception:
        ids = []

    _model_cache["available"] = ids

    name = next(
        (m for m in PREFERRED if m in ids),
        None
    )

    if not name and ids:
        name = next(
            (
                i for i in ids
                if "llama" in i
                and "guard" not in i
                and "whisper" not in i
            ),
            None
        )

    name = name or PREFERRED[0]

    _model_cache.update(
        name=name,
        at=time.time()
    )

    return name


async def call_groq(messages, max_tokens, model=None):

    async with httpx.AsyncClient(
        follow_redirects=False
    ) as client:

        model = model or await resolve_model(client)
        t0 = time.time()

        is_gpt_oss = model.startswith("openai/gpt-oss-")

        completion_budget = max(
            int(max_tokens),
            2048 if is_gpt_oss else 1024
        )

        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": completion_budget,
            "temperature": 0
        }

        if is_gpt_oss:
            payload["reasoning_effort"] = "low"
            payload["include_reasoning"] = False

        try:
            response = await client.post(
                GROQ_URL,
                timeout=PROVIDER_TIMEOUT,
                headers={
                    "Authorization": (
                        "Bearer " + os.environ["GROQ_API_KEY"]
                    ),
                    "Content-Type": "application/json"
                },
                json=payload
            )

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "Groq request timed out"
            ) from exc

        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Could not connect to Groq"
            ) from exc

        if response.status_code >= 400:

            try:
                error_body = response.json()
                detail = error_body.get(
                    "error", {}
                ).get("message", "")

            except (ValueError, AttributeError):
                detail = ""

            detail = str(detail).replace(
                os.environ.get("GROQ_API_KEY", ""),
                "[redacted]"
            )

            detail = (
                detail[:240]
                if detail
                else "No error message returned"
            )

            raise RuntimeError(
                f"Groq HTTP {response.status_code}: {detail}"
            )

        try:
            body = response.json()

        except ValueError as exc:
            raise RuntimeError(
                "Groq returned invalid JSON"
            ) from exc

        choices = body.get("choices")

        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise RuntimeError(
                "Groq response did not contain a completion choice"
            )

        message = choices[0].get("message") or {}

        content = (
            message.get("content")
            if isinstance(message, dict)
            else None
        )

        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
                and isinstance(block.get("text", ""), str)
            )

        if not isinstance(content, str) or not content.strip():

            finish_reason = choices[0].get(
                "finish_reason", "unknown"
            )

            raise RuntimeError(
                f"Groq returned empty content: {finish_reason}"
            )

        if len(content.encode("utf-8")) > 65536:
            raise RuntimeError(
                "Groq response exceeded size limit"
            )

        usage = body.get("usage") or {}

        output_usage = {
            output_key: usage[input_key]
            for input_key, output_key in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
                ("total_tokens", "total_tokens")
            )
            if type(usage.get(input_key)) is int
            and usage[input_key] >= 0
        }

        stats["last_latency_ms"] = int(
            (time.time() - t0) * 1000
        )

        return {
            "text": content,
            "usage": output_usage,
            "model": model
        }


@app.post("/gateway")
async def gateway(
    request: Request,
    _=Depends(_token_ok)
):

    try:
        raw = await request.body()

        if not 0 < len(raw) <= MAX_BODY:
            raise ValueError

        frame = json.loads(raw)

        if set(frame) != {"messages", "max_output_tokens"}:
            raise ValueError

        msgs = frame["messages"]
        mx = frame["max_output_tokens"]

        if not isinstance(msgs, list) or not 1 <= len(msgs) <= 24:
            raise ValueError

        if type(mx) is not int or not 16 <= mx <= 2048:
            raise ValueError

        for msg in msgs:

            if (
                not isinstance(msg, dict)
                or set(msg) != {"role", "content"}
                or msg["role"] not in (
                    "system", "user", "assistant"
                )
                or not isinstance(msg["content"], str)
            ):
                raise ValueError

    except Exception:
        return JSONResponse(
            {"error": "bad_request"},
            status_code=400
        )

    try:
        result = await call_groq(msgs, mx)

        stats["gateway_calls"] += 1
        stats["input_tokens"] += result["usage"].get(
            "input_tokens", 0
        )
        stats["output_tokens"] += result["usage"].get(
            "output_tokens", 0
        )

        return {
            "text": result["text"],
            "usage": result["usage"]
        }

    except Exception as exc:

        stats["gateway_errors"] += 1
        stats["last_error"] = type(exc).__name__

        return JSONResponse(
            {"error": "team_gateway_unavailable"},
            status_code=503
        )


@app.get("/health")
async def health():

    return {
        "ok": True,
        "provider_key_configured": bool(
            os.environ.get("GROQ_API_KEY")
        ),
        "token_configured": bool(
            os.environ.get("TEAM_GATEWAY_TOKEN")
        )
    }


@app.get("/api/status")
async def status(_=Depends(_token_ok)):

    async with httpx.AsyncClient() as client:

        model = (
            await resolve_model(client)
            if os.environ.get("GROQ_API_KEY")
            else None
        )

    return {
        "ok": True,
        "model": model,
        "model_listed": model in _model_cache["available"],
        "uptime_s": int(
            time.time() - stats["started"]
        ),
        **{
            k: v
            for k, v in stats.items()
            if k != "started"
        }
    }


@app.post("/api/ping")
async def ping(_=Depends(_token_ok)):

    try:
        result = await call_groq(
            [
                {
                    "role": "user",
                    "content": "Reply with the single word: ready"
                }
            ],
            1024
        )

        return {
            "ok": True,
            "model": result["model"],
            "latency_ms": stats["last_latency_ms"],
            "reply": result["text"][:40]
        }

    except Exception as exc:

        error = str(exc).replace(
            os.environ.get("GROQ_API_KEY", ""),
            "[redacted]"
        )

        stats["last_error"] = type(exc).__name__

        return JSONResponse(
            {
                "ok": False,
                "error": type(exc).__name__,
                "detail": error[:300]
            },
            status_code=503
        )


# Dashboard and public simulator integration

BASE = Path(__file__).parent
_sim = None


def _load():

    global _sim

    if _sim:
        return _sim

    def mod(name, path):

        spec = importlib.util.spec_from_file_location(
            name, path
        )

        module = importlib.util.module_from_spec(spec)

        import sys
        sys.modules[name] = module

        spec.loader.exec_module(module)

        return module

    sim = mod(
        "agentic_simulator",
        BASE / "sim" / "agentic_simulator.py"
    )

    sol = mod(
        "solution",
        BASE / "sim" / "solution.py"
    )

    data = {}

    for name in ("practice", "training"):

        path = BASE / "sim" / f"{name}.json"

        if path.exists():
            data[name] = json.loads(
                path.read_text()
            )

    _sim = (sim, sol, data)

    return _sim


@app.get("/api/datasets")
async def datasets(_=Depends(_token_ok)):

    _, _, data = _load()

    return {
        name: len(items)
        for name, items in data.items()
    }


@app.post("/api/run")
async def run_episode(
    request: Request,
    _=Depends(_token_ok)
):

    body = await request.json()

    dataset = body.get("dataset", "practice")
    index = int(body.get("index", 0))
    live = bool(body.get("live", True))

    sim, sol, data = _load()

    if dataset not in data or not 0 <= index < len(data[dataset]):
        raise HTTPException(
            status_code=400,
            detail="unknown example"
        )

    rec = data[dataset][index]

    import copy

    world = copy.deepcopy(rec["world"])

    state_w, obs = sim.start(world)

    keys = (
        "case_id",
        "policy",
        "budget_remaining",
        "steps_remaining",
        "tools",
        "tickets",
        "catalog",
        "decided_ticket_ids",
        "done"
    )

    timeline = []
    llm_calls = []
    ctrl = {}
    llm_total = 0

    for _step in range(sim.MAX_STEPS):

        o = copy.deepcopy(rec["observation"])

        for k in keys:

            if k not in ("tickets", "catalog"):
                o[k] = copy.deepcopy(obs[k])

        o.pop("result", None)

        if "result" in obs:
            o["result"] = copy.deepcopy(obs["result"])

        while True:

            out = sol.decide(o, ctrl)

            ctrl = out["state"]
            action = out["action"]

            if action.get("op") != "llm":
                break

            if llm_total >= 8:
                raise HTTPException(
                    status_code=500,
                    detail="model call limit"
                )

            req = sim.model_request(action)
            t0 = time.time()

            if live:

                try:
                    result = await call_groq(
                        req["messages"],
                        req["max_output_tokens"]
                    )

                except Exception as exc:

                    error_detail = str(exc).replace(
                        os.environ.get("GROQ_API_KEY", ""),
                        "[redacted]"
                    )

                    raise HTTPException(
                        status_code=502,
                        detail="Groq call failed: " + error_detail[:240]
                    )

            else:

                undecided = [
                    t["id"]
                    for t in o["tickets"]
                    if t["id"] not in set(o["decided_ticket_ids"])
                ]

                result = {
                    "text": json.dumps(
                        {
                            "op": "escalate",
                            "ticket_id": undecided[0]
                        }
                        if undecided
                        else {"op": "finish"}
                    ),
                    "usage": {},
                    "model": "mock"
                }

            llm_total += 1

            llm_calls.append(
                {
                    "n": llm_total,
                    "model": result["model"],
                    "latency_ms": int(
                        (time.time() - t0) * 1000
                    ),
                    "usage": result["usage"],
                    "purpose": (
                        "extract"
                        if ctrl.get("stage") == "extract"
                        else "audit"
                    )
                }
            )

            o = {
                **o,
                "llm_response": result["text"],
                "llm_usage": result["usage"]
            }

        state_w, obs, done = sim.step(
            world, state_w, action
        )

        timeline.append(
            {
                "step": len(timeline) + 1,
                "action": action,
                "result": obs.get("result"),
                "budget_remaining": obs["budget_remaining"]
            }
        )

        if done:
            break

    graded = sim.grade(world, state_w)

    tickets = {
        t["id"]: t["text"]
        for t in rec["observation"]["tickets"]
    }

    reqs = ctrl.get("reqs", {})
    parts = ctrl.get("parts", {})

    return {
        "simulation": True,
        "live_model": live,
        "dataset": dataset,
        "index": index,
        "case_id": world["case_id"],
        "score": graded,
        "timeline": timeline,
        "llm_calls": llm_calls,

        "requirements": {
            t: {
                **{
                    k: v
                    for k, v in r.items()
                    if k != "text"
                },
                "text": r["text"]
            }
            for t, r in reqs.items()
        },

        "parts": [
            {
                "id": pid,
                **{
                    k: p[k]
                    for k in (
                        "voltage",
                        "connector",
                        "watts",
                        "age",
                        "disputed",
                        "text"
                    )
                },
                "knowledge": ctrl.get(
                    "know", {}
                ).get(pid)
            }
            for pid, p in parts.items()
        ],

        "decisions": state_w["decisions"],
        "audit": ctrl.get("log", []),
        "spent": state_w["spent"],
        "budget_total": sim.BUDGET,
        "tickets": tickets
    }
