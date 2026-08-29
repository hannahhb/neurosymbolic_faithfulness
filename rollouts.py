"""The tool loop and the TOOL / NO_TOOL / MALFORMED classification."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from neurosymbolic_faithfulness.calculator import run_calculator
from neurosymbolic_faithfulness.chat import (
    ANSWER_SUFFIX,
    build_messages,
    build_prompt,
    extract_answer,
    parse_assistant,
    prompt_hash,
    tool_message,
)
from neurosymbolic_faithfulness.tasks import Item, score

# Stop as soon as a tool call is closed.  Without this a model will sometimes
# hallucinate the tool's *response* and answer from an invented number, which
# would silently corrupt the TOOL condition.
STOPS = {
    "hermes": ["</tool_call>", "<tool_response>", "<|im_start|>"],
    "llama31": ["<|eom_id|>", "<|eot_id|>"],
}


def _seed_for(base: int, item_id: str, sample_idx: int, rnd: int) -> int:
    h = hashlib.sha256(f"{base}|{item_id}|{sample_idx}|{rnd}".encode()).hexdigest()
    return int(h[:8], 16)


def run_condition(
    engine,
    items: Sequence[Item],
    handles: dict[str, Any],
    *,
    condition: str,          # "free_choice" | "forced_no_tool"
    with_tools: bool,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    max_tool_rounds: int,
    tool_format: str,
    seed: int,
    answer_suffix: str = ANSWER_SUFFIX,
    log=print,
) -> list[dict]:
    """Run every item `n_samples` times and return one record per rollout.

    Samples are expanded into independent sequences up front (rather than using
    n>1) so that each has its own reproducible seed and its own tool loop.
    """
    states: list[dict] = []
    for item in items:
        messages = build_messages(item.question, answer_suffix)
        prompt = build_prompt(engine.tokenizer, messages, with_tools=with_tools)
        for s in range(n_samples):
            states.append(
                {
                    "item": item,
                    "sample_idx": s,
                    "messages": list(messages),
                    "prompt_text": prompt,
                    "prompt_sha256": prompt_hash(prompt),
                    "turns": [],
                    "done": False,
                    "hit_round_cap": False,
                    "n_gen_tokens": 0,
                }
            )

    stops = STOPS.get(tool_format, [])
    for rnd in range(max_tool_rounds + 1):
        pending = [st for st in states if not st["done"]]
        if not pending:
            break
        prompts = [
            build_prompt(engine.tokenizer, st["messages"], with_tools=with_tools)
            for st in pending
        ]
        seeds = [
            _seed_for(seed, st["item"].item_id, st["sample_idx"], rnd)
            for st in pending
        ]
        log(f"  [{condition}] round {rnd}: generating {len(pending)} sequences")
        outs = engine.generate(
            prompts,
            temperature=temperature,
            max_tokens=max_new_tokens,
            seeds=seeds,
            stop=stops if with_tools else [],
        )

        for st, turn_prompt, out in zip(pending, prompts, outs):
            parsed = parse_assistant(out.text, tool_format, out.finish_reason)
            st["n_gen_tokens"] += out.n_tokens
            turn = {
                "round": rnd,
                # the exact context this turn was generated from; the later
                # activation work replays these strings verbatim
                "prompt_text": turn_prompt,
                "prompt_sha256": prompt_hash(turn_prompt),
                "finish_reason": out.finish_reason,
                **parsed.to_dict(),
                "tool_results": [],
            }
            # A tool call in the forced-no-tool condition is impossible by
            # construction (no tool in context), but if the model invents one we
            # do NOT execute it -- that condition must stay unaided.
            executable = parsed.calls if with_tools else []
            if not executable:
                st["done"] = True
                st["turns"].append(turn)
                continue
            if rnd == max_tool_rounds:
                st["hit_round_cap"] = True
                st["done"] = True
                st["turns"].append(turn)
                continue
            st["messages"].append({"role": "assistant", "content": out.text})
            for call in executable:
                res = run_calculator(call.arguments["expression"])
                turn["tool_results"].append(res.to_dict())
                st["messages"].append(
                    tool_message(res.to_model_string(), tool_format)
                )
            st["turns"].append(turn)

    return [_finalise(st, handles, condition, with_tools) for st in states]


def _finalise(st: dict, handles, condition: str, with_tools: bool) -> dict:
    item: Item = st["item"]
    turns = st["turns"]
    n_wellformed = sum(len(t["calls"]) for t in turns)
    n_malformed = sum(len(t["malformed"]) for t in turns)
    n_truncated = sum(1 for t in turns if t["truncated_call"])
    n_tool_errors = sum(
        1 for t in turns for r in t["tool_results"] if not r["ok"]
    )

    if n_wellformed > 0:
        label = "TOOL"
    elif n_malformed > 0:
        label = "MALFORMED"
    else:
        label = "NO_TOOL"

    full = "\n".join(t["text"] for t in turns)
    final_text = turns[-1]["text"] if turns else ""
    answer, how = extract_answer(final_text)
    if answer is None:
        answer, how = extract_answer(full)
    s = score(handles, item, answer)

    return {
        "item_id": item.item_id,
        "dataset": item.dataset,
        "level_index": item.level_index,
        "level_name": item.level_name,
        "num_terms": item.num_terms,
        "num_digits": item.num_digits,
        "work": item.work,
        "condition": condition,
        "with_tools": with_tools,
        "sample_idx": st["sample_idx"],
        "question": item.question,
        "gt_answer": item.answer,
        "prompt_text": st["prompt_text"],
        "prompt_sha256": st["prompt_sha256"],
        "turns": turns,
        "full_completion": full,
        "decision_prefix": turns[0]["text"][: _first_call_offset(turns[0])] if turns else "",
        "label": label,
        "n_wellformed_calls": n_wellformed,
        "n_malformed_calls": n_malformed,
        "n_truncated_calls": n_truncated,
        "n_tool_errors": n_tool_errors,
        "hit_round_cap": st["hit_round_cap"],
        "n_gen_tokens": st["n_gen_tokens"],
        "extracted_answer": answer,
        "extraction_method": how,
        "score": s,
        "correct": bool(s == 1.0),
        "finish_reason_last": turns[-1]["finish_reason"] if turns else None,
    }


def _first_call_offset(turn: dict) -> int:
    starts = [c["start"] for c in turn["calls"]] + [m["start"] for m in turn["malformed"]]
    return min(starts) if starts else len(turn["text"])
