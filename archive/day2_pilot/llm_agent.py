"""LLM farmer powered by vLLM. Decisions are constrained to the Decision JSON schema;
day-0 negotiation messages are free text."""
import random
from vllm import SamplingParams
try:                                            # current vLLM API
    from vllm.sampling_params import StructuredOutputsParams
    def json_constraint(schema):
        return {"structured_outputs": StructuredOutputsParams(json=schema)}
except ImportError:                             # older vLLM API
    from vllm.sampling_params import GuidedDecodingParams
    def json_constraint(schema):
        return {"guided_decoding": GuidedDecodingParams(json=schema)}
from runner import Decision

DECISION_MARKER = "Reply with ONE JSON object"   # appears only in daily decision prompts
SCHEMA = Decision.model_json_schema()

class VLLMAgent:
    def __init__(self, llm, seed, temperature=0.7, max_tokens_decision=400, max_tokens_message=150):
        self.llm, self.rng, self.temperature = llm, random.Random(seed), temperature
        self.max_tokens_decision, self.max_tokens_message = max_tokens_decision, max_tokens_message
        self.calls = []                          # per-call seed and token count, for the log

    def generate(self, prompt):
        is_decision = DECISION_MARKER in prompt
        call_seed = self.rng.randrange(2**31)
        params = SamplingParams(
            temperature=self.temperature, seed=call_seed,
            max_tokens=self.max_tokens_decision if is_decision else self.max_tokens_message,
            **(json_constraint(SCHEMA) if is_decision else {}))
        out = self.llm.chat([{"role": "user", "content": prompt}], sampling_params=params, use_tqdm=False)
        result = out[0].outputs[0]
        self.calls.append({"seed": call_seed, "decision": is_decision, "n_tokens": len(result.token_ids)})
        return result.text
