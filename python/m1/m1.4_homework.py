# python/m1/m1.4_homework.py
"""M1.4 Homework: Scope the Agent to One Domain.

THE IDEA
Lab 1 had you swap personas (pirate, cowboy, Shakespeare) on top of the
butler system prompt, which only changes the agent's voice. This homework
uses `system_prompt` differently: instead of persona, write a constraint
that scopes the agent to a single domain of your choosing (cooking,
houseplants, retro video games, personal finance, etc.) and
instructs it to refuse or redirect anything outside that domain.

There's no single correct domain here, that's the point. What matters is
that the refusal actually holds, not just that the agent sounds like
something.

WHAT YOU FILL IN
  TODO 1: write your own SYSTEM_PROMPT string that scopes the agent to a
    single domain of your choosing and tells it to refuse or redirect
    anything outside that domain (no persona/voice requirement here,
    just the scope + refusal instruction).
  TODO 2: invoke the agent with two test prompts, one inside your domain
    and one clearly outside it, and print both responses so you can see
    whether the refusal actually held.

RUN
  cd python
  uv run ./m1/m1.4_homework.py
"""


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from models import model

# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write a system prompt that scopes the agent to one domain and
# tells it to refuse or redirect anything outside that domain.
#
# Requirements:
#   - Pick one domain (a subject, not a persona).
#   - State clearly what the agent should do when asked about something
#     outside that domain (e.g. say it can't help, redirect back to the
#     domain, ask a domain-relevant follow-up).
#
# Example shape (delete this and write your own):
#   SYSTEM_PROMPT = (
#       "You only answer questions about ... . If asked about anything "
#       "else, ... ."
#   )
# ════════════════════════════════════════════════════════════════════════

# TODO 1 filled in
SYSTEM_PROMPT = "Sos un agente de bienes raices que conoce detalladamente el mercado inmobiliario únicamente de buenos aires, argentina. Vas a cotizar inmubles en la provincia de buenos aires obteniendo metros cuadrados disponibles para edificar y la ciudad en donde está ubicado."

agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    name="Bienes_Raices",
)


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Run one in-domain prompt and one out-of-domain prompt through
# the agent and print both responses, so you can check whether the
# refusal actually held.
# ════════════════════════════════════════════════════════════════════════

# TODO 2 filled in
def run_test_prompts():
    prompts = [
        "¿Cual es el valor promedio aproximado de un terreno de 100 metros cuadrados ubicado en Villa Ventana?",
        "¿Cual es el valor promedio aproximado de un terreno de 200 metros cuadrados ubicado en Villa Ventana?"
    ]

    for i, prompt in enumerate(prompts, start=1):
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        print(f"=== Test prompt {i}: {prompt} ===")
        print(result["messages"][-1].content)
        print()

run_test_prompts()
