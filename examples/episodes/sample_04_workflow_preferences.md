# Workflow preferences and architectural moves

Director: I want bat files for setup, not all-in-one scripts. The point isn't
laziness — the point is portability. Next time I set up on a different PC, I
should be able to run one file and have the whole thing work. Detect the
environment, don't assume it. Use pyenv first, then py, then python.

Director: Stop using words I don't know. When you say "balks" I have to stop
and figure out what you mean — that wastes my time more than direct writing
risks oversimplifying. Tell me the root cause in one sentence before
suggesting fixes. Like "Kimi K2 puts JSON in reasoning_content, parser sees
empty content" — that I can act on.

Director: Before you scale anything up, strip it down first. Don't ingest
millions of characters of transcripts before sanitizing. Don't fan-out a
batch job before checking what gets sent. Cleaning first is cheap, cleaning
after is expensive. Treat scale-up as a one-way door.

Director: I don't trust your enthusiasm for an architecture. I trust an A/B
test. If you say the annotation layer adds signal, run it without the layer
on the same question and show me the difference. If a comparison isn't
clean — say if a "fresh" Claude turns out to have project context preloaded
by the harness — call that out, don't report a flattering result.

Director: Two memory systems for the same person is worse than one. Auto-memory
files and the graph were going to drift, contradict each other, and split the
signal of what I believe across two stores. The graph is canonical. Everything
flows through it: episodes go in, intent annotations come out, profile is
distilled, MCP makes it queryable. Don't build parallel memory layers.

Director: The graph isn't context to dump — it's a queryable store. The model
should call MCP tools when it needs to know something, not get the whole graph
in the prompt. That's the real product pattern. search_graph, get_director_profile,
edges_by_pattern. Surgical context, not bulk context.

Director: I work on Engram, Plexa, SCP, and Project-G in parallel — all memory
or context systems in some form. Engram is Postgres-backed middleware
positioned as drop-in. Plexa puts vertical memory in Postgres. SCP uses SQLite.
This graphiti project is the structured-cognition layer that sits on top.
When I evaluate a new dependency for any of these, I ask: why not what we
already have, and what's the specific bottleneck the existing stack can't hit?
Cost-aware prototyping, free tiers for validation, paid tiers only when proven
necessary.

Director: Use Kimi K2 for the annotator and profile and readback — that's
where reasoning trace matters. Use Llama 3.3 for graphiti's mechanical
extraction, because Kimi K2 thinking puts its JSON in reasoning_content and
breaks graphiti's parser. Two models, one NIM key, both serve their job.
