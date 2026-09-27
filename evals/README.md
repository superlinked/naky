# Evaluations

`aggregate-results.json` contains the frozen aggregate comparison for this
version. It contains no task text, answer keys, model responses, record
identifiers, media, or provider receipts.

## GUI-World protocol

The comparison uses the `benchmark` split of
[`ONE-Lab/GUI-World`](https://huggingface.co/datasets/ONE-Lab/GUI-World) at
revision `7901fa3f0b9e36afa87d24ff9ecd48037c7fd6a0` (CC BY 4.0). Näky state text
and screenshots are read by the same `gpt-6-luna` policy with low reasoning,
16,384 maximum output tokens, high image detail, high service tier, no storage,
identical option permutations, and three byte-identical request-body trials per
task and arm.

The Näky arm prepends this exact syntax note before `screen.txt`:

> Representation: `@t` marks milliseconds; `S w h` resets the screen and
> `R w h` resizes it. `=`, `+`, `~`, `>`, and `-` set, add, change, move, and
> remove OCR elements. Number or quoted rows continue the prior operation;
> `a-b` is an inclusive ID range.

The top-level README documents the complete output grammar for product use.
The aggregate reports the evaluated prompt above unchanged.

The screenshot arm uses 20 chronologically ordered PNG frames capped to a
768-pixel long edge and sampled nearest equal-presentation-duration-bin
midpoints. This policy was selected on a separate TRAIN panel before any
benchmark response was opened. The aggregate separately binds the scored
product binary SHA-256, the unstripped public equivalence execution binary, the
stripped public release binary, the model bundle, and the public equivalence
receipt.

The released cohort keeps the upstream split census. A task-independent source
amendment separately identifies upstream zero-byte media and excludes those rows
from evaluable product and reader cohorts. The primary comparison additionally
excludes development-referenced record IDs and byte-identical media aliases.
All exclusions are frozen before responses are opened. The aggregate reports
both the released census and each evaluable cohort rather than silently
diminishing a denominator.

Results are paired by task. The headline includes three-trial majority accuracy,
pooled call accuracy, invalid-output rate, wins/losses/ties, input tokens, and a
95% percentile interval from 10,000 fixed-seed bootstrap replicates. Bootstrap
sampling is by media SHA-256 cluster within each of the six GUI-World scenarios.
Breakdowns cover scenario, MCQA versus Reasoning, mobile/desktop/multi/XR,
duration, orientation, and resolution. App slices require at least 30 distinct
media clusters.

Product metrics report authenticated per-record compute work, peak resident
memory, and complete-corpus state-text and screenshot-evidence sizes. Evidence
bytes are distinct from reasoning-model input tokens. Reader cost is an estimate
from the frozen price table, not a provider invoice.

The 1,819-record output-equivalence run executed an authenticated unstripped
build. The release executable is the byte-exact result of applying authenticated
GNU `strip --strip-all` to that executable in two deterministic replays. The
receipt records both binary identities and does not claim that the stripped
file was rerun across the full corpus.

`aggregate-results.schema.json` defines the closed public result contract. The
aggregate records exact cohort counts, policies, the three binary roles, model
identities, and an offline [`public-equivalence.json`](public-equivalence.json)
receipt. That receipt requires exact
event bytes and exact screen-text bytes after the authenticated first-line
format preamble. The evaluated preamble and public preamble are recorded
separately. The standalone comparison is in [`gui-world.html`](gui-world.html).
