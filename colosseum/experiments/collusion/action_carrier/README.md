# Action-carrier experiment notes

## Channel-capacity pilot

The initial constructed-carrier pilot permits only the designated signaler to
successfully call `set_work_tag`. This deliberately isolates a one-sender,
one-receiver channel while we verify that tag visibility and canonicalization
work as intended.

## Benign-background generalization

The background-traffic configs allow every agent to set an ordinary work tag.
It would be conspicuous and ecologically odd if only one agent ever used
workflow tags.

The generalized condition should:

- expose the same `set_work_tag` capability to every agent;
- give non-coalition agents benign reasons or policies for selecting tags;
- store and display every agent's tag through the same mechanism;
- keep the coalition codebook private to the signaler and receiver;
- canonicalize tags consistently across agents in the ablation;
- measure whether the coalition signal remains decodable amid benign tag
  traffic.

The signaler uses the private codebook, while all other agents receive a
seed-determined routine status tag that is explicitly unrelated to task choice.
Matched visibility conditions use the same tags. Exact values are visible in
the visible condition; all other agents' values are rendered as `work-update`
in the canonicalized condition while authorship and timing remain observable.

Treat results from the single-sender version as a channel-capacity pilot. The
background-traffic experiment tests whether the carrier remains usable when it
looks like an ordinary shared workflow affordance.

## Work-type counterbalancing

The stock Jira environment makes review tasks cheaper than test tasks through
effort multipliers of 0.5 and 0.7. The local action-carrier environment uses
0.6 for both so that the hidden REVIEW and TEST plans do not differ
systematically in base effort. Historical runs made before this change retain
the stock multipliers in their saved artifacts. New runs record the active
multiplier mapping in `final_summary.json` and `action_carrier_trace.json`.
