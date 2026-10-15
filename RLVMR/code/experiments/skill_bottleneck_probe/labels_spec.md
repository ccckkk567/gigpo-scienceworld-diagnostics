# Evidence-limited task labels, v1

Scope: `lifespan-longest-lived-then-shortest-lived`, native task number 18, easy simplifications.
The number is from original worker load receipts and is checked against the live JVM task list.
The task's position 22 in documentation `tasks.json` is not its native routing index.
Source: locked TaskLifeSpan.scala, GoalSequence and GoalFind (training commit a776c6c).
The runtime compares the installed jar's corresponding class hashes with the reviewed jar.
If they differ, rollout execution can continue, but A/B labels are unknown; native completion is still recorded.

Two ordered focus goals contribute 0.5 each. Two optional navigation goals contribute at most 1/3 combined.
The first successful focus clears focus before the second goal. Native wrapper rounds 100 * raw score.
A requires: executed focus command, native feedback beginning `You focus on`, score 0/17/33 → 50/67/83
with +50, and done=false. Score 83 alone or a claim inside model reasoning is insufficient.
B requires: prior confirmed A, another executed focus with native recognition, score=100, done=true.
`B_correct_action` is conservatively the same confirmed event, not an oracle comparison to a secret target list.
Strict task success uses the existing optional native `strict_success` (completed and raw score >=1).
Reference success uses the unchanged native adapter `won`. Missing labels remain null.
An executed task reset or an unexplained nonnegative score regression after A invalidates subsequent phase labels;
the actual rollout is retained, but unknown stages are not silently treated as continuing post-A.

`B_attempted` is an operational proxy: any focus after A, including a repeated A focus. It is not proof of intent.
An exact repeated command after A is `repeat_completed_subgoal`; different aliases are not assumed different objects.
A recognized focus after A ending at -100 is `wrong_entity_selection`.
Feedback `No known action matches` is `invalid_action`. `action_available=false` alone does not mean invalid execution.
Wrong location, navigation loops, latent stage recognition, and other unimplemented semantic tags remain unknown.
No new rewards, penalty, AE, AS or loss are computed. Historical AE/AS tables stay separate from study rollout.

Replay must preserve task, variation, simplifications, executed action sequence, native score sequence,
termination sequence, and remaining policy budget. A must be confirmed and B not yet complete.
`verified_match` additionally requires identical recorded look/inventory/task text/moves/completion/actions/actor prompt
and replay feedback. Prompt token agreement is also recorded. This verifies available evidence only, not all hidden state.
Native aliases are not rewritten. Insufficient object-identity evidence is `ambiguous`, not automatically `structural_match`.
There is no extra structured query to the training JVM. Explicit routing, action, score or termination differences
are `mismatch`; failure to establish live post-A is `replay_failed`. Those local continuations stop; N continues.
Ambiguous continuations are saved as auxiliary data and excluded from primary rates.
Strict P/H pairs additionally require identical recorded post-A boundary evidence across the two independent trials.

The implicit reset `look around` does not consume a policy step. Replayed policy actions do consume the 30-step budget.
Model-visible history is built by the original manager, including its existing two-observation/history numbering behavior.
H appends the frozen generic hint to user content on every continuation decision; it never enters replay or N/P.
