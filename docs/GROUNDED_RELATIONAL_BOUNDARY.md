# Learned-reader boundary for the continuing controller

`cpp/include/laminar/grounded_relational_bridge.h` preserves the grounded reader's
actual output when exposing a conservative observation to the structured
controller. It does not unify the two trainers or close the complete-learner gate.

`capture()` checks the existing report journal and archive, then retains all
alternative worlds, explicit negative facts, complete rows and columns, reader
names, and opaque relation-channel bindings. Its evidence stamp includes physical
and historical revisions, archived receipts, learned model signatures and exact
primitive/expansion values. Capturing state does not modify the reader or journal.
Source identities remain host metadata, not identities inferred from dialogue.

The reader's `complete` field means that interpretation is valid. It does **not**
mean that every fact has been observed. Its positive fact list can also be the
intersection of different possible worlds. The bridge therefore preserves the
full alternatives separately. Its explicitly conservative `Scene` projection
exposes only common positives and remains incomplete unless every source world
is explicitly closed and has identical facts. Explicit negatives and scoped
coverage stay in the envelope; the current controller view cannot use them.
An empty intersection never becomes a verified empty world.

Use `project_current(session, evidence, revision)` before supplying a captured
world to a controller. The caller owns the controller's monotonic observation
revision namespace; episode-local reader counters cannot be reused for that
purpose. The generic `project()` overload is a structural conversion and does
not authenticate externally constructed worlds or certify an old snapshot as
current. Capacity failures are explicit.

The bridge can index opaque dependent artifact IDs and report which stamps have
been invalidated. It does not mutate downstream training evidence. An owner must
withdraw or replay dependent receipts before using them again. A restoration
that recovers exactly the same physical facts still has new evidence versions.

## Focused validation

`cpp/tests/test_grounded_relational_bridge.cpp` uses the existing trainable
`GroundedSession`. Familiar observations are bootstrapped from transitions; one
atomic question is explicitly supplied. An unfamiliar report phrase is then
learned from raw text and verified final answers, with no supplied meaning for
that phrase. The same `ControlledRelationalSession` consumes the resulting
projection. Both relation-channel orientations are checked.

The contract checks ambiguous report withholding, recovery after teaching,
conservative withholding for partial-world negatives, lossless retention of
explicit negative/coverage evidence, current and archived correction/withdrawal,
stale snapshots, generated text/feedback rejection, clone isolation and changed
primitive priors. Separate structural cases check genuinely declared closed
worlds and reject disagreement among closed alternatives. The latter are not
claims that the partial-observation GroundedSession produces closed worlds.

Controller operations are forced in this protocol test, with its weights
restored immediately. This validates the boundary, not learned operation choice.
Release and ASAN/UBSAN contracts are the appropriate checks; no curriculum data
or archived experiment output is changed by this test.

## Remaining integration

The derived-teaching receipt registry and atomic invalidation/replay remain open.
Canonical language-receipt revision cannot currently replace an interpreted
teaching scene. Partial-world action planning also remains open: canonical
`Scene` cannot express the reader's alternatives and scoped coverage.

The full `Organism` byte predictor is not attached by this bridge. Its existing
grounded reader can learn symbolic report meanings alongside byte prediction,
but that is distinct from the byte predictor learning those meanings itself.
Existing journal/archive guards deliberately require question programs and
primitive weights to remain fixed. Sharing a continually repaired primitive
bank requires revalidating that prior and its derived evidence, not removing
the guards. A sustained mixed continuation must still test language, control,
representation growth, multiple repairs and retention together.
