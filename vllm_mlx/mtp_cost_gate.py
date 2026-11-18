"""Cost-aware gate for the MTP path.

Speculation is not free: a speculative round spends one extra target forward
to verify K drafted tokens. Whether that pays depends on the TARGET BUILD
and the machine - a fast target amortises the extra forward over too few
tokens to pay for it, a slow one does - and not on the batch size. The fork's
static switch ("the batch holds one UID, so speculate no more") answers a
different question than the one that matters, so this module answers the
right one: speculate only while the MEASURED cost and acceptance say the
round pays for itself.

The rule is the campaign's break-even rule with every term measured on this
machine, for this target, right now:

* F - the plain path's per-token step time. Measured as the wall-clock
  interval between two decode steps that both took the plain path.
* T - the speculative round's wall-clock cost, measured the same way.
* d - the drafter's own cost per round, measured on instrumented rounds
  (the wall-clock time of the drafter's forward).
* a - the observed per-draft acceptance (accepted drafts / drafted drafts).

A round commits 1 + a*K tokens for the cost of one plain token plus K extra
verified positions plus the drafter, so the marginal cost of one extra
verified position is

    s = (T - F - d) / K

and the round wins iff

    a * F  >  s + d/K

(the campaign's a*F > s + d with d counted per drafted token). The gate
keeps drafting while that holds and falls back to the plain path when it
does not.

Two properties matter as much as the rule itself:

* The decision is latched for the WHOLE request. A request either runs the
  speculative path from end to end or the plain path from end to end, so the
  gate never changes the arithmetic under a running request and never
  changes the tokens it emits.
* A missing measurement falls back to today's behaviour (speculate), never
  to something slower: the gate only ever falls back to the plain path when
  it has measured that the plain path is at least as fast.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

__all__ = ["MtpCostGate"]


class MtpCostGate:
    """Decide whether speculation pays, from measured cost and acceptance.

    The gate is fed by the MTP path (see install_mtp_mllm):

    * begin_request when a request starts (the decision is latched there),
    * observe_plain_step / observe_round with the wall-clock time of the
      step that just finished,
    * observe_draft_cost with the drafter's own forward time,
    * observe_acceptance with the drafts the last round accepted.
    """

    def __init__(
        self,
        depth: int,
        margin: float = 1.05,
        recheck_every: int = 8,
        instrument_every: int = 32,
        alpha: float = 0.25,
        min_round_samples: int = 16,
        min_plain_samples: int = 16,
        min_drafts: int = 32,
        skip_first_samples: int = 2,
    ) -> None:
        self.depth = max(1, int(depth))
        self.margin = float(margin)
        self.recheck_every = max(0, int(recheck_every))
        self.instrument_every = max(1, int(instrument_every))
        self.alpha = float(alpha)
        # How much measured evidence the gate wants before it trusts a
        # quantity. A request's first steps are not representative (the
        # target's first forward after a prefill, the drafter's first
        # forward with its weights still cold), so they are dropped.
        self.min_round_samples = max(1, int(min_round_samples))
        self.min_plain_samples = max(1, int(min_plain_samples))
        self.min_drafts = max(1, int(min_drafts))
        self.skip_first_samples = max(0, int(skip_first_samples))
        self._plain_s: Optional[float] = None
        self._round_s: Optional[float] = None
        self._draft_s: Optional[float] = None
        self._accepted = 0
        self._drafted = 0
        self._rounds = 0
        self._plain_steps = 0
        self._instrumented = 0
        self._steps = 0
        self._uid: Any = None
        self._plain_last: Optional[float] = None
        self._round_last: Optional[float] = None
        self._round_samples = 0
        self._plain_samples = 0
        self._skip_plain = 0
        self._skip_round = 0
        self._requests = 0
        self._decision = True
        self._decisions = 0

    # ------------------------------------------------------------------
    # observations

    def _ema(self, old: float, new: float) -> float:
        return old + self.alpha * (new - old)

    def observe_plain_step(self, seconds: float) -> None:
        """A plain (non-speculative) step took seconds."""
        if seconds <= 0:
            return
        if self._skip_plain > 0:
            # The first steps of a request are not representative.
            self._skip_plain -= 1
            return
        self._plain_samples += 1
        self._plain_s = (
            seconds if self._plain_s is None else self._ema(self._plain_s, seconds)
        )
        self._plain_last = seconds
        self._plain_steps += 1

    def observe_round(self, seconds: float) -> None:
        """A speculative round took seconds."""
        if seconds <= 0:
            return
        if self._skip_round > 0:
            # The first rounds of a request are not representative.
            self._skip_round -= 1
            return
        self._round_samples += 1
        self._round_s = (
            seconds if self._round_s is None else self._ema(self._round_s, seconds)
        )
        self._round_last = seconds
        self._rounds += 1

    def observe_draft_cost(self, seconds: float) -> None:
        """The drafter's own forward took seconds."""
        if seconds <= 0:
            return
        self._draft_s = (
            seconds if self._draft_s is None else self._ema(self._draft_s, seconds)
        )
        self._instrumented += 1

    def observe_acceptance(self, accepted: int, drafted: int) -> None:
        """The last round accepted accepted of drafted attempted drafts."""
        self._accepted += max(0, int(accepted))
        self._drafted += max(0, int(drafted))

    def note_step(self) -> None:
        """A decode step finished (whatever it did)."""
        self._steps += 1

    # ------------------------------------------------------------------
    # the measured quantities

    @property
    def acceptance(self) -> Optional[float]:
        """The observed per-draft acceptance, or None if nothing was drafted.

        _drafted counts the drafts the rounds ATTEMPTED (accepted plus
        rejected), so the acceptance is accepted / attempted.
        """
        if self._drafted <= 0:
            return None
        return self._accepted / self._drafted

    @property
    def round_samples(self) -> int:
        """How many measured rounds the gate has."""
        return self._round_samples

    @property
    def plain_samples(self) -> int:
        """How many measured plain steps the gate has."""
        return self._plain_samples

    @property
    def plain_seconds(self) -> Optional[float]:
        """The measured plain path's per-token step time (F)."""
        return self._plain_s

    @property
    def round_seconds(self) -> Optional[float]:
        """The measured speculative round's cost (T)."""
        return self._round_s

    def marginal_cost(self) -> Tuple[Optional[float], Optional[float]]:
        """The measured (s, d): the extra position's cost and the drafter's.

        s is the target's marginal cost of one extra verified position and d
        the drafter's cost per drafted token. Both are None until the plain
        path and the round have both been measured.
        """
        if self._plain_s is None or self._round_s is None:
            return None, None
        draft = self._draft_s or 0.0
        s = (self._round_s - self._plain_s - draft) / self.depth
        return s, draft / self.depth

    # ------------------------------------------------------------------
    # the decision

    def _decide(self) -> bool:
        """Whether the next request should speculate.

        The gate only decides from evidence it has actually measured, and it
        goes looking for that evidence in the order it needs it: the round's
        cost first (a speculative request), then the plain path's cost (a
        plain request), then the acceptance. Until it has that evidence it
        keeps measuring, and it never turns speculation off before it has
        measured that the plain path is at least as fast.
        """
        if self._round_samples < self.min_round_samples:
            return True
        if self._plain_samples < self.min_plain_samples:
            return False
        if self._drafted < self.min_drafts:
            return True
        # Every recheck_every requests the round is measured again, so a
        # target that got slower (or a drafter that got worse) is noticed.
        if (
            self.recheck_every
            and self._requests > 1
            and self._requests % self.recheck_every == 0
        ):
            return True
        acceptance = self.acceptance
        s, d = self.marginal_cost()
        if acceptance is None or s is None or self._plain_s is None:
            return True
        # a*F > margin*(s + d): speculation pays while the tokens it commits
        # are worth more than the round costs.
        return acceptance * self._plain_s > self.margin * (s + d)

    def begin_request(self, uid: Any) -> bool:
        """Latch the decision for the request that just started."""
        if uid is None or uid == self._uid:
            return False
        self._uid = uid
        self._requests += 1
        self._decisions += 1
        self._decision = self._decide()
        # A request's first steps are not representative: the target's first
        # forward after a prefill and the drafter's first forward with its
        # weights still cold are both slower than the steady state.
        self._skip_plain = self.skip_first_samples
        self._skip_round = self.skip_first_samples
        return True

    def should_speculate(self) -> bool:
        """Whether this request should speculate."""
        return self._decision

    @property
    def speculating(self) -> bool:
        """The gate's current decision."""
        return self._decision

    def should_instrument(self) -> bool:
        """Whether the next round should measure the drafter's own cost."""
        return self._steps % self.instrument_every == 0

    # ------------------------------------------------------------------
    # telemetry

    def snapshot(self) -> Dict[str, Any]:
        """The measured state, for the server's status endpoint."""
        s, d = self.marginal_cost()
        acceptance = self.acceptance
        plain = self._plain_s
        cost = None if s is None or d is None else (s + d) * 1000.0
        return {
            "enabled": True,
            "depth": self.depth,
            "margin": self.margin,
            "speculating": self._decision,
            "plain_step_ms": None if plain is None else plain * 1000.0,
            "plain_last_ms": (
                None if self._plain_last is None else self._plain_last * 1000.0
            ),
            "round_last_ms": (
                None if self._round_last is None else self._round_last * 1000.0
            ),
            "round_ms": None if self._round_s is None else self._round_s * 1000.0,
            "draft_cost_ms": None if self._draft_s is None else self._draft_s * 1000.0,
            "marginal_position_ms": None if s is None else s * 1000.0,
            "draft_per_token_ms": None if d is None else d * 1000.0,
            "acceptance": acceptance,
            "benefit_ms": (
                None
                if (acceptance is None or plain is None)
                else acceptance * plain * 1000.0
            ),
            "cost_ms": cost,
            "rounds": self._rounds,
            "plain_steps": self._plain_steps,
            "round_samples": self._round_samples,
            "plain_samples": self._plain_samples,
            "instrumented_rounds": self._instrumented,
            "requests": self._requests,
            "decisions": self._decisions,
            "accepted": self._accepted,
            "drafted": self._drafted,
        }
