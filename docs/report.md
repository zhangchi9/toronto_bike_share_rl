# Toronto Bike Share Rebalancing with Reinforcement Learning

## 1. Business Problem

Toronto's bike share system (Bike Share Toronto) runs about 1,070 docking stations
across the city, each with a fixed number of docks. Riders unlock a bike at one
station and return it to any other station with a free dock. This only works if
bikes and empty docks are both available where and when riders need them — and
left alone, the system drifts out of balance within hours:

- **Commuting patterns are one-directional.** Downtown stations fill with
  returned bikes every morning and empty out every evening (and the reverse at
  residential stations), so docks run out on one side of the trip and bikes run
  out on the other.
- **Two failures, both costly.** A rider who finds an **empty station** can't
  start a trip and the ride is lost outright. A rider who finds a **full
  station** can't end theirs; the system redirects them to the nearest station
  with space, but that rider pays for extra ride time they didn't want and the
  operator loses the normal return-triggered revenue.
- **The imbalance compounds.** Every lost rental and every blocked return is
  also a bike that never moved, so the same stations stay empty or full and the
  next rider hits the same wall.

The fix the industry uses is **rebalancing**: trucks drive bikes from
stations that have too many to stations that have too few. But running trucks
is itself expensive — a truck costs money whether it's driving, waiting, or
parked, through a fixed daily cost, driver hours, and fuel — and a truck sent
to the wrong zone, at the wrong time, or carrying the wrong number of bikes
does nothing for the stations that actually needed it. The operator is really
choosing, continuously, where each truck should go, when, and with how many
bikes, under a budget that makes the wrong choice a net loss.

### What we are solving for

**Decide, at every moment a truck is free, where it should drive and how
many bikes it should carry — to maximize system-wide profit, not just bikes
moved.**

Profit here nets three things against each other every minute:

1. **Revenue kept**, from unlocks and ride time that would otherwise have been
   lost to an empty station or degraded by a blocked return.
2. **Revenue still lost**, from requests no truck reached in time.
3. **The cost of running trucks** — a fixed cost and driver shift whether or
   not they are used that day, plus fuel for every kilometer driven.

A policy that moves a lot of bikes but drives too far, or keeps a truck on
payroll it barely uses, can lose money even while "helping." A good policy
has to weigh the dollar value of each potential rebalancing trip against what
that trip costs, continuously, across an entire city of stations and through
days of shifting demand — which is exactly the kind of sequential,
high-dimensional decision problem reinforcement learning is suited to, and
which a fixed human-written rule struggles to adapt to as demand shifts by
hour, day of week, and season.

### Why not a simpler fix

- **More docks everywhere** removes the constraint but is a multi-year capital
  project, not an operating decision, and doesn't fix the one-directional flow
  that causes the imbalance in the first place.
- **A fixed shuttle route or hand-written threshold rule** (e.g., "truck
  visits the same 5 stations every morning," or "move bikes whenever a
  station is below 20% full") is cheap to build but can't weigh a move's cost
  against its payoff, can't adapt to a Tuesday vs. a rainy Saturday, and we
  show later in this report that it struggles to beat even a random
  dispatcher on the same budget.
- **More trucks** scales cost linearly and still depends on dispatching them
  well — it doesn't answer the underlying question of *where* and *when*.

This report evaluates whether a **learned dispatch policy** — an agent that
observes the state of the system and decides truck routes and loads — can
outperform these simpler alternatives by enough to justify the additional
engineering, and under what conditions it does or doesn't hold up.
