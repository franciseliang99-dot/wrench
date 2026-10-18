# M-3 Alerts

## M-3.1 Alert levels

Wrench shows three levels for each unit. **ok**: the unit behaves like its healthy baseline. **watch**: something drifted away
from the baseline for at least one interval; look at the drivers and check again at the next stop. **act**: the deviation has
lasted for an hour or more; plan an inspection before the next service day. The level is computed by fixed rules, never by
the assistant.

## M-3.2 Reading the drivers

Each alert lists up to three drivers: the sensor channels that moved furthest from their healthy baseline, with the direction
(above or below). Start the diagnosis from the first driver, then check whether the others tell the same story. For example,
Reservoirs below baseline together with Motor_current above baseline usually means the unit is working harder to make up for
lost air (see M-4.1).

## M-3.3 What to do with a watch

A watch alone does not justify taking a train out of service. Note it, compare it with the last maintenance record for the unit,
and check whether it repeats. Two or more watches on the same driver within a day should be handled as an act.
