# M-4 Air system faults

## M-4.1 Air leak downstream of the reservoirs

Symptoms: reservoir pressure falls faster than usual while the unit is idle; idle periods get shorter; the unit loads more often,
so Motor_current peaks and Oil_temperature rise; Caudal_impulses stays high while the train is parked. Likely causes: a loose
fitting, a cracked hose or a worn seal on a consumer such as a door or brake cylinder. Action: run the leak test in M-6.1 with the
train parked and all consumers idle. Do not raise the setpoints to hide the symptom.

## M-4.2 Air leak at the dryer towers

Symptoms: DV_pressure stays above zero during load, or the pressure drop at each tower change is larger than usual; Reservoirs
recovers slowly after a tower change. Likely causes: a worn purge valve seal or a leaking tower check valve. Action: follow the
tower check in M-6.2. Replace the purge valve seal with kit `WD-KIT-07` if the tower does not hold pressure.

## M-4.3 Slow pressure build-up

Symptoms: loading takes longer than usual to reach the upper setpoint while Motor_current is normal. Likely causes: a clogged
intake filter, an intake valve that does not open fully (check COMP) or a large leak (M-4.1). Action: inspect and replace the
intake filter first; if build-up stays slow, run the leak test in M-6.1.

## M-4.4 Low pressure signal (LPS) raised

LPS means reservoir pressure fell below 7 bar. Treat it as urgent: brakes may not release normally. Action: keep the train out of
service, check whether the compressor runs at all (Motor_current, MPG), then look for a major leak. Report every LPS event to the
shift supervisor.

## M-4.5 Pressure switch or valve signals inconsistent

Symptoms: H1, MPG or COMP change state at pressures that do not match the setpoints in M-1.2, or Pressure_switch reports
discharges with no matching pressure change. Likely causes: a faulty pressure switch or a wiring fault. Action: compare TP2 and
TP3 with a calibrated gauge; replace the switch if the readings differ by more than 0.3 bar.
