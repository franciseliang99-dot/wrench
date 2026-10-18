# M-2 Sensor channels

## M-2.1 Pressure channels: TP2, TP3, Reservoirs, DV_pressure

TP2 is the pressure at the compressor outlet. TP3 is the pressure at the pneumatic panel. Reservoirs is the pressure in the main
air tanks; it normally tracks TP3 closely. DV_pressure is the pressure drop measured when a dryer tower vents; it reads near zero
while the compressor is under load. A gap that opens between TP3 and Reservoirs, or a Reservoirs trace that falls faster than
usual while the unit is idle, points to air escaping somewhere in the circuit.

## M-2.2 Temperature and current: Oil_temperature, Motor_current

Oil_temperature is the oil temperature in the compressor housing. It rises during load and falls during idle. Motor_current is
the current drawn by the drive motor. A unit that loads more often than usual shows more frequent current peaks and a higher
average oil temperature, even if each single cycle looks normal.

## M-2.3 Digital signals: COMP, DV_eletric, H1, MPG, LPS, Towers

COMP is the intake valve signal; it is active when the compressor is not taking in air (off or unloaded). DV_eletric drives the
outlet valve. H1 opens when pressure rises above the upper setpoint. MPG starts the compressor under load when pressure falls
below the lower setpoint. LPS is set when pressure falls below 7 bar, which should never happen in normal service. Towers shows
which dryer tower is currently drying.

## M-2.4 Pressure_switch, Oil_level, Caudal_impulses

Pressure_switch reports a discharge in the pilot-controlled valve. Oil_level is set when the oil level falls below the expected
mark. Caudal_impulses counts air-flow pulses into the reservoirs; a high pulse rate while the train is parked means air is being
replaced as fast as it leaks away.
