# Electric test vehicle: data needed for the simulation

Required items are needed before the simulation and training mean anything for this vehicle. Optional items have workable defaults and can be refined after the first flights. The values go into `configs/vehicles/electric_hopper.yaml`, `configs/motors/example_edf_90mm.yaml`, `configs/sensors/example_imu_gps.yaml` and `configs/missions/hop_50m.yaml`.

## Required

| What | Unit | Accuracy |
|---|---|---|
| Total mass ready to fly, battery in | g | +-10 g |
| Mass of the fan / motor / ESC unit | g | +-5 g |
| Balance point (CG) measured from the top, battery in | mm | +-5 mm |
| Overall length and body diameter | mm | +-2 mm |
| Max static thrust at full throttle, charged battery (thrust stand) | N | +-3 % |
| Thrust at 25 / 50 / 75 % throttle (same test) | N | +-3 % |
| Thrust response: time to reach 63 % after a throttle step | ms | +-20 ms |
| Battery time at hover throttle | s | +-10 s |
| Thrust vectoring: max angle each axis, and pivot (or vane) position from the top | deg, mm | +-0.5 deg, +-5 mm |
| Thrust vectoring servos: model, speed, PWM range at centre and limits | -, deg/s, us | datasheet |
| Roll control: what it is (fans or wheel), max roll torque or fan thrust x arm | N m | +-20 % |
| Legs: number, foot circle diameter, feet below the bottom of the body | -, mm, mm | +-5 mm |
| Touchdown speed and tilt the legs survive | m/s, deg | drop test |
| GPS: model, update rate (needed for the 10 m landing circle) | -, Hz | datasheet |
| IMU and barometer: model, sample rate | -, Hz | datasheet |
| Flight computer: board, control loop rate, language of the flight code | -, Hz, - | - |
| Mission rules: height, hover time, landing radius, measured from where | m, s, m | confirm |

## Optional

| What | Unit | Used until measured |
|---|---|---|
| Pitch inertia (CAD, or a swing test) | kg m2 | from size and mass |
| Roll inertia (CAD) | kg m2 | from size and mass |
| Drag coefficient and centre of pressure (OpenRocket) | -, mm | 0.8, mid-body |
| Fan reaction torque: roll torque per newton of thrust | N m / N | 0.003 |
| Servo delay and deadband | ms, deg | 20 ms, 0.1 deg |
| Sensor noise and bias: 30 s log with the board still on a table | csv | datasheet values |
| GPS accuracy at the field: 2 min log standing still | csv | 1 m |
| Site elevation and expected wind on test day | m, m/s | 300 m, 0-5 m/s |
