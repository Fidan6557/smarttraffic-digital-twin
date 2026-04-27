# SmartTraffic Digital Twin

SmartTraffic Digital Twin is an AI signal optimization platform for city intersections. The project demonstrates how a traffic authority can compare fixed-time traffic lights with an adaptive, data-driven controller before deploying changes on a real road.

The current prototype is built as a real-time digital twin: it simulates vehicles, pedestrians, buses, emergency vehicles, signal phases, safety clearance, queue growth, and live operational KPIs. The goal is not only to animate traffic, but to show a decision-support system that stakeholders can understand, test, and evaluate.

## Demo Preview

The dashboard presents the system as an operations product rather than a toy simulation:

- live intersection video stream
- adaptive signal state
- city scenario selector
- explainable controller decisions
- wait time, throughput, CO2, fuel, and pedestrian-risk KPIs
- downloadable PDF reports for stakeholder review

Recommended visual for GitHub: add a dashboard screenshot or short demo GIF here before final submission.

## Why This Matters for a Hackathon

SmartTraffic is designed to be easy for judges and stakeholders to evaluate in a live demo:

- **Visible impact:** the dashboard compares AI Adaptive control against a fixed-timer baseline.
- **Explainable AI:** every signal decision is logged with a human-readable reason.
- **Safety-first logic:** yellow and all-red clearance are enforced before conflicting movement is released.
- **Smart-city relevance:** the system includes public transport priority, emergency priority, pedestrian risk, CO2, and fuel estimates.
- **Stakeholder handoff:** reports can be downloaded as PDFs instead of only showing numbers on screen.
- **Integration-ready structure:** the API exposes metrics, scenarios, control mode, policy tuning, audit logs, and reports.

## Problem

Many intersections still operate with fixed timers. This creates several issues:

- Traffic lights do not react to changing demand.
- Long queues form during rush hour or event exits.
- Pedestrians may wait too long or cross during risky timing.
- Emergency vehicles and buses receive no intelligent priority.
- Cities often cannot test signal strategies safely before field deployment.

SmartTraffic addresses this by creating a controllable digital twin where signal policies can be compared through measurable impact.

## Solution

The system models an intersection and runs two controller modes:

- **Fixed Timer:** a baseline controller that switches phases using a static timing cycle.
- **AI Adaptive:** a demand-aware controller that reacts to queue length, waiting time, pedestrian demand, emergency vehicles, bus priority, and intersection safety.

The dashboard shows what the controller is doing and why. This makes the system explainable for judges, city stakeholders, and technical reviewers.

The prototype also includes operations-dashboard features that make it feel closer to a deployable product: policy tuning, audit logs, historical analytics, data-source readiness, and explicit safety guarantees.

## Core Features

### Digital Twin Scenarios

The project includes stakeholder-friendly city contexts:

- Baku, Ganjlik Intersection
- Baku, School Zone Crossing
- Baku Olympic Stadium Exit
- Baku Bus Priority Corridor
- Baku Emergency Response Route

These scenarios make the prototype feel like a city operations product instead of an abstract animation.

### Adaptive Signal Controller

The adaptive controller evaluates:

- North-South queue score
- East-West queue score
- vehicle waiting time
- pedestrian waiting time
- emergency priority requests
- bus priority requests
- intersection clearance status

It uses strict signal phases:

```text
GREEN -> YELLOW -> ALL_RED -> NEXT_GREEN
```

The all-red phase acts as a safety validator before releasing the next movement.

### Explainable Decisions

The dashboard explains each controller action, for example:

```text
Switching to EW
EW score 8.7 exceeds NS score 3.2.
```

For safety events:

```text
Incident detected
Vehicle blocked in intersection. Action: extend all-red phase.
```

### Before/After Report

The system generates a stakeholder-facing report comparing AI Adaptive against the fixed-timer baseline:

```text
AI Adaptive reduced avg wait by X%
Throughput increased by X%
Estimated CO2 reduced by X%
Pedestrian risk stayed LOW
```

The report can be downloaded as a PDF from the dashboard.

### Public Transport and Emergency Priority

The prototype includes special vehicle handling:

- buses receive public transport priority
- ambulances receive emergency priority
- priority vehicles trigger phase changes when safe
- the UI shows BUS PRIORITY or EMERGENCY PRIORITY overlays

### Incident Detection

If a vehicle is blocked inside the conflict zone, the controller treats it as an incident and extends the all-red phase until the intersection is clear.

### Operations Readiness

The dashboard includes product-oriented panels for traffic operators:

- **Policy Tuning:** min green, max green, yellow time, and all-red time can be adjusted from the UI.
- **Audit Log:** controller decisions and safety events are logged live.
- **Historical Analytics:** recent wait-time trend is shown as a compact chart.
- **Data Source Readiness:** simulation is active; CCTV, loop detector, and CSV inputs are marked integration-ready.
- **Safety Guarantees:** conflicting greens are impossible, all-red clearance is enforced, incidents block phase release, and priority events are logged.

### Sustainability Estimates

The system estimates environmental impact using saved idle seconds:

```text
CO2 saved = idle_seconds_saved * emission_factor
fuel saved = idle_seconds_saved * fuel_factor
```

These are simplified estimates for demo and stakeholder communication, not certified environmental measurements.

## Technical Architecture

```text
Simulation / Detection Input
        -> Queue and Wait Estimation
        -> Adaptive Signal Controller
        -> Safety Validator
        -> Signal Output
        -> KPI and Report Layer
```

### Main Components

- **Simulation Engine:** spawns vehicles, pedestrians, buses, and emergency vehicles.
- **Queue Estimator:** calculates queues by approach and tracks waiting time.
- **Adaptive Controller:** chooses signal phases using queue score and priority rules.
- **Safety Validator:** enforces yellow and all-red clearance before conflicting movement.
- **Incident Detector:** detects blocked vehicles in the intersection.
- **KPI Layer:** calculates wait time, throughput, risk, CO2, fuel, and improvement.
- **Policy Layer:** exposes controller thresholds for traffic-engineering tuning.
- **Audit Layer:** records signal decisions, priority requests, and safety actions.
- **History Layer:** keeps recent KPI samples for operational trend monitoring.
- **Report Layer:** exposes JSON and PDF reports.
- **API Layer:** provides integration-ready endpoints.

## API

```text
GET /api/v1/metrics
GET /api/v1/control/mode?mode=adaptive
GET /api/v1/control/mode?mode=fixed
GET /api/v1/scenario?name=morning_rush
GET /api/v1/scenario?name=school_crossing
GET /api/v1/scenario?name=event_exit
GET /api/v1/scenario?name=bus_corridor
GET /api/v1/scenario?name=emergency
GET /api/v1/report
GET /api/v1/report.pdf
GET /api/v1/operations-report.pdf
GET /api/v1/policy
GET /api/v1/audit-log
GET /api/v1/history
GET /api/v1/data-sources
```

## Run Locally

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the server:

```powershell
python main.py
```

Open the printed URL, usually:

```text
http://127.0.0.1:8003
```

## Recommended Demo Flow

1. Open the dashboard.
2. Select **Fixed Timer**.
3. Select **Morning Rush** and show how queues grow.
4. Switch to **AI Adaptive**.
5. Show the KPI improvement and explainable decision panel.
6. Select **Bus Corridor** to show public transport priority.
7. Select **Emergency Route** to show emergency vehicle priority.
8. Click **Generate Before/After Report**.
9. Download the PDF report for stakeholder handoff.

## Future Extensions

- Use real detector zones from camera feeds.
- Connect scenarios to real map geometry.
- Add historical traffic demand data.
- Add multi-intersection corridor coordination.
- Export signal timing recommendations for traffic engineers.
- Compare daily/weekly policy performance across scenarios.

## Current Limitations

This prototype is intentionally scoped for a hackathon demo. It demonstrates the product logic, safety rules, and stakeholder workflow, but it is not a certified traffic-engineering deployment system yet.

- Traffic demand is simulated, with optional camera-detection endpoints prepared for future integration.
- CO2 and fuel savings are simplified estimates based on saved idle time.
- The adaptive controller is a transparent rule-based decision engine, not a trained black-box model.
- Real deployment would require calibrated detector zones, local traffic regulations, field validation, and hardware integration.

## Positioning

SmartTraffic Digital Twin is designed as a smart-city decision-support prototype. It helps explain how adaptive traffic control can reduce waiting time, improve throughput, prioritize public transport and emergency vehicles, and keep safety rules visible to stakeholders.
