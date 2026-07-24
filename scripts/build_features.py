from __future__ import annotations

import argparse

import pandas as pd

from clinical_domains.core.landmark import restrict_events_to_landmark
from clinical_domains.features.aggregation import aggregate_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encounters", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--landmark-hours", type=float, default=24)
    args = parser.parse_args()
    encounters = pd.read_csv(args.encounters)
    events = pd.read_csv(args.events)
    landmarked = restrict_events_to_landmark(events, encounters, hours=args.landmark_hours)
    aggregate_events(landmarked).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
