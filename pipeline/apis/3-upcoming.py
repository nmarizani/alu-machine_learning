#!/usr/bin/env python3
"""Script to display the next SpaceX launch information."""

import requests
import datetime


def get_upcoming_launch():
    # Get upcoming launches
    launches_url = "https://api.spacexdata.com/v4/launches/upcoming"
    launches = requests.get(launches_url).json()

    # Sort by date_unix (ascending)
    launches.sort(key=lambda x: x.get("date_unix", float("inf")))

    # Get the earliest launch
    next_launch = launches[0]

    launch_name = next_launch.get("name", "Unknown Launch")
    date_utc = next_launch.get("date_utc")
    rocket_id = next_launch.get("rocket")
    launchpad_id = next_launch.get("launchpad")

    # Convert UTC to local time
    date_obj = datetime.datetime.strptime(date_utc, "%Y-%m-%dT%H:%M:%S.%fZ")
    local_date = date_obj.astimezone().isoformat()

    # Get rocket name
    rocket = requests.get("https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)).json()
    rocket_name = rocket.get("name", "Unknown Rocket")

    # Get launchpad name and locality
    launchpad = requests.get("https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id)).json()
    launchpad_name = launchpad.get("name", "Unknown Launchpad")
    locality = launchpad.get("locality", "Unknown Locality")

    # Final output
    print("{} ({}) {} - {} ({})".format(
        launch_name, local_date, rocket_name, launchpad_name, locality))


if __name__ == "__main__":
    get_upcoming_launch()
