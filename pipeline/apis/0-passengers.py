#!/usr/bin/env python3
"""Module to get ships that can carry a given number of passengers."""

import requests
import re


def availableShips(passengerCount):
    """
    Returns a list of starships that can carry at least `passengerCount`.

    Args:
        passengerCount (int): Minimum required passenger capacity.

    Returns:
        list: Names of qualifying starships.
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        results = data.get('results', [])

        for ship in results:
            passenger_str = ship.get('passengers', '0').replace(',', '')

            # Skip if not a clean integer (e.g., "n/a", "20-50", "unknown")
            if not re.fullmatch(r'\d+', passenger_str):
                continue

            if int(passenger_str) >= passengerCount:
                ships.append(ship.get('name'))

        url = data.get('next')

    return ships
