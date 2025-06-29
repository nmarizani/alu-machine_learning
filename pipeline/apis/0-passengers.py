#!/usr/bin/env python3
"""Module to get available ships based on passenger capacity."""

import requests


def availableShips(passengerCount):
    """
    Returns a list of starships from the SWAPI that can carry
    at least the given number of passengers.

    Args:
        passengerCount (int): The required number of passengers.

    Returns:
        list: List of ship names that meet the requirement.
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
            passengers = ship.get('passengers', '0').replace(',', '').replace('unknown', '0')
            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship.get('name'))
            except ValueError:
                continue

        url = data.get('next')

    return ships
