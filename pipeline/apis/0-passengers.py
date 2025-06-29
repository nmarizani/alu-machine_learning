#!/usr/bin/env python3
"""Module to get available ships based on passenger capacity."""

import requests


def availableShips(passengerCount):
    """
    Returns a list of starships from the SWAPI that can carry
    at least the given number of passengers.

    Args:
        passengerCount (int): Required passenger capacity.

    Returns:
        list: List of starship names meeting the requirement.
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        try:
            response = requests.get(url)
            data = response.json()
        except Exception:
            break

        results = data.get('results', [])

        for ship in results:
            raw_passengers = ship.get('passengers', '0')
            cleaned = raw_passengers.replace(',', '').replace('unknown', '0')
            try:
                num = int(cleaned)
                if num >= passengerCount:
                    ships.append(ship.get('name'))
            except ValueError:
                continue

        url = data.get('next')

    return ships
