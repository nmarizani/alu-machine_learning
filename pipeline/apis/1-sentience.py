#!/usr/bin/env python3
"""Module to get homeworlds of all sentient species."""

import requests


def sentientPlanets():
    """
    Fetches the list of home planet names for all sentient species.

    Returns:
        list: A list of planet names.
    """
    url = "https://swapi.dev/api/species/"
    planets = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        species_list = data.get('results', [])

        for species in species_list:
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()

            if 'sentient' in [classification, designation]:
                homeworld_url = species.get('homeworld')
                if homeworld_url:
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_name = planet_response.json().get('name', 'unknown')
                        planets.append(planet_name)
                    else:
                        planets.append('unknown')
                else:
                    planets.append('unknown')

        url = data.get('next')

    return planets
