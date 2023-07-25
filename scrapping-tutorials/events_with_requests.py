#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup as BS


def get_upcoming_events(url: str, body=None) -> dict:
    """ Returns a dictionary of upcoming events from python.org/events """
    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
            AppleWebKit/537.36(KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'
    }
    response = requests.get(url, headers=_headers, data=body)
    soup = BS(response.text, 'lxml')
    events = soup.find('ul', {'class': 'list-recent-events'}).findAll('li')
    event_details = dict()

    for event in events:
        event_details['name'] = event.find('h3').find('a').text
        event_details['location'] = event.find('span', {'class': 'event-location'}).text
        event_details['time'] = event.find('time').text

    return event_details


if __name__ == "__main__":
    URL = 'https://www.python.org/events/python-events/'
    result = get_upcoming_events(url=URL)
    print(result)