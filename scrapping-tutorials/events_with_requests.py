#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests

# from bs4 import BeautifulSoup as BS

url = 'https://www.python.org/events/python-events/'

# _headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
#         AppleWebKit/537.36(KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36' }
# response = requests.request('GET', url=url, headers=_headers)

# soup = BS(response.text, 'html.parser')
# events = soup.find('ul', {'class':'list-recent-events'}).findAll('li')

# event_details = {}
# for event in events:
#     event_details['name'] = event.find('h3').find('a').text
#     event_details['location'] = event.find('span', {'class': 'event-location'}).text
#     event_details['time'] = event.find('time').text


def get_upcoming_events(url: str, body=None) -> dict:
    _headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
        AppleWebKit/537.36(KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36' }
    response = requests.get(url, headers=_headers, data=body)
    return response.text


result = get_upcoming_events(url=url)
print(result[:200])