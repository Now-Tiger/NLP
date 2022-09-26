#!/usr/bin/env pypy3

import re

# Read/create the document or sentences.


def extract_email_ids(doc: str) -> str:
    addresses = re.findall(r'[\w\.-]+@[\w\.-]+', doc)
    print('----- Extracted email ids -----')
    for address in addresses:
        print(address)


def replace_mail_id(doc: str, email_address: str) -> str:
    replaced_id = re.sub(r'([\w\.-]+)@([\w\.-]+)', email_address, doc)
    print(f'\nreplaced email id:\n{replaced_id}')


if __name__ == '__main__':
    doc = "For more details please email us at: xyz@abc.com, pqr@mno.com"
    extract_email_ids(doc)

    new_doc= "For more details please email us at: xyz@abc.com"
    id = 'pqr@mno.com'
    replace_mail_id(new_doc, id)
