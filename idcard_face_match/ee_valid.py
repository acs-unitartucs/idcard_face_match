#!/usr/bin/env python3
# Copyright (c) 2021 Burak Can
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""Module for Estonian ID card related functions"""

import os
from pathlib import Path
from urllib import request
import subprocess
from queue import Queue


def check_validity(q: Queue, doc_num: str) -> None:
    """
    For Estonian documents, check the validity on "https://www2.politsei.ee/qr/?qr=" website
    """
    try:
        page = request.urlopen("https://www2.politsei.ee/qr/?qr=" + doc_num).read().decode("utf8")
    except Exception as e:
        q.put(["warning:online check failed", "exception"])
        return

    if f"The document {doc_num} is valid." in page:
        print(f"[+] check_validity(): The document {doc_num} is valid.")
    elif f"The document {doc_num} is invalid." in page:
        print(f"[-] check_validity(): The document {doc_num} is invalid.")
        q.put(["warning:online check failed", "INVALID"])
    elif f"The document {doc_num} has not been issued." in page:
        print(f"[-] check_validity(): The document {doc_num} has not been issued.")
        q.put(["warning:online check failed", "NOT ISSUED"])
    elif f"The document {doc_num} is a specimen." in page:
        print(f"[-] check_validity(): The document {doc_num} is a specimen.")
        q.put(["warning:online check failed", "SPECIMEN"])
    else:
        print("[-] check_validity(): politsei.ee response cannot be parsed!")
        q.put(["warning:online check failed", "cannot parse response"])


def download_certs(CSCA_certs_dir: Path, crls_dir: Path) -> None:
    """
    Download Estonian CSCA certificates and CRL
    """
    print("[+] Downloading CSCA certificates and CRLs.")
    csca_address = "https://pki.politsei.ee/"
    csca_certs_links = [
        "csca_Estonia_2007.cer",
        "csca_Estonia_2009.crt",
        "csca_Estonia_2012.cer",
        "csca_Estonia_2015.cer",
        "csca_Estonia_2016.cer",
        "csca_Estonia_2019.cer",
        "csca_Estonia_2020.der",
    ]

    # Get the crl
    subprocess.run(
        ["wget", "-N", "-P", os.path.abspath(crls_dir), csca_address + "csca.crl"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        check=True,
    )
    # Get csca certificates
    for link in csca_certs_links:
        subprocess.run(
            ["wget", "-N", "-P", os.path.abspath(CSCA_certs_dir), csca_address + link],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            check=True,
        )
