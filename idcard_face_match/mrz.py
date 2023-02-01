#!/usr/bin/env python3
# Copyright (c) 2021 Burak Can
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""Module for MRZ related calculations and functions"""

from typing import Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
from OpenSSL.crypto import load_certificate, FILETYPE_ASN1

from idcard_face_match.apdu import APDU
from idcard_face_match.card_comms import send
from idcard_face_match.byte_operations import nb
from idcard_face_match.secure_messaging_object import SMObject
from idcard_face_match.file_operations import read_data_from_ef


def calculate_check_digit(data: str) -> str:
    """Calculate MRZ check digits for data.

    :data data: Data to calculate the check digit of
    :returns: check digit
    """

    # fmt: off
    values = {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "<": 0, "A": 10, "B": 11,
        "C": 12, "D": 13, "E": 14, "F": 15, "G": 16, "H": 17,
        "I": 18, "J": 19, "K": 20, "L": 21, "M": 22, "N": 23,
        "O": 24, "P": 25, "Q": 26, "R": 27, "S": 28, "T": 29,
        "U": 30, "V": 31, "W": 32, "X": 33, "Y": 34, "Z": 35,
    }
    # fmt: on
    weights = [7, 3, 1]
    total = 0

    for counter, value in enumerate(data):
        total += weights[counter % 3] * values[value]
    return str(total % 10)


def estonia_read_mrz(sm_object: SMObject) -> Tuple[str, str]:
    """Read Estonian ID card information from personal data"""
    # reading personal data file (EstEID spec page 30)
    print("[+] estonia_read_mrz(): Selecting IAS ECC applet AID: A000000077010800070000FE00000100...")
    ias_ecc_aid = bytes.fromhex("A000000077010800070000FE00000100")

    # exception caught in main program loop
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x04", b"\x00", Lc=nb(len(ias_ecc_aid)), cdata=ias_ecc_aid))
    print("[+] estonia_read_mrz(): Selecting DF ID: 5000...")
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x01", b"\x0C", Lc=b"\x02", cdata=b"\x50\x00"))
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x01", b"\x0C", Lc=b"\x02", cdata=b"\x50\x07"))
    print("[+] estonia_read_mrz(): Reading personal data files...")
    document_number = send(sm_object, APDU(b"\x00", b"\xB0", b"\x00", b"\x00", Le=b"\x00")).decode("utf8")

    send(sm_object, APDU(b"\x00", b"\xA4", b"\x01", b"\x0C", Lc=b"\x02", cdata=b"\x50\x05"))
    date_of_birth = send(sm_object, APDU(b"\x00", b"\xB0", b"\x00", b"\x00", Le=b"\x00"))[:10].decode("utf8")
    date_of_birth = date_of_birth[-2:] + date_of_birth[3:5] + date_of_birth[:2]
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x01", b"\x0C", Lc=b"\x02", cdata=b"\x50\x08"))
    date_of_expiry = send(sm_object, APDU(b"\x00", b"\xB0", b"\x00", b"\x00", Le=b"\x00")).decode("utf8")
    date_of_expiry = date_of_expiry[-2:] + date_of_expiry[3:5] + date_of_expiry[:2]

    # Construct the 'MRZ information'
    print("[+] estonia_read_mrz(): Constructing the MRZ information...")
    mrz_information = (
        document_number
        + calculate_check_digit(document_number)
        + date_of_birth
        + calculate_check_digit(date_of_birth)
        + date_of_expiry
        + calculate_check_digit(date_of_expiry)
    )

    # Name is read from MRZ
#    send(sm_object, APDU(b"\x00", b"\xA4", b"\x01", b"\x0C", Lc=b"\x02", cdata=b"\x50\x01"))
#    surname = send(sm_object, APDU(b"\x00", b"\xB0", b"\x00", b"\x00", Le=b"\x00")).decode("utf8")
#    send(sm_object, APDU(b"\x00", b"\xA4", b"\x01", b"\x0C", Lc=b"\x02", cdata=b"\x50\x02"))
#    name = send(sm_object, APDU(b"\x00", b"\xB0", b"\x00", b"\x00", Le=b"\x00")).decode("utf8")
#    send(sm_object, APDU(b"\x00", b"\xA4", b"\x01", b"\x0C", Lc=b"\x02", cdata=b"\x50\x06"))
#    personal_id_code = send(sm_object, APDU(b"\x00", b"\xB0", b"\x00", b"\x00", Le=b"\x00")).decode("utf8")

    # Select LDS applet
    # A00000024710FF is applet id
    print("[+] estonia_read_mrz(): Selecting LDS AID: A00000024710FF...")
    aid = bytes.fromhex("A00000024710FF")
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x04", b"\x00", Lc=nb(len(aid)), cdata=aid))

    return mrz_information, document_number


def latvia_read_mrz(sm_object: SMObject) -> Tuple[str, str]:

    """Read and construct Latvian eID card MRZ information"""

    # Latvian eID does not contain a personal data file (like Estonian ID card) where MRZ-analog information required to construct BAC key is stored.
    # However:
    # (1) document_number is readable from IDEMIA EF.SN file
    # (2) date_of_birth can be extracted from personal ID code stored in the auth/sign certificates
    # (3) date_of_expiry can be calculated by adding 10 years to the notBefore date of certificates
    # This is just a heuristic that is prune to fail (points (2) and (3), in particular).

    print("[+] latvia_read_mrz(): Selecting IAS ECC applet AID: A000000077010800070000FE00000100...")
    ias_ecc_aid = bytes.fromhex("A000000077010800070000FE00000100")

    # exception caught in main program loop
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x04", b"\x00", Lc=nb(len(ias_ecc_aid)), cdata=ias_ecc_aid))

    print("[+] latvia_read_mrz(): Selecting EF.SN: D003...")
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x02", b"\x0C", Lc=b"\x02", cdata=b"\xD0\x03"))
    print("[+] latvia_read_mrz(): Reading IAS-ECC serial number...")
    document_number = send(sm_object, APDU(b"\x00", b"\xB0", b"\x00", b"\x00", Le=b"\x00"))[2:].decode("utf8")

    print("[+] latvia_read_mrz(): Reading auth certificate...")
    # AWP AID (for reading auth certificate)
    awp_aid = bytes.fromhex("E828BD080FF2504F5420415750")
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x04", b"\x0C", Lc=nb(len(awp_aid)), cdata=awp_aid))
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x02", b"\x0C", Lc=b"\x02", cdata=b"\x34\x01"))
    cert = read_data_from_ef(None, sm_object, b"\x34\x01", "EF:3401")
    cert = load_certificate(FILETYPE_ASN1, cert)

    # starting 2017-07-01 Latvian personal identification codes are randomly generated
    # and do not contain person's date of birth anymore
    # without knowing person's date of birth BAC key cannot be constructed :/
    PNO = cert.get_subject().serialNumber
    if not (PNO.startswith("PNOLV-") and int(PNO[6:8])<=31):
        print("[-] latvia_read_mrz(): personal ID code in the new format: the date of birth cannot be obtained!")
        return
    date_of_birth = PNO[10:12] + PNO[8:10] + PNO[6:8]
    print("[+] latvia_read_mrz(): date of birth from personal identification code:", date_of_birth)

    # expiration date is calculated by adding 10 years to the certificate validity starting day
    # works only if:
    # (1) the document that has been issued is valid for 10 years; and
    # (2) certificates have not been renewed and have been generated on the day of document issuance
    date_expiry = datetime.strptime(cert.get_notBefore().decode(),"%Y%m%d%H%M%SZ")  + relativedelta(years=10, days=-1)
    date_of_expiry = datetime.strftime(date_expiry, '%y%m%d')
    print("[+] latvia_read_mrz(): calculated date of expiry:", date_of_expiry)

    # Construct the 'MRZ information'
    print("[+] latvia_read_mrz(): Constructing the MRZ information...")
    mrz_information = (
        document_number
        + calculate_check_digit(document_number)
        + date_of_birth
        + calculate_check_digit(date_of_birth)
        + date_of_expiry
        + calculate_check_digit(date_of_expiry)
    )

    # Select LDS applet
    # A00000024710FF is applet id
    print("[+] latvia_read_mrz(): Selecting LDS AID: A00000024710FF...")
    aid = bytes.fromhex("A00000024710FF")
    send(sm_object, APDU(b"\x00", b"\xA4", b"\x04", b"\x00", Lc=nb(len(aid)), cdata=aid))

    return mrz_information, document_number


def check_expiration(expiry_date: bytes) -> bool:
    """Check if the MRZ expiry date is older than today's date."""
    date = expiry_date.decode("utf-8")
    date_obj = datetime.strptime(date, "%y%m%d")
    if date_obj.date() < datetime.now().date():
        return False
    return True
