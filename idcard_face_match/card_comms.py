#!/usr/bin/env python3
# Copyright (c) 2021 Burak Can
# Copyright (c) 2022 ACS research group, Institute of Computer Science, University of Tartu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""Functions related to card communication; APDU send etc."""

from smartcard.Exceptions import CardConnectionException
from smartcard.CardMonitoring import CardObserver
from smartcard.Exceptions import NoCardException
from smartcard.util import toHexString

from idcard_face_match.apdu import APDU
from idcard_face_match.secure_messaging import secure_messaging, process_rapdu, ReplyAPDUError
from idcard_face_match.secure_messaging_object import SMObject
from idcard_face_match.byte_operations import nb

import idcard_face_match.globals as globals

class CardWatcher(CardObserver):
    def __init__(self, q_main, q_camera, q_cardmon) -> None:
        super().__init__()
        self.q_main = q_main
        self.q_camera = q_camera
        self.q_cardmon = q_cardmon

        # support IDEMIA-powered Estonian ID cards
        self.validcards_est = [
            [0x3B, 0xDB, 0x96, 0x00, 0x80, 0xB1, 0xFE, 0x45, 0x1F, 0x83, 0x00, 0x12, 0x23, 0x3F, 0x53, 0x65, 0x49, 0x44, 0x0F, 0x90, 0x00, 0xF1],
        ]

        # support Latvian eID 2019
        self.validcards_lva = [
            [0x3B, 0xDB, 0x96, 0x00, 0x80, 0xB1, 0xFE, 0x45, 0x1F, 0x83, 0x00, 0x12, 0x42, 0x8F, 0x53, 0x65, 0x49, 0x44, 0x0F, 0x90, 0x00, 0x20], # LVeID on Oberthur COSMO v8.1 (2019)
        ]

    def update(self, _, actions):
        (addedcards, removedcards) = actions

        # validate the inserted card
        for card in addedcards:

            print("[+] CardWatcher(): Card ATR: " + toHexString(card.atr))

            # if card has a known good ATR
            if card.atr in self.validcards_est+self.validcards_lva:

                # try to establish connection
                print("[+] CardWatcher(): trying to establish connection...")
                try:
                    card.connection = card.createConnection()
                    card.connection.connect()
                except (NoCardException, CardConnectionException) as e:
                    print("[-] CardWatcher(): establishing connection failed!")
                    self.q_camera.put(["card communication failure", "failed to establish connection with the card"])
                    continue

                # check if the IDEMIA card has eMRTD applet
                # Selecting eMRTD Application ‘International AID’: A0000002471001
                aid = bytes.fromhex("A0000002471001")
                print("[+] CardWatcher(): selecting eMRTD applet...")
                try:
                    _, sw1, sw2 = card.connection.transmit([0x00, 0xA4, 0x04, 0x0C, len(aid)] + list(aid))
                    if [sw1, sw2] != [0x90, 0x00]:
                        print("[-] CardWatcher(): eMRTD applet selection unsuccessful!")
                        self.q_cardmon.put(["Unknown card"])
                        continue
                except CardCommunicationError:
                    self.q_camera.put(["card communication failure", "failed to establish connection with the card"])
                    self.q_main.put("-RAISED EXCEPTION-", "")
                    continue
                except CardConnectionException as ex:
                    self.q_camera.put(["card communication failure", "failed to establish connection with the card"])
                    print(ex)
                    self.q_main.put("-RAISED EXCEPTION-", "")
                    continue

                if card.atr in self.validcards_est:
                    self.q_cardmon.put(["Valid card EST", card.connection, card.atr])
                elif card.atr in self.validcards_lva:
                    self.q_cardmon.put(["Valid card LVA", card.connection, card.atr])

            elif not card.atr:
                print("[-] CardWatcher(): card read error!")
                self.q_cardmon.put(["Card read error"])
            else:
                print("[-] CardWatcher(): unknown card!")
                self.q_cardmon.put(["Unknown card"])

        for card in removedcards:
            print("[-] CardWatcher(): Removed card ATR: " + toHexString(card.atr))
            self.q_camera.put(["Disconnect"])
            self.q_cardmon.put("Disconnect")


class CardCommunicationError(Exception):
    """Exception to raise when an error occurs during card communication."""


def send(sm_object: SMObject, apdu: APDU) -> bytes:
    """
    Send APDU to the channel and return the data if there are no errors.
    """
    channel = sm_object.channel
    apdu_bytes = secure_messaging(sm_object, apdu)

    data, sw1, sw2 = channel.transmit(list(apdu_bytes))

    # update read bytes counter (for progress bar)
    if globals.bytes_total:
        globals.bytes_read+= len(data)
        globals.progress = min(100, round(globals.bytes_read*100/(globals.bytes_total), 1))
        print("[+] send(): increasing progress:", globals.progress)

    # success
    if [sw1, sw2] == [0x90, 0x00]:
        try:
            data = process_rapdu(sm_object, bytes(data))
        except ReplyAPDUError as ex:
            raise CardCommunicationError("[-] Reply APDU MAC doesn't match!") from ex
        else:
            return data
    # signals that there is more data to read
    if sw1 == 0x61:
        print("[=] TAKE A LOOK! More data to read:", sw2)
        return data + send(
            sm_object, APDU(b"\x00", b"\xC0", b"\x00", b"\x00", Le=nb(sw2))
        )  # GET RESPONSE of sw2 bytes
    if sw1 == 0x6C:
        print("[=] TAKE A LOOK! Resending with Le:", sw2)
        return send(
            sm_object, APDU(apdu.cla, apdu.ins, apdu.p1, apdu.p2, Le=nb(sw2))
        )  # resend APDU with Le = sw2
    # probably error condition
    # channel.disconnect()
    print("[-] Card communication error occured.")
    print(
        "Error: %02x %02x, sending APDU: %s"
        % (sw1, sw2, " ".join(["{:02x}".format(x) for x in apdu_bytes]).upper())
    )
    print(
        "Plain APDU: "
        + " ".join(
            [
                "{:02x}".format(x)
                for x in (
                    apdu.get_command_header()
                    + (apdu.Lc or b"")
                    + (apdu.cdata or b"")
                    + (apdu.Le or b"")
                )
            ]
        ).upper()
    )
    raise CardCommunicationError(
        "Error: %02x %02x, sending APDU: %s"
        % (sw1, sw2, " ".join(["{:02x}".format(x) for x in apdu_bytes]).upper())
    )
