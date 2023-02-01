#!/usr/bin/env python3
# Copyright (c) 2021 Burak Can
# Copyright (c) 2022 ACS research group, Institute of Computer Science, University of Tartu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""The main program loop idcard_face_match"""

import os

# import sys
# import termios
import argparse
from typing import Union, TextIO, BinaryIO
from pathlib import Path
from queue import Queue

from smartcard.Exceptions import CardConnectionException

from idcard_face_match.apdu import APDU
from idcard_face_match.card_comms import send, CardCommunicationError
from idcard_face_match.mrz import estonia_read_mrz, latvia_read_mrz, check_expiration
from idcard_face_match.bac import establish_bac_session_keys, SessionKeyEstablishmentError
from idcard_face_match.file_operations import (
    EFReadError,
    get_size_from_ef,
    read_data_from_ef,
    parse_efcom,
    get_dg_numbers,
    assert_dg_hash,
    get_dg1_content,
    parse_security_infos,
)
from idcard_face_match.passive_authentication import (
    passive_auth,
    PassiveAuthenticationCriticalError,
)
from idcard_face_match.face_compare import opencv_dnn_detector
from idcard_face_match.chip_authentication import chip_auth, ChipAuthenticationError
from idcard_face_match.active_authentication import active_auth, ActiveAuthenticationError
from idcard_face_match.image_operations import get_jpeg_im
from idcard_face_match.ee_valid import check_validity, download_certs
from idcard_face_match.secure_messaging_object import SMObject
from idcard_face_match.log_operations import create_output_folder
from idcard_face_match.icao_pkd_load import build_store
from idcard_face_match.byte_operations import nb

import idcard_face_match.globals as globals


def main_program_loop(
    q_main: Queue,
    q_camera: Queue,
    q_cardmon: Queue,
    args: argparse.Namespace,
    first_run: bool,
) -> None:
    """main function"""

    # Get dir arguments else fallback to EE certs
    CSCA_certs_dir = Path("certs/csca_certs")
    crls_dir = Path("certs/crls")
    output_dir = args.output
    output_files = not args.output is None
    outfile: Union[TextIO, BinaryIO]

    if ( not os.path.isdir(CSCA_certs_dir) or not os.path.isdir(crls_dir) ):
        print("[+] main_program_loop(): downloading CSCA and CRLs...")
        download_certs(CSCA_certs_dir, crls_dir)

    if first_run:
        dsccrl_dir = Path(os.path.join(os.path.dirname(CSCA_certs_dir), Path("icao_pkd_dsccrl")))
        ml_dir = Path(os.path.join(os.path.dirname(CSCA_certs_dir), Path("icao_pkd_ml")))
        print("[+] main_program_loop(): building CERT store...")
        build_store(CSCA_certs_dir, crls_dir, ml_dir, dsccrl_dir)

        # create face detector network
        opencv_dnn_detector()

        q_camera.put(['main thread ready'])

    # wait until card (possibly removed and) inserted
    while True:
        connection = q_cardmon.get()
        if connection[0].startswith("Valid card "):
            issuing_country = connection[0].replace('Valid card ', '')
            sm_object = SMObject(connection[1])
            q_camera.put(connection)
            break
        elif connection[0] in ["Unknown card", "Card read error"]:
            q_camera.put(connection)

    ## DERIVATION OF DOCUMENT BASIC ACCESS KEYS (KENC AND KMAC) ##
    q_camera.put(["activity:establishing secure channel"])
    try:
        if issuing_country == 'EST':
            mrz_information, document_number = estonia_read_mrz(sm_object)
        elif issuing_country == 'LVA':
            mrz_information, document_number = latvia_read_mrz(sm_object)
        else:
            print("[-] Unknown issuing country:", issuing_country)
            raise
    except CardCommunicationError:
        q_camera.put(["error:card communication failure", "reading from card MRZ information"])
        q_main.put("-RAISED EXCEPTION-", "")
        return
    except CardConnectionException as ex:
        print(ex)
        q_camera.put(["error:card communication failure", "reading from card MRZ information"])
        q_main.put("-RAISED EXCEPTION-", "")
        return
    except:
        q_camera.put(["error:BAC key failure", "constructing MRZ information"])
        q_main.put("-RAISED EXCEPTION-", "")
        return

    if output_files:
        folder_name = create_output_folder(output_dir, document_number)


    # Select eMRTD application
    print("[+] main_program_loop(): Selecting eMRTD Application ‘International AID’: A0000002471001...")
    aid = bytes.fromhex("A0000002471001")
    try:
        send(sm_object, APDU(b"\x00", b"\xA4", b"\x04", b"\x0C", Lc=nb(len(aid)), cdata=aid))
    except CardCommunicationError:
        q_camera.put(["error:card communication failure", "selecting eMRTD applet"])
        q_main.put("-RAISED EXCEPTION-", "")
        return
    except CardConnectionException as ex:
        q_camera.put(["error:card communication failure", "selecting eMRTD applet"])
        print(ex)
        q_main.put("-RAISED EXCEPTION-", "")
        return

    ## SECURE MESSAGING ##
    try:
        establish_bac_session_keys(sm_object, mrz_information.encode("utf-8"))
    except SessionKeyEstablishmentError as ex:
        print(ex)
        print("[-] main_program_loop(): Error while establishing BAC session keys")
        q_camera.put(["error:BAC key failure", "deriving BAC session keys"])
        q_main.put("-RAISED EXCEPTION-", "")
        return
    except CardCommunicationError:
        q_camera.put(["error:BAC key failure", "using BAC session keys"])
        q_main.put("-RAISED EXCEPTION-", "")
        return
    except CardConnectionException as ex:
        q_camera.put(["error:card communication failure", "establishing BAC session"])
        print(ex)
        q_main.put("-RAISED EXCEPTION-", "")
        return


    # Read EF.COM
    try:
        efcom = read_data_from_ef(q_main, sm_object, b"\x01\x1E", "EF.COM")
    except EFReadError as ex:
        q_camera.put(["error:card communication failure", "reading EF.COM"])
        print(ex)
        print("[-] main_program_loop(): Error while reading file EF.COM")
        q_main.put("-RAISED EXCEPTION-", "")
        return
    except CardCommunicationError:
        q_camera.put(["error:card communication failure", "reading EF.COM"])
        q_main.put("-RAISED EXCEPTION-", "")
        return
    except CardConnectionException as ex:
        q_camera.put(["error:card communication failure", "reading EF.COM"])
        print(ex)
        q_main.put("-RAISED EXCEPTION-", "")
        return
    else:
        if output_files:
            with open(os.path.join(folder_name, "EF_COM.BIN"), "wb") as outfile:
                outfile.write(efcom)
        ef_com_dg_list = parse_efcom(efcom)

    # read the size of each file to estimate the total amount of data that needs to be read from the card (needed for the progress bar)
    globals.bytes_total = 0
    bytes_total = 0
    for file_id in ef_com_dg_list.keys():

        # skip files we are not authorized to read
        if file_id[0] > 2 and file_id[0] < 14:
            continue

        try:
            _, data_len = get_size_from_ef(q_main, sm_object, b"\x01"+file_id, ef_com_dg_list[file_id])
        except EFReadError as ex:
            q_camera.put(["error:card communication failure", "obtaining file sizes"])
            print(ex)
            print("[-] main_program_loop(): Error while obtaining file sizes")
            q_main.put("-RAISED EXCEPTION-", "")
            return
        except CardCommunicationError:
            q_camera.put(["error:card communication failure", "obtaining file sizes"])
            q_main.put("-RAISED EXCEPTION-", "")
            return
        except CardConnectionException as ex:
            q_camera.put(["error:card communication failure", "obtaining file sizes"])
            print(ex)
            q_main.put("-RAISED EXCEPTION-", "")
            return
        else:
            bytes_total+= data_len

    globals.bytes_total = bytes_total + 2500
    print("[+] main_program_loop(): Total number of bytes to be read from the card:", globals.bytes_total)

    q_camera.put(["activity:verifying data authenticity"])

    # verifying data authenticity

    # Read EF.SOD
    try:
        efsod = read_data_from_ef(q_main, sm_object, b"\x01\x1D", "EF.SOD")
    except EFReadError as ex:
        q_camera.put(["error:card communication failure", "reading EF.SOD"])
        print(ex)
        print("[-] main_program_loop(): Error while reading file EF.SOD")
        q_main.put("-RAISED EXCEPTION-", "")
        return
    except CardCommunicationError:
        q_camera.put(["error:card communication failure", "reading EF.SOD"])
        q_main.put("-RAISED EXCEPTION-", "")
        return
    except CardConnectionException as ex:
        q_camera.put(["error:card communication failure", "reading EF.SOD"])
        print(ex)
        q_main.put("-RAISED EXCEPTION-", "")
        return
    else:
        if output_files:
            with open(os.path.join(folder_name, "EF_SOD.BIN"), "wb") as outfile:
                outfile.write(efsod)

    pa_error = False
    ee_deviant_doc = False
    try:
        passive_auth_return = passive_auth(efsod, ee_deviant_doc=ee_deviant_doc, dump=False)
    except PassiveAuthenticationCriticalError as ex:
        q_camera.put(["warning:data authenticity check failed", "PassiveAuthenticationCriticalError"])
        print(ex)
    else:
        if output_files:
            with open(os.path.join(folder_name, "CDS.der"), "wb") as outfile:
                outfile.write(passive_auth_return[2])
        if passive_auth_return[3] is None:
            pa_error = False
            hash_alg, data_group_hash_values, _, _ = passive_auth_return
        else:
            pa_error = True
            hash_alg, data_group_hash_values, _, exception = passive_auth_return
            q_camera.put(["warning:data authenticity check failed", ""])
            print(exception)

    ef_sod_dg_list = get_dg_numbers(data_group_hash_values)

    if ef_com_dg_list != ef_sod_dg_list:
        print("[-] main_program_loop(): EF.COM might have been changed, there are differences between EF_COM DGs and EF_SOD DGs!")
        q_camera.put(["warning:data authenticity check failed", "EF.COM does not match"])
        pa_error = True

    q_camera.put(["activity:verifying chip authenticity"])

    file_read_error = False
    security_infos = []

    # perform Chip Authentication
    if b"\x0e" in ef_sod_dg_list:
        try:
            DG = read_data_from_ef(q_main, sm_object, b"\x01" + b"\x0e", "EF.DG14")
        except EFReadError as ex:
            q_camera.put(["error:card communication failure", "reading EF.DG14"])
            print(ex)
            print("[-] main_program_loop(): Error while reading file EF.DG14")
            q_main.put("-RAISED EXCEPTION-", "")
            return
        except CardCommunicationError:
            q_camera.put(["error:card communication failure", "reading EF.DG14"])
            q_main.put("-RAISED EXCEPTION-", "")
            return
        except CardConnectionException as ex:
            q_camera.put(["error:card communication failure", "reading EF.DG14"])
            print(ex)
            q_main.put("-RAISED EXCEPTION-", "")
            return

        if output_files:
            with open(os.path.join(folder_name, "EF.DG14.BIN"), "wb") as outfile:
                outfile.write(DG)

        if not assert_dg_hash(DG, data_group_hash_values, hash_alg, b"\x0e"):
            pa_error = True
            q_camera.put(["warning:data authenticity check failed", "DG14 hash mismatch"])
            file_read_error = True
        else:
            pass

        security_infos = parse_security_infos(DG)
        try:
            chip_auth(security_infos, sm_object)
        except ChipAuthenticationError as ex:
            q_camera.put(["warning:chip authentication failure", "ChipAuthenticationError"])
            print(ex)
        except CardCommunicationError:
            q_camera.put(["error:card communication failure", "performing chip authentication"])
            q_main.put("-RAISED EXCEPTION-", "")
            return
        except CardConnectionException as ex:
            q_camera.put(["error:card communication failure", "performing chip authentication"])
            print(ex)
            q_main.put("-RAISED EXCEPTION-", "")
            return
        else:
            pass # chip authentication successful


    # perform Active Authentication (AA)
    # (in principle AA can be skipped since chip authentication provides the same security assurance)
    if b"\x0f" in ef_sod_dg_list:
        try:
            DG = read_data_from_ef(q_main, sm_object, b"\x01" + b"\x0f", "EF.DG15")
        except EFReadError as ex:
            q_camera.put(["error:card communication failure", "reading EF.DG15"])
            print(ex)
            print("[-] main_program_loop(): Error while reading file EF.DG15")
            q_main.put("-RAISED EXCEPTION-", "")
            return
        except CardCommunicationError:
            q_camera.put(["error:card communication failure", "reading EF.DG15"])
            q_main.put("-RAISED EXCEPTION-", "")
            return
        except CardConnectionException as ex:
            q_camera.put(["error:card communication failure", "reading EF.DG15"])
            print(ex)
            q_main.put("-RAISED EXCEPTION-", "")
            return

        if output_files:
            with open(os.path.join(folder_name, "EF.DG15.BIN"), "wb") as outfile:
                outfile.write(DG)

        if not assert_dg_hash(DG, data_group_hash_values, hash_alg, b"\x0f"):
            pa_error = True
            q_camera.put(["warning:data authenticity check failed", "DG15 hash mismatch"])
            file_read_error = True
        else:
            pass

        try:
            active_auth(DG, sm_object, security_infos)
        except ActiveAuthenticationError as ex:
            q_camera.put(["warning:chip authentication failure", "ActiveAuthenticationError"])
            print(ex)
        except CardCommunicationError:
            q_camera.put(["error:card communication failure", "performing active authentication"])
            q_main.put("-RAISED EXCEPTION-", "")
            return
        except CardConnectionException as ex:
            q_camera.put(["error:card communication failure", "performing active authentication"])
            print(ex)
            q_main.put("-RAISED EXCEPTION-", "")
            return
        else:
            pass

    # either Chip Authentication or Active Authentication must be performed for successful chip authentication
    if not (b"\x0e" in ef_sod_dg_list or b"\x0f" in ef_sod_dg_list):
            q_camera.put(["warning:chip authentication failure", "neither Chip Authentication or Active Authentication available"])

    # read other data files DG1 (MRZ), DG2 (facial image)
    for dg, dgname in ef_sod_dg_list.items():

        if dg == b"\x0f" or dg == b"\x0e":
            # Active Authentication and Chip Authentication assumed completed
            continue

        if dg == b"\x03" or dg == b"\x04":
            # Sensitive Data: Finger and iris image data stored in the LDS
            # Data Groups 3 and 4, respectively. These data are considered
            # to be more privacy sensitive than data stored in the other
            # Data Groups. We cannot read them.
            continue

        if dg == b"\x02":
            q_camera.put(["activity:downloading facial image"])

        try:
            DG = read_data_from_ef(q_main, sm_object, b"\x01" + dg, dgname)
        except EFReadError as ex:
            print(ex)
            print(f"[-] main_program_loop(): Error while reading file {dgname}")
            if dg in [b"\x01", b"\x02"]:
                q_camera.put(["error:card communication failure", f"reading {dgname}"])
                q_main.put("-RAISED EXCEPTION-", "")
                return
            continue
        except CardCommunicationError:
            q_camera.put(["error:card communication failure", f"reading {dgname}"])
            q_main.put("-RAISED EXCEPTION-", "")
            return
        except CardConnectionException as ex:
            q_camera.put(["error:card communication failure", f"reading {dgname}"])
            print(ex)
            q_main.put("-RAISED EXCEPTION-", "")
            return

        if output_files:
            with open(os.path.join(folder_name, dgname + ".BIN"), "wb") as outfile:
                outfile.write(DG)

        if not assert_dg_hash(DG, data_group_hash_values, hash_alg, dg):
            pa_error = True
            q_camera.put(["warning:data authenticity check failed", f"{dgname} hash mismatch"])
            file_read_error = True

        # DG1 (MRZ)
        if dg == b"\x01":
            mrz_read = get_dg1_content(DG)
            print("[+] MRZ:", mrz_read)
            mrz_expiration_date = b""
            if len(mrz_read) == 90:
                mrz_expiration_date = mrz_read[38:44]
            elif len(mrz_read) == 72:
                mrz_expiration_date = mrz_read[57:63]
            elif len(mrz_read) == 88:
                mrz_expiration_date = mrz_read[65:71]
            else:
                print("[-] main_program_loop(): Error in MRZ that was read from DG1")
                q_camera.put(["warning:document expired", f"bad MRZ read from DG1"])
            if mrz_expiration_date == b"":
                # Assume the document is expired
                q_camera.put(["warning:document expired", f"no expiration date in MRZ"])
            else:
                valid = check_expiration(mrz_expiration_date)
                if not valid:
                    q_camera.put(["warning:document expired", "expiration date in past"])

            # read name and surname from MRZ
            name = ""
            surname = ""
            row3 = mrz_read[-30:].decode()
            name_surname = row3.rstrip('<').replace('<',' ')
            if '  ' in name_surname:
                surname, name = name_surname.split('  ', 1)
            else:
                surname = name_surname
            q_camera.put(['ID name', f"{name} {surname}"])

            #if args.online_check:
            #    issuing_country = mrz_read[2:5]
            #    # online document validity check on politsei.ee for Estonian ID cards
            #    if issuing_country == b"EST":
            #        q_camera.put(["activity:online validity check"])
            #        check_validity(q, document_number)

        # DG2 (facial image)
        if dg == b"\x02":
            id_image = get_jpeg_im(DG)

            if output_files:
                with open(os.path.join(folder_name, "face.jpg"), "wb") as outfile:
                    outfile.write(id_image)

            q_camera.put(["ID image", id_image])


    # wait for card disconnect
    q_cardmon.get()

    q_main.put("-RUN COMPLETE-", "")
    return
