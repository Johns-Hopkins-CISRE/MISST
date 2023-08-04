#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""exception_handler.py: Contains methods for managing error messages."""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import sys

def short_err(msg: str, err: Exception):
    """
    Simplifies error messages for the user.
    The full exception can be invoked by typing "DEBUG".
    Only use this function for errors that might be raised by user error.
    If an exception is raise due to a program bug, this function should not
    be used.
    """
    # Print banner
    print("""
=======================================================================================
███╗   ███╗██╗███████╗███████╗████████╗       ███████╗██████╗ ██████╗  ██████╗ ██████╗ 
████╗ ████║██║██╔════╝██╔════╝╚══██╔══╝██╗    ██╔════╝██╔══██╗██╔══██╗██╔═══██╗██╔══██╗
██╔████╔██║██║███████╗███████╗   ██║   ╚═╝    █████╗  ██████╔╝██████╔╝██║   ██║██████╔╝
██║╚██╔╝██║██║╚════██║╚════██║   ██║   ██╗    ██╔══╝  ██╔══██╗██╔══██╗██║   ██║██╔══██╗
██║ ╚═╝ ██║██║███████║███████║   ██║   ╚═╝    ███████╗██║  ██║██║  ██║╚██████╔╝██║  ██║
=======================================================================================""")

    # Prints simplified error message
    print(msg)
    print("\nIf you need additional help, please contact Hudson Liu at hudsonliu0@gmail.com. " +
        "Furthermore, if you believe that this may be an issue with the source code, please " +
        "fill out a GitHub Issue Report using the following link: " +
        "\x1b]8;;https://github.com/Johns-Hopkins-CISRE/MISST/issues/new\x1b\\https://github.com/Johns-Hopkins-CISRE/MISST/issues/new\x1b]8;;\x1b\\")

    # Wait for user interaction and exit
    if input("Press any key to exit...") == "DEBUG":
        raise err
    sys.exit(0)
