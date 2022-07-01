# coding=utf-8
# Import libraries
import time


def print_passed_time(start_time):
    return "--- %s seconds ---" % round(time.clock() - start_time, 4)

