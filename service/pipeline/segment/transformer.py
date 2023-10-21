import logging
import os
import time
from multiprocessing import Process
from threading import Thread
from typing import List
import stressinjector as injector
import psutil

def transformCPU():
    injector.CPUStress(seconds=3)

def transformGPU():
    injector.MemoryStress(gigabytes=10)

def transform_model(args):
    injector.CPUStress(seconds=10)
    injector.MemoryStress(gigabytes=1_000)