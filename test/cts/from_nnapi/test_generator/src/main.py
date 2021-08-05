#!/usr/bin/python3

import os
import argparse
from test_generator import IndentedPrint
from test_generator import InitializeCtsTestFile
from test_generator import SmartOpen

def ParseCmdLine():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="the spec file/directory")
    parser.add_argument(
        "-t", "--test", help="the generated test file/directory")
    parser.add_argument(
        "-c", "--cts", help="the CTS generated tests in one file cts.js")
    args = parser.parse_args()
    return (args.spec, args.test, args.cts)

def ConvertNNAPITest(spec, test, fused = False):
    if os.path.isfile(spec):
        os.system(
            "python3 ./src/cts_generator.py %s -t %s -f %s" % \
            (spec, test, fused))
    elif os.path.isdir(spec):
        for version in sorted(os.listdir(spec)):
            specDir = os.path.join(spec, version)
            testDir = os.path.join(test, version)
            if not os.path.exists(testDir):
                os.makedirs(testDir)
            os.system(
                "python3 ./src/cts_generator.py %s -t %s -f %s" % \
                (specDir, testDir, fused))

def DumpAllInOneCtsTest(test, cts):
    versionList = os.listdir(test)
    with SmartOpen(cts, mode="a") as aioTest:
        InitializeCtsTestFile(aioTest, 3)
        for version in sorted(versionList):
            versionPath = os.path.join(test, version)
            for generatedTest in sorted(os.listdir(versionPath)):
                generatedTestPath = os.path.join(versionPath, generatedTest)
                with SmartOpen(generatedTestPath, mode="r") as readFile:
                    fileText = readFile.readlines()
                    for (lineNum, lineText) in enumerate(fileText):
                        if lineNum in range(6, len(fileText) - 2):
                            aioTest.write(lineText)
        IndentedPrint("});", file=aioTest)
        IndentedPrint("/* eslint-disable max-len */", file=aioTest)

if __name__ == "__main__":
    spec, test, cts = ParseCmdLine()
    if os.path.exists(cts):
        os.remove(cts)
    ConvertNNAPITest(spec, test)
    # generate tests with fused operations
    ConvertNNAPITest(spec, test, True)
    if os.path.isdir(test):
        DumpAllInOneCtsTest(test, cts)