#!/usr/bin/python3

# Copyright 2018, The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import copy
import math
import os
import sys
import numpy as np

# Stuff from test generator
import test_generator as tg
from test_generator import ActivationConverter
from test_generator import BoolScalar
from test_generator import Configuration
from test_generator import DataTypeConverter
from test_generator import DataLayoutConverter
from test_generator import Example
from test_generator import Float16Scalar
from test_generator import Float32Scalar
from test_generator import Float32Vector
from test_generator import IgnoredOutput
from test_generator import Input
from test_generator import Int32Scalar
from test_generator import Int32Vector
from test_generator import Internal
from test_generator import Model
from test_generator import Operand
from test_generator import Output
from test_generator import Parameter
from test_generator import ParameterAsInputConverter
from test_generator import RelaxedModeConverter
from test_generator import SmartOpen
from test_generator import SymmPerChannelQuantParams
from test_generator import IndentedPrint
from test_generator import InitializeCtsTestFile

import mapping_dict as md

msgTemplate = 'Unsupported convert "%s" test due to "%s"'

def ParseCmdLine():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="the spec file/directory")
    parser.add_argument(
        "-t", "--test", help="the generated test file/directory", default="-")
    args = parser.parse_args()
    tg.FileNames.InitializeFileLists(args.spec, args.test)

def CheckOperationWithImplicitPadding(nnapiOp, curOpInsList, nnapiOpInsList,
                                      nnapiOpOptionalInsLen):
    # Current this convert tool doesn't involve padding compute function for
    # 'VALID', 'SAME' implicit padding, this would be an enhancement feature.
    flag = False
    curOpInsLen = len(curOpInsList)
    nnapiOpInsLen = len(nnapiOpInsList)

    if nnapiOp in ['AVERAGE_POOL_2D', 'MAX_POOL_2D']:
        if curOpInsLen < nnapiOpInsLen:
            msg = msgTemplate % (nnapiOp, "implicit padding")
            # print(msg, file=sys.stderr)
            flag = True
    elif nnapiOp != 'CONCATENATION':
        if nnapiOpOptionalInsLen != 0:
            if curOpInsLen == nnapiOpInsLen:
                if nnapiOp == 'CONV_2D':
                    # Layout param index of inputs with implicit padding is 7
                    if curOpInsList[7].type.type == 'BOOL':
                        msg = msgTemplate % (nnapiOp, "implicit padding")
                        # print(msg, file=sys.stderr)
                        flag = True
                elif nnapiOp == 'DEPTHWISE_CONV_2D':
                    # Layout param index of inputs with implicit padding is 8
                    if curOpInsList[8].type.type == 'BOOL':
                        msg = msgTemplate % (nnapiOp, "implicit padding")
                        # print(msg, file=sys.stderr)
                        flag = True
            elif curOpInsLen < nnapiOpInsLen:
                msg = msgTemplate % (nnapiOp, "implicit padding")
                # print(msg, file=sys.stderr)
                flag = True

    return flag

def ClearMappingWebNNOpConfiguration():
    if Configuration.successedCounter == 0:
        Configuration.mappingWebNNOp.clear()

def SupportedConvertDepthwiseConv2D(inOprand, outOprand, layout):
    # Unsupport to convert DEPTHWISE_CONV_2D test with depth_multiplier not
    # being equal to 1, that is, input_channels != output_channels
    flag = True
    inDimensions = inOprand.type.dimensions
    outDimensions = outOprand.type.dimensions

    if layout:
        flag = inDimensions[1] == outDimensions[1]
    else:
        flag = inDimensions[3] == outDimensions[3]

    return flag

def GetOperandIndex(opInfoList, opName):
    index = 0
    found = False

    for opInfo in opInfoList:
        if opInfo['name'] == opName:
            found = True
            break
        index += 1

    return index if found else -1

def GetParamOperandValue(paramsList, insList, index):
    status = False
    value = None

    if index != -1:
        if index < len(insList):
            inOp = insList[index]
            if inOp in paramsList:
                status = True
                value = paramsList[paramsList.index(inOp)].value

    return (status, value)

def GetInputOperandValue(inputsDict, insList, index):
    status = False
    value = None

    if index != -1:
        inOp = insList[index]
        if inOp in inputsDict.keys():
            status = True
            value = inputsDict[inOp]

    return (status, value)


def SupportedConvertSoftmax(opInfoList, inOperand, paramsList, insList):
    # Unsupport to convert SOFTMAX tests with rank of input is greater than 2,
    # or SOFTMAX tests used 2D input with input1 as scaling factor for the
    # exponent being not equal to 1.0 or optional inupt2 as axis being not equal
    # to 1 or -1.
    flag = True
    dimLen = len(inOperand.dimensions)

    if dimLen != 2:
        flag = False
        return flag

    betaIndex = GetOperandIndex(opInfoList, 'beta')
    status, beta = GetParamOperandValue(paramsList, insList, betaIndex)

    if int(beta[0]) != 1:
        flag = False
        return flag

    if len(insList) == 3:
        # using optional axis parameter
        axisIndex = GetOperandIndex(opInfoList, 'axis')
        status, axis = GetParamOperandValue(paramsList, insList, axisIndex)
        if axis[0] not in [-1, 1]:
            flag = False
            return flag

    return flag

def UpdateMappingWebNNOpList(actValue):
    if Configuration.successedCounter == 0:
        if actValue == 1:
            Configuration.mappingWebNNOp.append('relu')
        else:
            Configuration.mappingWebNNOp.append('clamp')

def GetReluMappedInfo(actValue):
    info = None

    if actValue == 1:
        info = {
            'name': 'relu'
        }
    else:
        info = {
            'name': 'clamp'
        }
        options = \
            "{minValue: builder.constant(%d), maxValue: builder.constant(%d)}"
        if actValue == 2:
            # relu1
            info['options'] = options % (-1, 1)
        elif actValue == 3:
            # relu6
            info['options'] = options % (0, 6)

    return info

def ReorderFilterValue(operation, dimensions, values, layout):
    valueArray = np.array(values).reshape(dimensions)
    axes = (0, 3, 1, 2) if layout else (1, 2, 3, 0)

    if operation == 'DEPTHWISE_CONV_2D':
        axes = (3, 0, 1, 2) if layout else (1, 2, 0, 3)

    return list(np.transpose(valueArray, axes).ravel())

def GetWebNNFilterDimensions(operation, dimensions, layout):
    # https://webmachinelearning.github.io/webnn/#api-modelbuilder-conv2d
    # filter tensor:
    #   "nchw": [output_channels, input_channels/groups, height, width]
    #   "nhwc": [height, width, input_channels/groups, output_channels]
    if operation == 'CONV_2D':
        if layout:
            outputChannels = dimensions[-1]
            dimensions = dimensions[:-1]
            dimensions.insert(1, outputChannels)
        else:
            outputChannels = dimensions[0]
            dimensions = dimensions[1:]
            dimensions.append(outputChannels)
    elif operation == 'DEPTHWISE_CONV_2D':
        if layout:
            outputChannels = dimensions[-1]
            dimensions = dimensions[:-1]
            dimensions.insert(0, outputChannels)
        else:
            intputChannels = dimensions[0]
            dimensions = dimensions[1:]
            dimensions.insert(2, intputChannels)

    return dimensions

def FlattenedTo2D(in0Dims, in1Dims):
    inputSize = in1Dims[1]
    batchSize = int(np.product(in0Dims) / inputSize)
    return [batchSize, inputSize]

def GetWebNNOperandDesc(oprand, operation, opInsList, opInsInfoList, layout):
    operandType = oprand.type.mappingType
    operandDims = oprand.type.dimensions

    if operation == 'FULLY_CONNECTED':
        if opInsList.index(oprand) == 0:
            # input0
            if len(operandDims) > 2:
                input1Dims = opInsList[1].dimensions
                operandDims = FlattenedTo2D(operandDims, input1Dims)
        elif opInsList.index(oprand) == 1:
            operandDims = [operandDims[1], operandDims[0]]
    else:
        oprandName = opInsInfoList[opInsList.index(oprand)]['name']

        if oprandName == 'filter':
            operandDims = GetWebNNFilterDimensions(operation, operandDims, layout)
        elif oprandName == 'bias':
            if layout and len(operandDims) == 1:
                # Update operandDims likes [x] -> [1, x, 1, 1]
                operandDims = [1, operandDims[0], 1, 1]

    operandDesc = "{type: '%s', dimensions: %s}" % (operandType, operandDims)
    return operandDesc

def PrintInputOperand(oprand, operation, opInsList, opInsInfoList, layout,
                      test):
    opDesc = GetWebNNOperandDesc(oprand, operation, opInsList, opInsInfoList,
                                 layout)
    operand = "const %s = builder.input('%s', %s);" % (oprand, oprand, opDesc)
    IndentedPrint(operand, indent=4, file=test)

def GetOperandValue(oprand, operation, opInsList, name, layout, value=None):
    opValue = oprand.value

    if value is not None:
        opValue = value

    if name == 'filter':
        opValue = ReorderFilterValue(operation, oprand.type.dimensions, opValue,
                                     layout)

    if operation == 'FULLY_CONNECTED':
        if opInsList.index(oprand) == 1:
            # input1
            arrayValue = np.array(opValue).reshape(oprand.type.dimensions)
            opValue = list(np.transpose(arrayValue, (1,0)).ravel())

    return opValue

def PrintInputBuffer(oprand, operation, opInsList, name, value, layout, test):
    typedArray = oprand.type.mappingTypedArrayType
    opValue = GetOperandValue(oprand, operation, opInsList, name, layout, value)
    IndentedPrint('const %sBuffer = new %s(%s);' % (oprand, typedArray, opValue),
                  indent=4, file=test)

def PrintConstant(oprand, operation, opInsList, opInsInfoList, name, layout,
                  test):
    opDesc = GetWebNNOperandDesc(oprand, operation, opInsList, opInsInfoList,
                                 layout)
    opValue = GetOperandValue(oprand, operation, opInsList, name, layout)
    operand = "const %s = builder.constant(%s, new %s(%s));" % \
              (oprand, opDesc, oprand.type.mappingTypedArrayType, opValue)
    IndentedPrint(operand, indent=4, file=test)

def CheckDefaultParameterValue(op, inputFeedDict, paramsList):
    flag = False
    value = None

    if op in paramsList:
        value = paramsList[paramsList.index(op)].value
    else:
        value = inputFeedDict[op]

    if isinstance(value, list) and (len(value) == 0 or value[0] is None):
        flag = True

    return flag

def GetWebNNOperationParamsList(opInsInfoList, opInsList, inputFeedDict,
                                paramsList, operation):
    d = {}
    length = len(opInsList)

    if operation != 'CONCATENATION':
        counter = 0
        for nnOpInsInfo in opInsInfoList:
            index = nnOpInsInfo['mappingParamIndex']
            if index != -1:
                paramName = opInsList[counter]
                if paramName is None:
                    break
                value = d.get(index, None)
                if value is None:
                    key = nnOpInsInfo.get('optionsDictKey', None)
                    if key is None:
                        value = paramName
                    else:
                        if CheckDefaultParameterValue(paramName,
                                                      inputFeedDict,
                                                      paramsList):
                            continue
                        value = {}
                        value[key] = paramName
                    d[index] = value
                else:
                    sequenceIndex = nnOpInsInfo.get('sequenceIndex', -1)
                    if sequenceIndex != -1:
                        itemValue = value.get(nnOpInsInfo['optionsDictKey'],
                                              None)
                        if itemValue is None:
                            itemValue = paramName
                        else:
                            if isinstance(itemValue, list):
                                itemValue.append(paramName)
                            else:
                                itemValue = [itemValue, paramName]
                        value[nnOpInsInfo['optionsDictKey']] = itemValue
                    else:
                        value[nnOpInsInfo['optionsDictKey']] = paramName
            counter += 1
            if counter == length:
                break
    else:
        # Refer to
        #   https://webmachinelearning.github.io/webnn/#api-modelbuilder-concat
        # WebNN API concat Op has sequence<Operand> inputs
        d[0] = []
        for ins in opInsList[:-1]:
            d[0].append(ins)
        d[1] = opInsList[-1]

    paramsList = sorted(d.items(), key = lambda item:item[0])
    lastParam = paramsList[-1][1]

    if isinstance(lastParam, dict):
        keyList = lastParam.keys()
        if 'padding' in keyList:
            leftRight = lastParam['padding'][:2]
            topBottom = lastParam['padding'][2:]
            lastParam['padding'] = topBottom + leftRight
        if 'strides' in keyList:
            lastParam['strides'].reverse()
        if 'windowDimensions' in keyList:
            lastParam['windowDimensions'].reverse()
        if 'dilations' in keyList:
            lastParam['dilations'].reverse()
    return paramsList

def UpdateWebNNOperationOptionalParamValue(targetValue, kvList):
    if isinstance(targetValue, dict):
        for key, value in kvList:
            if targetValue.get(key, None) is None:
                targetValue[key] = value
                if key == 'layout':
                    targetValue[key] = 'nchw' if value else 'nhwc'

def GetWebNNParamsString(params):
    paramsList = [p[1] for p in params]
    paramsStr = '%s' % paramsList
    return paramsStr[1:-1]

def PrintMappedReluOpertions(fusedReluMappedInfo, outputOp, operandName):
        mappedWebNNOpName = fusedReluMappedInfo['name']
        options = fusedReluMappedInfo.get('options', None)

        if options is None:
            IndentedPrint("const %s = builder.%s(%s);" % \
                          (outputOp, mappedWebNNOpName, operandName),
                          indent=4, file=test)
        else:
            IndentedPrint("const %s = builder.%s(%s, %s);" % \
                          (outputOp, mappedWebNNOpName, operandName, options),
                          indent=4, file=test)

def PrintOperations(biasOp, webnnOpType, webnnParamsStr, fusedReluMappedInfo,
                    outputOp, test):
    if biasOp is not None:
        IndentedPrint("const interOut0 = builder.%s(%s);" % \
                      (webnnOpType, webnnParamsStr),
                      indent=4, file=test)
        if fusedReluMappedInfo is not None:
            # Add 'add' operation
            IndentedPrint("const interOut1 = builder.add(interOut0, %s);" % \
                          biasOp, indent=4, file=test)
            # Add 'relu' or 'clamp' operation
            PrintMappedReluOpertions(fusedReluMappedInfo[1], outputOp,
                                     'interOut1')
        else:
            # Add 'add' operation
            IndentedPrint("const %s = builder.add(interOut0, %s);" % \
                          (outputOp, biasOp), indent=4, file=test)
    else:
        if fusedReluMappedInfo is not None:
            if fusedReluMappedInfo[0]:
                IndentedPrint("const interOut0 = builder.%s(%s);" % \
                              (webnnOpType, webnnParamsStr), indent=4,
                              file=test)
                # Add 'relu' or 'clamp' operation
                PrintMappedReluOpertions(fusedReluMappedInfo[1], outputOp,
                                         'interOut0')
            else:
                PrintMappedReluOpertions(fusedReluMappedInfo[1], outputOp,
                                         webnnParamsStr)
        else:
            IndentedPrint("const %s = builder.%s(%s);" % \
                          (outputOp, webnnOpType, webnnParamsStr),
                          indent=4, file=test)

# Dump Test file for Cts tests
def DumpCtsTest(example, test):
    model = example.model

    if len(model.operations) > 1:
        msg = 'Not convert complicated tests with multi-operations'
        # print(msg, file=sys.stderr)
        return

    nnapiOp = model.operations[0].optype

    if nnapiOp not in md.MappingDict.keys():
        msg = msgTemplate % (nnapiOp, 'none mapped WebNN Opeartion')
        # print(msg, file=sys.stderr)
        return

    # WebNN polyfill API cur supports 'int32' and 'float32'
    unSupportedTypesList = ['int8', 'uint8', 'float16']
    operandTypeList = model.GetMappedOperandTypes()
    usedUnsupportedType = list(set(unSupportedTypesList) & set(operandTypeList))

    if len(usedUnsupportedType) > 0:
        msg = msgTemplate % \
              (nnapiOp, 'unsupported %s Operand Type' % usedUnsupportedType)
        # print(msg, file=sys.stderr)
        return

    mappingOpDict = md.MappingDict[nnapiOp]
    mappedWebNNOp = mappingOpDict['webnnOperation']

    if Configuration.successedCounter == 0:
        # Update mappingWebNNOp by first time
        Configuration.mappingWebNNOp.append(mappedWebNNOp)

    nnapiOpInsList = copy.deepcopy(mappingOpDict['insList'])
    nnapiOpOptionalInsList = mappingOpDict.get('optionalInsList', [])
    curOpInsList = model.operations[0].ins

    if CheckOperationWithImplicitPadding(nnapiOp, curOpInsList,
                                         nnapiOpInsList,
                                         len(nnapiOpOptionalInsList)):
        ClearMappingWebNNOpConfiguration()
        return

    if GetOperandIndex(nnapiOpInsList, 'bias') != -1:
        if Configuration.successedCounter == 0:
            Configuration.mappingWebNNOp.append('add')

    curInputsList = example.model.GetInputs()
    curOutputsList = example.model.GetOutputs()
    curParamsList = example.model.GetParameters()

    fusedReluMappedInfo = None
    actIndex = GetOperandIndex(nnapiOpInsList, 'activation')
    actStatus, actValue = GetParamOperandValue(curParamsList, curOpInsList,
                                               actIndex)

    if actStatus:
        UpdateMappingWebNNOpList(actValue[0])
        fusedReluMappedInfo = (True, GetReluMappedInfo(actValue[0]))

    if nnapiOp == 'RELU1':
        fusedReluMappedInfo = (False, GetReluMappedInfo(2))

    if nnapiOp == 'RELU6':
        fusedReluMappedInfo = (False, GetReluMappedInfo(3))

    nnapiOpInsList.extend(nnapiOpOptionalInsList)
    layoutIndex = GetOperandIndex(nnapiOpInsList, 'layout')
    layoutStatus, layoutValue = GetParamOperandValue(curParamsList,
                                                     curOpInsList,
                                                     layoutIndex)
    # True: 'nchw', False: 'nhwc'
    layout = False if not layoutStatus else layoutValue[0]

    if nnapiOp == 'DEPTHWISE_CONV_2D':
        if not SupportedConvertDepthwiseConv2D(curInputsList[0],
                                               curOutputsList[0],
                                               layout):
            ClearMappingWebNNOpConfiguration()
            return

    biasOp = None
    testIndex = 1 if len(example.feedDicts)>1 else 0

    for inputFeedDict, outputFeedDict in example.feedDicts:
        if nnapiOp == 'SOFTMAX':
            if not SupportedConvertSoftmax(nnapiOpInsList, curInputsList[0],
                                           curParamsList, curOpInsList):
                ClearMappingWebNNOpConfiguration()
                return
        IndentedPrint("", file=test) # Add blank line
        testPurpose = 'test %s converted from %s test' % \
                      (' + '.join(Configuration.mappingWebNNOp),
                       str(example.testName))
        if testIndex > 0:
            testPurpose = "%s/%d" % (testPurpose, testIndex)
        IndentedPrint("it('%s', async function() {" % testPurpose,
                      indent=2, file=test)
        IndentedPrint("// Converted test case (from: %s/%s)" % \
                      (tg.FileNames.version,
                       os.path.basename(tg.FileNames.specFile)),
                      indent=4, file=test)
        IndentedPrint("const builder = nn.createModelBuilder();",
                      indent=4, file=test)
        computeParamsList = []
        # Create operand(s) by ModelBuilder.input
        for op in curInputsList:
            opInsDict = nnapiOpInsList[curOpInsList.index(op)]
            mappingParamIndex = opInsDict['mappingParamIndex']
            if mappingParamIndex != -1:
                rule = md.MappingRule(opInsDict['mappingRuleType'])
                if rule == md.MappingRule.OPERAND_OPERAND:
                    PrintInputOperand(op, nnapiOp, curOpInsList, nnapiOpInsList,
                                      layout, test)
                    PrintInputBuffer(op, nnapiOp, curOpInsList,
                                     opInsDict['name'], inputFeedDict[op],
                                     layout, test)
                    computeParamsList.append("'%s': {buffer: %sBuffer}" % \
                                             (op, op))
                elif rule == md.MappingRule.OPERAND_VARIABLE:
                    varValue = inputFeedDict[op]
                    if len(varValue) != 0 and varValue[0] is not None:
                        IndentedPrint('const %s = %s;' % (op, varValue),
                                      indent=4, file=test)
                elif rule == md.MappingRule.OPERAND_ARRAY:
                    varValue = inputFeedDict[op]
                    if len(varValue) != 0:
                        IndentedPrint('const %s = %s;' % (op, varValue),
                                      indent=4, file=test)
            else:
                if opInsDict['name'] == 'bias':
                    biasOp = op
                    PrintInputOperand(op, nnapiOp, curOpInsList, nnapiOpInsList,
                                      layout, test)
                    PrintInputBuffer(op, nnapiOp, curOpInsList,
                                     opInsDict['name'], inputFeedDict[op],
                                     layout, test)
                    computeParamsList.append("'%s': {buffer: %sBuffer}" % \
                                             (op, op))
        # Create operand(s) by ModelBuilder.constant, or define variable(s)
        for op in curParamsList:
            opInsDict = nnapiOpInsList[curOpInsList.index(op)]
            mappingParamIndex = opInsDict['mappingParamIndex']
            if mappingParamIndex != -1:
                rule = md.MappingRule(opInsDict['mappingRuleType'])
                if rule == md.MappingRule.OPERAND_OPERAND:
                    PrintConstant(op, nnapiOp, curOpInsList, nnapiOpInsList,
                                  opInsDict['name'], layout, test)
                elif rule == md.MappingRule.VARIABLE_VARIABLE:
                    varValue = curParamsList[curParamsList.index(op)].value[0]
                    if opInsDict['name'] == 'layout':
                        if varValue:
                            varValue = "'nchw'"
                        else:
                            varValue = "'nhwc'"
                    IndentedPrint('const %s = %s;' % (op, varValue), indent=4,
                                  file=test)
                elif rule == md.MappingRule.OPERAND_VARIABLE:
                    varValue = curParamsList[curParamsList.index(op)].value
                    if len(varValue) != 0 and varValue[0] is not None:
                        IndentedPrint('const %s = %s;' % (op, varValue),
                                      indent=4, file=test)
                elif rule == md.MappingRule.OPERAND_ARRAY:
                    varValue = curParamsList[curParamsList.index(op)].value
                    if len(varValue) != 0:
                        IndentedPrint('const %s = %s;' % (op, varValue),
                                      indent=4, file=test)
            else:
                if opInsDict['name'] == 'bias':
                    biasOp = op
                    PrintConstant(op, nnapiOp, curOpInsList, nnapiOpInsList,
                                  opInsDict['name'], layout, test)
        if len(curOutputsList) == 1:
            outputOp = curOutputsList[0]
            IndentedPrint("const expected = %s;" % outputFeedDict[outputOp],
                          indent=4, file=test)
        elif len(curOutputsList) > 1:
            outputOp = curOutputsList
            expectedValueList = [outputFeedDict[k] for k in outputOp]
            IndentedPrint("const expected = %s;" % expectedValueList, indent=4,
                          file=test)
        # Update optional parameter value
        optionsKeyValueList = []
        hasLayoutOption = False
        for optionalIns in nnapiOpOptionalInsList:
            if optionalIns['name'] == 'layout':
                hasLayoutOption = True
                break
        if hasLayoutOption:
            if not layout:
                # Default 'nchw' layout with WebNN API
                optionsKeyValueList.append(('layout', False))
        if nnapiOp == 'DEPTHWISE_CONV_2D':
            # True: 'nchw' False: 'nhwc'
            chanelIndex = 1 if layout else 3
            groups = outputOp.type.dimensions[chanelIndex]
            optionsKeyValueList.append(('groups', groups))
        mappingParams = GetWebNNOperationParamsList(nnapiOpInsList,
                                                    curOpInsList,
                                                    inputFeedDict,
                                                    curParamsList,
                                                    nnapiOp)
        UpdateWebNNOperationOptionalParamValue(mappingParams[-1][1],
                                               optionsKeyValueList)
        webnnParamsStr = GetWebNNParamsString(mappingParams)
        PrintOperations(biasOp, mappedWebNNOp, webnnParamsStr,
                        fusedReluMappedInfo, outputOp, test)
        if len(curOutputsList) == 1:
            IndentedPrint("const model = builder.createModel({%s});" % outputOp,
                          indent=4, file=test)
        elif len(curOutputsList) > 1:
            outputOpNameList = [item.name for item in outputOp]
            IndentedPrint("const model = builder.createModel({%s});" % \
                          ', '.join(outputOpNameList), indent=4, file=test)
        IndentedPrint("const compilation = await model.compile();", indent=4,
                      file=test)
        IndentedPrint("const outputs = await compilation.compute({%s});" % \
                      ', '.join(computeParamsList), indent=4, file=test)
        # Check compute output
        criteria = 'utils.ctsFp32RestrictAccuracyCriteria'
        if model.isRelaxed:
            criteria = 'utils.ctsFp32RelaxedAccuracyCriteria'
        if len(curOutputsList) == 1:
            IndentedPrint(
                "utils.checkValue(outputs.%s.buffer, expected, %s);" % \
                (outputOp, criteria), indent=4, file=test)
        elif len(curOutputsList) > 1:
            IndentedPrint('for (let i = 0; i < %d; i++) {' % \
                          len(curOutputsList), indent=4, file=test)
            bufferStr = 'outputs[%s[i]].buffer' % ['%s' % k for k in outputOp]
            IndentedPrint(
                "utils.checkValue(%s, expected[i], %s);" % \
                (bufferStr, criteria), indent=6, file=test)
            IndentedPrint("}", indent=4, file=test)
        IndentedPrint("});", indent=2, file=test)
        testIndex += 1

    Configuration.successedCounter += 1

if __name__ == '__main__':
    ParseCmdLine()
    while tg.FileNames.NextFile():
        Configuration.mappingWebNNOp = []
        Configuration.successedCounter = 0
        # print("Generating test(s) from spec: %s" % tg.FileNames.specFile,
        #       file=sys.stderr)
        exec(open(tg.FileNames.specFile, "r").read())
        testFile = tg.FileNames.testFile
        with SmartOpen(testFile) as test:
            InitializeCtsTestFile(test, 4)
            Example.DumpAllExamples(DumpTest=DumpCtsTest, test=test)
            IndentedPrint("});", file=test)
            IndentedPrint("/* eslint-disable max-len */", file=test)
        if Configuration.successedCounter == 0:
            os.remove(testFile)
        else:
            newName = 'test_%s_converted_from_%s' % \
                      ('_'.join(Configuration.mappingWebNNOp),
                       os.path.basename(testFile))
            renamedFile = os.path.join(os.path.dirname(testFile), newName)
            os.rename(testFile, renamedFile)
            # print("Successfully generated CTS test: %s" % renamedFile,
            #       file=sys.stderr)
