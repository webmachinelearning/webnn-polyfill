#!/usr/bin/python3

from enum import IntEnum

class MappingRule(IntEnum):
    OPERAND_OPERAND = 0
    VARIABLE_VARIABLE = 1
    OPERAND_VARIABLE = 2
    OPERAND_ARRAY = 3

# NN-API Operations mapping WebNN API Operations
MappingDict = {
    'ADD': {
        'webnnOperation': 'add',
        'insList': [
            {
                'name': 'input0',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'input1',
                'mappingParamIndex': 1,
                'mappingRuleType': 0
            },
            {
                'name': 'activation',
                'mappingParamIndex': -1
            }
        ]
    },
    'AVERAGE_POOL_2D': {
        'webnnOperation': 'averagePool2d',
        'insList': [ # only support for explicit paddings
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'paddingLeft',
                'mappingParamIndex': 1,
                'optionsDictKey': 'padding',
                'sequenceIndex': 2, # [beginning_height, ending_height, beginning_width, ending_width]
                'mappingRuleType': 1
            },
            {
                'name': 'paddingRight',
                'mappingParamIndex': 1,
                'optionsDictKey': 'padding',
                'sequenceIndex': 3,
                'mappingRuleType': 1
            },
            {
                'name': 'paddingTop',
                'mappingParamIndex': 1,
                'optionsDictKey': 'padding',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'paddingBottom',
                'mappingParamIndex': 1,
                'optionsDictKey': 'padding',
                'sequenceIndex': 1,
                'mappingRuleType': 1
            },
            {
                'name': 'strideWidth',
                'mappingParamIndex': 1,
                'optionsDictKey': 'strides',
                'sequenceIndex': 1, # [stride_height, stride_width]
                'mappingRuleType': 1
            },
            {
                'name': 'strideHeight',
                'mappingParamIndex': 1,
                'optionsDictKey': 'strides',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'filterWidth',
                'mappingParamIndex': 1,
                'optionsDictKey': 'windowDimensions',
                'sequenceIndex': 1, # [window_height, window_width]
                'mappingRuleType': 1
            },
            {
                'name': 'filterHeight',
                'mappingParamIndex': 1,
                'optionsDictKey': 'windowDimensions',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'activation',
                'mappingParamIndex': -1
            }
        ],
        'optionalInsList': [
            {
                'name': 'layout',
                'mappingParamIndex': 1,
                'optionsDictKey': 'layout',
                'mappingRuleType': 1
            }
        ]
    },
    'CONCATENATION': {
        'webnnOperation': 'concat',
        'insList': [
            {
                'name': 'input0',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'input1',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'axis',
                'mappingParamIndex': 1,
                'mappingRuleType': 1
            }
        ]
    },
    'CONV_2D': {
        'webnnOperation': 'conv2d',
        'insList': [ # only support for explicit paddings
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'filter',
                'mappingParamIndex': 1,
                'mappingRuleType': 0
            },
            {
                'name': 'bias',
                'mappingParamIndex': -1,
                'mappingRuleType': 0
            },
            {
                'name': 'paddingLeft',
                'mappingParamIndex': 2,
                'optionsDictKey': 'padding',
                'sequenceIndex': 2, # [beginning_height, ending_height, beginning_width, ending_width]
                'mappingRuleType': 1
            },
            {
                'name': 'paddingRight',
                'mappingParamIndex': 2,
                'optionsDictKey': 'padding',
                'sequenceIndex': 3,
                'mappingRuleType': 1
            },
            {
                'name': 'paddingTop',
                'mappingParamIndex': 2,
                'optionsDictKey': 'padding',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'paddingBottom',
                'mappingParamIndex': 2,
                'optionsDictKey': 'padding',
                'sequenceIndex': 1,
                'mappingRuleType': 1
            },
            {
                'name': 'strideWidth',
                'mappingParamIndex': 2,
                'optionsDictKey': 'strides',
                'sequenceIndex': 1, # [stride_height, stride_width]
                'mappingRuleType': 1
            },
            {
                'name': 'strideHeight',
                'mappingParamIndex': 2,
                'optionsDictKey': 'strides',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'activation',
                'mappingParamIndex': -1
            }
        ],
        'optionalInsList': [
            {
                'name': 'layout',
                'mappingParamIndex': 2,
                'optionsDictKey': 'layout',
                'mappingRuleType': 1
            },
            {
                'name': 'dilationWidth',
                'mappingParamIndex': 2,
                'optionsDictKey': 'dilations',
                'sequenceIndex': 1, # [dilation_height, dilation_width]
                'mappingRuleType': 1
            },
            {
                'name': 'dilationHeight',
                'mappingParamIndex': 2,
                'optionsDictKey': 'dilations',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            }
        ]
    },
    'DEPTHWISE_CONV_2D': {
        'webnnOperation': 'conv2d',
        'insList': [ # only support for explicit paddings
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'filter',
                'mappingParamIndex': 1,
                'mappingRuleType': 0
            },
            {
                'name': 'bias',
                'mappingParamIndex': -1,
                'mappingRuleType': 0
            },
            {
                'name': 'paddingLeft',
                'mappingParamIndex': 2,
                'optionsDictKey': 'padding',
                'sequenceIndex': 2, # [beginning_height, ending_height, beginning_width, ending_width]
                'mappingRuleType': 1
            },
            {
                'name': 'paddingRight',
                'mappingParamIndex': 2,
                'optionsDictKey': 'padding',
                'sequenceIndex': 3,
                'mappingRuleType': 1
            },
            {
                'name': 'paddingTop',
                'mappingParamIndex': 2,
                'optionsDictKey': 'padding',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'paddingBottom',
                'mappingParamIndex': 2,
                'optionsDictKey': 'padding',
                'sequenceIndex': 1,
                'mappingRuleType': 1
            },
            {
                'name': 'strideWidth',
                'mappingParamIndex': 2,
                'optionsDictKey': 'strides',
                'sequenceIndex': 1, # [stride_height, stride_width]
                'mappingRuleType': 1
            },
            {
                'name': 'strideHeight',
                'mappingParamIndex': 2,
                'optionsDictKey': 'strides',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'multiplier',
                'mappingParamIndex': -1
            },
            {
                'name': 'activation',
                'mappingParamIndex': -1
            }
        ],
        'optionalInsList': [
            {
                'name': 'layout',
                'mappingParamIndex': 2,
                'optionsDictKey': 'layout',
                'mappingRuleType': 1
            },
            {
                'name': 'dilationWidth',
                'mappingParamIndex': 2,
                'optionsDictKey': 'dilations',
                'sequenceIndex': 1, # [dilation_height, dilation_width]
                'mappingRuleType': 1
            },
            {
                'name': 'dilationHeight',
                'mappingParamIndex': 2,
                'optionsDictKey': 'dilations',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            }
        ]
    },
    'DIV': {
        'webnnOperation': 'div',
        'insList': [
            {
                'name': 'input0',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'input1',
                'mappingParamIndex': 1,
                'mappingRuleType': 0
            },
            {
                'name': 'activation',
                'mappingParamIndex': -1
            }
        ]
    },
    'EXP': {
        'webnnOperation': 'exp',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            }
        ]
    },
    'FULLY_CONNECTED': {
        'webnnOperation': 'matmul',
        'insList': [
            {
                'name': 'input0',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'input1',
                'mappingParamIndex': 1,
                'mappingRuleType': 0
            },
            {
                'name': 'bias',
                'mappingParamIndex': -1,
                'mappingRuleType': 0
            },
            {
                'name': 'activation',
                'mappingParamIndex': -1
            }
        ]
    },
    'LOGISTIC': {
        'webnnOperation': 'sigmoid',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            }
        ]
    },
    'MAX_POOL_2D': {
        'webnnOperation': 'maxPool2d',
        'insList': [ # only support for explicit paddings
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'paddingLeft',
                'mappingParamIndex': 1,
                'optionsDictKey': 'padding',
                'sequenceIndex': 2, # [beginning_height, ending_height, beginning_width, ending_width]
                'mappingRuleType': 1
            },
            {
                'name': 'paddingRight',
                'mappingParamIndex': 1,
                'optionsDictKey': 'padding',
                'sequenceIndex': 3,
                'mappingRuleType': 1
            },
            {
                'name': 'paddingTop',
                'mappingParamIndex': 1,
                'optionsDictKey': 'padding',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'paddingBottom',
                'mappingParamIndex': 1,
                'optionsDictKey': 'padding',
                'sequenceIndex': 1,
                'mappingRuleType': 1
            },
            {
                'name': 'strideWidth',
                'mappingParamIndex': 1,
                'optionsDictKey': 'strides',
                'sequenceIndex': 1, # [stride_height, stride_width]
                'mappingRuleType': 1
            },
            {
                'name': 'strideHeight',
                'mappingParamIndex': 1,
                'optionsDictKey': 'strides',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'filterWidth',
                'mappingParamIndex': 1,
                'optionsDictKey': 'windowDimensions',
                'sequenceIndex': 1, # [window_height, window_width]
                'mappingRuleType': 1
            },
            {
                'name': 'filterHeight',
                'mappingParamIndex': 1,
                'optionsDictKey': 'windowDimensions',
                'sequenceIndex': 0,
                'mappingRuleType': 1
            },
            {
                'name': 'activation',
                'mappingParamIndex': -1
            }
        ],
        'optionalInsList': [
            {
                'name': 'layout',
                'mappingParamIndex': 1,
                'optionsDictKey': 'layout',
                'mappingRuleType': 1
            }
        ]
    },
    'MAXIMUM': {
        'webnnOperation': 'max',
        'insList': [
            {
                'name': 'input0',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'input1',
                'mappingParamIndex': 1,
                'mappingRuleType': 0
            }
        ]
    },
    'MINIMUM': {
        'webnnOperation': 'min',
        'insList': [
            {
                'name': 'input0',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'input1',
                'mappingParamIndex': 1,
                'mappingRuleType': 0
            }
        ]
    },
    'MUL': {
        'webnnOperation': 'mul',
        'insList': [
            {
                'name': 'input0',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'input1',
                'mappingParamIndex': 1,
                'mappingRuleType': 0
            },
            {
                'name': 'activation',
                'mappingParamIndex': -1
            }
        ]
    },
    'RELU': {
        'webnnOperation': 'relu',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            }
        ]
    },
    'RELU1': {
        'webnnOperation': 'clamp',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            }
        ]
    },
    'RELU6': {
        'webnnOperation': 'clamp',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            }
        ]
    },
    'RESHAPE': {
        'webnnOperation': 'reshape',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'shape',
                'mappingParamIndex': 1,
                'mappingRuleType': 2
            }
        ]
    },
    'SLICE': {
        'webnnOperation': 'slice',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'starts',
                'mappingParamIndex': 1,
                'mappingRuleType': 3
            },
            {
                'name': 'sizes',
                'mappingParamIndex': 2,
                'mappingRuleType': 3
            }
        ]
    },
    'SOFTMAX': {
        'webnnOperation': 'softmax',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'beta',
                'mappingParamIndex': -1,
                'mappingRuleType': 1
            },
            {
                'name': 'axis',
                'mappingParamIndex': -1,
                'mappingRuleType': 1
            }
        ]
    },
    'SPLIT': {
        'webnnOperation': 'split',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'axis',
                'mappingParamIndex': 2,
                'optionsDictKey': 'axis',
                'mappingRuleType': 1
            },
            {
                'name': 'splitsNumber',
                'mappingParamIndex': 1,
                'mappingRuleType': 1
            }
        ]
    },
    'SQRT': {
        'webnnOperation': 'pow',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            }
        ]
    },
    'SQUEEZE': {
        'webnnOperation': 'squeeze',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            }
        ],
        'optionalInsList': [
            {
                'name': 'axes',
                'mappingParamIndex': 1,
                'optionsDictKey': 'axes',
                'mappingRuleType': 2
            }
        ]
    },
    'SUB': {
        'webnnOperation': 'sub',
        'insList': [
            {
                'name': 'input0',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            },
            {
                'name': 'input1',
                'mappingParamIndex': 1,
                'mappingRuleType': 0
            },
            {
                'name': 'activation',
                'mappingParamIndex': -1
            }
        ]
    },
    'TANH': {
        'webnnOperation': 'tanh',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            }
        ]
    },
    'TRANSPOSE': {
        'webnnOperation': 'transpose',
        'insList': [
            {
                'name': 'input',
                'mappingParamIndex': 0,
                'mappingRuleType': 0
            }
        ],
        'optionalInsList': [
            {
                'name': 'permutation',
                'mappingParamIndex': 1,
                'optionsDictKey': 'permutation',
                'mappingRuleType': 2
            }
        ]
    }
}