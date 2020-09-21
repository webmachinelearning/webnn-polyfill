!function(t){var e={};function n(r){if(e[r])return e[r].exports;var o=e[r]={i:r,l:!1,exports:{}};return t[r].call(o.exports,o,o.exports,n),o.l=!0,o.exports}n.m=t,n.c=e,n.d=function(t,e,r){n.o(t,e)||Object.defineProperty(t,e,{enumerable:!0,get:r})},n.r=function(t){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},n.t=function(t,e){if(1&e&&(t=n(t)),8&e)return t;if(4&e&&"object"==typeof t&&t&&t.__esModule)return t;var r=Object.create(null);if(n.r(r),Object.defineProperty(r,"default",{enumerable:!0,value:t}),2&e&&"string"!=typeof t)for(var o in t)n.d(r,o,function(e){return t[e]}.bind(null,o));return r},n.n=function(t){var e=t&&t.__esModule?function(){return t.default}:function(){return t};return n.d(e,"a",e),e},n.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},n.p="",n(n.s=92)}([function(t,e,n){"use strict";n.r(e),n.d(e,"shuffle",(function(){return o})),n.d(e,"clamp",(function(){return s})),n.d(e,"nearestLargerEven",(function(){return a})),n.d(e,"sum",(function(){return i})),n.d(e,"randUniform",(function(){return u})),n.d(e,"distSquared",(function(){return c})),n.d(e,"assert",(function(){return l})),n.d(e,"assertShapesMatch",(function(){return h})),n.d(e,"assertNonNull",(function(){return d})),n.d(e,"flatten",(function(){return p})),n.d(e,"sizeFromShape",(function(){return f})),n.d(e,"isScalarShape",(function(){return g})),n.d(e,"arraysEqual",(function(){return m})),n.d(e,"isInt",(function(){return b})),n.d(e,"tanh",(function(){return x})),n.d(e,"sizeToSquarishShape",(function(){return y})),n.d(e,"createShuffledIndices",(function(){return v})),n.d(e,"rightPad",(function(){return w})),n.d(e,"repeatedTry",(function(){return C})),n.d(e,"inferFromImplicitShape",(function(){return $})),n.d(e,"parseAxisParam",(function(){return O})),n.d(e,"squeezeShape",(function(){return I})),n.d(e,"getTypedArrayFromDType",(function(){return S})),n.d(e,"getArrayFromDType",(function(){return E})),n.d(e,"checkConversionForErrors",(function(){return R})),n.d(e,"isValidDtype",(function(){return A})),n.d(e,"hasEncodingLoss",(function(){return k})),n.d(e,"isTypedArray",(function(){return T})),n.d(e,"bytesPerElement",(function(){return F})),n.d(e,"bytesFromStringArray",(function(){return N})),n.d(e,"isString",(function(){return D})),n.d(e,"isBoolean",(function(){return _})),n.d(e,"isNumber",(function(){return B})),n.d(e,"inferDtype",(function(){return j})),n.d(e,"isFunction",(function(){return M})),n.d(e,"nearestDivisor",(function(){return P})),n.d(e,"computeStrides",(function(){return L})),n.d(e,"toTypedArray",(function(){return W})),n.d(e,"toNestedArray",(function(){return z})),n.d(e,"makeOnesTypedArray",(function(){return U})),n.d(e,"makeZerosTypedArray",(function(){return V})),n.d(e,"makeZerosNestedTypedArray",(function(){return G})),n.d(e,"now",(function(){return H})),n.d(e,"assertNonNegativeIntegerDimensions",(function(){return K})),n.d(e,"fetch",(function(){return q})),n.d(e,"encodeString",(function(){return X})),n.d(e,"decodeString",(function(){return Y})),n.d(e,"locToIndex",(function(){return Q})),n.d(e,"indexToLoc",(function(){return Z}));var r=n(6);
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function o(t){let e=t.length,n=0,r=0;for(;e>0;)r=Math.random()*e|0,e--,n=t[e],t[e]=t[r],t[r]=n}function s(t,e,n){return Math.max(t,Math.min(e,n))}function a(t){return t%2==0?t:t+1}function i(t){let e=0;for(let n=0;n<t.length;n++)e+=t[n];return e}function u(t,e){const n=Math.random();return e*n+(1-n)*t}function c(t,e){let n=0;for(let r=0;r<t.length;r++){const o=Number(t[r])-Number(e[r]);n+=o*o}return n}function l(t,e){if(!t)throw new Error("string"==typeof e?e:e())}function h(t,e,n=""){l(m(t,e),()=>n+` Shapes ${t} and ${e} must match`)}function d(t){l(null!=t,()=>"The input to the tensor constructor must be a non-null value.")}function p(t,e=[],n=!1){if(null==e&&(e=[]),Array.isArray(t)||T(t)&&!n)for(let r=0;r<t.length;++r)p(t[r],e,n);else e.push(t);return e}function f(t){if(0===t.length)return 1;let e=t[0];for(let n=1;n<t.length;n++)e*=t[n];return e}function g(t){return 0===t.length}function m(t,e){if(t===e)return!0;if(null==t||null==e)return!1;if(t.length!==e.length)return!1;for(let n=0;n<t.length;n++)if(t[n]!==e[n])return!1;return!0}function b(t){return t%1==0}function x(t){if(null!=Math.tanh)return Math.tanh(t);if(t===1/0)return 1;if(t===-1/0)return-1;{const e=Math.exp(2*t);return(e-1)/(e+1)}}function y(t){const e=Math.ceil(Math.sqrt(t));return[e,Math.ceil(t/e)]}function v(t){const e=new Uint32Array(t);for(let n=0;n<t;++n)e[n]=n;return o(e),e}function w(t,e){return e<=t.length?t:t+" ".repeat(e-t.length)}function C(t,e=(t=>0),n){return new Promise((r,o)=>{let s=0;const a=()=>{if(t())return void r();s++;const i=e(s);null!=n&&s>=n?o():setTimeout(a,i)};a()})}function $(t,e){let n=1,r=-1;for(let e=0;e<t.length;++e)if(t[e]>=0)n*=t[e];else if(-1===t[e]){if(-1!==r)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${e}`);r=e}else if(t[e]<0)throw Error(`Shapes can not be < 0. Found ${t[e]} at dim ${e}`);if(-1===r){if(e>0&&e!==n)throw Error(`Size(${e}) must match the product of shape ${t}`);return t}if(0===n)throw Error(`Cannot infer the missing size in [${t}] when there are 0 elements`);if(e%n!=0)throw Error(`The implicit shape can't be a fractional number. Got ${e} / ${n}`);const o=t.slice();return o[r]=e/n,o}function O(t,e){const n=e.length;return l((t=null==t?e.map((t,e)=>e):[].concat(t)).every(t=>t>=-n&&t<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis `+t),l(t.every(t=>b(t)),()=>"All values in axis param must be integers but got axis "+t),t.map(t=>t<0?n+t:t)}function I(t,e){const n=[],r=[],o=null!=e&&Array.isArray(e)&&0===e.length,s=null==e||o?null:O(e,t).sort();let a=0;for(let e=0;e<t.length;++e){if(null!=s){if(s[a]===e&&1!==t[e])throw new Error(`Can't squeeze axis ${e} since its dim '${t[e]}' is not 1`);(null==s[a]||s[a]>e)&&1===t[e]&&(n.push(t[e]),r.push(e)),s[a]<=e&&a++}1!==t[e]&&(n.push(t[e]),r.push(e))}return{newShape:n,keptDims:r}}function S(t,e){let n=null;if(null==t||"float32"===t)n=new Float32Array(e);else if("int32"===t)n=new Int32Array(e);else{if("bool"!==t)throw new Error("Unknown data type "+t);n=new Uint8Array(e)}return n}function E(t,e){let n=null;if(null==t||"float32"===t)n=new Float32Array(e);else if("int32"===t)n=new Int32Array(e);else if("bool"===t)n=new Uint8Array(e);else{if("string"!==t)throw new Error("Unknown data type "+t);n=new Array(e)}return n}function R(t,e){for(let n=0;n<t.length;n++){const r=t[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${e} being uploaded contains ${r}.`)}}function A(t){return"bool"===t||"complex64"===t||"float32"===t||"int32"===t||"string"===t}function k(t,e){return"complex64"!==e&&(("float32"!==e||"complex64"===t)&&(("int32"!==e||"float32"===t||"complex64"===t)&&("bool"!==e||"bool"!==t)))}function T(t){return t instanceof Float32Array||t instanceof Int32Array||t instanceof Uint8Array}function F(t){if("float32"===t||"int32"===t)return 4;if("complex64"===t)return 8;if("bool"===t)return 1;throw new Error("Unknown dtype "+t)}function N(t){if(null==t)return 0;let e=0;return t.forEach(t=>e+=t.length),e}function D(t){return"string"==typeof t||t instanceof String}function _(t){return"boolean"==typeof t}function B(t){return"number"==typeof t}function j(t){return Array.isArray(t)?j(t[0]):t instanceof Float32Array?"float32":t instanceof Int32Array||t instanceof Uint8Array?"int32":B(t)?"float32":D(t)?"string":_(t)?"bool":"float32"}function M(t){return!!(t&&t.constructor&&t.call&&t.apply)}function P(t,e){for(let n=e;n<t;++n)if(t%n==0)return n;return t}function L(t){const e=t.length;if(e<2)return[];const n=new Array(e-1);n[e-2]=t[e-1];for(let r=e-3;r>=0;--r)n[r]=n[r+1]*t[r+1];return n}function W(t,e){if("string"===e)throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(t)&&(t=p(t)),Object(r.b)().getBool("DEBUG")&&R(t,e),function(t,e){return t instanceof Float32Array&&"float32"===e||t instanceof Int32Array&&"int32"===e||t instanceof Uint8Array&&"bool"===e}(t,e))return t;if(null==e||"float32"===e||"complex64"===e)return new Float32Array(t);if("int32"===e)return new Int32Array(t);if("bool"===e){const e=new Uint8Array(t.length);for(let n=0;n<e.length;++n)0!==Math.round(t[n])&&(e[n]=1);return e}throw new Error("Unknown data type "+e)}function z(t,e){if(0===t.length)return e[0];const n=t.reduce((t,e)=>t*e);if(0===n)return[];if(n!==e.length)throw new Error(`[${t}] does not match the input size ${e.length}.`);return function t(e,n,r){const o=new Array;if(1===n.length){const t=n[0];for(let n=0;n<t;n++)o[n]=r[e+n]}else{const s=n[0],a=n.slice(1),i=a.reduce((t,e)=>t*e);for(let n=0;n<s;n++)o[n]=t(e+n*i,a,r)}return o}(0,t,e)}function U(t,e){const n=V(t,e);for(let t=0;t<n.length;t++)n[t]=1;return n}function V(t,e){if(null==e||"float32"===e||"complex64"===e)return new Float32Array(t);if("int32"===e)return new Int32Array(t);if("bool"===e)return new Uint8Array(t);throw new Error("Unknown data type "+e)}function G(t,e){const n=t.reduce((t,e)=>t*e,1);if(null==e||"float32"===e)return z(t,new Float32Array(n));if("int32"===e)return z(t,new Int32Array(n));if("bool"===e)return z(t,new Uint8Array(n));throw new Error("Unknown data type "+e)}function H(){return Object(r.b)().platform.now()}function K(t){t.forEach(e=>{l(Number.isInteger(e)&&e>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${t}].`)})}function q(t,e){return Object(r.b)().platform.fetch(t,e)}function X(t,e="utf-8"){return e=e||"utf-8",Object(r.b)().platform.encode(t,e)}function Y(t,e="utf-8"){return e=e||"utf-8",Object(r.b)().platform.decode(t,e)}function Q(t,e,n){if(0===e)return 0;if(1===e)return t[0];let r=t[t.length-1];for(let e=0;e<t.length-1;++e)r+=n[e]*t[e];return r}function Z(t,e,n){if(0===e)return[];if(1===e)return[t];const r=new Array(e);for(let e=0;e<r.length-1;++e)r[e]=Math.floor(t/n[e]),t-=r[e]*n[e];return r[r.length-1]=t,r}},function(t,e,n){"use strict";n.d(e,"a",(function(){return r})),n.d(e,"b",(function(){return o})),n.d(e,"c",(function(){return s})),n.d(e,"d",(function(){return a})),n.d(e,"e",(function(){return i})),n.d(e,"f",(function(){return u})),n.d(e,"g",(function(){return c})),n.d(e,"h",(function(){return l})),n.d(e,"i",(function(){return h})),n.d(e,"j",(function(){return d})),n.d(e,"k",(function(){return p})),n.d(e,"l",(function(){return f})),n.d(e,"n",(function(){return g})),n.d(e,"m",(function(){return m})),n.d(e,"o",(function(){return b})),n.d(e,"r",(function(){return x})),n.d(e,"p",(function(){return y})),n.d(e,"q",(function(){return v})),n.d(e,"s",(function(){return w})),n.d(e,"t",(function(){return C})),n.d(e,"u",(function(){return $})),n.d(e,"v",(function(){return O})),n.d(e,"w",(function(){return I})),n.d(e,"x",(function(){return S})),n.d(e,"y",(function(){return E})),n.d(e,"z",(function(){return R})),n.d(e,"A",(function(){return A})),n.d(e,"B",(function(){return k})),n.d(e,"C",(function(){return T})),n.d(e,"D",(function(){return F})),n.d(e,"E",(function(){return N})),n.d(e,"F",(function(){return D})),n.d(e,"G",(function(){return _})),n.d(e,"H",(function(){return B})),n.d(e,"J",(function(){return j})),n.d(e,"I",(function(){return M})),n.d(e,"K",(function(){return P})),n.d(e,"L",(function(){return L})),n.d(e,"M",(function(){return W})),n.d(e,"N",(function(){return z})),n.d(e,"O",(function(){return U})),n.d(e,"Q",(function(){return V})),n.d(e,"P",(function(){return G})),n.d(e,"R",(function(){return H})),n.d(e,"S",(function(){return K})),n.d(e,"T",(function(){return q})),n.d(e,"V",(function(){return X})),n.d(e,"U",(function(){return Y})),n.d(e,"W",(function(){return Q})),n.d(e,"X",(function(){return Z})),n.d(e,"Y",(function(){return J})),n.d(e,"Z",(function(){return tt})),n.d(e,"ab",(function(){return et})),n.d(e,"bb",(function(){return nt})),n.d(e,"cb",(function(){return rt})),n.d(e,"eb",(function(){return ot})),n.d(e,"fb",(function(){return st})),n.d(e,"gb",(function(){return at})),n.d(e,"hb",(function(){return it})),n.d(e,"jb",(function(){return ut})),n.d(e,"ib",(function(){return ct})),n.d(e,"kb",(function(){return lt})),n.d(e,"lb",(function(){return ht})),n.d(e,"mb",(function(){return dt})),n.d(e,"nb",(function(){return pt})),n.d(e,"qb",(function(){return ft})),n.d(e,"rb",(function(){return gt})),n.d(e,"sb",(function(){return mt})),n.d(e,"tb",(function(){return bt})),n.d(e,"vb",(function(){return xt})),n.d(e,"wb",(function(){return yt})),n.d(e,"xb",(function(){return vt})),n.d(e,"ub",(function(){return wt})),n.d(e,"ob",(function(){return Ct})),n.d(e,"pb",(function(){return $t})),n.d(e,"yb",(function(){return Ot})),n.d(e,"Eb",(function(){return It})),n.d(e,"zb",(function(){return St})),n.d(e,"Cb",(function(){return Et})),n.d(e,"Ab",(function(){return Rt})),n.d(e,"Bb",(function(){return At})),n.d(e,"Db",(function(){return kt})),n.d(e,"Fb",(function(){return Tt})),n.d(e,"Gb",(function(){return Ft})),n.d(e,"Hb",(function(){return Nt})),n.d(e,"Ib",(function(){return Dt})),n.d(e,"Jb",(function(){return _t})),n.d(e,"Nb",(function(){return Bt})),n.d(e,"Kb",(function(){return jt})),n.d(e,"Lb",(function(){return Mt})),n.d(e,"Mb",(function(){return Pt})),n.d(e,"Pb",(function(){return Lt})),n.d(e,"Ob",(function(){return Wt})),n.d(e,"Qb",(function(){return zt})),n.d(e,"Rb",(function(){return Ut})),n.d(e,"Sb",(function(){return Vt})),n.d(e,"Tb",(function(){return Gt})),n.d(e,"Ub",(function(){return Ht})),n.d(e,"Vb",(function(){return Kt})),n.d(e,"Wb",(function(){return qt})),n.d(e,"Xb",(function(){return Xt})),n.d(e,"Zb",(function(){return Yt})),n.d(e,"cc",(function(){return Qt})),n.d(e,"dc",(function(){return Zt})),n.d(e,"ac",(function(){return Jt})),n.d(e,"bc",(function(){return te})),n.d(e,"Yb",(function(){return ee})),n.d(e,"ec",(function(){return ne})),n.d(e,"gc",(function(){return re})),n.d(e,"hc",(function(){return oe})),n.d(e,"ic",(function(){return se})),n.d(e,"jc",(function(){return ae})),n.d(e,"oc",(function(){return ie})),n.d(e,"mc",(function(){return ue})),n.d(e,"nc",(function(){return ce})),n.d(e,"lc",(function(){return le})),n.d(e,"kc",(function(){return he})),n.d(e,"qc",(function(){return de})),n.d(e,"tc",(function(){return pe})),n.d(e,"zc",(function(){return fe})),n.d(e,"rc",(function(){return ge})),n.d(e,"sc",(function(){return me})),n.d(e,"pc",(function(){return be})),n.d(e,"vc",(function(){return xe})),n.d(e,"uc",(function(){return ye})),n.d(e,"yc",(function(){return ve})),n.d(e,"xc",(function(){return we})),n.d(e,"Ac",(function(){return Ce})),n.d(e,"Bc",(function(){return $e})),n.d(e,"Cc",(function(){return Oe})),n.d(e,"Dc",(function(){return Ie})),n.d(e,"Ec",(function(){return Se})),n.d(e,"Fc",(function(){return Ee})),n.d(e,"Gc",(function(){return Re})),n.d(e,"Hc",(function(){return Ae})),n.d(e,"wc",(function(){return ke})),n.d(e,"db",(function(){return Te})),n.d(e,"fc",(function(){return Fe}));const r="Abs",o="Acos",s="Acosh",a="Add",i="AddN",u="All",c="Any",l="ArgMax",h="ArgMin",d="Asin",p="Asinh",f="Atan",g="Atanh",m="Atan2",b="AvgPool",x="AvgPoolBackprop",y="AvgPool3D",v="AvgPool3DBackprop",w="BatchMatMul",C="BatchToSpaceND",$="BroadcastTo",O="Cast",I="Ceil",S="ClipByValue",E="Complex",R="Concat",A="Conv2D",k="Conv2DBackpropFilter",T="Conv2DBackpropInput",F="Conv3D",N="Conv3DBackpropFilterV2",D="Conv3DBackpropInputV2",_="Cos",B="Cosh",j="Cumsum",M="CropAndResize",P="DepthToSpace",L="DepthwiseConv2dNative",W="DepthwiseConv2dNativeBackpropFilter",z="DepthwiseConv2dNativeBackpropInput",U="Dilation2D",V="Dilation2DBackpropInput",G="Dilation2DBackpropFilter",H="Div",K="Elu",q="EluGrad",X="Erf",Y="Equal",Q="Exp",Z="Expm1",J="FFT",tt="Fill",et="FlipLeftRight",nt="Floor",rt="FloorDiv",ot="FusedBatchNorm",st="GatherV2",at="Greater",it="GreaterEqual",ut="Identity",ct="IFFT",lt="Imag",ht="IsFinite",dt="IsInf",pt="IsNan",ft="Less",gt="LessEqual",mt="Log",bt="Log1p",xt="LogicalAnd",yt="LogicalNot",vt="LogicalOr",wt="LogSoftmax",Ct="LRN",$t="LRNBackprop",Ot="Max",It="Maximum",St="MaxPool",Et="MaxPoolBackprop",Rt="MaxPool3D",At="MaxPool3DBackprop",kt="MaxPoolWithArgmax",Tt="Min",Ft="Minimum",Nt="Mod",Dt="Multiply",_t="Negate",Bt="NotEqual",jt="NonMaxSuppressionV3",Mt="NonMaxSuppressionV4",Pt="NonMaxSuppressionV5",Lt="OnesLike",Wt="OneHot",zt="PadV2",Ut="Pow",Vt="Prelu",Gt="Prod",Ht="Range",Kt="Real",qt="Reciprocal",Xt="Relu",Yt="Reshape",Qt="ResizeNearestNeighbor",Zt="ResizeNearestNeighborGrad",Jt="ResizeBilinear",te="ResizeBilinearGrad",ee="Relu6",ne="Reverse",re="Round",oe="Rsqrt",se="SelectV2",ae="Selu",ie="Slice",ue="Sin",ce="Sinh",le="Sign",he="Sigmoid",de="Softplus",pe="Sqrt",fe="Sum",ge="SpaceToBatchND",me="SplitV",be="Softmax",xe="SquaredDifference",ye="Square",ve="Sub",we="StridedSlice",Ce="Tan",$e="Tanh",Oe="Tile",Ie="TopK",Se="Transpose",Ee="Unpack",Re="UnsortedSegmentSum",Ae="ZerosLike",ke="Step",Te="FromPixels",Fe="RotateWithOffset"},function(t,e,n){"use strict";n.d(e,"c",(function(){return i})),n.d(e,"a",(function(){return c})),n.d(e,"b",(function(){return l}));var r=n(5),o=n(6),s=n(4),a=n(0);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function i(t,e){let n=t;if(Object(a.isTypedArray)(t))return"string"===e?[]:[t.length];if(!Array.isArray(t))return[];const r=[];for(;Array.isArray(n)||Object(a.isTypedArray)(n)&&"string"!==e;)r.push(n.length),n=n[0];return Array.isArray(t)&&Object(o.b)().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&function t(e,n,r){if(r=r||[],!Array.isArray(e)&&!Object(a.isTypedArray)(e))return void Object(a.assert)(0===n.length,()=>`Element arr[${r.join("][")}] is a primitive, but should be an array/TypedArray of ${n[0]} elements`);Object(a.assert)(n.length>0,()=>`Element arr[${r.join("][")}] should be a primitive, but is an array of ${e.length} elements`),Object(a.assert)(e.length===n[0],()=>`Element arr[${r.join("][")}] should have ${n[0]} elements, but has ${e.length} elements`);const o=n.slice(1);for(let n=0;n<e.length;++n)t(e[n],o,r.concat(n))}(t,r,[]),r}function u(t,e,n,r){if(null!=t&&("numeric"!==t&&t!==e||"numeric"===t&&"string"===e))throw new Error(`Argument '${n}' passed to '${r}' must be ${t} tensor, but got ${e} tensor`)}function c(t,e,n,o="numeric"){if(t instanceof s.a)return u(o,t.dtype,e,n),t;let c=Object(a.inferDtype)(t);if("string"!==c&&["bool","int32","float32"].indexOf(o)>=0&&(c=o),u(o,c,e,n),null==t||!Object(a.isTypedArray)(t)&&!Array.isArray(t)&&"number"!=typeof t&&"boolean"!=typeof t&&"string"!=typeof t){const r=null==t?"null":t.constructor.name;throw new Error(`Argument '${e}' passed to '${n}' must be a Tensor or TensorLike, but got '${r}'`)}const l=i(t,c);Object(a.isTypedArray)(t)||Array.isArray(t)||(t=[t]);const h="string"!==c?Object(a.toTypedArray)(t,c):Object(a.flatten)(t,[],!0);return r.a.makeTensor(h,l,c)}function l(t,e,n,r="numeric"){if(!Array.isArray(t))throw new Error(`Argument ${e} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return t.map((t,r)=>c(t,`${e}[${r}]`,n),r)}},function(t,e,n){"use strict";n.d(e,"a",(function(){return o}));var r=n(5);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function o(t){const e=Object.keys(t);if(1!==e.length)throw new Error("Please provide an object with a single key (operation name) mapping to a function. Got an object with "+e.length+" keys.");let n=e[0];const o=t[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n+="__op";const s=(...t)=>{r.a.startScope(n);try{const e=o(...t);return e instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r.a.endScope(e),e}catch(t){throw r.a.endScope(null),t}};return Object.defineProperty(s,"name",{value:n,configurable:!0}),s}},function(t,e,n){"use strict";n.d(e,"b",(function(){return u})),n.d(e,"f",(function(){return d})),n.d(e,"e",(function(){return p})),n.d(e,"d",(function(){return f})),n.d(e,"a",(function(){return g})),n.d(e,"c",(function(){return m}));var r=n(0);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function o(t,e,n,o){const u=Object(r.computeStrides)(e),c=function(t,e,n,o){const a=Object(r.sizeFromShape)(e),u=o[o.length-1],c=new Array(u).fill(0),l=e.length,h="complex64"===n?i(t):t;if(l>1)for(let t=0;t<a/u;t++){const e=t*u;for(let t=0;t<u;t++)c[t]=Math.max(c[t],s(h[e+t],0,n).length)}return c}(t,e,n,u),l=e.length,h=function t(e,n,r,o,u,c=!0){const l="complex64"===r?2:1,h=n[0],d=n.length;if(0===d){if("complex64"===r){return[s(i(e)[0],0,r)]}return"bool"===r?[a(e[0])]:[e[0].toString()]}if(1===d){if(h>20){const t=3*l;let n=Array.from(e.slice(0,t)),o=Array.from(e.slice((h-3)*l,h*l));return"complex64"===r&&(n=i(n),o=i(o)),["["+n.map((t,e)=>s(t,u[e],r)).join(", ")+", ..., "+o.map((t,e)=>s(t,u[h-3+e],r)).join(", ")+"]"]}return["["+("complex64"===r?i(e):Array.from(e)).map((t,e)=>s(t,u[e],r)).join(", ")+"]"]}const p=n.slice(1),f=o.slice(1),g=o[0]*l,m=[];if(h>20){for(let n=0;n<3;n++){const o=n*g,s=o+g;m.push(...t(e.slice(o,s),p,r,f,u,!1))}m.push("...");for(let n=h-3;n<h;n++){const o=n*g,s=o+g;m.push(...t(e.slice(o,s),p,r,f,u,n===h-1))}}else for(let n=0;n<h;n++){const o=n*g,s=o+g;m.push(...t(e.slice(o,s),p,r,f,u,n===h-1))}const b=2===d?",":"";m[0]="["+m[0]+b;for(let t=1;t<m.length-1;t++)m[t]=" "+m[t]+b;let x=",\n";for(let t=2;t<d;t++)x+="\n";return m[m.length-1]=" "+m[m.length-1]+"]"+(c?"":x),m}(t,e,n,u,c),d=["Tensor"];return o&&(d.push("  dtype: "+n),d.push("  rank: "+l),d.push(`  shape: [${e}]`),d.push("  values:")),d.push(h.map(t=>"    "+t).join("\n")),d.join("\n")}function s(t,e,n){let o;return o=Array.isArray(t)?parseFloat(t[0].toFixed(7))+" + "+parseFloat(t[1].toFixed(7))+"j":Object(r.isString)(t)?`'${t}'`:"bool"===n?a(t):parseFloat(t.toFixed(7)).toString(),Object(r.rightPad)(o,e)}function a(t){return 0===t?"false":"true"}function i(t){const e=[];for(let n=0;n<t.length;n+=2)e.push([t[n],t[n+1]]);return e}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class u{constructor(t,e,n){if(this.dtype=e,this.shape=t.slice(),this.size=r.sizeFromShape(t),null!=n){const t=n.length;r.assert(t===this.size,()=>`Length of values '${t}' does not match the size inferred by the shape '${this.size}'.`)}if("complex64"===e)throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=n||r.getArrayFromDType(e,this.size),this.strides=Object(r.computeStrides)(t)}set(t,...e){0===e.length&&(e=[0]),r.assert(e.length===this.rank,()=>`The number of provided coordinates (${e.length}) must match the rank (${this.rank})`);const n=this.locToIndex(e);this.values[n]=t}get(...t){0===t.length&&(t=[0]);let e=0;for(const n of t){if(n<0||n>=this.shape[e]){const e=`Requested out of range element at ${t}.   Buffer shape=`+this.shape;throw new Error(e)}e++}let n=t[t.length-1];for(let e=0;e<t.length-1;++e)n+=this.strides[e]*t[e];return this.values[n]}locToIndex(t){if(0===this.rank)return 0;if(1===this.rank)return t[0];let e=t[t.length-1];for(let n=0;n<t.length-1;++n)e+=this.strides[n]*t[n];return e}indexToLoc(t){if(0===this.rank)return[];if(1===this.rank)return[t];const e=new Array(this.shape.length);for(let n=0;n<e.length-1;++n)e[n]=Math.floor(t/this.strides[n]),t-=e[n]*this.strides[n];return e[e.length-1]=t,e}get rank(){return this.shape.length}toTensor(){return c().makeTensor(this.values,this.shape,this.dtype)}}let c=null,l=null,h=null;function d(t){c=t}function p(t){l=t}function f(t){h=t}class g{constructor(t,e,n,o){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=e||"float32",this.size=r.sizeFromShape(t),this.strides=Object(r.computeStrides)(t),this.dataId=n,this.id=o,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return l.buffer(this.shape,this.dtype,t)}bufferSync(){return l.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return Object(r.toNestedArray)(this.shape,t)}arraySync(){return Object(r.toNestedArray)(this.shape,this.dataSync())}async data(){this.throwIfDisposed();const t=c().read(this.dataId);if("string"===this.dtype){const e=await t;try{return e.map(t=>r.decodeString(t))}catch(t){throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataSync(){this.throwIfDisposed();const t=c().readSync(this.dataId);if("string"===this.dtype)try{return t.map(t=>r.decodeString(t))}catch(t){throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await c().read(this.dataId);return"string"===this.dtype?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(c().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return l.print(this,t)}clone(){return this.throwIfDisposed(),l.clone(this)}toString(t=!1){return o(this.dataSync(),this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),l.cast(this,t)}variable(t=!0,e,n){return this.throwIfDisposed(),c().makeVariable(this,t,e,n)}}Object.defineProperty(g,Symbol.hasInstance,{value:t=>!!t&&null!=t.data&&null!=t.dataSync&&null!=t.throwIfDisposed});class m extends g{constructor(t,e,n,r){super(t.shape,t.dtype,t.dataId,r),this.trainable=e,this.name=n}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!r.arraysEqual(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);c().disposeTensor(this),this.dataId=t.dataId,c().incRef(this,null)}dispose(){c().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(m,Symbol.hasInstance,{value:t=>t instanceof g&&null!=t.assign&&t.assign instanceof Function})},function(t,e,n){"use strict";n.d(e,"b",(function(){return g})),n.d(e,"a",(function(){return m}));var r=n(6),o=n(18),s=n(1),a=n(13),i=n(0);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class u{constructor(t,e){this.backendTimer=t,this.logger=e,null==e&&(this.logger=new l)}profileKernel(t,e,n){let r;const o=this.backendTimer.time(()=>{r=n()});for(let e=0;e<r.length;e++){const n=r[e];n.data().then(e=>{c(e,n.dtype,t)})}return{kernelName:t,outputs:r,inputs:e,timeMs:o.then(t=>t.kernelMs),extraInfo:o.then(t=>null!=t.getExtraProfileInfo?t.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:e,outputs:n,timeMs:r,inputs:o,extraInfo:s}=t;n.forEach(t=>{Promise.all([t.data(),r,s]).then(n=>{this.logger.logKernelProfile(e,t,n[0],n[1],o,n[2])})})}}function c(t,e,n){if("float32"!==e)return!1;for(let e=0;e<t.length;e++){const r=t[e];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}class l{logKernelProfile(t,e,n,r,o,s){const a="number"==typeof r?i.rightPad(r+"ms",9):r.error,u=i.rightPad(t,25),c=e.rank,l=e.size,h=i.rightPad(e.shape.toString(),14);let d="";for(const t in o){const n=o[t];if(null!=n){const r=n.shape||e.shape,o=r.length;d+=`${t}: ${o}D ${o>0?r:""} `}}console.log(`%c${u}\t%c${a}\t%c${c}D ${h}\t%c${l}\t%c${d}\t%c${s}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var h=n(4),d=n(7);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class p{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class f{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new p}async ready(){if(null!=this.pendingBackendInit)return this.pendingBackendInit.then(()=>{});if(null!=this.backendInstance)return;const t=this.getSortedBackends();for(let e=0;e<t.length;e++){const n=t[e];if(await this.initializeBackend(n).success)return void await this.setBackend(n)}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(null!=this.pendingBackendInit)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(null==this.backendInstance){const{name:t,asyncInit:e}=this.initializeBackendsAndReturnBest();if(e)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry)){if(!(t in this.registryFactory))return null;{const{asyncInit:e}=this.initializeBackend(t);if(e)return null}}return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,e,n=1){return t in this.registryFactory?(console.warn(t+" backend was already registered. Reusing existing backend factory."),!1):(this.registryFactory[t]={factory:e,priority:n},!0)}async setBackend(t){if(null==this.registryFactory[t])throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,null==this.registry[t]){this.backendInstance=null;const{success:e,asyncInit:n}=this.initializeBackend(t);if(!(n?await e:e))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new u(this.backendInstance),!0}setupRegisteredKernels(){Object(a.c)(this.backendName).forEach(t=>{null!=t.setupFunc&&t.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){Object(a.c)(t).forEach(e=>{null!=e.disposeFunc&&e.disposeFunc(this.registry[t])})}initializeBackend(t){const e=this.registryFactory[t];if(null==e)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const n=e.factory();if(Promise.resolve(n)===n){const e=++this.pendingBackendInitId,r=n.then(n=>!(e<this.pendingBackendInitId)&&(this.registry[t]=n,this.pendingBackendInit=null,!0)).catch(n=>(e<this.pendingBackendInitId||(this.pendingBackendInit=null,console.warn(`Initialization of backend ${t} failed`),console.warn(n.stack||n.message)),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}return this.registry[t]=n,{success:!0,asyncInit:!1}}catch(e){return console.warn(`Initialization of backend ${t} failed`),console.warn(e.stack||e.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(t+" backend not found in registry");this.backendName===t&&null!=this.pendingBackendInit&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(0===Object.keys(this.registryFactory).length)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,e)=>this.registryFactory[e].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let e=0;e<t.length;e++){const n=t[e],{success:r,asyncInit:o}=this.initializeBackend(n);if(o||r)return{name:n,asyncInit:o}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,e){const n=this.state.tensorInfo.get(e),r=n.backend,o=this.readSync(e);r.disposeData(e),n.backend=t,t.move(e,o,n.shape,n.dtype),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,e){let n,r=null;if(null==e){if("function"!=typeof t)throw new Error("Please provide a function to tidy()");e=t}else{if("string"!=typeof t&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if("function"!=typeof e)throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");r=t}return this.scopedRun(()=>this.startScope(r),()=>this.endScope(n),()=>(n=e(),n instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),n))}scopedRun(t,e,n){t();try{const t=n();return e(),t}catch(t){throw e(),t}}nextTensorId(){return f.nextTensorId++}nextVariableId(){return f.nextVariableId++}clone(t){const e=this.makeTensorFromDataId(t.dataId,t.shape,t.dtype),n={x:t};return this.addTapeNode(this.state.activeScope.name,n,[e],t=>({x:()=>{const e={x:t},n={dtype:"float32"};return m.runKernelFunc(e=>e.cast(t,"float32"),e,null,s.v,n)}}),[],{}),e}runKernel(t,e,n,r,o){return this.runKernelFunc(null,e,null,t,n,r,o)}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,e,n){const r=this.backend.numDataIds();let o=0;n.forEach(t=>{o+="complex64"===t.dtype?3:1});const s=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],a=r-e-o-s;if(a>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${a} data ids) after running '${t}'`)}runKernelFunc(t,e,n,r,o,s,i){let u,c=[];const l=this.isTapeOn();null==r&&(r=null!=this.state.activeScope?this.state.activeScope.name:"");const h=this.state.numBytes,d=this.state.numTensors;let p;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);const f=Object(a.b)(r,this.backendName);let g,m;if(null!=f)p=()=>{const t=this.backend.numDataIds();g=f.kernelFunc({inputs:e,attrs:o,backend:this.backend});const n=Array.isArray(g)?g:[g];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(r,t,n);const a=n.map(({dataId:t,shape:e,dtype:n})=>this.makeTensorFromDataId(t,e,n));if(l){let t=this.getTensorsForGradient(r,e,a);if(null==t){null==i&&(i=[]);const e=a.filter((t,e)=>i[e]);t=(s||[]).slice().concat(e)}c=this.saveTensorsForBackwardMode(t)}return a};else{const e=t=>{l&&(c=t.map(t=>this.keep(this.clone(t))))};p=()=>{const n=this.backend.numDataIds();g=this.tidy(()=>t(this.backend,e));const o=Array.isArray(g)?g:[g];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(r,n,o),o}}return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{this.ENV.getBool("DEBUG")||this.state.profiling?(m=this.profiler.profileKernel(r,e,()=>p()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(m),u=m.outputs):u=p()}),l&&this.addTapeNode(r,e,u,n,c,o),this.state.profiling&&this.state.activeProfile.kernels.push({name:r,bytesAdded:this.state.numBytes-h,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-d,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(e).map(t=>null!=e[t]?e[t].shape:null),outputShapes:u.map(t=>t.shape),kernelTimeMs:m.timeMs,extraInfo:m.extraInfo}),Array.isArray(g)?u:u[0]}saveTensorsForBackwardMode(t){return t.map(t=>this.keep(this.clone(t)))}getTensorsForGradient(t,e,n){const r=Object(a.a)(t);if(null!=r){const t=r.inputsToSave||[],o=r.outputsToSave||[];let s;r.saveAllInputs?(i.assert(Array.isArray(e),()=>"saveAllInputs is true, expected inputs to be an array."),s=Object.keys(e).map(t=>e[t])):s=t.map(t=>e[t]);const a=n.filter((t,e)=>o[e]);return s.concat(a)}return null}makeTensor(t,e,n,r){if(null==t)throw new Error("Values passed to engine.makeTensor() are null");n=n||"float32",r=r||this.backend;let o=t;"string"===n&&i.isString(t[0])&&(o=t.map(t=>i.encodeString(t)));const s=r.write(o,e,n),a=new h.a(e,n,s,this.nextTensorId());if(this.incRef(a,r),"string"===n){const t=this.state.tensorInfo.get(s),e=Object(i.bytesFromStringArray)(o);this.state.numBytes+=e-t.bytes,t.bytes=e}return a}makeTensorFromDataId(t,e,n,r){n=n||"float32";const o=new h.a(e,n,t,this.nextTensorId());return this.incRef(o,r),o}makeVariable(t,e=!0,n,r){n=n||this.nextVariableId().toString(),null!=r&&r!==t.dtype&&(t=t.cast(r));const o=new h.c(t,e,n,this.nextTensorId());if(null!=this.state.registeredVariables[o.name])throw new Error(`Variable with name ${o.name} was already registered`);return this.state.registeredVariables[o.name]=o,this.incRef(o,this.backend),o}incRef(t,e){const n=this.state.tensorInfo.has(t.dataId)?this.state.tensorInfo.get(t.dataId).refCount:0;if(this.state.numTensors++,"string"===t.dtype&&this.state.numStringTensors++,0===n){this.state.numDataBuffers++;let n=0;"complex64"!==t.dtype&&"string"!==t.dtype&&(n=t.size*i.bytesPerElement(t.dtype)),this.state.tensorInfo.set(t.dataId,{backend:e||this.backend,dtype:t.dtype,shape:t.shape,bytes:n,refCount:0}),this.state.numBytes+=n}this.state.tensorInfo.get(t.dataId).refCount++,t instanceof h.c||this.track(t)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;this.state.numTensors--,"string"===t.dtype&&this.state.numStringTensors--;const e=this.state.tensorInfo.get(t.dataId);e.refCount<=1?("complex64"!==t.dtype&&(this.state.numBytes-=e.bytes),this.state.numDataBuffers--,e.backend.disposeData(t.dataId),this.state.tensorInfo.delete(t.dataId)):this.state.tensorInfo.get(t.dataId).refCount--}disposeVariables(){for(const t in this.state.registeredVariables){const e=this.state.registeredVariables[t];this.disposeVariable(e)}}disposeVariable(t){this.disposeTensor(t),null!=this.state.registeredVariables[t.name]&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,null==t.reasons&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const e=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(t=>t.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-e,this.state.activeProfile.newTensors=this.state.numTensors-n;for(const t of this.state.activeProfile.kernels)t.kernelTimeMs=await t.kernelTimeMs,t.extraInfo=await t.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&0===this.state.kernelDepth}addTapeNode(t,e,n,r,o,s){const u={id:this.state.nextTapeNodeId++,kernelName:t,inputs:e,outputs:n,saved:o},c=Object(a.a)(t);null!=c&&(r=c.gradFunc),null!=r&&(u.gradient=t=>(t=t.map((t,e)=>{if(null==t){const t=n[e],r=i.makeZerosTypedArray(t.size,t.dtype);return this.makeTensor(r,t.shape,t.dtype)}return t}),r(t.length>1?t:t[0],o,s))),this.state.activeTape.push(u)}keep(t){return t.kept=!0,t}startTape(){0===this.state.gradientDepth&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const e={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(e.name=t),this.state.scopeStack.push(e),this.state.activeScope=e}endScope(t){const e=Object(d.a)(t),n=new Set(e.map(t=>t.id));for(let t=0;t<this.state.activeScope.track.length;t++){const e=this.state.activeScope.track[t];e.kept||n.has(e.id)||e.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=0===this.state.scopeStack.length?null:this.state.scopeStack[this.state.scopeStack.length-1],e.forEach(t=>{t.kept||t.scopeId!==r.id||this.track(t)})}gradients(t,e,n,r=!1){if(i.assert(e.length>0,()=>"gradients() received an empty list of xs."),null!=n&&"float32"!==n.dtype)throw new Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);const o=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));i.assert(o instanceof h.a,()=>"The result y returned by f() must be a tensor.");const s=function(t,e,n){const r={},o={};for(let t=0;t<e.length;t++)r[e[t].id]=!0;for(let n=0;n<t.length;n++){const s=t[n],a=s.inputs;for(const t in a){const n=a[t];let i=!1;for(let t=0;t<e.length;t++)if(r[n.id]){s.outputs.forEach(t=>r[t.id]=!0),i=!0,o[s.id]=!0;break}if(i)break}}const s={};s[n.id]=!0;const a={};for(let e=t.length-1;e>=0;e--){const n=t[e],r=n.inputs;for(let t=0;t<n.outputs.length;t++)if(s[n.outputs[t].id]){for(const t in r)s[r[t].id]=!0,a[n.id]=!0;break}}const i=[];for(let e=0;e<t.length;e++){const n=t[e];if(o[n.id]&&a[n.id]){const t={};for(const e in n.inputs){const o=n.inputs[e];r[o.id]&&(t[e]=o)}const e=Object.assign({},n);e.inputs=t,e.outputs=n.outputs,i.push(e)}}return i}(this.state.activeTape,e,o);if(!r&&0===s.length&&e.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const t={};t[o.id]=null==n?function(t){const e=Object(i.makeOnesTypedArray)(Object(i.sizeFromShape)(t),"float32");return m.makeTensor(e,t,"float32")}(o.shape):n,function(t,e,n,r){for(let o=e.length-1;o>=0;o--){const s=e[o],a=[];if(s.outputs.forEach(e=>{const n=t[e.id];null!=n?a.push(n):a.push(null)}),null==s.gradient)throw new Error(`Cannot compute gradient: gradient function not found for ${s.kernelName}.`);const u=s.gradient(a);for(const e in s.inputs){if(!(e in u))throw new Error(`Cannot backprop through input ${e}. Available gradients found: ${Object.keys(u)}.`);const o=n(()=>u[e]());if("float32"!==o.dtype)throw new Error(`Error in gradient for op ${s.kernelName}. The gradient of input ${e} must have 'float32' dtype, but has '${o.dtype}'`);const a=s.inputs[e];if(!i.arraysEqual(o.shape,a.shape))throw new Error(`Error in gradient for op ${s.kernelName}. The gradient of input '${e}' has shape '${o.shape}', which does not match the shape of the input '${a.shape}'`);if(null==t[a.id])t[a.id]=o;else{const e=t[a.id];t[a.id]=r(e,o),e.dispose()}}}}(t,s,t=>this.tidy(t),b);const r=e.map(e=>t[e.id]);return 0===this.state.gradientDepth&&(this.state.activeTape.forEach(t=>{for(const e of t.saved)e.dispose()}),this.state.activeTape=null),{value:o,grads:r}})}customGrad(t){return i.assert(i.isFunction(t),()=>"The f passed in customGrad(f) must be a function."),(...e)=>{let n;i.assert(e.every(t=>t instanceof h.a),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");const r={};return e.forEach((t,e)=>{r[e]=t}),this.runKernelFunc((r,o)=>(n=t(...e,o),i.assert(n.value instanceof h.a,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),i.assert(i.isFunction(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),r,(t,r)=>{const o=n.gradFunc(t,r),s=Array.isArray(o)?o:[o];i.assert(s.length===e.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),i.assert(s.every(t=>t instanceof h.a),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const a={};return s.forEach((t,e)=>{a[e]=()=>t}),a})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}async time(t){const e=Object(i.now)(),n=await this.backend.time(t);return n.wallMs=Object(i.now)()-e,n}track(t){return null!=this.state.activeScope&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new p;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}function g(){const t=Object(o.b)();if(null==t._tfengine){const e=new r.a(t);t._tfengine=new f(e)}return Object(r.c)(t._tfengine.ENV),Object(h.f)(()=>t._tfengine),t._tfengine}f.nextTensorId=0,f.nextVariableId=0;const m=g();function b(t,e){const n={a:t,b:e};return m.runKernelFunc((n,r)=>{const o=n.add(t,e);return r([t,e]),o},n,null,s.d)}},function(t,e,n){"use strict";n.d(e,"a",(function(){return r})),n.d(e,"b",(function(){return o})),n.d(e,"c",(function(){return a}));class r{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.populateURLFlags()}setPlatform(t,e){null!=this.platform&&console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`),this.platformName=t,this.platform=e}registerFlag(t,e,n){if(this.flagRegistry[t]={evaluationFn:e,setHook:n},null!=this.urlFlags[t]){const e=this.urlFlags[t];console.warn(`Setting feature override from URL ${t}: ${e}.`),this.set(t,e)}}async getAsync(t){return t in this.flags||(this.flags[t]=await this.evaluateFlag(t)),this.flags[t]}get(t){if(t in this.flags)return this.flags[t];const e=this.evaluateFlag(t);if(e instanceof Promise)throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=e,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,e){if(null==this.flagRegistry[t])throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=e,null!=this.flagRegistry[t].setHook&&this.flagRegistry[t].setHook(e)}evaluateFlag(t){if(null==this.flagRegistry[t])throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(void 0===this.global||void 0===this.global.location||void 0===this.global.location.search)return;const t=function(t){const e={};return t.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(t,...n)=>(function(t,e,n){t[decodeURIComponent(e)]=decodeURIComponent(n||"")}(e,n[0],n[1]),n.join("="))),e}(this.global.location.search);if("tfjsflags"in t){t.tfjsflags.split(",").forEach(t=>{const[e,n]=t.split(":");this.urlFlags[e]=function(t,e){if("true"===(e=e.toLowerCase())||"false"===e)return"true"===e;if(""+ +e===e)return+e;throw new Error(`Could not parse value flag value ${e} for flag ${t}.`)}(e,n)})}}}function o(){return s}let s=null;function a(t){s=t}},function(t,e,n){"use strict";n.d(e,"b",(function(){return s})),n.d(e,"a",(function(){return a}));var r=n(4),o=n(14);n(0);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function s(t,e){if(t.dtype===e.dtype)return[t,e];const n=Object(o.c)(t.dtype,e.dtype);return[t.cast(n),e.cast(n)]}function a(t){const e=[];return function t(e,n,o){if(null==e)return;if(e instanceof r.a)return void n.push(e);if(s=e,!Array.isArray(s)&&"object"!=typeof s)return;var s;const a=e;for(const e in a){const r=a[e];o.has(r)||(o.add(r),t(r,n,o))}}(t,e,new Set),e}},function(t,e,n){"use strict";(function(t){n.d(e,"e",(function(){return i})),n.d(e,"a",(function(){return l})),n.d(e,"b",(function(){return h})),n.d(e,"d",(function(){return d})),n.d(e,"c",(function(){return p})),n.d(e,"f",(function(){return f}));var r=n(9),o=n(11),s=n(0),a=n(19);function i(t,e){const n={};let i,u=0;for(const c of e){const e=c.name,l=c.dtype,h=c.shape,d=Object(s.sizeFromShape)(h);let p;if("quantization"in c){const n=c.quantization;if("uint8"===n.dtype||"uint16"===n.dtype){if(!("min"in n)||!("scale"in n))throw new Error(`Weight ${c.name} with quantization ${n.dtype} doesn't have corresponding metadata min and scale.`)}else{if("float16"!==n.dtype)throw new Error(`Weight ${c.name} has unknown quantization dtype ${n.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);if("float32"!==l)throw new Error(`Weight ${c.name} is quantized with ${n.dtype} which only supports weights of type float32 not ${l}.`)}const r=a.a[n.dtype],o=t.slice(u,u+d*r),s="uint8"===n.dtype?new Uint8Array(o):new Uint16Array(o);if("float32"===l)if("uint8"===n.dtype||"uint16"===n.dtype){p=new Float32Array(s.length);for(let t=0;t<s.length;t++){const e=s[t];p[t]=e*n.scale+n.min}}else{if("float16"!==n.dtype)throw new Error(`Unsupported quantization type ${n.dtype} for weight type float32.`);void 0===i&&(i=g()),p=i(s)}else{if("int32"!==l)throw new Error(`Unsupported dtype in weight '${e}': ${l}`);if("uint8"!==n.dtype&&"uint16"!==n.dtype)throw new Error(`Unsupported quantization type ${n.dtype} for weight type int32.`);p=new Int32Array(s.length);for(let t=0;t<s.length;t++){const e=s[t];p[t]=Math.round(e*n.scale+n.min)}}u+=d*r}else if("string"===l){const e=Object(s.sizeFromShape)(c.shape);p=[];for(let n=0;n<e;n++){const e=new Uint32Array(t.slice(u,u+4))[0];u+=4;const n=new Uint8Array(t.slice(u,u+e));p.push(n),u+=e}}else{const s=a.a[l],i=t.slice(u,u+d*s);if("float32"===l)p=new Float32Array(i);else if("int32"===l)p=new Int32Array(i);else if("bool"===l)p=new Uint8Array(i);else{if("complex64"!==l)throw new Error(`Unsupported dtype in weight '${e}': ${l}`);{p=new Float32Array(i);const t=new Float32Array(p.length/2),s=new Float32Array(p.length/2);for(let e=0;e<t.length;e++)t[e]=p[2*e],s[e]=p[2*e+1];const a=Object(o.a)(t,h,"float32"),u=Object(o.a)(s,h,"float32");n[e]=Object(r.a)(a,u)}}u+=d*s}"complex64"!==l&&(n[e]=Object(o.a)(p,h,l))}return n}const u=void 0!==t&&("undefined"==typeof Blob||"undefined"==typeof atob||"undefined"==typeof btoa);function c(e){return u?t.byteLength(e):new Blob([e]).size}function l(e){if(u)return t.from(e).toString("base64");const n=new Uint8Array(e);let r="";for(let t=0,e=n.length;t<e;t++)r+=String.fromCharCode(n[t]);return btoa(r)}function h(e){if(u){const n=t.from(e,"base64");return n.buffer.slice(n.byteOffset,n.byteOffset+n.byteLength)}const n=atob(e),r=new Uint8Array(n.length);for(let t=0;t<n.length;++t)r.set([n.charCodeAt(t)],t);return r.buffer}function d(t){if(1===t.length)return t[0];let e=0;t.forEach(t=>{e+=t.byteLength});const n=new Uint8Array(e);let r=0;return t.forEach(t=>{n.set(new Uint8Array(t),r),r+=t.byteLength}),n.buffer}function p(t){for(t=t.trim();t.endsWith("/");)t=t.slice(0,t.length-1);const e=t.split("/");return e[e.length-1]}function f(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:null==t.modelTopology?0:c(JSON.stringify(t.modelTopology)),weightSpecsBytes:null==t.weightSpecs?0:c(JSON.stringify(t.weightSpecs)),weightDataBytes:null==t.weightData?0:t.weightData.byteLength}}function g(){const t=function(){const t=t=>{let e=t<<13,n=0;for(;0==(8388608&e);)n-=8388608,e<<=1;return e&=-8388609,n+=947912704,e|n},e=new Uint32Array(2048);e[0]=0;for(let n=1;n<1024;n++)e[n]=t(n);for(let t=1024;t<2048;t++)e[t]=939524096+(t-1024<<13);return e}(),e=function(){const t=new Uint32Array(64);t[0]=0,t[31]=1199570944,t[32]=2147483648,t[63]=3347054592;for(let e=1;e<31;e++)t[e]=e<<23;for(let e=33;e<63;e++)t[e]=2147483648+(e-32<<23);return t}(),n=function(){const t=new Uint32Array(64);for(let e=0;e<64;e++)t[e]=1024;return t[0]=t[32]=0,t}();return r=>{const o=new ArrayBuffer(4*r.length),s=new Uint32Array(o);for(let o=0;o<r.length;o++){const a=r[o],i=t[n[a>>10]+(1023&a)]+e[a>>10];s[o]=i}return new Float32Array(o)}}}).call(this,n(93).Buffer)},function(t,e,n){"use strict";n.d(e,"a",(function(){return u}));var r=n(5),o=n(1),s=n(2),a=n(0),i=n(3);const u=Object(i.a)({complex_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){const n=Object(s.a)(t,"real","complex"),i=Object(s.a)(e,"imag","complex");a.assertShapesMatch(n.shape,i.shape,`real and imag shapes, ${n.shape} and ${i.shape}, must match in call to tf.complex().`);const u={real:n,imag:i};return r.a.runKernelFunc(t=>t.complex(n,i),u,null,o.y)}})},function(t,e){t.exports=function(){throw new Error("define cannot be used indirect")}},function(t,e,n){"use strict";n.d(e,"a",(function(){return s}));var r=n(2),o=n(12);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function s(t,e,n){const s=Object(r.c)(t,n);return Object(o.a)(t,e,s,n)}},function(t,e,n){"use strict";n.d(e,"a",(function(){return s}));var r=n(5),o=n(0);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function s(t,e,n,s){if(null==s&&(s=Object(o.inferDtype)(t)),"complex64"===s)throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(!Object(o.isTypedArray)(t)&&!Array.isArray(t)&&"number"!=typeof t&&"boolean"!=typeof t&&"string"!=typeof t)throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(null!=e){Object(o.assertNonNegativeIntegerDimensions)(e);const t=Object(o.sizeFromShape)(e),r=Object(o.sizeFromShape)(n);Object(o.assert)(t===r,()=>`Based on the provided shape, [${e}], the tensor should have ${t} values but has ${r}`);for(let t=0;t<n.length;++t){const r=n[t],s=t!==n.length-1||r!==Object(o.sizeFromShape)(e.slice(t));Object(o.assert)(n[t]===e[t]||!s,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${e}). `)}}return Object(o.isTypedArray)(t)||Array.isArray(t)||(t=[t]),e=e||n,t="string"!==s?Object(o.toTypedArray)(t,s):Object(o.flatten)(t,[],!0),r.a.makeTensor(t,e,s)}},function(t,e,n){"use strict";n.d(e,"b",(function(){return i})),n.d(e,"a",(function(){return u})),n.d(e,"c",(function(){return c})),n.d(e,"e",(function(){return l})),n.d(e,"d",(function(){return h}));var r=n(6),o=n(18);
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const s=Object(o.a)("kernelRegistry",()=>new Map),a=Object(o.a)("gradRegistry",()=>new Map);function i(t,e){const n=d(t,e);return s.get(n)}function u(t){return a.get(t)}function c(t){const e=s.entries(),n=[];for(;;){const{done:r,value:o}=e.next();if(r)break;const[s,a]=o,[i]=s.split("_");i===t&&n.push(a)}return n}function l(t){const{kernelName:e,backendName:n}=t,r=d(e,n);s.has(r)&&console.warn(`The kernel '${e}' for backend '${n}' is already registered`),s.set(r,t)}function h(t){const{kernelName:e}=t;a.has(e)&&Object(r.b)().getBool("DEBUG")&&console.warn(`Overriding the gradient for '${e}'`),a.set(e,t)}function d(t,e){return`${e}_${t}`}},function(t,e,n){"use strict";
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var r,o,s,a,i;n.d(e,"a",(function(){return r})),n.d(e,"c",(function(){return c})),n.d(e,"b",(function(){return l})),function(t){t.R0="R0",t.R1="R1",t.R2="R2",t.R3="R3",t.R4="R4",t.R5="R5",t.R6="R6"}(r||(r={})),function(t){t.float32="float32",t.int32="int32",t.bool="int32",t.complex64="complex64"}(o||(o={})),function(t){t.float32="float32",t.int32="int32",t.bool="bool",t.complex64="complex64"}(s||(s={})),function(t){t.float32="float32",t.int32="float32",t.bool="float32",t.complex64="complex64"}(a||(a={})),function(t){t.float32="complex64",t.int32="complex64",t.bool="complex64",t.complex64="complex64"}(i||(i={}));const u={float32:a,int32:o,bool:s,complex64:i};function c(t,e){if("string"===t||"string"===e){if("string"===t&&"string"===e)return"string";throw new Error(`Can not upcast ${t} with ${e}`)}return u[t][e]}function l(t){return c(t,"int32")}},function(t,e){t.exports=function(t){return t.webpackPolyfill||(t.deprecate=function(){},t.paths=[],t.children||(t.children=[]),Object.defineProperty(t,"loaded",{enumerable:!0,get:function(){return t.l}}),Object.defineProperty(t,"id",{enumerable:!0,get:function(){return t.i}}),t.webpackPolyfill=1),t}},function(t,e){(function(e){t.exports=e}).call(this,{})},function(t,e,n){"use strict";(function(t){var e=n(20),r=n(6);
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const o=Object(r.b)();o.registerFlag("DEBUG",()=>!1,t=>{t&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")}),o.registerFlag("IS_BROWSER",()=>e.isBrowser()),o.registerFlag("IS_NODE",()=>void 0!==t&&void 0!==t.versions&&void 0!==t.versions.node),o.registerFlag("IS_CHROME",()=>"undefined"!=typeof navigator&&null!=navigator&&null!=navigator.userAgent&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),o.registerFlag("PROD",()=>!1),o.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>o.getBool("DEBUG")),o.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0),o.registerFlag("IS_TEST",()=>!1)}).call(this,n(21))},function(t,e,n){"use strict";(function(t,r){
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
let o;function s(){if(null==o){let e;if("undefined"!=typeof window)e=window;else if(void 0!==t)e=t;else if(void 0!==r)e=r;else{if("undefined"==typeof self)throw new Error("Could not find a global object");e=self}o=e}return o}function a(t,e){const n=function(){const t=s();return null==t._tfGlobals&&(t._tfGlobals=new Map),t._tfGlobals}();if(n.has(t))return n.get(t);{const r=e();return n.set(t,r),n.get(t)}}n.d(e,"b",(function(){return s})),n.d(e,"a",(function(){return a}))}).call(this,n(22),n(21))},function(t,e,n){"use strict";n.d(e,"a",(function(){return r}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const r={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8}},function(t,e,n){"use strict";function r(){if("undefined"!=typeof navigator&&null!=navigator){const t=navigator.userAgent||navigator.vendor||window.opera;return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function o(){return"undefined"!=typeof window&&null!=window.document||"undefined"!=typeof WorkerGlobalScope}n.r(e),n.d(e,"isMobile",(function(){return r})),n.d(e,"isBrowser",(function(){return o}))},function(t,e){var n,r,o=t.exports={};function s(){throw new Error("setTimeout has not been defined")}function a(){throw new Error("clearTimeout has not been defined")}function i(t){if(n===setTimeout)return setTimeout(t,0);if((n===s||!n)&&setTimeout)return n=setTimeout,setTimeout(t,0);try{return n(t,0)}catch(e){try{return n.call(null,t,0)}catch(e){return n.call(this,t,0)}}}!function(){try{n="function"==typeof setTimeout?setTimeout:s}catch(t){n=s}try{r="function"==typeof clearTimeout?clearTimeout:a}catch(t){r=a}}();var u,c=[],l=!1,h=-1;function d(){l&&u&&(l=!1,u.length?c=u.concat(c):h=-1,c.length&&p())}function p(){if(!l){var t=i(d);l=!0;for(var e=c.length;e;){for(u=c,c=[];++h<e;)u&&u[h].run();h=-1,e=c.length}u=null,l=!1,function(t){if(r===clearTimeout)return clearTimeout(t);if((r===a||!r)&&clearTimeout)return r=clearTimeout,clearTimeout(t);try{r(t)}catch(e){try{return r.call(null,t)}catch(e){return r.call(this,t)}}}(t)}}function f(t,e){this.fun=t,this.array=e}function g(){}o.nextTick=function(t){var e=new Array(arguments.length-1);if(arguments.length>1)for(var n=1;n<arguments.length;n++)e[n-1]=arguments[n];c.push(new f(t,e)),1!==c.length||l||i(p)},f.prototype.run=function(){this.fun.apply(null,this.array)},o.title="browser",o.browser=!0,o.env={},o.argv=[],o.version="",o.versions={},o.on=g,o.addListener=g,o.once=g,o.off=g,o.removeListener=g,o.removeAllListeners=g,o.emit=g,o.prependListener=g,o.prependOnceListener=g,o.listeners=function(t){return[]},o.binding=function(t){throw new Error("process.binding is not supported")},o.cwd=function(){return"/"},o.chdir=function(t){throw new Error("process.chdir is not supported")},o.umask=function(){return 0}},function(t,e){var n;n=function(){return this}();try{n=n||new Function("return this")()}catch(t){"object"==typeof window&&(n=window)}t.exports=n},function(t,e,n){var r=n(100),o=n(101),s=n(102),a=n(103),i=n(104),u=n(105),c=n(106);c.alea=r,c.xor128=o,c.xorwow=s,c.xorshift7=a,c.xor4096=i,c.tychei=u,t.exports=c},,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,function(t,e,n){t.exports=n(108)},function(t,e,n){"use strict";(function(t){
/*!
 * The buffer module from node.js, for the browser.
 *
 * @author   Feross Aboukhadijeh <http://feross.org>
 * @license  MIT
 */
var r=n(94),o=n(95),s=n(96);function a(){return u.TYPED_ARRAY_SUPPORT?2147483647:1073741823}function i(t,e){if(a()<e)throw new RangeError("Invalid typed array length");return u.TYPED_ARRAY_SUPPORT?(t=new Uint8Array(e)).__proto__=u.prototype:(null===t&&(t=new u(e)),t.length=e),t}function u(t,e,n){if(!(u.TYPED_ARRAY_SUPPORT||this instanceof u))return new u(t,e,n);if("number"==typeof t){if("string"==typeof e)throw new Error("If encoding is specified then the first argument must be a string");return h(this,t)}return c(this,t,e,n)}function c(t,e,n,r){if("number"==typeof e)throw new TypeError('"value" argument must not be a number');return"undefined"!=typeof ArrayBuffer&&e instanceof ArrayBuffer?function(t,e,n,r){if(e.byteLength,n<0||e.byteLength<n)throw new RangeError("'offset' is out of bounds");if(e.byteLength<n+(r||0))throw new RangeError("'length' is out of bounds");e=void 0===n&&void 0===r?new Uint8Array(e):void 0===r?new Uint8Array(e,n):new Uint8Array(e,n,r);u.TYPED_ARRAY_SUPPORT?(t=e).__proto__=u.prototype:t=d(t,e);return t}(t,e,n,r):"string"==typeof e?function(t,e,n){"string"==typeof n&&""!==n||(n="utf8");if(!u.isEncoding(n))throw new TypeError('"encoding" must be a valid string encoding');var r=0|f(e,n),o=(t=i(t,r)).write(e,n);o!==r&&(t=t.slice(0,o));return t}(t,e,n):function(t,e){if(u.isBuffer(e)){var n=0|p(e.length);return 0===(t=i(t,n)).length||e.copy(t,0,0,n),t}if(e){if("undefined"!=typeof ArrayBuffer&&e.buffer instanceof ArrayBuffer||"length"in e)return"number"!=typeof e.length||(r=e.length)!=r?i(t,0):d(t,e);if("Buffer"===e.type&&s(e.data))return d(t,e.data)}var r;throw new TypeError("First argument must be a string, Buffer, ArrayBuffer, Array, or array-like object.")}(t,e)}function l(t){if("number"!=typeof t)throw new TypeError('"size" argument must be a number');if(t<0)throw new RangeError('"size" argument must not be negative')}function h(t,e){if(l(e),t=i(t,e<0?0:0|p(e)),!u.TYPED_ARRAY_SUPPORT)for(var n=0;n<e;++n)t[n]=0;return t}function d(t,e){var n=e.length<0?0:0|p(e.length);t=i(t,n);for(var r=0;r<n;r+=1)t[r]=255&e[r];return t}function p(t){if(t>=a())throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x"+a().toString(16)+" bytes");return 0|t}function f(t,e){if(u.isBuffer(t))return t.length;if("undefined"!=typeof ArrayBuffer&&"function"==typeof ArrayBuffer.isView&&(ArrayBuffer.isView(t)||t instanceof ArrayBuffer))return t.byteLength;"string"!=typeof t&&(t=""+t);var n=t.length;if(0===n)return 0;for(var r=!1;;)switch(e){case"ascii":case"latin1":case"binary":return n;case"utf8":case"utf-8":case void 0:return L(t).length;case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return 2*n;case"hex":return n>>>1;case"base64":return W(t).length;default:if(r)return L(t).length;e=(""+e).toLowerCase(),r=!0}}function g(t,e,n){var r=!1;if((void 0===e||e<0)&&(e=0),e>this.length)return"";if((void 0===n||n>this.length)&&(n=this.length),n<=0)return"";if((n>>>=0)<=(e>>>=0))return"";for(t||(t="utf8");;)switch(t){case"hex":return A(this,e,n);case"utf8":case"utf-8":return S(this,e,n);case"ascii":return E(this,e,n);case"latin1":case"binary":return R(this,e,n);case"base64":return I(this,e,n);case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return k(this,e,n);default:if(r)throw new TypeError("Unknown encoding: "+t);t=(t+"").toLowerCase(),r=!0}}function m(t,e,n){var r=t[e];t[e]=t[n],t[n]=r}function b(t,e,n,r,o){if(0===t.length)return-1;if("string"==typeof n?(r=n,n=0):n>2147483647?n=2147483647:n<-2147483648&&(n=-2147483648),n=+n,isNaN(n)&&(n=o?0:t.length-1),n<0&&(n=t.length+n),n>=t.length){if(o)return-1;n=t.length-1}else if(n<0){if(!o)return-1;n=0}if("string"==typeof e&&(e=u.from(e,r)),u.isBuffer(e))return 0===e.length?-1:x(t,e,n,r,o);if("number"==typeof e)return e&=255,u.TYPED_ARRAY_SUPPORT&&"function"==typeof Uint8Array.prototype.indexOf?o?Uint8Array.prototype.indexOf.call(t,e,n):Uint8Array.prototype.lastIndexOf.call(t,e,n):x(t,[e],n,r,o);throw new TypeError("val must be string, number or Buffer")}function x(t,e,n,r,o){var s,a=1,i=t.length,u=e.length;if(void 0!==r&&("ucs2"===(r=String(r).toLowerCase())||"ucs-2"===r||"utf16le"===r||"utf-16le"===r)){if(t.length<2||e.length<2)return-1;a=2,i/=2,u/=2,n/=2}function c(t,e){return 1===a?t[e]:t.readUInt16BE(e*a)}if(o){var l=-1;for(s=n;s<i;s++)if(c(t,s)===c(e,-1===l?0:s-l)){if(-1===l&&(l=s),s-l+1===u)return l*a}else-1!==l&&(s-=s-l),l=-1}else for(n+u>i&&(n=i-u),s=n;s>=0;s--){for(var h=!0,d=0;d<u;d++)if(c(t,s+d)!==c(e,d)){h=!1;break}if(h)return s}return-1}function y(t,e,n,r){n=Number(n)||0;var o=t.length-n;r?(r=Number(r))>o&&(r=o):r=o;var s=e.length;if(s%2!=0)throw new TypeError("Invalid hex string");r>s/2&&(r=s/2);for(var a=0;a<r;++a){var i=parseInt(e.substr(2*a,2),16);if(isNaN(i))return a;t[n+a]=i}return a}function v(t,e,n,r){return z(L(e,t.length-n),t,n,r)}function w(t,e,n,r){return z(function(t){for(var e=[],n=0;n<t.length;++n)e.push(255&t.charCodeAt(n));return e}(e),t,n,r)}function C(t,e,n,r){return w(t,e,n,r)}function $(t,e,n,r){return z(W(e),t,n,r)}function O(t,e,n,r){return z(function(t,e){for(var n,r,o,s=[],a=0;a<t.length&&!((e-=2)<0);++a)n=t.charCodeAt(a),r=n>>8,o=n%256,s.push(o),s.push(r);return s}(e,t.length-n),t,n,r)}function I(t,e,n){return 0===e&&n===t.length?r.fromByteArray(t):r.fromByteArray(t.slice(e,n))}function S(t,e,n){n=Math.min(t.length,n);for(var r=[],o=e;o<n;){var s,a,i,u,c=t[o],l=null,h=c>239?4:c>223?3:c>191?2:1;if(o+h<=n)switch(h){case 1:c<128&&(l=c);break;case 2:128==(192&(s=t[o+1]))&&(u=(31&c)<<6|63&s)>127&&(l=u);break;case 3:s=t[o+1],a=t[o+2],128==(192&s)&&128==(192&a)&&(u=(15&c)<<12|(63&s)<<6|63&a)>2047&&(u<55296||u>57343)&&(l=u);break;case 4:s=t[o+1],a=t[o+2],i=t[o+3],128==(192&s)&&128==(192&a)&&128==(192&i)&&(u=(15&c)<<18|(63&s)<<12|(63&a)<<6|63&i)>65535&&u<1114112&&(l=u)}null===l?(l=65533,h=1):l>65535&&(l-=65536,r.push(l>>>10&1023|55296),l=56320|1023&l),r.push(l),o+=h}return function(t){var e=t.length;if(e<=4096)return String.fromCharCode.apply(String,t);var n="",r=0;for(;r<e;)n+=String.fromCharCode.apply(String,t.slice(r,r+=4096));return n}(r)}e.Buffer=u,e.SlowBuffer=function(t){+t!=t&&(t=0);return u.alloc(+t)},e.INSPECT_MAX_BYTES=50,u.TYPED_ARRAY_SUPPORT=void 0!==t.TYPED_ARRAY_SUPPORT?t.TYPED_ARRAY_SUPPORT:function(){try{var t=new Uint8Array(1);return t.__proto__={__proto__:Uint8Array.prototype,foo:function(){return 42}},42===t.foo()&&"function"==typeof t.subarray&&0===t.subarray(1,1).byteLength}catch(t){return!1}}(),e.kMaxLength=a(),u.poolSize=8192,u._augment=function(t){return t.__proto__=u.prototype,t},u.from=function(t,e,n){return c(null,t,e,n)},u.TYPED_ARRAY_SUPPORT&&(u.prototype.__proto__=Uint8Array.prototype,u.__proto__=Uint8Array,"undefined"!=typeof Symbol&&Symbol.species&&u[Symbol.species]===u&&Object.defineProperty(u,Symbol.species,{value:null,configurable:!0})),u.alloc=function(t,e,n){return function(t,e,n,r){return l(e),e<=0?i(t,e):void 0!==n?"string"==typeof r?i(t,e).fill(n,r):i(t,e).fill(n):i(t,e)}(null,t,e,n)},u.allocUnsafe=function(t){return h(null,t)},u.allocUnsafeSlow=function(t){return h(null,t)},u.isBuffer=function(t){return!(null==t||!t._isBuffer)},u.compare=function(t,e){if(!u.isBuffer(t)||!u.isBuffer(e))throw new TypeError("Arguments must be Buffers");if(t===e)return 0;for(var n=t.length,r=e.length,o=0,s=Math.min(n,r);o<s;++o)if(t[o]!==e[o]){n=t[o],r=e[o];break}return n<r?-1:r<n?1:0},u.isEncoding=function(t){switch(String(t).toLowerCase()){case"hex":case"utf8":case"utf-8":case"ascii":case"latin1":case"binary":case"base64":case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return!0;default:return!1}},u.concat=function(t,e){if(!s(t))throw new TypeError('"list" argument must be an Array of Buffers');if(0===t.length)return u.alloc(0);var n;if(void 0===e)for(e=0,n=0;n<t.length;++n)e+=t[n].length;var r=u.allocUnsafe(e),o=0;for(n=0;n<t.length;++n){var a=t[n];if(!u.isBuffer(a))throw new TypeError('"list" argument must be an Array of Buffers');a.copy(r,o),o+=a.length}return r},u.byteLength=f,u.prototype._isBuffer=!0,u.prototype.swap16=function(){var t=this.length;if(t%2!=0)throw new RangeError("Buffer size must be a multiple of 16-bits");for(var e=0;e<t;e+=2)m(this,e,e+1);return this},u.prototype.swap32=function(){var t=this.length;if(t%4!=0)throw new RangeError("Buffer size must be a multiple of 32-bits");for(var e=0;e<t;e+=4)m(this,e,e+3),m(this,e+1,e+2);return this},u.prototype.swap64=function(){var t=this.length;if(t%8!=0)throw new RangeError("Buffer size must be a multiple of 64-bits");for(var e=0;e<t;e+=8)m(this,e,e+7),m(this,e+1,e+6),m(this,e+2,e+5),m(this,e+3,e+4);return this},u.prototype.toString=function(){var t=0|this.length;return 0===t?"":0===arguments.length?S(this,0,t):g.apply(this,arguments)},u.prototype.equals=function(t){if(!u.isBuffer(t))throw new TypeError("Argument must be a Buffer");return this===t||0===u.compare(this,t)},u.prototype.inspect=function(){var t="",n=e.INSPECT_MAX_BYTES;return this.length>0&&(t=this.toString("hex",0,n).match(/.{2}/g).join(" "),this.length>n&&(t+=" ... ")),"<Buffer "+t+">"},u.prototype.compare=function(t,e,n,r,o){if(!u.isBuffer(t))throw new TypeError("Argument must be a Buffer");if(void 0===e&&(e=0),void 0===n&&(n=t?t.length:0),void 0===r&&(r=0),void 0===o&&(o=this.length),e<0||n>t.length||r<0||o>this.length)throw new RangeError("out of range index");if(r>=o&&e>=n)return 0;if(r>=o)return-1;if(e>=n)return 1;if(this===t)return 0;for(var s=(o>>>=0)-(r>>>=0),a=(n>>>=0)-(e>>>=0),i=Math.min(s,a),c=this.slice(r,o),l=t.slice(e,n),h=0;h<i;++h)if(c[h]!==l[h]){s=c[h],a=l[h];break}return s<a?-1:a<s?1:0},u.prototype.includes=function(t,e,n){return-1!==this.indexOf(t,e,n)},u.prototype.indexOf=function(t,e,n){return b(this,t,e,n,!0)},u.prototype.lastIndexOf=function(t,e,n){return b(this,t,e,n,!1)},u.prototype.write=function(t,e,n,r){if(void 0===e)r="utf8",n=this.length,e=0;else if(void 0===n&&"string"==typeof e)r=e,n=this.length,e=0;else{if(!isFinite(e))throw new Error("Buffer.write(string, encoding, offset[, length]) is no longer supported");e|=0,isFinite(n)?(n|=0,void 0===r&&(r="utf8")):(r=n,n=void 0)}var o=this.length-e;if((void 0===n||n>o)&&(n=o),t.length>0&&(n<0||e<0)||e>this.length)throw new RangeError("Attempt to write outside buffer bounds");r||(r="utf8");for(var s=!1;;)switch(r){case"hex":return y(this,t,e,n);case"utf8":case"utf-8":return v(this,t,e,n);case"ascii":return w(this,t,e,n);case"latin1":case"binary":return C(this,t,e,n);case"base64":return $(this,t,e,n);case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return O(this,t,e,n);default:if(s)throw new TypeError("Unknown encoding: "+r);r=(""+r).toLowerCase(),s=!0}},u.prototype.toJSON=function(){return{type:"Buffer",data:Array.prototype.slice.call(this._arr||this,0)}};function E(t,e,n){var r="";n=Math.min(t.length,n);for(var o=e;o<n;++o)r+=String.fromCharCode(127&t[o]);return r}function R(t,e,n){var r="";n=Math.min(t.length,n);for(var o=e;o<n;++o)r+=String.fromCharCode(t[o]);return r}function A(t,e,n){var r=t.length;(!e||e<0)&&(e=0),(!n||n<0||n>r)&&(n=r);for(var o="",s=e;s<n;++s)o+=P(t[s]);return o}function k(t,e,n){for(var r=t.slice(e,n),o="",s=0;s<r.length;s+=2)o+=String.fromCharCode(r[s]+256*r[s+1]);return o}function T(t,e,n){if(t%1!=0||t<0)throw new RangeError("offset is not uint");if(t+e>n)throw new RangeError("Trying to access beyond buffer length")}function F(t,e,n,r,o,s){if(!u.isBuffer(t))throw new TypeError('"buffer" argument must be a Buffer instance');if(e>o||e<s)throw new RangeError('"value" argument is out of bounds');if(n+r>t.length)throw new RangeError("Index out of range")}function N(t,e,n,r){e<0&&(e=65535+e+1);for(var o=0,s=Math.min(t.length-n,2);o<s;++o)t[n+o]=(e&255<<8*(r?o:1-o))>>>8*(r?o:1-o)}function D(t,e,n,r){e<0&&(e=4294967295+e+1);for(var o=0,s=Math.min(t.length-n,4);o<s;++o)t[n+o]=e>>>8*(r?o:3-o)&255}function _(t,e,n,r,o,s){if(n+r>t.length)throw new RangeError("Index out of range");if(n<0)throw new RangeError("Index out of range")}function B(t,e,n,r,s){return s||_(t,0,n,4),o.write(t,e,n,r,23,4),n+4}function j(t,e,n,r,s){return s||_(t,0,n,8),o.write(t,e,n,r,52,8),n+8}u.prototype.slice=function(t,e){var n,r=this.length;if((t=~~t)<0?(t+=r)<0&&(t=0):t>r&&(t=r),(e=void 0===e?r:~~e)<0?(e+=r)<0&&(e=0):e>r&&(e=r),e<t&&(e=t),u.TYPED_ARRAY_SUPPORT)(n=this.subarray(t,e)).__proto__=u.prototype;else{var o=e-t;n=new u(o,void 0);for(var s=0;s<o;++s)n[s]=this[s+t]}return n},u.prototype.readUIntLE=function(t,e,n){t|=0,e|=0,n||T(t,e,this.length);for(var r=this[t],o=1,s=0;++s<e&&(o*=256);)r+=this[t+s]*o;return r},u.prototype.readUIntBE=function(t,e,n){t|=0,e|=0,n||T(t,e,this.length);for(var r=this[t+--e],o=1;e>0&&(o*=256);)r+=this[t+--e]*o;return r},u.prototype.readUInt8=function(t,e){return e||T(t,1,this.length),this[t]},u.prototype.readUInt16LE=function(t,e){return e||T(t,2,this.length),this[t]|this[t+1]<<8},u.prototype.readUInt16BE=function(t,e){return e||T(t,2,this.length),this[t]<<8|this[t+1]},u.prototype.readUInt32LE=function(t,e){return e||T(t,4,this.length),(this[t]|this[t+1]<<8|this[t+2]<<16)+16777216*this[t+3]},u.prototype.readUInt32BE=function(t,e){return e||T(t,4,this.length),16777216*this[t]+(this[t+1]<<16|this[t+2]<<8|this[t+3])},u.prototype.readIntLE=function(t,e,n){t|=0,e|=0,n||T(t,e,this.length);for(var r=this[t],o=1,s=0;++s<e&&(o*=256);)r+=this[t+s]*o;return r>=(o*=128)&&(r-=Math.pow(2,8*e)),r},u.prototype.readIntBE=function(t,e,n){t|=0,e|=0,n||T(t,e,this.length);for(var r=e,o=1,s=this[t+--r];r>0&&(o*=256);)s+=this[t+--r]*o;return s>=(o*=128)&&(s-=Math.pow(2,8*e)),s},u.prototype.readInt8=function(t,e){return e||T(t,1,this.length),128&this[t]?-1*(255-this[t]+1):this[t]},u.prototype.readInt16LE=function(t,e){e||T(t,2,this.length);var n=this[t]|this[t+1]<<8;return 32768&n?4294901760|n:n},u.prototype.readInt16BE=function(t,e){e||T(t,2,this.length);var n=this[t+1]|this[t]<<8;return 32768&n?4294901760|n:n},u.prototype.readInt32LE=function(t,e){return e||T(t,4,this.length),this[t]|this[t+1]<<8|this[t+2]<<16|this[t+3]<<24},u.prototype.readInt32BE=function(t,e){return e||T(t,4,this.length),this[t]<<24|this[t+1]<<16|this[t+2]<<8|this[t+3]},u.prototype.readFloatLE=function(t,e){return e||T(t,4,this.length),o.read(this,t,!0,23,4)},u.prototype.readFloatBE=function(t,e){return e||T(t,4,this.length),o.read(this,t,!1,23,4)},u.prototype.readDoubleLE=function(t,e){return e||T(t,8,this.length),o.read(this,t,!0,52,8)},u.prototype.readDoubleBE=function(t,e){return e||T(t,8,this.length),o.read(this,t,!1,52,8)},u.prototype.writeUIntLE=function(t,e,n,r){(t=+t,e|=0,n|=0,r)||F(this,t,e,n,Math.pow(2,8*n)-1,0);var o=1,s=0;for(this[e]=255&t;++s<n&&(o*=256);)this[e+s]=t/o&255;return e+n},u.prototype.writeUIntBE=function(t,e,n,r){(t=+t,e|=0,n|=0,r)||F(this,t,e,n,Math.pow(2,8*n)-1,0);var o=n-1,s=1;for(this[e+o]=255&t;--o>=0&&(s*=256);)this[e+o]=t/s&255;return e+n},u.prototype.writeUInt8=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,1,255,0),u.TYPED_ARRAY_SUPPORT||(t=Math.floor(t)),this[e]=255&t,e+1},u.prototype.writeUInt16LE=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,2,65535,0),u.TYPED_ARRAY_SUPPORT?(this[e]=255&t,this[e+1]=t>>>8):N(this,t,e,!0),e+2},u.prototype.writeUInt16BE=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,2,65535,0),u.TYPED_ARRAY_SUPPORT?(this[e]=t>>>8,this[e+1]=255&t):N(this,t,e,!1),e+2},u.prototype.writeUInt32LE=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,4,4294967295,0),u.TYPED_ARRAY_SUPPORT?(this[e+3]=t>>>24,this[e+2]=t>>>16,this[e+1]=t>>>8,this[e]=255&t):D(this,t,e,!0),e+4},u.prototype.writeUInt32BE=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,4,4294967295,0),u.TYPED_ARRAY_SUPPORT?(this[e]=t>>>24,this[e+1]=t>>>16,this[e+2]=t>>>8,this[e+3]=255&t):D(this,t,e,!1),e+4},u.prototype.writeIntLE=function(t,e,n,r){if(t=+t,e|=0,!r){var o=Math.pow(2,8*n-1);F(this,t,e,n,o-1,-o)}var s=0,a=1,i=0;for(this[e]=255&t;++s<n&&(a*=256);)t<0&&0===i&&0!==this[e+s-1]&&(i=1),this[e+s]=(t/a>>0)-i&255;return e+n},u.prototype.writeIntBE=function(t,e,n,r){if(t=+t,e|=0,!r){var o=Math.pow(2,8*n-1);F(this,t,e,n,o-1,-o)}var s=n-1,a=1,i=0;for(this[e+s]=255&t;--s>=0&&(a*=256);)t<0&&0===i&&0!==this[e+s+1]&&(i=1),this[e+s]=(t/a>>0)-i&255;return e+n},u.prototype.writeInt8=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,1,127,-128),u.TYPED_ARRAY_SUPPORT||(t=Math.floor(t)),t<0&&(t=255+t+1),this[e]=255&t,e+1},u.prototype.writeInt16LE=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,2,32767,-32768),u.TYPED_ARRAY_SUPPORT?(this[e]=255&t,this[e+1]=t>>>8):N(this,t,e,!0),e+2},u.prototype.writeInt16BE=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,2,32767,-32768),u.TYPED_ARRAY_SUPPORT?(this[e]=t>>>8,this[e+1]=255&t):N(this,t,e,!1),e+2},u.prototype.writeInt32LE=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,4,2147483647,-2147483648),u.TYPED_ARRAY_SUPPORT?(this[e]=255&t,this[e+1]=t>>>8,this[e+2]=t>>>16,this[e+3]=t>>>24):D(this,t,e,!0),e+4},u.prototype.writeInt32BE=function(t,e,n){return t=+t,e|=0,n||F(this,t,e,4,2147483647,-2147483648),t<0&&(t=4294967295+t+1),u.TYPED_ARRAY_SUPPORT?(this[e]=t>>>24,this[e+1]=t>>>16,this[e+2]=t>>>8,this[e+3]=255&t):D(this,t,e,!1),e+4},u.prototype.writeFloatLE=function(t,e,n){return B(this,t,e,!0,n)},u.prototype.writeFloatBE=function(t,e,n){return B(this,t,e,!1,n)},u.prototype.writeDoubleLE=function(t,e,n){return j(this,t,e,!0,n)},u.prototype.writeDoubleBE=function(t,e,n){return j(this,t,e,!1,n)},u.prototype.copy=function(t,e,n,r){if(n||(n=0),r||0===r||(r=this.length),e>=t.length&&(e=t.length),e||(e=0),r>0&&r<n&&(r=n),r===n)return 0;if(0===t.length||0===this.length)return 0;if(e<0)throw new RangeError("targetStart out of bounds");if(n<0||n>=this.length)throw new RangeError("sourceStart out of bounds");if(r<0)throw new RangeError("sourceEnd out of bounds");r>this.length&&(r=this.length),t.length-e<r-n&&(r=t.length-e+n);var o,s=r-n;if(this===t&&n<e&&e<r)for(o=s-1;o>=0;--o)t[o+e]=this[o+n];else if(s<1e3||!u.TYPED_ARRAY_SUPPORT)for(o=0;o<s;++o)t[o+e]=this[o+n];else Uint8Array.prototype.set.call(t,this.subarray(n,n+s),e);return s},u.prototype.fill=function(t,e,n,r){if("string"==typeof t){if("string"==typeof e?(r=e,e=0,n=this.length):"string"==typeof n&&(r=n,n=this.length),1===t.length){var o=t.charCodeAt(0);o<256&&(t=o)}if(void 0!==r&&"string"!=typeof r)throw new TypeError("encoding must be a string");if("string"==typeof r&&!u.isEncoding(r))throw new TypeError("Unknown encoding: "+r)}else"number"==typeof t&&(t&=255);if(e<0||this.length<e||this.length<n)throw new RangeError("Out of range index");if(n<=e)return this;var s;if(e>>>=0,n=void 0===n?this.length:n>>>0,t||(t=0),"number"==typeof t)for(s=e;s<n;++s)this[s]=t;else{var a=u.isBuffer(t)?t:L(new u(t,r).toString()),i=a.length;for(s=0;s<n-e;++s)this[s+e]=a[s%i]}return this};var M=/[^+\/0-9A-Za-z-_]/g;function P(t){return t<16?"0"+t.toString(16):t.toString(16)}function L(t,e){var n;e=e||1/0;for(var r=t.length,o=null,s=[],a=0;a<r;++a){if((n=t.charCodeAt(a))>55295&&n<57344){if(!o){if(n>56319){(e-=3)>-1&&s.push(239,191,189);continue}if(a+1===r){(e-=3)>-1&&s.push(239,191,189);continue}o=n;continue}if(n<56320){(e-=3)>-1&&s.push(239,191,189),o=n;continue}n=65536+(o-55296<<10|n-56320)}else o&&(e-=3)>-1&&s.push(239,191,189);if(o=null,n<128){if((e-=1)<0)break;s.push(n)}else if(n<2048){if((e-=2)<0)break;s.push(n>>6|192,63&n|128)}else if(n<65536){if((e-=3)<0)break;s.push(n>>12|224,n>>6&63|128,63&n|128)}else{if(!(n<1114112))throw new Error("Invalid code point");if((e-=4)<0)break;s.push(n>>18|240,n>>12&63|128,n>>6&63|128,63&n|128)}}return s}function W(t){return r.toByteArray(function(t){if((t=function(t){return t.trim?t.trim():t.replace(/^\s+|\s+$/g,"")}(t).replace(M,"")).length<2)return"";for(;t.length%4!=0;)t+="=";return t}(t))}function z(t,e,n,r){for(var o=0;o<r&&!(o+n>=e.length||o>=t.length);++o)e[o+n]=t[o];return o}}).call(this,n(22))},function(t,e,n){"use strict";e.byteLength=function(t){var e=c(t),n=e[0],r=e[1];return 3*(n+r)/4-r},e.toByteArray=function(t){var e,n,r=c(t),a=r[0],i=r[1],u=new s(function(t,e,n){return 3*(e+n)/4-n}(0,a,i)),l=0,h=i>0?a-4:a;for(n=0;n<h;n+=4)e=o[t.charCodeAt(n)]<<18|o[t.charCodeAt(n+1)]<<12|o[t.charCodeAt(n+2)]<<6|o[t.charCodeAt(n+3)],u[l++]=e>>16&255,u[l++]=e>>8&255,u[l++]=255&e;2===i&&(e=o[t.charCodeAt(n)]<<2|o[t.charCodeAt(n+1)]>>4,u[l++]=255&e);1===i&&(e=o[t.charCodeAt(n)]<<10|o[t.charCodeAt(n+1)]<<4|o[t.charCodeAt(n+2)]>>2,u[l++]=e>>8&255,u[l++]=255&e);return u},e.fromByteArray=function(t){for(var e,n=t.length,o=n%3,s=[],a=0,i=n-o;a<i;a+=16383)s.push(l(t,a,a+16383>i?i:a+16383));1===o?(e=t[n-1],s.push(r[e>>2]+r[e<<4&63]+"==")):2===o&&(e=(t[n-2]<<8)+t[n-1],s.push(r[e>>10]+r[e>>4&63]+r[e<<2&63]+"="));return s.join("")};for(var r=[],o=[],s="undefined"!=typeof Uint8Array?Uint8Array:Array,a="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",i=0,u=a.length;i<u;++i)r[i]=a[i],o[a.charCodeAt(i)]=i;function c(t){var e=t.length;if(e%4>0)throw new Error("Invalid string. Length must be a multiple of 4");var n=t.indexOf("=");return-1===n&&(n=e),[n,n===e?0:4-n%4]}function l(t,e,n){for(var o,s,a=[],i=e;i<n;i+=3)o=(t[i]<<16&16711680)+(t[i+1]<<8&65280)+(255&t[i+2]),a.push(r[(s=o)>>18&63]+r[s>>12&63]+r[s>>6&63]+r[63&s]);return a.join("")}o["-".charCodeAt(0)]=62,o["_".charCodeAt(0)]=63},function(t,e){e.read=function(t,e,n,r,o){var s,a,i=8*o-r-1,u=(1<<i)-1,c=u>>1,l=-7,h=n?o-1:0,d=n?-1:1,p=t[e+h];for(h+=d,s=p&(1<<-l)-1,p>>=-l,l+=i;l>0;s=256*s+t[e+h],h+=d,l-=8);for(a=s&(1<<-l)-1,s>>=-l,l+=r;l>0;a=256*a+t[e+h],h+=d,l-=8);if(0===s)s=1-c;else{if(s===u)return a?NaN:1/0*(p?-1:1);a+=Math.pow(2,r),s-=c}return(p?-1:1)*a*Math.pow(2,s-r)},e.write=function(t,e,n,r,o,s){var a,i,u,c=8*s-o-1,l=(1<<c)-1,h=l>>1,d=23===o?Math.pow(2,-24)-Math.pow(2,-77):0,p=r?0:s-1,f=r?1:-1,g=e<0||0===e&&1/e<0?1:0;for(e=Math.abs(e),isNaN(e)||e===1/0?(i=isNaN(e)?1:0,a=l):(a=Math.floor(Math.log(e)/Math.LN2),e*(u=Math.pow(2,-a))<1&&(a--,u*=2),(e+=a+h>=1?d/u:d*Math.pow(2,1-h))*u>=2&&(a++,u/=2),a+h>=l?(i=0,a=l):a+h>=1?(i=(e*u-1)*Math.pow(2,o),a+=h):(i=e*Math.pow(2,h-1)*Math.pow(2,o),a=0));o>=8;t[n+p]=255&i,p+=f,i/=256,o-=8);for(a=a<<o|i,c+=o;c>0;t[n+p]=255&a,p+=f,a/=256,c-=8);t[n+p-f]|=128*g}},function(t,e){var n={}.toString;t.exports=Array.isArray||function(t){return"[object Array]"==n.call(t)}},function(t,e,n){"use strict";(function(t){var e=n(6);
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const r=()=>n(98);let o;class s{constructor(){this.util=n(99),this.textEncoder=new this.util.TextEncoder}fetch(t,n){return null!=Object(e.b)().global.fetch?Object(e.b)().global.fetch(t,n):(null==o&&(o=r()),o(t,n))}now(){const e=t.hrtime();return 1e3*e[0]+e[1]/1e6}encode(t,e){if("utf-8"!==e&&"utf8"!==e)throw new Error("Node built-in encoder only supports utf-8, but got "+e);return this.textEncoder.encode(t)}decode(t,e){return 0===t.length?"":new this.util.TextDecoder(e).decode(t)}}Object(e.b)().get("IS_NODE")&&Object(e.b)().setPlatform("node",new s)}).call(this,n(21))},function(t,e){},function(t,e){},function(t,e,n){(function(t){var r;!function(t,o,s){function a(t){var e,n=this,r=(e=4022871197,function(t){t=t.toString();for(var n=0;n<t.length;n++){var r=.02519603282416938*(e+=t.charCodeAt(n));r-=e=r>>>0,e=(r*=e)>>>0,e+=4294967296*(r-=e)}return 2.3283064365386963e-10*(e>>>0)});n.next=function(){var t=2091639*n.s0+2.3283064365386963e-10*n.c;return n.s0=n.s1,n.s1=n.s2,n.s2=t-(n.c=0|t)},n.c=1,n.s0=r(" "),n.s1=r(" "),n.s2=r(" "),n.s0-=r(t),n.s0<0&&(n.s0+=1),n.s1-=r(t),n.s1<0&&(n.s1+=1),n.s2-=r(t),n.s2<0&&(n.s2+=1),r=null}function i(t,e){return e.c=t.c,e.s0=t.s0,e.s1=t.s1,e.s2=t.s2,e}function u(t,e){var n=new a(t),r=e&&e.state,o=n.next;return o.int32=function(){return 4294967296*n.next()|0},o.double=function(){return o()+11102230246251565e-32*(2097152*o()|0)},o.quick=o,r&&("object"==typeof r&&i(r,n),o.state=function(){return i(n,{})}),o}o&&o.exports?o.exports=u:n(10)&&n(16)?void 0===(r=function(){return u}.call(e,n,e,o))||(o.exports=r):this.alea=u}(0,t,n(10))}).call(this,n(15)(t))},function(t,e,n){(function(t){var r;!function(t,o,s){function a(t){var e=this,n="";e.x=0,e.y=0,e.z=0,e.w=0,e.next=function(){var t=e.x^e.x<<11;return e.x=e.y,e.y=e.z,e.z=e.w,e.w^=e.w>>>19^t^t>>>8},t===(0|t)?e.x=t:n+=t;for(var r=0;r<n.length+64;r++)e.x^=0|n.charCodeAt(r),e.next()}function i(t,e){return e.x=t.x,e.y=t.y,e.z=t.z,e.w=t.w,e}function u(t,e){var n=new a(t),r=e&&e.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do{var t=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21)}while(0===t);return t},o.int32=n.next,o.quick=o,r&&("object"==typeof r&&i(r,n),o.state=function(){return i(n,{})}),o}o&&o.exports?o.exports=u:n(10)&&n(16)?void 0===(r=function(){return u}.call(e,n,e,o))||(o.exports=r):this.xor128=u}(0,t,n(10))}).call(this,n(15)(t))},function(t,e,n){(function(t){var r;!function(t,o,s){function a(t){var e=this,n="";e.next=function(){var t=e.x^e.x>>>2;return e.x=e.y,e.y=e.z,e.z=e.w,e.w=e.v,(e.d=e.d+362437|0)+(e.v=e.v^e.v<<4^t^t<<1)|0},e.x=0,e.y=0,e.z=0,e.w=0,e.v=0,t===(0|t)?e.x=t:n+=t;for(var r=0;r<n.length+64;r++)e.x^=0|n.charCodeAt(r),r==n.length&&(e.d=e.x<<10^e.x>>>4),e.next()}function i(t,e){return e.x=t.x,e.y=t.y,e.z=t.z,e.w=t.w,e.v=t.v,e.d=t.d,e}function u(t,e){var n=new a(t),r=e&&e.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do{var t=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21)}while(0===t);return t},o.int32=n.next,o.quick=o,r&&("object"==typeof r&&i(r,n),o.state=function(){return i(n,{})}),o}o&&o.exports?o.exports=u:n(10)&&n(16)?void 0===(r=function(){return u}.call(e,n,e,o))||(o.exports=r):this.xorwow=u}(0,t,n(10))}).call(this,n(15)(t))},function(t,e,n){(function(t){var r;!function(t,o,s){function a(t){var e=this;e.next=function(){var t,n,r=e.x,o=e.i;return t=r[o],n=(t^=t>>>7)^t<<24,n^=(t=r[o+1&7])^t>>>10,n^=(t=r[o+3&7])^t>>>3,n^=(t=r[o+4&7])^t<<7,t=r[o+7&7],n^=(t^=t<<13)^t<<9,r[o]=n,e.i=o+1&7,n},function(t,e){var n,r=[];if(e===(0|e))r[0]=e;else for(e=""+e,n=0;n<e.length;++n)r[7&n]=r[7&n]<<15^e.charCodeAt(n)+r[n+1&7]<<13;for(;r.length<8;)r.push(0);for(n=0;n<8&&0===r[n];++n);for(8==n?r[7]=-1:r[n],t.x=r,t.i=0,n=256;n>0;--n)t.next()}(e,t)}function i(t,e){return e.x=t.x.slice(),e.i=t.i,e}function u(t,e){null==t&&(t=+new Date);var n=new a(t),r=e&&e.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do{var t=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21)}while(0===t);return t},o.int32=n.next,o.quick=o,r&&(r.x&&i(r,n),o.state=function(){return i(n,{})}),o}o&&o.exports?o.exports=u:n(10)&&n(16)?void 0===(r=function(){return u}.call(e,n,e,o))||(o.exports=r):this.xorshift7=u}(0,t,n(10))}).call(this,n(15)(t))},function(t,e,n){(function(t){var r;!function(t,o,s){function a(t){var e=this;e.next=function(){var t,n,r=e.w,o=e.X,s=e.i;return e.w=r=r+1640531527|0,n=o[s+34&127],t=o[s=s+1&127],n^=n<<13,t^=t<<17,n^=n>>>15,t^=t>>>12,n=o[s]=n^t,e.i=s,n+(r^r>>>16)|0},function(t,e){var n,r,o,s,a,i=[],u=128;for(e===(0|e)?(r=e,e=null):(e+="\0",r=0,u=Math.max(u,e.length)),o=0,s=-32;s<u;++s)e&&(r^=e.charCodeAt((s+32)%e.length)),0===s&&(a=r),r^=r<<10,r^=r>>>15,r^=r<<4,r^=r>>>13,s>=0&&(a=a+1640531527|0,o=0==(n=i[127&s]^=r+a)?o+1:0);for(o>=128&&(i[127&(e&&e.length||0)]=-1),o=127,s=512;s>0;--s)r=i[o+34&127],n=i[o=o+1&127],r^=r<<13,n^=n<<17,r^=r>>>15,n^=n>>>12,i[o]=r^n;t.w=a,t.X=i,t.i=o}(e,t)}function i(t,e){return e.i=t.i,e.w=t.w,e.X=t.X.slice(),e}function u(t,e){null==t&&(t=+new Date);var n=new a(t),r=e&&e.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do{var t=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21)}while(0===t);return t},o.int32=n.next,o.quick=o,r&&(r.X&&i(r,n),o.state=function(){return i(n,{})}),o}o&&o.exports?o.exports=u:n(10)&&n(16)?void 0===(r=function(){return u}.call(e,n,e,o))||(o.exports=r):this.xor4096=u}(0,t,n(10))}).call(this,n(15)(t))},function(t,e,n){(function(t){var r;!function(t,o,s){function a(t){var e=this,n="";e.next=function(){var t=e.b,n=e.c,r=e.d,o=e.a;return t=t<<25^t>>>7^n,n=n-r|0,r=r<<24^r>>>8^o,o=o-t|0,e.b=t=t<<20^t>>>12^n,e.c=n=n-r|0,e.d=r<<16^n>>>16^o,e.a=o-t|0},e.a=0,e.b=0,e.c=-1640531527,e.d=1367130551,t===Math.floor(t)?(e.a=t/4294967296|0,e.b=0|t):n+=t;for(var r=0;r<n.length+20;r++)e.b^=0|n.charCodeAt(r),e.next()}function i(t,e){return e.a=t.a,e.b=t.b,e.c=t.c,e.d=t.d,e}function u(t,e){var n=new a(t),r=e&&e.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do{var t=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21)}while(0===t);return t},o.int32=n.next,o.quick=o,r&&("object"==typeof r&&i(r,n),o.state=function(){return i(n,{})}),o}o&&o.exports?o.exports=u:n(10)&&n(16)?void 0===(r=function(){return u}.call(e,n,e,o))||(o.exports=r):this.tychei=u}(0,t,n(10))}).call(this,n(15)(t))},function(t,e,n){var r;!function(o,s){var a,i=this,u=s.pow(256,6),c=s.pow(2,52),l=2*c;function h(t,e,n){var r=[],h=f(function t(e,n){var r,o=[],s=typeof e;if(n&&"object"==s)for(r in e)try{o.push(t(e[r],n-1))}catch(t){}return o.length?o:"string"==s?e:e+"\0"}((e=1==e?{entropy:!0}:e||{}).entropy?[t,g(o)]:null==t?function(){try{var t;return a&&(t=a.randomBytes)?t=t(256):(t=new Uint8Array(256),(i.crypto||i.msCrypto).getRandomValues(t)),g(t)}catch(t){var e=i.navigator,n=e&&e.plugins;return[+new Date,i,n,i.screen,g(o)]}}():t,3),r),m=new d(r),b=function(){for(var t=m.g(6),e=u,n=0;t<c;)t=256*(t+n),e*=256,n=m.g(1);for(;t>=l;)t/=2,e/=2,n>>>=1;return(t+n)/e};return b.int32=function(){return 0|m.g(4)},b.quick=function(){return m.g(4)/4294967296},b.double=b,f(g(m.S),o),(e.pass||n||function(t,e,n,r){return r&&(r.S&&p(r,m),t.state=function(){return p(m,{})}),n?(s.random=t,e):t})(b,h,"global"in e?e.global:this==s,e.state)}function d(t){var e,n=t.length,r=this,o=0,s=r.i=r.j=0,a=r.S=[];for(n||(t=[n++]);o<256;)a[o]=o++;for(o=0;o<256;o++)a[o]=a[s=255&s+t[o%n]+(e=a[o])],a[s]=e;(r.g=function(t){for(var e,n=0,o=r.i,s=r.j,a=r.S;t--;)e=a[o=255&o+1],n=256*n+a[255&(a[o]=a[s=255&s+e])+(a[s]=e)];return r.i=o,r.j=s,n})(256)}function p(t,e){return e.i=t.i,e.j=t.j,e.S=t.S.slice(),e}function f(t,e){for(var n,r=t+"",o=0;o<r.length;)e[255&o]=255&(n^=19*e[255&o])+r.charCodeAt(o++);return g(e)}function g(t){return String.fromCharCode.apply(0,t)}if(s.seedrandom=h,f(s.random(),o),t.exports){t.exports=h;try{a=n(107)}catch(t){}}else void 0===(r=function(){return h}.call(e,n,e,t))||(t.exports=r)}([],Math)},function(t,e){},function(t,e,n){"use strict";n.r(e);var r={};n.r(r),n.d(r,"assertParamsValid",(function(){return X})),n.d(r,"maskToAxes",(function(){return Y})),n.d(r,"computeOutShape",(function(){return Q})),n.d(r,"stridesWithElidedDims",(function(){return Z})),n.d(r,"getNormalizedAxes",(function(){return et})),n.d(r,"startIndicesWithElidedDims",(function(){return nt})),n.d(r,"stopIndicesWithElidedDims",(function(){return rt})),n.d(r,"stridesForAxis",(function(){return ot})),n.d(r,"startForAxis",(function(){return st})),n.d(r,"stopForAxis",(function(){return at})),n.d(r,"isSliceContinous",(function(){return it})),n.d(r,"computeFlatOffset",(function(){return ut})),n.d(r,"parseSliceParams",(function(){return ct}));var o={};n.r(o),n.d(o,"segOpComputeOptimalWindowSize",(function(){return dr})),n.d(o,"computeOutShape",(function(){return pr})),n.d(o,"collectGatherOpShapeInfo",(function(){return fr}));var s={};n.r(s),n.d(s,"axesAreInnerMostDims",(function(){return Pt})),n.d(s,"combineLocations",(function(){return Lt})),n.d(s,"computeOutAndReduceShapes",(function(){return Wt})),n.d(s,"expandShapeToKeepDim",(function(){return zt})),n.d(s,"assertAxesAreInnerMostDims",(function(){return Ut})),n.d(s,"getAxesPermutation",(function(){return Vt})),n.d(s,"getUndoAxesPermutation",(function(){return Gt})),n.d(s,"getInnerMostAxes",(function(){return Ht})),n.d(s,"getBroadcastDims",(function(){return Dt})),n.d(s,"getReductionAxes",(function(){return _t})),n.d(s,"assertAndGetBroadcastShape",(function(){return Bt})),n.d(s,"assertParamsConsistent",(function(){return ce})),n.d(s,"computeOutShape",(function(){return le})),n.d(s,"computeDilation2DInfo",(function(){return ft})),n.d(s,"computePool2DInfo",(function(){return gt})),n.d(s,"computePool3DInfo",(function(){return mt})),n.d(s,"computeConv2DInfo",(function(){return bt})),n.d(s,"computeConv3DInfo",(function(){return xt})),n.d(s,"computeDefaultPad",(function(){return yt})),n.d(s,"tupleValuesAreOne",(function(){return Ot})),n.d(s,"eitherStridesOrDilationsAreOne",(function(){return It})),n.d(s,"convertConv2DDataFormat",(function(){return St})),n.d(s,"getFusedDyActivation",(function(){return Fn})),n.d(s,"getFusedBiasGradient",(function(){return Nn})),n.d(s,"applyActivation",(function(){return Dn})),n.d(s,"shouldFuse",(function(){return _n})),n.d(s,"PARALLELIZE_THRESHOLD",(function(){return Bn})),n.d(s,"computeOptimalWindowSize",(function(){return jn})),n.d(s,"slice_util",(function(){return r})),n.d(s,"upcastType",(function(){return lt.c})),n.d(s,"getImageCenter",(function(){return Mn})),n.d(s,"getReshaped",(function(){return Pn})),n.d(s,"getPermuted",(function(){return Ln})),n.d(s,"getReshapedPermuted",(function(){return Wn})),n.d(s,"getSliceBeginCoords",(function(){return zn})),n.d(s,"getSliceSize",(function(){return Un})),n.d(s,"prepareAndValidate",(function(){return Vn})),n.d(s,"validateUpdateShape",(function(){return Gn})),n.d(s,"validateInput",(function(){return Hn})),n.d(s,"calculateShapes",(function(){return Kn})),n.d(s,"SELU_SCALEALPHA",(function(){return qn})),n.d(s,"SELU_SCALE",(function(){return Xn})),n.d(s,"ERF_P",(function(){return Yn})),n.d(s,"ERF_A1",(function(){return Qn})),n.d(s,"ERF_A2",(function(){return Zn})),n.d(s,"ERF_A3",(function(){return Jn})),n.d(s,"ERF_A4",(function(){return tr})),n.d(s,"ERF_A5",(function(){return er})),n.d(s,"warn",(function(){return nr})),n.d(s,"log",(function(){return rr})),n.d(s,"mergeRealAndImagArrays",(function(){return or})),n.d(s,"splitRealAndImagArrays",(function(){return sr})),n.d(s,"complexWithEvenIndex",(function(){return ar})),n.d(s,"complexWithOddIndex",(function(){return ir})),n.d(s,"getComplexWithIndex",(function(){return ur})),n.d(s,"assignToTypedArray",(function(){return cr})),n.d(s,"exponents",(function(){return lr})),n.d(s,"exponent",(function(){return hr})),n.d(s,"prepareSplitSize",(function(){return xe})),n.d(s,"segment_util",(function(){return o})),n.d(s,"castTensor",(function(){return gr})),n.d(s,"reshapeTensor",(function(){return mr})),n.d(s,"linspaceImpl",(function(){return br}));var a={};n.r(a),n.d(a,"nonMaxSuppressionV3Impl",(function(){return Ne})),n.d(a,"nonMaxSuppressionV4Impl",(function(){return De})),n.d(a,"nonMaxSuppressionV5Impl",(function(){return _e})),n.d(a,"split",(function(){return yr})),n.d(a,"tile",(function(){return vr})),n.d(a,"topkImpl",(function(){return wr})),n.d(a,"whereImpl",(function(){return Cr}));var i,u={};n.r(u),n.d(u,"maxImpl",(function(){return Al})),n.d(u,"transposeImpl",(function(){return kl}));class c{}!function(t){t.float32="float32",t.float16="float16",t.int32="int32",t.uint32="uint32",t["tensor-float32"]="tensor-float32",t["tensor-float16"]="tensor-float16",t["tensor-int32"]="tensor-int32"}(i||(i={}));var l=n(5),h=(n(17),n(6)),d=n(8);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class p{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return null==p.instance&&(p.instance=new p),p.instance}static registerSaveRouter(t){p.getInstance().saveRouters.push(t)}static registerLoadRouter(t){p.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return p.getHandlers(t,"save")}static getLoadHandlers(t,e){return p.getHandlers(t,"load",e)}static getHandlers(t,e,n){const r=[];return("load"===e?p.getInstance().loadRouters:p.getInstance().saveRouters).forEach(e=>{const o=e(t,n);null!==o&&r.push(o)}),r}}function f(){if(!Object(h.b)().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const t="undefined"==typeof window?self:window,e=t.indexedDB||t.mozIndexedDB||t.webkitIndexedDB||t.msIndexedDB||t.shimIndexedDB;if(null==e)throw new Error("The current browser does not appear to support IndexedDB.");return e}function g(t){const e=t.result;e.createObjectStore("models_store",{keyPath:"modelPath"}),e.createObjectStore("model_info_store",{keyPath:"modelPath"})}class m{constructor(t){if(this.indexedDB=f(),null==t||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,e){return new Promise((t,n)=>{const r=this.indexedDB.open("tensorflowjs",1);r.onupgradeneeded=()=>g(r),r.onsuccess=()=>{const o=r.result;if(null==e){const e=o.transaction("models_store","readonly"),r=e.objectStore("models_store").get(this.modelPath);r.onsuccess=()=>{if(null==r.result)return o.close(),n(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));t(r.result.modelArtifacts)},r.onerror=t=>(o.close(),n(r.error)),e.oncomplete=()=>o.close()}else{const r=Object(d.f)(e),s=o.transaction("model_info_store","readwrite");let a=s.objectStore("model_info_store");const i=a.put({modelPath:this.modelPath,modelArtifactsInfo:r});let u;i.onsuccess=()=>{u=o.transaction("models_store","readwrite");const i=u.objectStore("models_store").put({modelPath:this.modelPath,modelArtifacts:e,modelArtifactsInfo:r});i.onsuccess=()=>t({modelArtifactsInfo:r}),i.onerror=t=>{a=s.objectStore("model_info_store");const e=a.delete(this.modelPath);e.onsuccess=()=>(o.close(),n(i.error)),e.onerror=t=>(o.close(),n(i.error))}},i.onerror=t=>(o.close(),n(i.error)),s.oncomplete=()=>{null==u?o.close():u.oncomplete=()=>o.close()}}},r.onerror=t=>n(r.error)})}}m.URL_SCHEME="indexeddb://";const b=t=>{return Object(h.b)().getBool("IS_BROWSER")&&!Array.isArray(t)&&t.startsWith(m.URL_SCHEME)?(e=t.slice(m.URL_SCHEME.length),new m(e)):null;var e};p.registerSaveRouter(b),p.registerLoadRouter(b);class x{constructor(){this.indexedDB=f()}async listModels(){return new Promise((t,e)=>{const n=this.indexedDB.open("tensorflowjs",1);n.onupgradeneeded=()=>g(n),n.onsuccess=()=>{const r=n.result,o=r.transaction("model_info_store","readonly"),s=o.objectStore("model_info_store").getAll();s.onsuccess=()=>{const e={};for(const t of s.result)e[t.modelPath]=t.modelArtifactsInfo;t(e)},s.onerror=t=>(r.close(),e(s.error)),o.oncomplete=()=>r.close()},n.onerror=t=>e(n.error)})}async removeModel(t){var e;return t=(e=t).startsWith(m.URL_SCHEME)?e.slice(m.URL_SCHEME.length):e,new Promise((e,n)=>{const r=this.indexedDB.open("tensorflowjs",1);r.onupgradeneeded=()=>g(r),r.onsuccess=()=>{const o=r.result,s=o.transaction("model_info_store","readwrite"),a=s.objectStore("model_info_store"),i=a.get(t);let u;i.onsuccess=()=>{if(null==i.result)return o.close(),n(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const r=a.delete(t),s=()=>{u=o.transaction("models_store","readwrite");const r=u.objectStore("models_store").delete(t);r.onsuccess=()=>e(i.result.modelArtifactsInfo),r.onerror=t=>n(i.error)};r.onsuccess=s,r.onerror=t=>(s(),o.close(),n(i.error))}},i.onerror=t=>(o.close(),n(i.error)),s.oncomplete=()=>{null==u?o.close():u.oncomplete=()=>o.close()}},r.onerror=t=>n(r.error)})}}var y=n(0);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const v="tensorflowjs_models",w="info",C="model_topology",$="weight_specs",O="weight_data",I="model_metadata";function S(t){return{info:[v,t,w].join("/"),topology:[v,t,C].join("/"),weightSpecs:[v,t,$].join("/"),weightData:[v,t,O].join("/"),modelMetadata:[v,t,I].join("/")}}function E(t){const e=t.split("/");if(e.length<3)throw new Error("Invalid key format: "+t);return e.slice(1,e.length-1).join("/")}class R{constructor(t){if(!Object(h.b)().getBool("IS_BROWSER")||"undefined"==typeof window||void 0===window.localStorage)throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,null==t||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=S(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const e=JSON.stringify(t.modelTopology),n=JSON.stringify(t.weightSpecs),r=Object(d.f)(t);try{return this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,e),this.LS.setItem(this.keys.weightSpecs,n),this.LS.setItem(this.keys.weightData,Object(d.a)(t.weightData)),this.LS.setItem(this.keys.modelMetadata,JSON.stringify({format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,userDefinedMetadata:t.userDefinedMetadata})),{modelArtifactsInfo:r}}catch(t){throw this.LS.removeItem(this.keys.info),this.LS.removeItem(this.keys.topology),this.LS.removeItem(this.keys.weightSpecs),this.LS.removeItem(this.keys.weightData),this.LS.removeItem(this.keys.modelMetadata),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(null==t)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if("JSON"!==t.modelTopologyType)throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const e={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(null==n)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);e.modelTopology=n;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(null==r)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);e.weightSpecs=r;const o=this.LS.getItem(this.keys.modelMetadata);if(null!=o){const t=JSON.parse(o);e.format=t.format,e.generatedBy=t.generatedBy,e.convertedBy=t.convertedBy,e.userDefinedMetadata=t.userDefinedMetadata}const s=this.LS.getItem(this.keys.weightData);if(null==s)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return e.weightData=Object(d.b)(s),e}}R.URL_SCHEME="localstorage://";const A=t=>{return Object(h.b)().getBool("IS_BROWSER")&&!Array.isArray(t)&&t.startsWith(R.URL_SCHEME)?(e=t.slice(R.URL_SCHEME.length),new R(e)):null;var e};p.registerSaveRouter(A),p.registerLoadRouter(A);class k{constructor(){Object(y.assert)(Object(h.b)().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),Object(y.assert)("undefined"==typeof window||void 0!==window.localStorage,()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},e=v+"/",n="/"+w;for(let r=0;r<this.LS.length;++r){const o=this.LS.key(r);if(o.startsWith(e)&&o.endsWith(n)){t[E(o)]=JSON.parse(this.LS.getItem(o))}}return t}async removeModel(t){var e;const n=S(t=(e=t).startsWith(R.URL_SCHEME)?e.slice(R.URL_SCHEME.length):e);if(null==this.LS.getItem(n.info))throw new Error(`Cannot find model at path '${t}'`);const r=JSON.parse(this.LS.getItem(n.info));return this.LS.removeItem(n.info),this.LS.removeItem(n.topology),this.LS.removeItem(n.weightSpecs),this.LS.removeItem(n.weightData),r}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class T{constructor(){this.managers={}}static getInstance(){return null==T.instance&&(T.instance=new T),T.instance}static registerManager(t,e){Object(y.assert)(null!=t,()=>"scheme must not be undefined or null."),t.endsWith("://")&&(t=t.slice(0,t.indexOf("://"))),Object(y.assert)(t.length>0,()=>"scheme must not be an empty string.");const n=T.getInstance();Object(y.assert)(null==n.managers[t],()=>`A model store manager is already registered for scheme '${t}'.`),n.managers[t]=e}static getManager(t){const e=this.getInstance().managers[t];if(null==e)throw new Error(`Cannot find model manager for scheme '${t}'`);return e}static getSchemes(){return Object.keys(this.getInstance().managers)}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class F{fetch(t,e){return fetch(t,e)}now(){return performance.now()}encode(t,e){if("utf-8"!==e&&"utf8"!==e)throw new Error("Browser's encoder only supports utf-8, but got "+e);return null==this.textEncoder&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,e){return new TextDecoder(e).decode(t)}}if(Object(h.b)().get("IS_BROWSER")){Object(h.b)().setPlatform("browser",new F);try{T.registerManager(R.URL_SCHEME,new k)}catch(t){}try{T.registerManager(m.URL_SCHEME,new x)}catch(t){}}n(97);var N=n(4);
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function D(t,e="float32",n){return e=e||"float32",y.assertNonNegativeIntegerDimensions(t),new N.b(t,e,n)}var _=n(1),B=n(2),j=n(3);const M=Object(j.a)({cast_:
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){const n=Object(B.a)(t,"x","cast");if(!y.isValidDtype(e))throw new Error("Failed to cast to unknown dtype "+e);if("string"===e&&"string"!==n.dtype||"string"!==e&&"string"===n.dtype)throw new Error("Only strings can be casted to strings");const r={x:n},o={dtype:e};return l.a.runKernelFunc(t=>t.cast(n,e),r,null,_.v,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const P=Object(j.a)({clone_:function(t){const e=Object(B.a)(t,"x","clone",null),n={x:e};return l.a.runKernelFunc(()=>l.a.makeTensorFromDataId(e.dataId,e.shape,e.dtype),n,null,_.jb)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
Object(l.b)();const L={buffer:D,cast:M,clone:P,print:function(t,e=!1){console.log(t.toString(e))}};Object(N.e)(L);function W(t){return new Promise(t=>setTimeout(t)).then(t)}class z{constructor(t){if(!Object(h.b)().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");t.startsWith(z.URL_SCHEME)&&(t=t.slice(z.URL_SCHEME.length)),null!=t&&0!==t.length||(t="model"),this.modelTopologyFileName=t+".json",this.weightDataFileName=t+".weights.bin"}async save(t){if("undefined"==typeof document)throw new Error("Browser downloads are not supported in this environment since `document` is not present");const e=window.URL.createObjectURL(new Blob([t.weightData],{type:"application/octet-stream"}));if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const n=[{paths:["./"+this.weightDataFileName],weights:t.weightSpecs}],r={modelTopology:t.modelTopology,format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,weightsManifest:n},o=window.URL.createObjectURL(new Blob([JSON.stringify(r)],{type:"application/json"})),s=null==this.jsonAnchor?document.createElement("a"):this.jsonAnchor;if(s.download=this.modelTopologyFileName,s.href=o,await W(()=>s.dispatchEvent(new MouseEvent("click"))),null!=t.weightData){const t=null==this.weightDataAnchor?document.createElement("a"):this.weightDataAnchor;t.download=this.weightDataFileName,t.href=e,await W(()=>t.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:Object(d.f)(t)}}}}z.URL_SCHEME="downloads://";
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function U(t,e,n,r){!function(t){Object(y.assert)(null!=t&&Array.isArray(t)&&t.length>0,()=>"promises must be a none empty array")}(t),function(t,e){Object(y.assert)(t>=0&&t<=1,()=>"Progress fraction must be in range [0, 1], but got startFraction "+t),Object(y.assert)(e>=0&&e<=1,()=>"Progress fraction must be in range [0, 1], but got endFraction "+e),Object(y.assert)(e>=t,()=>`startFraction must be no more than endFraction, but got startFraction ${t} and endFraction `+e)}(n=null==n?0:n,r=null==r?1:r);let o=0;return Promise.all(t.map(s=>(s.then(s=>{const a=n+ ++o/t.length*(r-n);return e(a),s}),s)))}p.registerSaveRouter(t=>Object(h.b)().getBool("IS_BROWSER")&&!Array.isArray(t)&&t.startsWith(z.URL_SCHEME)?function(t="model"){return new z(t)}(t.slice(z.URL_SCHEME.length)):null);n(19);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function V(t,e){null==e&&(e={});const n=null==e.fetchFunc?Object(h.b)().platform.fetch:e.fetchFunc,r=t.map(t=>n(t,e.requestInit,{isBinary:!0})),o=(null==e.onProgress?await Promise.all(r):await U(r,e.onProgress,0,.5)).map(t=>t.arrayBuffer());return null==e.onProgress?await Promise.all(o):await U(o,e.onProgress,.5,1)}class G{constructor(t,e){if(this.DEFAULT_METHOD="POST",null==e&&(e={}),this.weightPathPrefix=e.weightPathPrefix,this.onProgress=e.onProgress,this.weightUrlConverter=e.weightUrlConverter,null!=e.fetchFunc?(Object(y.assert)("function"==typeof e.fetchFunc,()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=e.fetchFunc):this.fetch=Object(h.b)().platform.fetch,Object(y.assert)(null!=t&&t.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(t)&&Object(y.assert)(2===t.length,()=>`URL paths for http must have a length of 2, (actual length is ${t.length}).`),this.path=t,null!=e.requestInit&&null!=e.requestInit.body)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=e.requestInit||{}}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const e=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);e.body=new FormData;const n=[{paths:["./model.weights.bin"],weights:t.weightSpecs}],r={modelTopology:t.modelTopology,format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,userDefinedMetadata:t.userDefinedMetadata,weightsManifest:n};e.body.append("model.json",new Blob([JSON.stringify(r)],{type:"application/json"}),"model.json"),null!=t.weightData&&e.body.append("model.weights.bin",new Blob([t.weightData],{type:"application/octet-stream"}),"model.weights.bin");const o=await this.fetch(this.path,e);if(o.ok)return{modelArtifactsInfo:Object(d.f)(t),responses:[o]};throw new Error("BrowserHTTPRequest.save() failed due to HTTP response status "+o.status+".")}async load(){const t=await this.fetch(this.path,this.requestInit);if(!t.ok)throw new Error(`Request to ${this.path} failed with status code `+t.status+". Please verify this URL points to the model JSON of the model to load.");let e;try{e=await t.json()}catch(t){let e=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?e+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":e+=" Please make sure the server is serving valid JSON for this request.",new Error(e)}const n=e.modelTopology,r=e.weightsManifest,o=e.generatedBy,s=e.convertedBy,a=e.format,i=e.userDefinedMetadata;if(null==n&&null==r)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);let u,c;if(null!=r){const t=await this.loadWeights(r);[u,c]=t}return{modelTopology:n,weightSpecs:u,weightData:c,userDefinedMetadata:i,generatedBy:o,convertedBy:s,format:a}}async loadWeights(t){const e=Array.isArray(this.path)?this.path[1]:this.path,[n,r]=function(t){const e=t.lastIndexOf("/"),n=t.lastIndexOf("?"),r=t.substring(0,e),o=n>e?t.substring(n):"";return[r+"/",o]}(e),o=this.weightPathPrefix||n,s=[];for(const e of t)s.push(...e.weights);const a=[],i=[];for(const e of t)for(const t of e.paths)null!=this.weightUrlConverter?i.push(this.weightUrlConverter(t)):a.push(o+t+r);this.weightUrlConverter&&a.push(...await Promise.all(i));const u=await V(a,{requestInit:this.requestInit,fetchFunc:this.fetch,onProgress:this.onProgress});return[s,Object(d.d)(u)]}}function H(t){return null!=t.match(G.URL_SCHEME_REGEX)}G.URL_SCHEME_REGEX=/^https?:\/\//;const K=(t,e)=>{if("undefined"==typeof fetch&&(null==e||null==e.fetchFunc))return null;{let n=!0;if(n=Array.isArray(t)?t.every(t=>H(t)):H(t),n)return q(t,e)}return null};function q(t,e){return new G(t,e)}p.registerSaveRouter(K),p.registerLoadRouter(K);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function X(t,e,n){y.assert(t.rank===e.length,()=>`Error in slice${t.rank}D: Length of begin ${e} must match the rank of the array (${t.rank}).`),y.assert(t.rank===n.length,()=>`Error in slice${t.rank}D: Length of size ${n} must match the rank of the array (${t.rank}).`);for(let r=0;r<t.rank;++r)y.assert(e[r]+n[r]<=t.shape[r],()=>`Error in slice${t.rank}D: begin[${r}] + size[${r}] (${e[r]+n[r]}) would overflow input.shape[${r}] (${t.shape[r]})`)}function Y(t){const e=[];let n=0;for(;t>0;)1&t&&e.push(n),t/=2,n++;return e}function Q(t,e,n){const r=[];for(let o=0;o<t.length;o++)r[o]=Math.ceil((e[o]-t[o])/n[o]);return r}function Z(t,e,n,r){const o=[...t];for(let t=o.length;t<r.length;t++)o.push(1);for(let t=0;t<n;t++)0===t?o[e]=1:(o.splice(e,0,1),o.pop());return o}function J(t,e,n){return n<=t?n:n-(e-1)}function tt(t,e){const n=[];for(let r=0;r<t;r++)n.push(e+r);return n}function et(t,e,n,r,o,s,a,i,u){const c=t.length;let l=new Array(c),h=new Array(c),d=new Array(c);if(e.length&&n>0){const u=e[0],c=n+1;l=nt(a,u,c,r,t),h=rt(i,u,c,o,t),d=Z(s,u,c,t)}else for(let e=0;e<c;e++)l[e]=st(a,r,s,t,e,u),h[e]=at(i,o,s,t,e,u),d[e]=ot(s,e,u);return{begin:l,end:h,strides:d}}function nt(t,e,n,r,o){const s=[...o],a=tt(n,e);for(let o=0;o<s.length;o++)if(a.indexOf(o)>-1)s[o]=0;else{const a=J(e,n,o);let i=r[a];t&1<<a&&(i=0),s[o]=i}return s}function rt(t,e,n,r,o){const s=[...o],a=tt(n,e);for(let o=0;o<s.length;o++)if(a.indexOf(o)>-1)s[o]=Number.MAX_SAFE_INTEGER;else{const a=J(e,n,o);let i=r[a];t&1<<a&&(i=Number.MAX_SAFE_INTEGER),s[o]=i}for(let t=0;t<s.length;t++){const e=o[t];s[t]<0&&(s[t]+=e),s[t]=y.clamp(0,s[t],o[t])}return s}function ot(t,e,n){let r=t[e];return(n&1<<e||null==r)&&(r=1),r}function st(t,e,n,r,o,s){let a=e[o];const i=n[o]||1;(t&1<<o||s&1<<o||null==a)&&(a=i>0?Number.MIN_SAFE_INTEGER:Number.MAX_SAFE_INTEGER);const u=r[o];return a<0&&(a+=u),a=y.clamp(0,a,u-1),a}function at(t,e,n,r,o,s){let a=e[o];const i=n[o]||1;(t&1<<o||s&1<<o||null==a)&&(a=i>0?Number.MAX_SAFE_INTEGER:Number.MIN_SAFE_INTEGER);const u=r[o];return a<0&&(a+=u),a=i>0?y.clamp(0,a,u):y.clamp(-1,a,u-1),a}function it(t,e,n){let r=n.length;for(let t=0;t<n.length;t++)if(n[t]>1){r=t;break}for(let o=r+1;o<n.length;o++)if(e[o]>0||n[o]!==t[o])return!1;return!0}function ut(t,e){let n=t.length>0?t[t.length-1]:1;for(let r=0;r<t.length-1;r++)n+=t[r]*e[r];return n}function ct(t,e,n){let r,o;return r="number"==typeof e?[e,...new Array(t.rank-1).fill(0)]:e.length<t.rank?e.concat(new Array(t.rank-e.length).fill(0)):e.slice(),r.forEach(t=>{y.assert(-1!==t,()=>"slice() does not support negative begin indexing.")}),o=null==n?new Array(t.rank).fill(-1):"number"==typeof n?[n,...new Array(t.rank-1).fill(-1)]:n.length<t.rank?n.concat(new Array(t.rank-n.length).fill(-1)):n,o=o.map((e,n)=>e>=0?e:(y.assert(-1===e,()=>`Negative size values should be exactly -1 but got ${e} for the slice() size at index ${n}.`),t.shape[n]-r[n])),[r,o]}var lt=n(14),ht=n(7);const dt=Object(j.a)({add_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){let n=Object(B.a)(t,"a","add"),r=Object(B.a)(e,"b","add");[n,r]=Object(ht.b)(n,r);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.add(n,r);return e([n,r]),o},o,null,_.d)}});var pt=n(9);
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ft(t,e,n,r,o="NHWC",s){return bt(t,[...e,t[3]],n,s,r,null,null,St(o))}function gt(t,e,n,r,o,s,a="channelsLast"){const[i,u]=vt(e);let c;if("channelsLast"===a)c=[i,u,t[3],t[3]];else{if("channelsFirst"!==a)throw new Error("Unknown dataFormat "+a);c=[i,u,t[1],t[1]]}return bt(t,c,n,r,o,s,!1,a)}function mt(t,e,n,r,o,s,a="NDHWC"){const[i,u,c]=wt(e);let l,h;if("NDHWC"===a)h="channelsLast",l=[i,u,c,t[4],t[4]];else{if("NCDHW"!==a)throw new Error("Unknown dataFormat "+a);h="channelsFirst",l=[i,u,c,t[1],t[1]]}return xt(t,l,n,r,o,!1,h,s)}function bt(t,e,n,r,o,s,a=!1,i="channelsLast"){let[u,c,l,h]=[-1,-1,-1,-1];if("channelsLast"===i)[u,c,l,h]=t;else{if("channelsFirst"!==i)throw new Error("Unknown dataFormat "+i);[u,h,c,l]=t}const[d,p,,f]=e,[g,m]=vt(n),[b,x]=vt(r),v=Ct(d,b),w=Ct(p,x),{padInfo:C,outHeight:$,outWidth:O}=function(t,e,n,r,o,s,a,i,u){let c,l,h;if("number"==typeof t){c={top:t,bottom:t,left:t,right:t,type:0===t?"VALID":"NUMBER"};const o=function(t,e,n,r,o){null==r&&(r=yt(t,e,n));const s=t[0],a=t[1],i=$t((s-e+2*r)/n+1,o);y.assert(y.isInt(i),()=>`The output # of rows (${i}) must be an integer. Change the stride and/or zero pad parameters`);const u=$t((a-e+2*r)/n+1,o);return y.assert(y.isInt(u),()=>`The output # of columns (${u}) must be an integer. Change the stride and/or zero pad parameters`),[i,u]}([e,n],s,r,t,i);l=o[0],h=o[1]}else if("same"===t){l=Math.ceil(e/r),h=Math.ceil(n/o);const t=Math.max(0,(l-1)*r+s-e),i=Math.max(0,(h-1)*o+a-n),u=Math.floor(t/2),d=t-u,p=Math.floor(i/2);c={top:u,bottom:d,left:p,right:i-p,type:"SAME"}}else if("valid"===t)c={top:0,bottom:0,left:0,right:0,type:"VALID"},l=Math.ceil((e-s+1)/r),h=Math.ceil((n-a+1)/o);else{if("object"!=typeof t)throw Error("Unknown padding parameter: "+t);{const d="channelsLast"===u?t[1][0]:t[2][0],p="channelsLast"===u?t[1][1]:t[2][1],f="channelsLast"===u?t[2][0]:t[3][0],g="channelsLast"===u?t[2][1]:t[3][1];c={top:d,bottom:p,left:f,right:g,type:0===d&&0===p&&0===f&&0===g?"VALID":"EXPLICIT"},l=$t((e-s+d+p)/r+1,i),h=$t((n-a+f+g)/o+1,i)}}return{padInfo:c,outHeight:l,outWidth:h}}(o,c,l,g,m,v,w,s,i),I=a?f*h:f;let S;return"channelsFirst"===i?S=[u,I,$,O]:"channelsLast"===i&&(S=[u,$,O,I]),{batchSize:u,dataFormat:i,inHeight:c,inWidth:l,inChannels:h,outHeight:$,outWidth:O,outChannels:I,padInfo:C,strideHeight:g,strideWidth:m,filterHeight:d,filterWidth:p,effectiveFilterHeight:v,effectiveFilterWidth:w,dilationHeight:b,dilationWidth:x,inShape:t,outShape:S,filterShape:e}}function xt(t,e,n,r,o,s=!1,a="channelsLast",i){let[u,c,l,h,d]=[-1,-1,-1,-1,-1];if("channelsLast"===a)[u,c,l,h,d]=t;else{if("channelsFirst"!==a)throw new Error("Unknown dataFormat "+a);[u,d,c,l,h]=t}const[p,f,g,,m]=e,[b,x,v]=wt(n),[w,C,$]=wt(r),O=Ct(p,w),I=Ct(f,C),S=Ct(g,$),{padInfo:E,outDepth:R,outHeight:A,outWidth:k}=function(t,e,n,r,o,s,a,i,u,c,l){let h,d,p,f;if("number"==typeof t){h={top:t,bottom:t,left:t,right:t,front:t,back:t,type:0===t?"VALID":"NUMBER"};const s=function(t,e,n,r,o,s){null==o&&(o=yt(t,e,r));const a=t[0],i=t[1],u=t[2],c=$t((a-e+2*o)/r+1,s);y.assert(y.isInt(c),()=>`The output # of depths (${c}) must be an integer. Change the stride and/or zero pad parameters`);const l=$t((i-e+2*o)/r+1,s);y.assert(y.isInt(l),()=>`The output # of rows (${l}) must be an integer. Change the stride and/or zero pad parameters`);const h=$t((u-e+2*o)/r+1,s);return y.assert(y.isInt(h),()=>`The output # of columns (${h}) must be an integer. Change the stride and/or zero pad parameters`),[c,l,h,n]}([e,n,r,1],i,1,o,t,l);d=s[0],p=s[1],f=s[2]}else if("same"===t){d=Math.ceil(e/o),p=Math.ceil(n/s),f=Math.ceil(r/a);const t=(d-1)*o+i-e,l=(p-1)*s+u-n,g=(f-1)*a+c-r,m=Math.floor(t/2),b=t-m,x=Math.floor(l/2),y=l-x,v=Math.floor(g/2);h={top:x,bottom:y,left:v,right:g-v,front:m,back:b,type:"SAME"}}else{if("valid"!==t)throw Error("Unknown padding parameter: "+t);h={top:0,bottom:0,left:0,right:0,front:0,back:0,type:"VALID"},d=Math.ceil((e-i+1)/o),p=Math.ceil((n-u+1)/s),f=Math.ceil((r-c+1)/a)}return{padInfo:h,outDepth:d,outHeight:p,outWidth:f}}(o,c,l,h,b,x,v,O,I,S,i),T=s?m*d:m;let F;return"channelsFirst"===a?F=[u,T,R,A,k]:"channelsLast"===a&&(F=[u,R,A,k,T]),{batchSize:u,dataFormat:a,inDepth:c,inHeight:l,inWidth:h,inChannels:d,outDepth:R,outHeight:A,outWidth:k,outChannels:T,padInfo:E,strideDepth:b,strideHeight:x,strideWidth:v,filterDepth:p,filterHeight:f,filterWidth:g,effectiveFilterDepth:O,effectiveFilterHeight:I,effectiveFilterWidth:S,dilationDepth:w,dilationHeight:C,dilationWidth:$,inShape:t,outShape:F,filterShape:e}}function yt(t,e,n,r=1){const o=Ct(e,r);return Math.floor((t[0]*(n-1)-n+o)/2)}function vt(t){return"number"==typeof t?[t,t,t]:2===t.length?[t[0],t[1],1]:t}function wt(t){return"number"==typeof t?[t,t,t]:t}function Ct(t,e){return e<=1?t:t+(t-1)*(e-1)}function $t(t,e){if(!e)return t;switch(e){case"round":return Math.round(t);case"ceil":return Math.ceil(t);case"floor":return Math.floor(t);default:throw new Error("Unknown roundingMode "+e)}}function Ot(t){const[e,n,r]=vt(t);return 1===e&&1===n&&1===r}function It(t,e){return Ot(t)||Ot(e)}function St(t){if("NHWC"===t)return"channelsLast";if("NCHW"===t)return"channelsFirst";throw new Error("Unknown dataFormat "+t)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Et=Object(j.a)({reshape_:function(t,e){const n=Object(B.a)(t,"x","reshape",null);e=y.inferFromImplicitShape(e,n.size),y.assert(n.size===y.sizeFromShape(e),()=>"new shape and old shape must have the same number of elements.");const r={x:n},o={shape:e};return l.a.runKernelFunc((t,r)=>(r([n]),t.reshape(n,e)),r,null,_.Zb,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rt=Object(j.a)({conv2d_:function(t,e,n,r,o="NHWC",s=[1,1],a){const i=Object(B.a)(t,"x","conv2d"),u=Object(B.a)(e,"filter","conv2d");let c=i,h=!1;3===i.rank&&(h=!0,c=Et(i,[1,i.shape[0],i.shape[1],i.shape[2]])),y.assert(4===c.rank,()=>`Error in conv2d: input must be rank 4, but got rank ${c.rank}.`),y.assert(4===u.rank,()=>"Error in conv2d: filter must be rank 4, but got rank "+u.rank+"."),null!=a&&y.assert(y.isInt(r),()=>`Error in conv2d: pad must be an integer when using, dimRoundingMode ${a} but got pad ${r}.`);const d="NHWC"===o?c.shape[3]:c.shape[1];y.assert(d===u.shape[2],()=>`Error in conv2d: depth of input (${d}) must match input depth for filter ${u.shape[2]}.`),y.assert(It(n,s),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${s}'`);const p={x:c,filter:u},f={strides:n,pad:r,dataFormat:o,dilations:s,dimRoundingMode:a},g=l.a.runKernelFunc((t,e)=>{const i=St(o),l=bt(c.shape,u.shape,n,s,r,a,!1,i),h=t.conv2d(c,u,l);return e([c,u]),h},p,null,_.A,f);return h?Et(g,[g.shape[1],g.shape[2],g.shape[3]]):g}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const At=Object(j.a)({depthwiseConv2d_:function(t,e,n,r,o="NHWC",s=[1,1],a){const i=Object(B.a)(t,"x","depthwiseConv2d"),u=Object(B.a)(e,"filter","depthwiseConv2d");let c=i,h=!1;3===i.rank&&(h=!0,c=Et(i,[1,i.shape[0],i.shape[1],i.shape[2]])),y.assert(4===c.rank,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${c.rank}.`),y.assert(4===u.rank,()=>"Error in depthwiseConv2d: filter must be rank 4, but got rank "+u.rank+"."),y.assert(c.shape[3]===u.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${c.shape[3]}) must match the inChannels dimension in filter ${u.shape[2]}.`),null!=a&&y.assert(y.isInt(r),()=>`Error in depthwiseConv2d: pad must be an integer when using, dimRoundingMode ${a} but got pad ${r}.`);const d={x:c,filter:u},p={strides:n,pad:r,dataFormat:o,dilations:s,dimRoundingMode:a},f=l.a.runKernelFunc((t,e)=>{null==s&&(s=[1,1]),y.assert(It(n,s),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${s}'`);const o=bt(c.shape,u.shape,n,s,r,a,!0),i=t.depthwiseConv2D(c,u,o);return e([c,u]),i},d,null,_.L,p);return h?Et(f,[f.shape[1],f.shape[2],f.shape[3]]):f}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kt=Object(j.a)({floorDiv_:function(t,e){let n=Object(B.a)(t,"a","floorDiv"),r=Object(B.a)(e,"b","floorDiv");[n,r]=Object(ht.b)(n,r);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.floorDiv(n,r);return e([n,r]),o},o,null,_.cb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tt=Object(j.a)({div_:function(t,e){let n=Object(B.a)(t,"a","div"),r=Object(B.a)(e,"b","div");if([n,r]=Object(ht.b)(n,r),"int32"===n.dtype&&"int32"===r.dtype)return kt(n,r);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.realDivide(n,r);return e([n,r]),o},o,null,_.R,{})}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ft=Object(j.a)({matMul_:function(t,e,n=!1,r=!1){let o=Object(B.a)(t,"a","matMul"),s=Object(B.a)(e,"b","matMul");[o,s]=Object(ht.b)(o,s),y.assert(o.rank>=2&&s.rank>=2&&o.rank===s.rank,()=>`Error in matMul: inputs must have the same rank of at least 2, got ranks ${o.rank} and ${s.rank}.`);const a=n?o.shape[o.rank-2]:o.shape[o.rank-1],i=r?s.shape[s.rank-1]:s.shape[s.rank-2],u=n?o.shape[o.rank-1]:o.shape[o.rank-2],c=r?s.shape[s.rank-2]:s.shape[s.rank-1],h=o.shape.slice(0,-2),d=s.shape.slice(0,-2),p=y.sizeFromShape(h),f=y.sizeFromShape(d);y.assert(y.arraysEqual(h,d),()=>`Error in matMul: outer dimensions (${h}) and (${d}) of Tensors with shapes ${o.shape} and `+s.shape+" must match."),y.assert(a===i,()=>`Error in matMul: inner shapes (${a}) and (${i}) of Tensors with shapes ${o.shape} and ${s.shape} and transposeA=${n} and transposeB=${r} must match.`);const g=o.shape.slice(0,-2).concat([u,c]),m=Et(o,n?[p,a,u]:[p,u,a]),b=Et(s,r?[f,c,i]:[f,i,c]),x={a:m,b:b},v={transposeA:n,transposeB:r},w=l.a.runKernelFunc((t,e)=>(e([m,b]),t.batchMatMul(m,b,n,r)),x,null,_.s,v);return Et(w,g)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nt=Object(j.a)({dot_:function(t,e){const n=Object(B.a)(t,"t1","dot"),r=Object(B.a)(e,"t2","dot");y.assert(!(1!==n.rank&&2!==n.rank||1!==r.rank&&2!==r.rank),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${r.rank}.`);const o=1===n.rank?n.size:n.shape[1],s=1===r.rank?r.size:r.shape[0];if(y.assert(o===s,()=>`Error in dot: inner dimensions of inputs must match, but got ${o} and ${s}.`),1===n.rank&&1===r.rank){const t=Et(n,[1,-1]),e=Et(r,[-1,1]),o=Ft(t,e);return Et(o,[])}if(1===n.rank&&2===r.rank){const t=Et(n,[1,-1]),e=Et(r,[r.shape[0],r.shape[1]]),o=Ft(t,e);return Et(o,[o.size])}if(2===n.rank&&1===r.rank){const t=Et(r,[-1,1]),e=Ft(n,t);return Et(e,[e.size])}{const t=Et(r,[r.shape[0],r.shape[1]]);return Ft(n,t)}}});
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dt(t,e){const n=t.length,r=[];for(let o=0;o<n;o++){const s=n-1-o,a=t[s]||1;(e[e.length-1-o]||1)>1&&1===a&&r.unshift(s)}return r}function _t(t,e){const n=[];for(let r=0;r<e.length;r++){const o=t[t.length-r-1],s=e.length-r-1,a=e[s];(null==o||1===o&&a>1)&&n.unshift(s)}return n}function Bt(t,e){const n=[],r=Math.max(t.length,e.length);for(let o=0;o<r;o++){let r=t[t.length-o-1];null==r&&(r=1);let s=e[e.length-o-1];if(null==s&&(s=1),1===r)n.unshift(s);else if(1===s)n.unshift(r);else{if(r!==s){throw Error(`Operands could not be broadcast together with shapes ${t} and ${e}.`)}n.unshift(r)}}return n}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jt=Object(j.a)({equal_:function(t,e){let n=Object(B.a)(t,"a","equal"),r=Object(B.a)(e,"b","equal");[n,r]=Object(ht.b)(n,r),Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc(t=>t.equal(n,r),o,null,_.U)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mt=Object(j.a)({imag_:function(t){const e=Object(B.a)(t,"input","imag"),n={input:e};return l.a.runKernelFunc(t=>t.imag(e),n,null,_.kb)}});
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pt(t,e){for(let n=0;n<t.length;++n)if(t[t.length-n-1]!==e-1-n)return!1;return!0}function Lt(t,e,n){const r=t.length+e.length,o=[];let s=0,a=0;for(let i=0;i<r;i++)-1===n.indexOf(i)?o.push(t[s++]):o.push(e[a++]);return o}function Wt(t,e){const n=[],r=t.length;for(let o=0;o<r;o++)-1===e.indexOf(o)&&n.push(t[o]);return[n,e.map(e=>t[e])]}function zt(t,e){return Lt(t,e.map(t=>1),e)}function Ut(t,e,n){y.assert(Pt(e,n),()=>t+" supports only inner-most axes for now. "+`Got axes ${e} and rank-${n} input.`)}function Vt(t,e){if(Pt(t,e))return null;const n=[];for(let r=0;r<e;++r)-1===t.indexOf(r)&&n.push(r);return t.forEach(t=>n.push(t)),n}function Gt(t){return t.map((t,e)=>[e,t]).sort((t,e)=>t[1]-e[1]).map(t=>t[0])}function Ht(t,e){const n=[];for(let r=e-t;r<e;++r)n.push(r);return n}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kt=Object(j.a)({transpose_:function(t,e){const n=Object(B.a)(t,"x","transpose");if(null==e&&(e=n.shape.map((t,e)=>e).reverse()),y.assert(n.rank===e.length,()=>`Error in transpose: rank of input ${n.rank} must match length of perm ${e}.`),e.forEach(t=>{y.assert(t>=0&&t<n.rank,()=>"All entries in 'perm' must be between 0 and "+(n.rank-1)+" but got "+e)}),n.rank<=1)return n.clone();const r={x:n},o={perm:e};return l.a.runKernelFunc(t=>t.transpose(n,e),r,null,_.Ec,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qt=Object(j.a)({max_:function(t,e=null,n=!1){const r=Object(B.a)(t,"x","max"),o={x:r},s={reductionIndices:e,keepDims:n};return l.a.runKernelFunc((t,o)=>{let s=y.parseAxisParam(e,r.shape);const a=Vt(s,r.rank);let i=r;null!=a&&(i=Kt(r,a),s=Ht(s.length,i.rank));const u=t.max(i,s);null!=a&&i.dispose();let c=u;if(n){const t=zt(c.shape,y.parseAxisParam(e,r.shape));c=Et(c,t),u.dispose()}return o([r,c]),c},o,null,_.yb,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xt=Object(j.a)({mul_:function(t,e){let n=Object(B.a)(t,"a","mul"),r=Object(B.a)(e,"b","mul");[n,r]=Object(ht.b)(n,r);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.multiply(n,r);return e([n,r]),o},o,null,_.Ib)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yt=Object(j.a)({avgPool_:function(t,e,n,r,o){const s=Object(B.a)(t,"x","avgPool","float32");y.assert(It(n,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`);let a=s,i=!1;3===s.rank&&(i=!0,a=Et(s,[1,s.shape[0],s.shape[1],s.shape[2]])),y.assert(4===a.rank,()=>`Error in avgPool: x must be rank 4 but got rank ${a.rank}.`),null!=o&&y.assert(y.isInt(r),()=>`Error in avgPool: pad must be an integer when using, dimRoundingMode ${o} but got pad ${r}.`);const u={x:a},c={filterSize:e,strides:n,pad:r,dimRoundingMode:o};let h=l.a.runKernelFunc((t,s)=>{const i=gt(a.shape,e,n,1,r,o);return s([a]),1===i.filterWidth&&1===i.filterHeight&&y.arraysEqual(i.inShape,i.outShape)?a.clone():t.avgPool(a,i)},u,null,_.o,c);return h=M(h,s.dtype),i?Et(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qt=Object(j.a)({batchToSpaceND_:function(t,e,n){const r=Object(B.a)(t,"x","batchToSpaceND"),o=e.reduce((t,e)=>t*e);y.assert(r.rank>=1+e.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${e.length}`),y.assert(n.length===e.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${e.length}`),y.assert(r.shape[0]%o==0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${e.join(" * ")} === ${o}`);const s={x:r},a={blockShape:e,crops:n};return l.a.runKernelFunc(t=>t.batchToSpaceND(r,e,n),s,null,_.t,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zt=Object(j.a)({maxPool_:function(t,e,n,r,o){const s=Object(B.a)(t,"x","maxPool");let a=s,i=!1;3===s.rank&&(i=!0,a=Et(s,[1,s.shape[0],s.shape[1],s.shape[2]])),y.assert(4===a.rank,()=>`Error in maxPool: input must be rank 4 but got rank ${a.rank}.`),y.assert(It(n,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`),null!=o&&y.assert(y.isInt(r),()=>`Error in maxPool: pad must be an integer when using, dimRoundingMode ${o} but got pad ${r}.`);const u={x:a},c={filterSize:e,strides:n,pad:r,dimRoundingMode:o},h=l.a.runKernelFunc((t,s)=>{const i=gt(a.shape,e,n,1,r,o);let u;return u=1===i.filterWidth&&1===i.filterHeight&&y.arraysEqual(i.inShape,i.outShape)?a.clone():t.maxPool(a,i),s([a,u]),u},u,null,_.zb,c);return i?Et(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jt=Object(j.a)({spaceToBatchND_:function(t,e,n){const r=Object(B.a)(t,"x","spaceToBatchND");y.assert(r.rank>=1+e.length,()=>`input rank ${r.rank} should be > than [blockShape] ${e.length}`),y.assert(n.length===e.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${e.length}`),y.assert(r.shape.reduce((t,r,o)=>o>0&&o<=e.length?t&&(r+n[o-1][0]+n[o-1][1])%e[o-1]==0:t,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${e.toString()}`);const o={x:r},s={blockShape:e,paddings:n};return l.a.runKernelFunc(t=>t.spaceToBatchND(r,e,n),o,null,_.rc,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const te=Object(j.a)({pool_:function(t,e,n,r,o,s){null==o&&(o=[1,1]),null==s&&(s=1),0===r&&(r="valid");const a=Object(B.a)(t,"x","maxPool");let i=a,u=!1;3===a.rank&&(u=!0,i=Et(a,[1,a.shape[0],a.shape[1],a.shape[2]])),y.assert(It(s,o),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${s} and dilations '${o}'`);const c=gt(i.shape,e,s,o,r),l=[c.dilationHeight,c.dilationWidth];let h;h="same"===r?function(t,e){const n=t.map((t,n)=>t+(t-1)*(e[n]-1)).map(t=>t-1),r=n.map(t=>Math.floor(t/2)),o=n.map((t,e)=>t-r[e]);return n.map((t,e)=>[r[e],o[e]])}([c.filterHeight,c.filterWidth],l):[[0,0],[0,0]];const d=1===l[0]&&1===l[1],[p,f]=function(t,e,n){const r=n.map(t=>t[0]),o=n.map(t=>t[1]),s=t.concat(r,o),a=e.map((t,e)=>(t-s[e]%t)%t),i=o.map((t,e)=>t+a[e]),u=e.map((t,e)=>[r[e],i[e]]),c=e.map((t,e)=>[0,a[e]]);return[u,c]}([c.inHeight,c.inWidth],l,h),g=d?r:"valid",m=d?i:Jt(i,l,p),b=("avg"===n?()=>Yt(m,e,s,g):()=>Zt(m,e,s,g))(),x=d?b:Qt(b,l,f);return u?Et(x,[x.shape[1],x.shape[2],x.shape[3]]):x}});var ee=n(12);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ne(t,e){Object(y.assertNonNull)(t);const n=Object(B.c)(t,e);if(1!==n.length)throw new Error("tensor1d() requires values to be a flat/TypedArray");return Object(ee.a)(t,null,n,e)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function re(t,e="float32"){if("complex64"===e){const e=re(t,"float32"),n=re(t,"float32");return Object(pt.a)(e,n)}const n=Object(y.makeZerosTypedArray)(Object(y.sizeFromShape)(t),e);return l.a.makeTensor(n,t,e)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oe(t,e,n=1,r="float32"){if(0===n)throw new Error("Cannot have a step of zero");const o={start:t,stop:e,step:n,dtype:r};return l.a.runKernelFunc(()=>{if(t===e||t<e&&n<0||e<t&&n>1)return re([0],r);const o=Math.abs(Math.ceil((e-t)/n)),s=Object(y.makeZerosTypedArray)(o,r);e<t&&1===n&&(n=-1),s[0]=t;for(let t=1;t<s.length;t++)s[t]=s[t-1]+n;return ne(s,r)},{},null,_.Ub,o)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const se=Object(j.a)({real_:function(t){const e=Object(B.a)(t,"input","real"),n={input:e};return l.a.runKernelFunc(t=>t.real(e),n,null,_.Vb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ae=Object(j.a)({relu_:function(t){const e=Object(B.a)(t,"x","relu"),n={x:e};return l.a.runKernelFunc((t,n)=>(n([e]),"bool"===e.dtype?M(e,"int32"):t.relu(e)),n,null,_.Xb)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ie(t,e){if((Object(y.isTypedArray)(t)&&"string"!==e||Array.isArray(t))&&"complex64"!==e)throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if("string"===e&&Object(y.isTypedArray)(t)&&!(t instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return Object(ee.a)(t,[],[],e)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ue=Object(j.a)({softmax_:function(t,e=-1){const n=Object(B.a)(t,"logits","softmax","float32");if(-1===e&&(e=n.rank-1),e!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${e}`);const r={logits:n},o={dim:e};return l.a.runKernelFunc((t,r)=>{const o=t.softmax(n,e);return r([o]),o},r,null,_.pc,o)}});
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ce(t,e){const n=t[0].length;t.forEach((t,e)=>{y.assert(t.length===n,()=>`Error in concat${n}D: rank of tensors[${e}] must be the same as the rank of the rest (${n})`)}),y.assert(e>=0&&e<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);const r=t[0];t.forEach((t,o)=>{for(let s=0;s<n;s++)y.assert(s===e||t[s]===r[s],()=>`Error in concat${n}D: Shape of tensors[${o}] (${t}) does not match the shape of the rest (${r}) along the non-concatenated axis ${o}.`)})}function le(t,e){const n=t[0].slice();for(let r=1;r<t.length;r++)n[e]+=t[r][e];return n}var he=n(11);
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const de=Object(j.a)({concat_:function(t,e=0){Object(y.assert)(t.length>=1,()=>"Pass at least one tensor to concat");let n=Object(B.b)(t,"tensors","concat");"complex64"===n[0].dtype&&n.forEach(t=>{if("complex64"!==t.dtype)throw new Error(`Cannot concatenate complex64 tensors with a tensor\n          with dtype ${t.dtype}. `)});const r=Object(y.parseAxisParam)(e,n[0].shape)[0],o=le(n.map(t=>t.shape),r);if(0===Object(y.sizeFromShape)(o))return Object(he.a)([],o);if(n=n.filter(t=>t.size>0),1===n.length)return n[0];ce(n.map(t=>t.shape),r);const s=n,a={axis:e};return l.a.runKernelFunc((t,e)=>{const o=t.concat(n,r);return e(n),o},s,null,_.z,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pe=Object(j.a)({expandDims_:function(t,e=0){const n=Object(B.a)(t,"x","expandDims",null);y.assert(e<=n.rank,()=>"Axis must be <= rank of the tensor");const r=n.shape.slice();return e<0&&(y.assert(-(n.rank+1)<=e,()=>`Axis must be in the interval [${-(n.rank+1)}, ${n.rank}]`),e=n.rank+e+1),r.splice(e,0,1),Et(n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fe=Object(j.a)({stack_:function(t,e=0){const n=Object(B.b)(t,"tensors","stack");if(y.assert(n.length>=1,()=>"Pass at least one tensor to tf.stack"),1===n.length)return pe(n[0],e);const r=n[0].rank,o=n[0].shape,s=n[0].dtype;y.assert(e<=r,()=>"Axis must be <= rank of the tensor"),n.forEach(t=>{y.assertShapesMatch(o,t.shape,"All tensors passed to stack must have matching shapes"),y.assert(s===t.dtype,()=>"All tensors passed to stack must have matching dtypes")});const a=n.map(t=>pe(t,e));return de(a,e)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ge(t,e,n){if(Object(y.assertNonNull)(t),null!=e&&2!==e.length)throw new Error("tensor2d() requires shape to have two numbers");const r=Object(B.c)(t,n);if(2!==r.length&&1!==r.length)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(1===r.length&&null==e)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return Object(ee.a)(t,e,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function me(t,e,n){if(Object(y.assertNonNull)(t),null!=e&&4!==e.length)throw new Error("tensor4d() requires shape to have four numbers");const r=Object(B.c)(t,n);if(4!==r.length&&1!==r.length)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(1===r.length&&null==e)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return Object(ee.a)(t,e,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const be=Object(j.a)({slice_:function(t,e,n){const r=Object(B.a)(t,"x","slice");if(0===r.rank)throw new Error("Slicing scalar is not possible");const[o,s]=ct(r,e,n);X(r,o,s);const a={x:r},i={begin:e,size:n};return l.a.runKernelFunc((t,e)=>(e([r]),t.slice(r,o,s)),a,null,_.oc,i)}});function xe(t,e,n=0){let r=[];if("number"==typeof e)Object(y.assert)(t.shape[n]%e==0,()=>"Number of splits must evenly divide the axis."),r=new Array(e).fill(t.shape[n]/e);else{const o=e.reduce((t,e)=>(-1===e&&(t+=1),t),0);Object(y.assert)(o<=1,()=>"There should be only one negative value in split array.");const s=e.indexOf(-1);if(-1!==s){const r=e.reduce((t,e)=>e>0?t+e:t);e[s]=t.shape[n]-r}Object(y.assert)(t.shape[n]===e.reduce((t,e)=>t+e),()=>"The sum of sizes must match the size of the axis dimension."),r=e}return r}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ye=Object(j.a)({split_:function(t,e,n=0){const r=Object(B.a)(t,"x","split"),o={x:r},s={numOrSizeSplits:e,axis:n};return l.a.runKernelFunc((t,o)=>{const s=Object(y.parseAxisParam)(n,r.shape)[0],a=xe(r,e,s);return t.split(r,a,s)},o,null,_.sc,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ve=Object(j.a)({zerosLike_:function(t){const e=Object(B.a)(t,"x","zerosLike"),n={x:e};return l.a.runKernelFunc(t=>t.zerosLike(e),n,null,_.Hc)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const we=Object(j.a)({fft_:function(t){Object(y.assert)("complex64"===t.dtype,()=>`The dtype for tf.spectral.fft() must be complex64 but got ${t.dtype}.`);const e={input:t};return l.a.runKernelFunc(e=>{const n=t.shape[t.shape.length-1],r=t.size/n,o=t.as2D(r,n);return e.fft(o).reshape(t.shape)},e,null,_.Y)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ce=Object(j.a)({rfft_:function(t,e){Object(y.assert)("float32"===t.dtype,()=>"The dtype for rfft() must be real value but got "+t.dtype);let n=t.shape[t.shape.length-1];const r=t.size/n;let o;if(null!=e&&e<n){const r=t.shape.map(t=>0),s=t.shape.map(t=>t);s[t.shape.length-1]=e,o=be(t,r,s),n=e}else if(null!=e&&e>n){const r=t.shape.map(t=>t);r[t.shape.length-1]=e-n,o=de([t,re(r)],t.shape.length-1),n=e}else o=t;const s=ve(o),a=Et(Object(pt.a)(o,s),[r,n]),i=we(a),u=Math.floor(n/2)+1,c=se(i),l=Mt(i),h=ye(c,[u,n-u],c.shape.length-1),d=ye(l,[u,n-u],l.shape.length-1),p=o.shape.slice();return p[o.shape.length-1]=u,Et(Object(pt.a)(h[0],d[0]),p)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $e=Object(j.a)({ifft_:function(t){Object(y.assert)("complex64"===t.dtype,()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${t.dtype}.`);const e={input:t};return l.a.runKernelFunc(e=>{const n=t.shape[t.shape.length-1],r=t.size/n,o=Et(t,[r,n]),s=e.ifft(o);return Et(s,t.shape)},e,null,_.ib)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Oe=Object(j.a)({reverse_:function(t,e){const n=Object(B.a)(t,"x","reverse"),r={x:n},o={dims:e};return l.a.runKernelFunc(t=>{const r=Object(y.parseAxisParam)(e,n.shape);if(0===n.rank)return P(n);const o=t.reverse(n,r);return Et(o,n.shape)},r,null,_.ec,o)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ie=Object(j.a)({irfft_:function(t){const e=t.shape[t.shape.length-1],n=t.size/e;let r;if(e<=2){const o=Et(t,[n,e]);r=$e(o)}else{const o=[n,2*(e-1)],s=Et(se(t),[n,e]),a=Et(Mt(t),[n,e]),i=Oe(be(s,[0,1],[n,e-2]),1),u=Xt(Oe(be(a,[0,1],[n,e-2]),1),ie(-1)),c=de([s,i],1),l=de([a,u],1),h=Et(Object(pt.a)(c,l),[o[0],o[1]]);r=$e(h)}if(r=se(r),3===t.rank&&0!==t.shape[0]){const e=r,n=t.shape[0];r=Et(r,[n,r.shape[0]/n,r.shape[1]]),e.dispose()}return r}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Se(t,e,n){const r=1-t%2,o=new Float32Array(t);for(let s=0;s<t;++s){const a=2*Math.PI*s/(t+r-1);o[s]=e-n*Math.cos(a)}return ne(o,"float32")}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({hammingWindow_:function(t){return Se(t,.54,.46)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ee=Object(j.a)({hannWindow_:function(t){return Se(t,.5,.5)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Re(t,e,n){const r={shape:t,value:e,dtype:n};return l.a.runKernelFunc(r=>r.fill(t,e,n),{},null,_.Z,r)}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ae=Object(j.a)({frame_:function(t,e,n,r=!1,o=0){let s=0;const a=[];for(;s+e<=t.size;)a.push(be(t,s,e)),s+=n;if(r)for(;s<t.size;){const r=s+e-t.size,i=de([be(t,s,e-r),Re([r],o)]);a.push(i),s+=n}return 0===a.length?ge([],[0,e]):Et(de(a),[a.length,e])}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({stft_:function(t,e,n,r,o=Ee){var s;null==r&&(s=e,r=Math.floor(Math.pow(2,Math.ceil(Math.log(s)/Math.log(2)))));const a=Ae(t,e,n),i=Xt(a,o(e)),u=[];for(let t=0;t<a.shape[0];t++)u.push(Ce(be(i,[t,0],[1,e]),r));return de(u)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({cropAndResize_:function(t,e,n,r,o,s){const a=Object(B.a)(t,"image","cropAndResize"),i=Object(B.a)(e,"boxes","cropAndResize","float32"),u=Object(B.a)(n,"boxInd","cropAndResize","int32");o=o||"bilinear",s=s||0;const c=i.shape[0];y.assert(4===a.rank,()=>`Error in cropAndResize: image must be rank 4,but got rank ${a.rank}.`),y.assert(2===i.rank&&4===i.shape[1],()=>`Error in cropAndResize: boxes must be have size [${c},4] but had shape ${i.shape}.`),y.assert(1===u.rank&&u.shape[0]===c,()=>`Error in cropAndResize: boxInd must be have size [${c}] but had shape ${i.shape}.`),y.assert(2===r.length,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),y.assert(r[0]>=1&&r[1]>=1,()=>"cropSize must be atleast [1,1], but was "+r),y.assert("bilinear"===o||"nearest"===o,()=>"method must be bilinear or nearest, but was "+o);const h={image:a,boxes:i,boxInd:u},d={method:o,extrapolationValue:s,cropSize:r};return l.a.runKernelFunc(t=>t.cropAndResize(a,i,u,r,o,s),h,null,_.I,d)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({flipLeftRight_:function(t){const e=Object(B.a)(t,"image","flipLeftRight","float32");y.assert(4===e.rank,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${e.rank}.`);const n={image:e};return l.a.runKernel(_.ab,n,{})}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({rotateWithOffset_:function(t,e,n=0,r=.5){const o=Object(B.a)(t,"image","rotateWithOffset","float32");y.assert(4===o.rank,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${o.rank}.`);const s={image:o},a={radians:e,fillValue:n,center:r};return l.a.runKernel(_.fc,s,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ke(t,e,n,r,o,s){null==r&&(r=.5),null==o&&(o=Number.NEGATIVE_INFINITY),null==s&&(s=0);const a=t.shape[0];return n=Math.min(n,a),y.assert(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),y.assert(2===t.rank,()=>`boxes must be a 2D tensor, but was of rank '${t.rank}'`),y.assert(4===t.shape[1],()=>"boxes must have 4 columns, but 2nd dimension was "+t.shape[1]),y.assert(1===e.rank,()=>"scores must be a 1D tensor"),y.assert(e.shape[0]===a,()=>`scores has incompatible shape with boxes. Expected ${a}, but was `+e.shape[0]),y.assert(0<=s&&s<=1,()=>`softNmsSigma must be in [0, 1], but was '${s}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:o,softNmsSigma:s}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({nonMaxSuppression_:function(t,e,n,r=.5,o=Number.NEGATIVE_INFINITY){const s=Object(B.a)(t,"boxes","nonMaxSuppression"),a=Object(B.a)(e,"scores","nonMaxSuppression"),i=ke(s,a,n,r,o);n=i.maxOutputSize,r=i.iouThreshold,o=i.scoreThreshold;const u={maxOutputSize:n,iouThreshold:r,scoreThreshold:o};return l.a.runKernelFunc(t=>t.nonMaxSuppression(s,a,n,r,o),{boxes:s,scores:a},null,_.Kb,u)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Te(t,e,n){const r=function(t,e,n){return function(t,e,n){let r=0,o=t.length,s=0,a=!1;for(;r<o;){s=r+(o-r>>>1);const i=n(e,t[s]);i>0?r=s+1:(o=s,a=!i)}return a?r:-r-1}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(t,e,n||Fe)}(t,e,n),o=r<0?-(r+1):r;t.splice(o,0,e)}function Fe(t,e){return t>e?1:t<e?-1:0}function Ne(t,e,n,r,o){return Be(t,e,n,r,o,0).selectedIndices}function De(t,e,n,r,o,s){return Be(t,e,n,r,o,0,!1,s,!0)}function _e(t,e,n,r,o,s){return Be(t,e,n,r,o,s,!0)}function Be(t,e,n,r,o,s,a=!1,i=!1,u=!1){const c=[];for(let t=0;t<e.length;t++)e[t]>o&&c.push({score:e[t],boxIndex:t,suppressBeginIndex:0});c.sort(Pe);const l=s>0?-.5/s:0,h=[],d=[];for(;h.length<n&&c.length>0;){const e=c.pop(),{score:n,boxIndex:s,suppressBeginIndex:a}=e;if(n<o)break;let i=!1;for(let n=h.length-1;n>=a;--n){const a=je(t,s,h[n]);if(a>=r){i=!0;break}if(e.score=e.score*Me(r,l,a),e.score<=o)break}e.suppressBeginIndex=h.length,i||(e.score===n?(h.push(s),d.push(e.score)):e.score>o&&Te(c,e,Pe))}const p=h.length,f=n-p;i&&f>0&&(h.push(...new Array(f).fill(0)),d.push(...new Array(f).fill(0)));const g={selectedIndices:ne(h,"int32")};return a&&(g.selectedScores=ne(d,"float32")),u&&(g.validOutputs=ie(p,"int32")),g}function je(t,e,n){const r=t.subarray(4*e,4*e+4),o=t.subarray(4*n,4*n+4),s=Math.min(r[0],r[2]),a=Math.min(r[1],r[3]),i=Math.max(r[0],r[2]),u=Math.max(r[1],r[3]),c=Math.min(o[0],o[2]),l=Math.min(o[1],o[3]),h=Math.max(o[0],o[2]),d=Math.max(o[1],o[3]),p=(i-s)*(u-a),f=(h-c)*(d-l);if(p<=0||f<=0)return 0;const g=Math.max(s,c),m=Math.max(a,l),b=Math.min(i,h),x=Math.min(u,d),y=Math.max(b-g,0)*Math.max(x-m,0);return y/(p+f-y)}function Me(t,e,n){const r=Math.exp(e*n*n);return n<=t?r:0}function Pe(t,e){return t.score-e.score||t.score===e.score&&e.boxIndex-t.boxIndex}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({nonMaxSuppressionWithScore_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r=.5,o=Number.NEGATIVE_INFINITY,s=0){const a=Object(B.a)(t,"boxes","nonMaxSuppression"),i=Object(B.a)(e,"scores","nonMaxSuppression"),u=ke(a,i,n,r,o,s),c={boxes:a,scores:i},h={maxOutputSize:n=u.maxOutputSize,iouThreshold:r=u.iouThreshold,scoreThreshold:o=u.scoreThreshold,softNmsSigma:s=u.softNmsSigma},d=l.a.runKernel(_.Mb,c,h);return{selectedIndices:d[0],selectedScores:d[1]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({nonMaxSuppressionPadded_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r=.5,o=Number.NEGATIVE_INFINITY,s=!1){const a=Object(B.a)(t,"boxes","nonMaxSuppression"),i=Object(B.a)(e,"scores","nonMaxSuppression"),u=ke(a,i,n,r,o,null),c={boxes:a,scores:i},h={maxOutputSize:u.maxOutputSize,iouThreshold:u.iouThreshold,scoreThreshold:u.scoreThreshold,padToMaxOutputSize:s},d=l.a.runKernel(_.Lb,c,h);return{selectedIndices:d[0],validOutputs:d[1]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Le=Object(j.a)({resizeBilinear_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n=!1){const r=Object(B.a)(t,"images","resizeBilinear");y.assert(3===r.rank||4===r.rank,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${r.rank}.`),y.assert(2===e.length,()=>"Error in resizeBilinear: new shape must 2D, but got shape "+e+".");let o=r,s=!1;3===r.rank&&(s=!0,o=Et(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const[a,i]=e,u={images:o},c={alignCorners:n,size:e},h=l.a.runKernelFunc((t,e)=>(e([o]),t.resizeBilinear(o,a,i,n)),u,null,_.ac,c);return s?Et(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const We=Object(j.a)({resizeNearestNeighbor_:function(t,e,n=!1){const r=Object(B.a)(t,"images","resizeNearestNeighbor");y.assert(3===r.rank||4===r.rank,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${r.rank}.`),y.assert(2===e.length,()=>"Error in resizeNearestNeighbor: new shape must 2D, but got shape "+e+"."),y.assert("float32"===r.dtype||"int32"===r.dtype,()=>"`images` must have `int32` or `float32` as dtype");let o=r,s=!1;3===r.rank&&(s=!0,o=Et(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const[a,i]=e,u={images:o},c={alignCorners:n,size:e},h=l.a.runKernelFunc((t,e)=>(e([o]),t.resizeNearestNeighbor(o,a,i,n)),u,null,_.cc,c);return s?Et(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ze=Object(j.a)({greaterEqual_:function(t,e){let n=Object(B.a)(t,"a","greaterEqual"),r=Object(B.a)(e,"b","greaterEqual");[n,r]=Object(ht.b)(n,r),Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.greaterEqual(n,r);return e([n,r]),o},o,null,_.hb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ue=Object(j.a)({lessEqual_:function(t,e){let n=Object(B.a)(t,"a","lessEqual"),r=Object(B.a)(e,"b","lessEqual");[n,r]=Object(ht.b)(n,r),Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.lessEqual(n,r);return e([n,r]),o},o,null,_.rb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ve=Object(j.a)({logicalAnd_:function(t,e){const n=Object(B.a)(t,"a","logicalAnd","bool"),r=Object(B.a)(e,"b","logicalAnd","bool");Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc(t=>t.logicalAnd(n,r),o,null,_.vb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ge=Object(j.a)({sub_:function(t,e){let n=Object(B.a)(t,"a","sub"),r=Object(B.a)(e,"b","sub");[n,r]=Object(ht.b)(n,r);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.subtract(n,r);return e([n,r]),o},o,null,_.yc)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const He=Object(j.a)({unstack_:function(t,e=0){const n=Object(B.a)(t,"x","unstack");y.assert(e>=-n.shape.length&&e<n.shape.length,()=>`Axis = ${e} is not in [-${n.shape.length}, ${n.shape.length})`),e<0&&(e+=n.shape.length);const r={value:n},o={axis:e};return l.a.runKernelFunc(t=>t.unstack(n,e),r,null,_.Fc,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ke=Object(j.a)({broadcastTo_:function(t,e){let n=Object(B.a)(t,"broadcastTo","x");const r=n.shape;if(e.some(t=>!(t>0)||t%1!=0))throw new Error(`broadcastTo(): Invalid broadcast shape [${e}].`);if(e.length<n.rank)throw new Error(`broadcastTo(): shape.length=${e.length} < input.rank=${n.rank}.`);if(e.length>n.rank){const t=n.shape.slice();for(;t.length<e.length;)t.unshift(1);n=Et(n,t)}const o=n.shape,s=Array.from(e);for(let t=e.length-1;t>=0;t--)if(o[t]===e[t])s[t]=1;else if(1!==n.shape[t])throw new Error(`broadcastTo(): [${r}] cannot be broadcast to [${e}].`);if(0===s.map((t,e)=>t>1?e:-1).filter(t=>t>=0).length)return P(n);const a={x:n},i={shape:e,inputShape:o};return l.a.runKernelFunc(t=>t.tile(n,s),a,null,_.u,i)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qe=Object(j.a)({where_:function(t,e,n){const r=Object(B.a)(e,"a","where"),o=Object(B.a)(n,"b","where"),s=Object(B.a)(t,"condition","where","bool"),a=Bt(r.shape,o.shape),i=Ke(r,a),u=Ke(o,a);1===s.rank&&Object(y.assert)(s.shape[0]===r.shape[0],()=>"The first dimension of `a` must match the size of `condition`."),1!==s.rank&&Object(y.assertShapesMatch)(s.shape,u.shape,"Error in where: ");const c={condition:s,t:i,e:u};return l.a.runKernelFunc((t,e)=>{const n=t.select(s,i,u);return e([s]),n},c,null,_.ic)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({bandPart_:function(t,e,n){Object(y.assert)(e%1==0,()=>`bandPart(): numLower must be an integer, got ${e}.`),Object(y.assert)(n%1==0,()=>`bandPart(): numUpper must be an integer, got ${n}.`);const r=Object(B.a)(t,"a","bandPart");Object(y.assert)(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);const o=r.shape,[s,a]=r.shape.slice(-2);if(!(e<=s))throw new Error(`bandPart(): numLower (${e}) must not be greater than the number of rows (${s}).`);if(!(n<=a))throw new Error(`bandPart(): numUpper (${n}) must not be greater than the number of columns (${a}).`);e<0&&(e=s),n<0&&(n=a);const i=Et(oe(0,s,1,"int32"),[-1,1]),u=oe(0,a,1,"int32"),c=Ge(i,u),l=Ve(Ue(c,ie(+e,"int32")),ze(c,ie(-n,"int32"))),h=re([s,a],r.dtype);return Et(fe(He(Et(r,[-1,s,a])).map(t=>qe(l,t,h))),o)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xe=Object(j.a)({abs_:function(t){const e=Object(B.a)(t,"x","abs"),n={x:e};return l.a.runKernelFunc((t,n)=>(n([e]),"complex64"===e.dtype?t.complexAbs(e):t.abs(e)),n,null,_.a)}});const Ye=Object(j.a)({min_:function(t,e=null,n=!1){const r=Object(B.a)(t,"x","min"),o={x:r},s={axis:e,keepDims:n};return l.a.runKernelFunc((t,o)=>{const s=Object(y.parseAxisParam)(e,r.shape);let a=s;const i=Vt(a,r.rank);let u=r;null!=i&&(u=Kt(r,i),a=Ht(a.length,r.rank));const c=t.min(u,a);null!=i&&u.dispose();let l=c;if(n){const t=zt(l.shape,s);l=Et(c,t),c.dispose()}return o([r,l]),l},o,null,_.Fb,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qe=Object(j.a)({pow_:function(t,e){let n=Object(B.a)(t,"base","pow"),r=Object(B.a)(e,"exp","pow");[n,r]=Object(ht.b)(n,r);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.pow(n,r);return e([n,r,o]),o},o,null,_.Rb)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ze=Object(j.a)({sqrt_:function(t){const e=Object(B.a)(t,"x","sqrt"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.sqrt(e);return n([e]),r},n,null,_.tc)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Je=Object(j.a)({square_:function(t){const e=Object(B.a)(t,"x","square"),n=[e];return l.a.runKernelFunc((t,n)=>(n([e]),t.square(e)),{x:e},null,"Square",{},n,[])}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tn=Object(j.a)({sum_:function(t,e=null,n=!1){let r=Object(B.a)(t,"x","sum");"bool"===r.dtype&&(r=M(r,"int32"));const o={x:r},s={axis:e,keepDims:n};return l.a.runKernelFunc((t,o)=>{o([r]);const s=Object(y.parseAxisParam)(e,r.shape),a=Vt(s,r.rank);let i=s,u=r;null!=a&&(u=Kt(r,a),i=Ht(i.length,r.rank));let c=t.sum(u,i);if(n){const t=zt(c.shape,s);c=Et(c,t)}return c},o,null,_.zc,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const en=Object(j.a)({norm_:function(t,e="euclidean",n=null,r=!1){const o=function t(e,n,r=null){if(0===e.rank)return Xe(e);if(1!==e.rank&&null===r)return t(Et(e,[-1]),n,r);if(1===e.rank||"number"==typeof r||Array.isArray(r)&&1===r.length){if(1===n)return tn(Xe(e),r);if(n===1/0)return qt(Xe(e),r);if(n===-1/0)return Ye(Xe(e),r);if("euclidean"===n||2===n)return Ze(tn(Qe(Xe(e),ie(2,"int32")),r));throw new Error("Error in norm: invalid ord value: "+n)}if(Array.isArray(r)&&2===r.length){if(1===n)return qt(tn(Xe(e),r[0]),r[1]-1);if(n===1/0)return qt(tn(Xe(e),r[1]),r[0]);if(n===-1/0)return Ye(tn(Xe(e),r[1]),r[0]);if("fro"===n||"euclidean"===n)return Ze(tn(Je(e),r));throw new Error("Error in norm: invalid ord value: "+n)}throw new Error("Error in norm: invalid axis: "+r)}(t=Object(B.a)(t,"x","norm"),e,n);let s=o.shape;if(r){const e=Object(y.parseAxisParam)(n,t.shape);s=zt(o.shape,e)}return Et(o,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nn=Object(j.a)({squeeze_:function(t,e){const n=Object(B.a)(t,"x","squeeze");return Et(n,Object(y.squeezeShape)(n.shape,e).newShape)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({gramSchmidt_:function(t){let e;if(Array.isArray(t)){e=!1,Object(y.assert)(null!=t&&t.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const n=t[0].shape[0];for(let e=1;e<t.length;++e)Object(y.assert)(t[e].shape[0]===n,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${t[e].shape[0]} vs. ${n})`)}else e=!0,t=ye(t,t.shape[0],0).map(t=>nn(t,[0]));Object(y.assert)(t.length<=t[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${t.length}) exceeds number of dimensions (${t[0].shape[0]}).`);const n=[],r=t;for(let e=0;e<t.length;++e)n.push(l.a.tidy(()=>{let t=r[e];if(e>0)for(let r=0;r<e;++r){const e=Xt(tn(Xt(n[r],t)),n[r]);t=Ge(t,e)}return Tt(t,en(t,"euclidean"))}));return e?fe(n,0):n}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rn(t){Object(h.b)().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(t+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function on(){return l.a}function sn(t,e){return l.a.tidy(t,e)}function an(t){Object(ht.a)(t).forEach(t=>t.dispose())}function un(t){return l.a.setBackend(t)}function cn(t,e,n=1){return l.a.registerBackend(t,e,n)}Object(N.d)(rn);const ln=Object(j.a)({tile_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){const n=Object(B.a)(t,"x","tile",null);y.assert(n.rank===e.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${e}.`);const r=[n],o={x:n},s={reps:e};return l.a.runKernelFunc((t,r)=>{const o=t.tile(n,e);return r([n]),o},o,null,_.Cc,s,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hn=Object(j.a)({eye_:function(t,e,n,r="float32"){null==e&&(e=t);const o=D([t,e],r),s=t<=e?t:e;for(let t=0;t<s;++t)o.set(1,t,t);const a=Et(o.toTensor(),[t,e]);if(null==n)return a;if(1===n.length)return ln(pe(a,0),[n[0],1,1]);if(2===n.length)return ln(pe(pe(a,0),0),[n[0],n[1],1,1]);if(3===n.length)return ln(pe(pe(pe(a,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dn=Object(j.a)({greater_:function(t,e){let n=Object(B.a)(t,"a","greater"),r=Object(B.a)(e,"b","greater");[n,r]=Object(ht.b)(n,r),Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc(t=>t.greater(n,r),o,null,_.gb)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pn=Object(j.a)({neg_:function(t){const e=Object(B.a)(t,"x","neg"),n={x:e};return l.a.runKernelFunc(t=>t.neg(e),n,null,_.Jb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fn(t,e=!1){return l.a.tidy(()=>{Object(y.assert)(2===t.shape.length,()=>`qr2d() requires a 2D Tensor, but got a ${t.shape.length}D Tensor.`);const n=t.shape[0],r=t.shape[1];let o=hn(n),s=P(t);const a=ge([[1]],[1,1]);let i=P(a);const u=n>=r?r:n;for(let t=0;t<u;++t){const e=s,u=i,c=o;[i,s,o]=l.a.tidy(()=>{const e=be(s,[t,t],[n-t,1]),u=en(e),c=be(s,[t,t],[1,1]),l=qe(dn(c,0),ge([[-1]]),ge([[1]])),h=Ge(c,Xt(l,u)),d=Tt(e,h);i=1===d.shape[0]?P(a):de([a,be(d,[1,0],[d.shape[0]-1,d.shape[1]])],0);const p=pn(Tt(Ft(l,h),u)),f=be(s,[t,0],[n-t,r]),g=Xt(p,i),m=Kt(i);if(0===t)s=Ge(f,Ft(g,Ft(m,f)));else{const e=Ge(f,Ft(g,Ft(m,f)));s=de([be(s,[0,0],[t,r]),e],0)}const b=Kt(g),x=be(o,[0,t],[n,o.shape[1]-t]);if(0===t)o=Ge(x,Ft(Ft(x,i),b));else{const e=Ge(x,Ft(Ft(x,i),b));o=de([be(o,[0,0],[n,t]),e],1)}return[i,s,o]}),an([e,u,c])}return!e&&n>r&&(o=be(o,[0,0],[n,r]),s=be(s,[0,0],[r,r])),[o,s]})}Object(j.a)({qr_:function(t,e=!1){if(Object(y.assert)(t.rank>=2,()=>"qr() requires input tensor to have a rank >= 2, but got rank "+t.rank),2===t.rank)return fn(t,e);{const n=t.shape.slice(0,t.shape.length-2).reduce((t,e)=>t*e),r=He(Et(t,[n,t.shape[t.shape.length-2],t.shape[t.shape.length-1]]),0),o=[],s=[];r.forEach(t=>{const[n,r]=fn(t,e);o.push(n),s.push(r)});return[Et(fe(o,0),t.shape),Et(fe(s,0),t.shape)]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var gn;function mn(t){return l.a.customGrad(t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function bn(t,e="float32"){if("complex64"===e){const e=bn(t,"float32"),n=re(t,"float32");return Object(pt.a)(e,n)}const n=Object(y.makeOnesTypedArray)(Object(y.sizeFromShape)(t),e);return l.a.makeTensor(n,t,e)}
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */!function(t){t[t.NONE=0]="NONE",t[t.MEAN=1]="MEAN",t[t.SUM=2]="SUM",t[t.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"}(gn||(gn={}));const xn=Object(j.a)({mean_:function(t,e=null,n=!1){const r=Object(B.a)(t,"x","mean"),o=Object(y.parseAxisParam)(e,r.shape),s=Wt(r.shape,o)[1],a=Object(y.sizeFromShape)(s);return mn(t=>{const r=ie(a),s=r.dtype===t.dtype?t:M(t,r.dtype),i=Tt(s,r);return{value:tn(i,e,n),gradFunc:e=>{const n=t.shape.slice();o.forEach(t=>{n[t]=1});const r=Et(e,n);return Tt(Xt(r,bn(t.shape,"float32")),a)}}})(r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yn=Object(j.a)({notEqual_:function(t,e){let n=Object(B.a)(t,"a","notEqual"),r=Object(B.a)(e,"b","notEqual");[n,r]=Object(ht.b)(n,r),Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc(t=>t.notEqual(n,r),o,null,_.Nb)}});const vn=Object(j.a)({computeWeightedLoss_:function(t,e,n=gn.SUM_BY_NONZERO_WEIGHTS){const r=Object(B.a)(t,"losses","computeWeightedLoss");let o=null;null!=e&&(o=Object(B.a)(e,"weights","computeWeightedLoss"));const s=null==o?r:Xt(r,o);if(n===gn.NONE)return s;if(n===gn.SUM)return tn(s);if(n===gn.MEAN){if(null==o)return xn(s);{const t=r.size/o.size,e=Tt(tn(s),tn(o));return t>1?Tt(e,ie(t)):e}}if(n===gn.SUM_BY_NONZERO_WEIGHTS){if(null==o)return Tt(tn(s),ie(r.size));{const t=Xt(o,bn(r.shape)),e=M(tn(yn(t,ie(0))),"float32");return Tt(tn(s),e)}}throw Error("Unknown reduction: "+n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({absoluteDifference_:function(t,e,n,r=gn.SUM_BY_NONZERO_WEIGHTS){const o=Object(B.a)(t,"labels","absoluteDifference"),s=Object(B.a)(e,"predictions","absoluteDifference");let a=null;null!=n&&(a=Object(B.a)(n,"weights","absoluteDifference")),Object(y.assertShapesMatch)(o.shape,s.shape,"Error in absoluteDifference: ");const i=Xe(Ge(o,s));return vn(i,a,r)}});Object(j.a)({cosineDistance_:function(t,e,n,r,o=gn.SUM_BY_NONZERO_WEIGHTS){const s=Object(B.a)(t,"labels","cosineDistance"),a=Object(B.a)(e,"predictions","cosineDistance");let i=null;null!=r&&(i=Object(B.a)(r,"weights","cosineDistance")),Object(y.assertShapesMatch)(s.shape,a.shape,"Error in cosineDistance: ");const u=ie(1),c=Ge(u,tn(Xt(s,a),n,!0));return vn(c,i,o)}});Object(j.a)({hingeLoss_:function(t,e,n,r=gn.SUM_BY_NONZERO_WEIGHTS){let o=Object(B.a)(t,"labels","hingeLoss");const s=Object(B.a)(e,"predictions","hingeLoss");let a=null;null!=n&&(a=Object(B.a)(n,"weights","hingeLoss")),Object(y.assertShapesMatch)(o.shape,s.shape,"Error in hingeLoss: ");const i=ie(1);o=Ge(Xt(ie(2),o),i);const u=ae(Ge(i,Xt(o,s)));return vn(u,a,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wn=Object(j.a)({minimum_:function(t,e){let n=Object(B.a)(t,"a","minimum"),r=Object(B.a)(e,"b","minimum");[n,r]=Object(ht.b)(n,r),"bool"===n.dtype&&(n=M(n,"int32"),r=M(r,"int32")),Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.minimum(n,r);return e([n,r]),o},o,null,_.Gb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({huberLoss_:function(t,e,n,r=1,o=gn.SUM_BY_NONZERO_WEIGHTS){const s=Object(B.a)(t,"labels","huberLoss"),a=Object(B.a)(e,"predictions","huberLoss");let i=null;null!=n&&(i=Object(B.a)(n,"weights","huberLoss")),Object(y.assertShapesMatch)(s.shape,a.shape,"Error in huberLoss: ");const u=ie(r),c=Xe(Ge(a,s)),l=wn(c,u),h=Ge(c,l),d=dt(Xt(ie(.5),Je(l)),Xt(u,h));return vn(d,i,o)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cn=Object(j.a)({log_:function(t){const e=Object(B.a)(t,"x","log"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.log(e);return n([e]),r},n,null,_.sb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({logLoss_:function(t,e,n,r=1e-7,o=gn.SUM_BY_NONZERO_WEIGHTS){const s=Object(B.a)(t,"labels","logLoss"),a=Object(B.a)(e,"predictions","logLoss");let i=null;null!=n&&(i=Object(B.a)(n,"weights","logLoss")),Object(y.assertShapesMatch)(s.shape,a.shape,"Error in logLoss: ");const u=ie(1),c=ie(r),l=pn(Xt(s,Cn(dt(a,c)))),h=Xt(Ge(u,s),Cn(dt(Ge(u,a),c))),d=Ge(l,h);return vn(d,i,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $n=Object(j.a)({squaredDifference_:function(t,e){let n=Object(B.a)(t,"a","squaredDifference"),r=Object(B.a)(e,"b","squaredDifference");[n,r]=Object(ht.b)(n,r),Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.squaredDifference(n,r);return e([n,r]),o},o,null,_.vc,{})}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({meanSquaredError_:function(t,e,n,r=gn.SUM_BY_NONZERO_WEIGHTS){const o=Object(B.a)(t,"labels","meanSquaredError"),s=Object(B.a)(e,"predictions","meanSquaredError");let a=null;null!=n&&(a=Object(B.a)(n,"weights","meanSquaredError")),Object(y.assertShapesMatch)(o.shape,s.shape,"Error in meanSquaredError: ");const i=$n(o,s);return vn(i,a,r)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const On=Object(j.a)({exp_:function(t){const e=Object(B.a)(t,"x","exp"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.exp(e);return n([r]),r},n,null,_.W)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const In=Object(j.a)({log1p_:function(t){const e=Object(B.a)(t,"x","log1p"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.log1p(e);return n([e]),r},n,null,_.tb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({sigmoidCrossEntropy_:function(t,e,n,r=0,o=gn.SUM_BY_NONZERO_WEIGHTS){let s=Object(B.a)(t,"multiClassLabels","sigmoidCrossEntropy");const a=Object(B.a)(e,"logits","sigmoidCrossEntropy");let i=null;if(null!=n&&(i=Object(B.a)(n,"weights","sigmoidCrossEntropy")),Object(y.assertShapesMatch)(s.shape,a.shape,"Error in sigmoidCrossEntropy: "),r>0){const t=ie(r),e=ie(1),n=ie(.5);s=dt(Xt(s,Ge(e,t)),Xt(n,t))}const u=function(t,e){const n=Object(B.a)(t,"labels","sigmoidCrossEntropyWithLogits"),r=Object(B.a)(e,"logits","sigmoidCrossEntropyWithLogits");Object(y.assertShapesMatch)(n.shape,r.shape,"Error in sigmoidCrossEntropyWithLogits: ");const o=ae(r),s=Xt(r,n),a=In(On(pn(Xe(r))));return dt(Ge(o,s),a)}(s,a);return vn(u,i,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sn=Object(j.a)({logSumExp_:function(t,e=null,n=!1){const r=Object(B.a)(t,"x","logSumExp"),o=Object(y.parseAxisParam)(e,r.shape),s=qt(r,o,!0),a=Ge(r,s),i=On(a),u=tn(i,o),c=Cn(u),l=dt(Et(s,c.shape),c);if(n){const t=zt(l.shape,o);return Et(l,t)}return l}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Object(j.a)({softmaxCrossEntropy_:function(t,e,n,r=0,o=gn.SUM_BY_NONZERO_WEIGHTS){let s=Object(B.a)(t,"onehotLabels","softmaxCrossEntropy");const a=Object(B.a)(e,"logits","softmaxCrossEntropy");let i=null;if(null!=n&&(i=Object(B.a)(n,"weights","softmaxCrossEntropy")),Object(y.assertShapesMatch)(s.shape,a.shape,"Error in softmaxCrossEntropy: "),r>0){const t=ie(r),e=ie(1),n=ie(s.shape[1]);s=dt(Xt(s,Ge(e,t)),Tt(t,n))}const u=function(t,e,n=-1){if(-1===n&&(n=e.rank-1),n!==e.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${e.rank} and dim was `+n);return mn((t,e,r)=>{const o=Sn(e,[n],!0),s=Ge(M(e,"float32"),o);r([t,s]);const a=pn(Xt(s,t));return{value:tn(a,[n]),gradFunc:(t,e)=>{const[r,o]=e,s=zt(t.shape,[n]);return[Xt(Et(t,s),Ge(M(r,"float32"),On(o))),Xt(Et(t,s),Ge(On(o),M(r,"float32")))]}}})(t,e)}(s,a);return vn(u,i,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var En=n(13);
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rn=Object(j.a)({elu_:function(t){const e=Object(B.a)(t,"x","elu"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.elu(e);return n([r]),r},n,null,_.S)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const An=Object(j.a)({prelu_:function(t,e){const n=Object(B.a)(t,"x","prelu"),r=Object(B.a)(e,"alpha","prelu"),o={x:n,alpha:r};return l.a.runKernelFunc((t,e)=>{const o=t.prelu(n,r);return e([n,r]),o},o,null,_.Sb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kn=Object(j.a)({relu6_:function(t){const e=Object(B.a)(t,"x","relu6"),n={x:e};return l.a.runKernelFunc((t,n)=>(n([e]),"bool"===e.dtype?M(e,"int32"):t.relu6(e)),n,null,_.Yb)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tn=Object(j.a)({step_:function(t,e=0){const n=Object(B.a)(t,"x","step"),r={x:n},o={alpha:e};return l.a.runKernelFunc(t=>t.step(n,e),r,null,_.wc,o)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fn(t,e,n){if(null==n||"linear"===n)return t;if("relu"===n)return Xt(t,Tn(e));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function Nn(t,e){let n=e;const r=_t(t.shape,e.shape);return r.length>0&&(n=tn(n,r)),Et(n,t.shape)}function Dn(t,e,n){if("linear"===e)return t;if("relu"===e)return ae(t);if("elu"===e)return Rn(t);if("relu6"===e)return kn(t);if("prelu"===e)return An(t,n);throw new Error(`Unknown fused activation ${e}.`)}const _n=(t,e)=>!(t>0)||"linear"===e,Bn=30;
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jn(t){return t<=Bn?t:Object(y.nearestDivisor)(t,Math.floor(Math.sqrt(t)))}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mn(t,e,n){return[n*("number"==typeof t?t:t[0]),e*("number"==typeof t?t:t[1])]}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pn(t,e,n,r=!0){let o=[];if(r)o=o.concat(e.slice(0)),o.push(t[0]/n),o=o.concat(t.slice(1));else{o=o.concat(t[0]);const n=e.length;for(let r=0;r<n;++r)o=o.concat([t[r+1]/e[r],e[r]]);o=o.concat(t.slice(n+1))}return o}function Ln(t,e,n=!0){const r=[];if(n){r.push(e);for(let n=e+1;n<t;++n)n<=2*e?(r.push(n),r.push(n-(e+1))):r.push(n)}else{const n=[],o=[];for(let r=1;r<t;++r)r>=2*e+1||r%2==1?o.push(r):n.push(r);r.push(...n),r.push(0),r.push(...o)}return r}function Wn(t,e,n,r=!0){const o=[];r?o.push(t[0]/n):o.push(t[0]*n);for(let n=1;n<t.length;++n)n<=e.length?r?o.push(e[n-1]*t[n]):o.push(t[n]/e[n-1]):o.push(t[n]);return o}function zn(t,e){const n=[0];for(let r=0;r<e;++r)n.push(t[r][0]);return n}function Un(t,e,n){const r=t.slice(0,1);for(let o=0;o<n;++o)r.push(t[o+1]-e[o][0]-e[o][1]);return r}function Vn(t,e){if(t.rank<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${e.rank}.`);if("int32"!==e.dtype)throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.shape[e.rank-1]>t.rank)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${e.shape[e.rank-1]} vs. ${t.rank}`);if(0===t.size)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${t.shape}.`);const n=e.shape,r=n[n.length-1];let o=1;for(let t=0;t<n.length-1;++t)o*=n[t];const s=t.shape,a=n.slice();a.pop();let i=1;for(let e=r;e<t.rank;++e)i*=s[e],a.push(s[e]);const u=[...Object(y.computeStrides)(t.shape).map(t=>t/i),1].slice(0,r);return[a,o,i,u]}function Gn(t,e,n){const r=e.rank>1?e.shape[e.rank-1]:1,o=e.rank>1?e.rank-1:1,s="Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: "+n.shape+`, indices.shape: ${e.shape}, shape: ${t}`+`, sliceDim: ${r}, and batchDim: ${o}.`;if(n.rank<o)throw new Error(s+` update.rank < ${o}. `);if(t.length<r+(n.rank-o))throw new Error(s+" Output shape length < "+(r+(n.rank-o)));if(n.rank!==o+t.length-r)throw new Error(s+" update.rank != "+(o+t.length-r));for(let t=0;t<o;++t)if(n.shape[t]!==e.shape[t])throw new Error(s+` updates.shape[${t}] (${n.shape[t]}) != indices.shape[${t}] (${e.shape[t]}).`);for(let e=0;e<n.rank-o;++e)if(n.shape[e+o]!==t[e+r])throw new Error(s+` updates.shape[${e+o}] (${n.shape[e+o]}) != shape[${e+o}] (${t[e+o]})`)}function Hn(t,e,n){if(e.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${t.rank}.`);if("int32"!==e.dtype)throw new Error("The dtype of 'indices' should be int32, but got dtype: "+e.dtype);if(n.length<1)throw new Error("Output rank must be greater or equal to 1, but got shape: "+n);if(0===n.length){if(0===e.size)throw new Error("Indices specified for empty output. indices shape: "+e.shape);if(0===t.size)throw new Error("Updates specified for empty output. updates shape: "+t.shape)}Gn(n,e,t)}function Kn(t,e,n){const r=e.shape.length,o=r>1?e.shape[r-1]:1,s=n.length;let a=1;for(let t=o;t<s;++t)a*=n[t];const i=o<1?1:o;return{sliceRank:o,numUpdates:Object(y.sizeFromShape)(e.shape)/i,sliceSize:a,strides:[...Object(y.computeStrides)(n.slice(0,o)),1],outputSize:Object(y.sizeFromShape)(n)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qn=1.7580993408473768,Xn=1.0507009873554805,Yn=.3275911,Qn=.254829592,Zn=-.284496736,Jn=1.421413741,tr=-1.453152027,er=1.061405429;
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function nr(...t){Object(h.b)().getBool("IS_TEST")||console.warn(...t)}function rr(...t){Object(h.b)().getBool("IS_TEST")||console.log(...t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function or(t,e){if(t.length!==e.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${t.length}, imag: ${e.length}.`);const n=new Float32Array(2*t.length);for(let r=0;r<n.length;r+=2)n[r]=t[r/2],n[r+1]=e[r/2];return n}function sr(t){const e=new Float32Array(t.length/2),n=new Float32Array(t.length/2);for(let r=0;r<t.length;r+=2)e[r/2]=t[r],n[r/2]=t[r+1];return{real:e,imag:n}}function ar(t){const e=Math.ceil(t.length/4),n=new Float32Array(e),r=new Float32Array(e);for(let e=0;e<t.length;e+=4)n[Math.floor(e/4)]=t[e],r[Math.floor(e/4)]=t[e+1];return{real:n,imag:r}}function ir(t){const e=Math.floor(t.length/4),n=new Float32Array(e),r=new Float32Array(e);for(let e=2;e<t.length;e+=4)n[Math.floor(e/4)]=t[e],r[Math.floor(e/4)]=t[e+1];return{real:n,imag:r}}function ur(t,e){return{real:t[2*e],imag:t[2*e+1]}}function cr(t,e,n,r){t[2*r]=e,t[2*r+1]=n}function lr(t,e){const n=new Float32Array(t/2),r=new Float32Array(t/2);for(let o=0;o<Math.ceil(t/2);o++){const s=(e?2:-2)*Math.PI*(o/t);n[o]=Math.cos(s),r[o]=Math.sin(s)}return{real:n,imag:r}}function hr(t,e,n){const r=(n?2:-2)*Math.PI*(t/e);return{real:Math.cos(r),imag:Math.sin(r)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dr(t,e){let n,r=!1;for(t<=Bn?(n=t,r=!0):n=Object(y.nearestDivisor)(t,Math.floor(Math.sqrt(t)));!r;)n>e||n===t?r=!0:n=Object(y.nearestDivisor)(t,n+1);return n}function pr(t,e,n){const r=[],o=t.length;for(let s=0;s<o;s++)s!==e?r.push(t[s]):r.push(n);return r}function fr(t,e,n){const r=t.shape[n],o=[];let s=1,a=1;for(let e=0;e<n;e++)o.push(t.shape[e]),s*=t.shape[e];for(let t=0;t<e.rank;t++)o.push(e.shape[t]);for(let e=n+1;e<t.rank;e++)o.push(t.shape[e]),a*=t.shape[e];return{batchSize:s,sliceSize:a,dimSize:r,outputShape:o}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gr(t,e,n){if("complex64"===e){if("complex64"===t.dtype)return t.clone();const e=re(t.shape),r=M(t,"float32"),o=n.complex(r,e);return e.dispose(),r.dispose(),o}if(!Object(y.hasEncodingLoss)(t.dtype,e))return l.a.makeTensorFromDataId(t.dataId,t.shape,e);if("complex64"===t.dtype){const r=n.real(t),o=M(r,e);return r.dispose(),o}if("int32"===e)return n.int(t);if("bool"===e){const e=ie(0,t.dtype),r=n.notEqual(t,e);return e.dispose(),r}throw new Error(`Error in Cast: failed to cast ${t.dtype} to ${e}`)}function mr(t,e){return l.a.makeTensorFromDataId(t.dataId,e,t.dtype)}function br(t,e,n){const r=(e-t)/(n-1),o=Object(y.makeZerosTypedArray)(n,"float32");o[0]=t;for(let t=1;t<o.length;t++)o[t]=o[t-1]+r;return ne(o,"float32")}var xr=n(20);
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yr(t,e,n){const r=new Array(t.rank).fill(0),o=t.shape.slice();return e.map(e=>{const s=[...o];s[n]=e;const a=be(t,r,s);return r[n]+=e,a})}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vr(t,e){const n=new Array(t.rank);for(let r=0;r<n.length;r++)n[r]=t.shape[r]*e[r];const r=D(n,t.dtype);for(let e=0;e<r.values.length;++e){const n=r.indexToLoc(e),o=new Array(t.rank);for(let e=0;e<o.length;e++)o[e]=n[e]%t.shape[e];const s=t.locToIndex(o);r.values[e]=t.values[s]}return r.toTensor()}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wr(t,e,n,r,o){const s=e[e.length-1],[a,i]=[t.length/s,s],u=Object(y.getTypedArrayFromDType)(n,a*r),c=Object(y.getTypedArrayFromDType)("int32",a*r);for(let e=0;e<a;e++){const n=e*i,o=t.subarray(n,n+i),s=[];for(let t=0;t<o.length;t++)s.push({value:o[t],index:t});s.sort((t,e)=>e.value-t.value);const a=e*r,l=u.subarray(a,a+r),h=c.subarray(a,a+r);for(let t=0;t<r;t++)l[t]=s[t].value,h[t]=s[t].index}const l=e.slice();return l[l.length-1]=r,[Object(he.a)(u,l,n),Object(he.a)(c,l,"int32")]}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cr(t,e){const n=[];for(let t=0;t<e.length;t++)e[t]&&n.push(t);const r=D(t,"int32"),o=D([n.length,t.length],"int32");for(let e=0;e<n.length;e++){const s=r.indexToLoc(n[e]),a=e*t.length;o.values.set(s,a)}return o.toTensor()}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $r{constructor(t,e){this.backend=t,this.dataMover=e,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,e){this.dataIdsCount++,this.data.set(t,e)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class Or{time(t){return Ir("time")}read(t){return Ir("read")}readSync(t){return Ir("readSync")}numDataIds(){return Ir("numDataIds")}disposeData(t){return Ir("disposeData")}write(t,e,n){return Ir("write")}move(t,e,n,r){return Ir("move")}memory(){return Ir("memory")}floatPrecision(){return Ir("floatPrecision")}epsilon(){return 32===this.floatPrecision()?1e-7:1e-4}batchMatMul(t,e,n,r){return Ir("batchMatMul")}fusedBatchMatMul({a:t,b:e,transposeA:n,transposeB:r,bias:o,activation:s,preluActivationWeights:a}){return Ir("fusedBatchMatMul")}slice(t,e,n){return Ir("slice")}stridedSlice(t,e,n,r){return Ir("stridedSlice")}unstack(t,e){return Ir("unstack")}reverse(t,e){return Ir("reverse")}concat(t,e){return Ir("concat")}neg(t){return Ir("neg")}add(t,e){return Ir("add")}addN(t){return Ir("addN")}subtract(t,e){return Ir("subtract")}multiply(t,e){return Ir("multiply")}realDivide(t,e){return Ir("realDivide")}floorDiv(t,e){return Ir("floorDiv")}sum(t,e){return Ir("sum")}prod(t,e){return Ir("prod")}unsortedSegmentSum(t,e,n){return Ir("unsortedSegmentSum")}argMin(t,e){return Ir("argMin")}argMax(t,e){return Ir("argMax")}equal(t,e){return Ir("equal")}notEqual(t,e){return Ir("notEqual")}less(t,e){return Ir("less")}lessEqual(t,e){return Ir("lessEqual")}greater(t,e){return Ir("greater")}greaterEqual(t,e){return Ir("greaterEqual")}logicalNot(t){return Ir("logicalNot")}logicalAnd(t,e){return Ir("logicalAnd")}logicalOr(t,e){return Ir("logicalOr")}where(t){return Ir("where")}select(t,e,n){return Ir("select")}topk(t,e,n){return Ir("topk")}min(t,e){return Ir("min")}minimum(t,e){return Ir("minimum")}mod(t,e){return Ir("mod")}max(t,e){return Ir("max")}maximum(t,e){return Ir("maximum")}all(t,e){return Ir("all")}any(t,e){return Ir("any")}squaredDifference(t,e){return Ir("squaredDifference")}ceil(t){return Ir("ceil")}floor(t){return Ir("floor")}round(t){return Ir("round")}sign(t){return Ir("sign")}isNaN(t){return Ir("isNaN")}isInf(t){return Ir("isInf")}isFinite(t){return Ir("isFinite")}pow(t,e){return Ir("pow")}exp(t){return Ir("exp")}expm1(t){return Ir("expm1")}softmax(t,e){return Ir("softmax")}log(t){return Ir("log")}log1p(t){return Ir("log1p")}sqrt(t){return Ir("sqrt")}rsqrt(t){return Ir("rsqrt")}square(t){return Ir("square")}reciprocal(t){return Ir("reciprocal")}relu(t){return Ir("relu")}relu6(t){return Ir("relu6")}prelu(t,e){return Ir("prelu")}elu(t){return Ir("elu")}eluDer(t,e){return Ir("eluDer")}selu(t){return Ir("selu")}int(t){return Ir("int")}clip(t,e,n){return Ir("clip")}abs(t){return Ir("abs")}complexAbs(t){return Ir("complexAbs")}sigmoid(t){return Ir("sigmoid")}softplus(t){return Ir("softplus")}sin(t){return Ir("sin")}cos(t){return Ir("cos")}tan(t){return Ir("tan")}asin(t){return Ir("asin")}acos(t){return Ir("acos")}atan(t){return Ir("atan")}atan2(t,e){return Ir("atan2")}sinh(t){return Ir("sinh")}cosh(t){return Ir("cosh")}tanh(t){return Ir("tanh")}asinh(t){return Ir("asinh")}acosh(t){return Ir("acosh")}atanh(t){return Ir("atanh")}erf(t){return Ir("erf")}step(t,e){return Ir("step")}fusedConv2d({input:t,filter:e,convInfo:n,bias:r,activation:o,preluActivationWeights:s}){return Ir("fusedConv2d")}conv2d(t,e,n){return Ir("conv2d")}conv2dDerInput(t,e,n){return Ir("conv2dDerInput")}conv2dDerFilter(t,e,n){return Ir("conv2dDerFilter")}fusedDepthwiseConv2D({input:t,filter:e,convInfo:n,bias:r,activation:o,preluActivationWeights:s}){return Ir("fusedDepthwiseConv2D")}depthwiseConv2D(t,e,n){return Ir("depthwiseConv2D")}depthwiseConv2DDerInput(t,e,n){return Ir("depthwiseConv2DDerInput")}depthwiseConv2DDerFilter(t,e,n){return Ir("depthwiseConv2DDerFilter")}conv3d(t,e,n){return Ir("conv3d")}conv3dDerInput(t,e,n){return Ir("conv3dDerInput")}conv3dDerFilter(t,e,n){return Ir("conv3dDerFilter")}maxPool(t,e){return Ir("maxPool")}maxPoolBackprop(t,e,n,r){return Ir("maxPoolBackprop")}avgPool(t,e){return Ir("avgPool")}avgPoolBackprop(t,e,n){return Ir("avgPoolBackprop")}avgPool3d(t,e){return Ir("avgPool3d")}avgPool3dBackprop(t,e,n){return Ir("avgPool3dBackprop")}maxPool3d(t,e){return Ir("maxPool3d")}maxPool3dBackprop(t,e,n,r){return Ir("maxPool3dBackprop")}reshape(t,e){return Ir("reshape")}cast(t,e){return Ir("cast")}tile(t,e){return Ir("tile")}pad(t,e,n){return Ir("pad")}transpose(t,e){return Ir("transpose")}gather(t,e,n){return Ir("gather")}gatherND(t,e){return Ir("gatherND")}scatterND(t,e,n){return Ir("scatterND")}batchToSpaceND(t,e,n){return Ir("batchToSpaceND")}spaceToBatchND(t,e,n){return Ir("spaceToBatchND")}resizeBilinear(t,e,n,r){return Ir("resizeBilinear")}resizeBilinearBackprop(t,e,n){return Ir("resizeBilinearBackprop")}resizeNearestNeighbor(t,e,n,r){return Ir("resizeNearestNeighbor")}resizeNearestNeighborBackprop(t,e,n){return Ir("resizeNearestNeighborBackprop")}batchNorm(t,e,n,r,o,s){return Ir("batchNorm")}localResponseNormalization4D(t,e,n,r,o){return Ir("localResponseNormalization4D")}LRNGrad(t,e,n,r,o,s,a){return Ir("LRNGrad")}multinomial(t,e,n,r){return Ir("multinomial")}oneHot(t,e,n,r){return Ir("oneHot")}cumsum(t,e,n,r){return Ir("cumsum")}nonMaxSuppression(t,e,n,r,o){return Ir("nonMaxSuppression")}fft(t){return Ir("fft")}ifft(t){return Ir("ifft")}complex(t,e){return Ir("complex")}real(t){return Ir("real")}imag(t){return Ir("imag")}cropAndResize(t,e,n,r,o,s){return Ir("cropAndResize")}depthToSpace(t,e,n){return Ir("depthToSpace")}split(t,e,n){return Ir("split")}sparseToDense(t,e,n,r){return Ir("sparseToDense")}diag(t){return Ir("diag")}fill(t,e,n){return Ir("fill")}onesLike(t){return Ir("onesLike")}zerosLike(t){return Ir("zerosLike")}linspace(t,e,n){return Ir("linspace")}dispose(){return Ir("dispose")}}function Ir(t){throw new Error(`'${t}' not yet implemented or not found in the registry. Did you forget to import the kernel?`)}
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sr={kernelName:_.a,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(t,Tn(M(n,"float32"),-1))}}},Er={kernelName:_.b,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>{const e=Je(M(n,"float32")),r=Ze(Ge(ie(1),e));return pn(Tt(t,r))}}}},Rr={kernelName:_.c,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>{const e=Ze(Ge(Je(M(n,"float32")),1));return Tt(t,e)}}}},Ar={kernelName:_.d,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e,o=Bt(n.shape,r.shape);return{a:()=>{let e=t;const r=_t(n.shape,o);return r.length>0&&(e=tn(e,r)),Et(e,n.shape)},b:()=>{let e=t;const n=_t(r.shape,o);return n.length>0&&(e=tn(e,n)),Et(e,r.shape)}}}},kr={kernelName:_.e,saveAllInputs:!0,gradFunc:(t,e)=>{const n={};return e.forEach((e,r)=>{n[r]=()=>t.clone()}),n}},Tr={kernelName:_.h,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>ve(n)}}},Fr={kernelName:_.i,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>ve(n)}}},Nr={kernelName:_.j,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Tt(t,Ze(Ge(ie(1),Je(M(n,"float32")))))}}},Dr={kernelName:_.k,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>{const e=Ze(dt(ie(1),Je(M(n,"float32"))));return Tt(t,e)}}}},_r={kernelName:_.m,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e,o=Bt(n.shape,r.shape);return{a:()=>{const e=dt(Je(n),Je(r));let s=Xt(t,Tt(r,e));const a=_t(n.shape,o);return a.length>0&&(s=tn(s,a)),Et(s,n.shape)},b:()=>{const e=dt(Je(n),Je(r));let s=pn(Xt(t,Tt(n,e)));const a=_t(r.shape,o);return a.length>0&&(s=tn(s,a)),Et(s,r.shape)}}}},Br={kernelName:_.l,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Tt(t,dt(Je(M(n,"float32")),1))}}},jr={kernelName:_.n,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Tt(t,Ge(ie(1),Je(M(n,"float32"))))}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mr=Object(j.a)({avgPool3dBackprop_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o=[1,1,1],s,a){const i=Object(B.a)(t,"dy","avgPool3dBackprop"),u=Object(B.a)(e,"input","avgPool3dBackprop");let c=i,h=u,d=!1;4===u.rank&&(d=!0,c=Et(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]]),h=Et(u,[1,u.shape[0],u.shape[1],u.shape[2],u.shape[3]])),y.assert(5===c.rank,()=>"Error in avgPool3dBackprop: dy must be rank 5 but got rank "+c.rank+"."),y.assert(5===h.rank,()=>"Error in avgPool3dBackprop: input must be rank 5 but got rank "+h.rank+"."),y.assert(It(r,o),()=>`Error in avgPool3dBackprop: Either strides or dilations must be 1. Got strides ${r} and dilations '${o}'`),null!=a&&y.assert(y.isInt(s),()=>`Error in maxPool3dBackprop: pad must be an integer when using, dimRoundingMode ${a} but got pad ${s}.`);const p={dy:c,input:h},f={filterSize:n,strides:r,dilations:o,pad:s,dimRoundingMode:a},g=l.a.runKernelFunc(t=>{const e=mt(h.shape,n,r,o,s,a);return t.avgPool3dBackprop(c,h,e)},p,null,_.q,f);return d?Et(g,[g.shape[1],g.shape[2],g.shape[3],g.shape[4]]):g}}),Pr={kernelName:_.p,inputsToSave:["x"],gradFunc:(t,e,n)=>{const[r]=e,{filterSize:o,strides:s,dilations:a,pad:i,dimRoundingMode:u}=n,c=null==a?[1,1,1]:a;return{x:()=>Mr(t,r,o,s,c,i,u)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Lr=Object(j.a)({avgPoolBackprop_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o){const s=Object(B.a)(t,"dy","avgPoolBackprop"),a=Object(B.a)(e,"input","avgPoolBackprop");y.assert(a.rank===s.rank,()=>`Rank of input (${a.rank}) does not match rank of dy (${s.rank})`);let i=a,u=s,c=!1;3===a.rank&&(c=!0,i=Et(a,[1,a.shape[0],a.shape[1],a.shape[2]]),u=Et(s,[1,s.shape[0],s.shape[1],s.shape[2]])),y.assert(4===u.rank,()=>"Error in avgPoolBackprop: dy must be rank 4 but got rank "+u.rank+"."),y.assert(4===i.rank,()=>"Error in avgPoolBackprop: input must be rank 4 but got rank "+i.rank+".");const h={dy:u,input:i},d={filterSize:n,strides:r,pad:o},p=l.a.runKernelFunc(t=>{const e=gt(i.shape,n,r,1,o);return t.avgPoolBackprop(u,i,e)},h,null,_.r,d);return c?Et(p,[p.shape[1],p.shape[2],p.shape[3]]):p}}),Wr={kernelName:_.o,inputsToSave:["x"],gradFunc:(t,e,n)=>{const[r]=e,{filterSize:o,strides:s,pad:a}=n;return{x:()=>Lr(t,r,o,s,a)}}},zr={kernelName:_.s,inputsToSave:["a","b"],gradFunc:(t,e,n)=>{const[r,o]=e,{transposeA:s,transposeB:a}=n;return s||a?!s&&a?{a:()=>Ft(t,o,!1,!1),b:()=>Ft(t,r,!0,!1)}:s&&!a?{a:()=>Ft(o,t,!1,!0),b:()=>Ft(r,t,!1,!1)}:{a:()=>Ft(o,t,!0,!0),b:()=>Ft(t,r,!0,!0)}:{a:()=>Ft(t,o,!1,!0),b:()=>Ft(r,t,!0,!1)}}},Ur={kernelName:_.t,gradFunc:(t,e,n)=>{const{blockShape:r,crops:o}=n;return{x:()=>Jt(t,r,o)}}},Vr={kernelName:_.u,gradFunc:(t,e,n)=>{const r=n,o=r.inputShape,s=r.shape,a=Array.from(s);for(let t=o.length-1;t>=0;t--)if(o[t]===s[t])a[t]=1;else if(1!==o[t])throw new Error(`broadcastTo(): [${o}] cannot be broadcast to [${s}].`);const i=[];for(let t=0;t<a.length;t++)a[t]>1&&i.push(t);return{x:()=>tn(t,i,!0)}}},Gr={kernelName:_.v,gradFunc:t=>({x:()=>t.clone()})},Hr={kernelName:_.w,gradFunc:t=>({x:()=>ve(t)})},Kr={kernelName:_.x,inputsToSave:["x"],gradFunc:(t,e,n)=>{const[r]=e,{clipValueMin:o,clipValueMax:s}=n;return{x:()=>qe(Ve(ze(r,o),Ue(r,s)),t,ve(t))}}},qr={kernelName:_.z,saveAllInputs:!0,gradFunc:(t,e,n)=>{const r=e.map(t=>t.shape),{axis:o}=n,s=Object(y.parseAxisParam)(o,e[0].shape)[0],a=r.map(t=>t[s]);return ye(t,a,s).map(t=>()=>t)}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xr=Object(j.a)({conv2DBackpropFilter_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o,s="NHWC",a){let i=t;3===t.rank&&(i=Et(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let u=e;3===u.rank&&(u=Et(e,[1,e.shape[0],e.shape[1],e.shape[2]])),y.assert(4===i.rank,()=>"Error in conv2dDerFilter: input must be rank 4, but got shape "+i.shape+"."),y.assert(4===u.rank,()=>"Error in conv2dDerFilter: dy must be rank 4, but got shape "+u.shape+"."),y.assert(4===n.length,()=>"Error in conv2dDerFilter: filterShape must be length 4, but got "+n+".");const c="NHWC"===s?i.shape[3]:i.shape[1],h="NHWC"===s?u.shape[3]:u.shape[1];y.assert(c===n[2],()=>`Error in conv2dDerFilter: depth of input ${c}) must match input depth in filter (${n[2]}.`),y.assert(h===n[3],()=>`Error in conv2dDerFilter: depth of dy (${h}) must match output depth for filter (${n[3]}).`),null!=a&&y.assert(y.isInt(o),()=>`Error in conv2dDerFilter: pad must be an integer when using, dimRoundingMode ${a} but got pad ${o}.`);const d={x:i,dy:u},p={strides:r,pad:o,dataFormat:s,dimRoundingMode:a};return l.a.runKernelFunc(t=>{const e=St(s),c=bt(i.shape,n,r,1,o,a,!1,e);return t.conv2dDerFilter(i,u,c)},d,null,_.B,p)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yr=Object(j.a)({conv2DBackpropInput_:function(t,e,n,r,o,s="NHWC",a){y.assert(t.length===e.rank,()=>`Length of inShape (${t.length}) and rank of dy (${e.rank}) must match`);let i=t,u=e,c=!1;3===e.rank&&(c=!0,u=Et(e,[1,e.shape[0],e.shape[1],e.shape[2]]),i=[1,t[0],t[1],t[2]]),y.assert(4===i.length,()=>"Error in conv2dDerInput: inShape must be length 4, but got length "+i.length+"."),y.assert(4===u.rank,()=>"Error in conv2dDerInput: dy must be rank 4, but got rank "+u.rank),y.assert(4===n.rank,()=>"Error in conv2dDerInput: filter must be rank 4, but got rank "+n.rank);const h="NHWC"===s?i[3]:i[1],d="NHWC"===s?u.shape[3]:u.shape[1];y.assert(h===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${h}) must match input depth for filter ${n.shape[2]}.`),y.assert(d===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${d}) must match output depth for filter ${n.shape[3]}.`),null!=a&&y.assert(y.isInt(o),()=>`Error in conv2dDerInput: pad must be an integer when using, dimRoundingMode ${a} but got pad ${o}.`);const p={dy:u,filter:n},f={strides:r,pad:o,dataFormat:s,dimRoundingMode:a,inputShape:i},g=l.a.runKernelFunc((t,e)=>{const c=St(s),l=bt(i,n.shape,r,1,o,a,!1,c),h=t.conv2dDerInput(u,n,l);return e([u,n]),h},p,null,_.C,f);return c?Et(g,[g.shape[1],g.shape[2],g.shape[3]]):g}}),Qr={kernelName:_.A,inputsToSave:["x","filter"],gradFunc:(t,e,n)=>{const[r,o]=e,{dilations:s,strides:a,pad:i,dataFormat:u}=n;return y.assert(Ot(s),()=>`Error in gradient of conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${s}'`),{x:()=>Yr(r.shape,t,o,a,i,u),filter:()=>Xr(r,t,o.shape,a,i,u)}}},Zr={kernelName:_.C,inputsToSave:["dy","filter"],gradFunc:(t,e,n)=>{const[r,o]=e,{strides:s,pad:a,dataFormat:i,dimRoundingMode:u}=n;return{dy:()=>Rt(t,o,s,a,i,1,u),filter:()=>Xr(t,r,o.shape,s,a,i,u)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jr=Object(j.a)({conv3DBackpropFilter_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o){let s=t;4===t.rank&&(s=Et(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]));let a=e;4===a.rank&&(a=Et(e,[1,e.shape[0],e.shape[1],e.shape[2],e.shape[3]])),y.assert(5===s.rank,()=>"Error in conv3dDerFilter: input must be rank 5, but got shape "+s.shape+"."),y.assert(5===a.rank,()=>"Error in conv3dDerFilter: dy must be rank 5, but got shape "+a.shape+"."),y.assert(5===n.length,()=>"Error in conv3dDerFilter: filterShape must be length 5, but got "+n+"."),y.assert(s.shape[4]===n[3],()=>`Error in conv3dDerFilter: depth of input ${s.shape[4]}) must match input depth in filter (${n[3]}.`),y.assert(a.shape[4]===n[4],()=>`Error in conv3dDerFilter: depth of dy (${a.shape[4]}) must match output depth for filter (${n[4]}).`);const i={x:s,y:a},u={strides:r,pad:o};return l.a.runKernelFunc(t=>{const e=xt(s.shape,n,r,1,o);return t.conv3dDerFilter(s,a,e)},i,null,_.E,u)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const to=Object(j.a)({conv3DBackpropInput_:function(t,e,n,r,o){y.assert(t.length===e.rank,()=>`Length of inShape (${t.length}) and rank of dy (${e.rank}) must match`);let s=t,a=e,i=!1;4===e.rank&&(i=!0,a=Et(e,[1,e.shape[0],e.shape[1],e.shape[2],e.shape[3]]),s=[1,t[0],t[1],t[2],t[3]]);const u=s[4],c=a.shape[4];y.assert(5===s.length,()=>"Error in conv3dDerInput: inShape must be length 5, but got length "+s.length+"."),y.assert(5===a.rank,()=>"Error in conv3dDerInput: dy must be rank 5, but got rank "+a.rank),y.assert(5===n.rank,()=>"Error in conv3dDerInput: filter must be rank 5, but got rank "+n.rank),y.assert(u===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[3]}.`),y.assert(c===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${c}) must match output depth for filter ${n.shape[4]}.`);const h={dy:a},d={pad:o},p=l.a.runKernelFunc(t=>{const e=xt(s,n.shape,r,1,o);return t.conv3dDerInput(a,n,e)},h,null,_.F,d);return i?Et(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}}),eo={kernelName:_.D,inputsToSave:["x","filter"],gradFunc:(t,e,n)=>{const{dilations:r,strides:o,pad:s}=n;y.assert(Ot(r),()=>`Error in gradient of conv3D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${r}'`);const[a,i]=e;return{x:()=>to(a.shape,t,i,o,s),filter:()=>Jr(a,t,i.shape,o,s)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const no=Object(j.a)({sin_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","sin"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.sin(e);return n([e]),r},n,null,_.mc)}}),ro={kernelName:_.G,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(pn(no(M(n,"float32"))),t)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oo=Object(j.a)({sinh_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","sinh"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.sinh(e);return n([e]),r},n,null,_.nc)}}),so={kernelName:_.H,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(oo(M(n,"float32")),t)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ao=Object(j.a)({cumsum_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=0,n=!1,r=!1){const o=Object(B.a)(t,"x","cumsum"),s={x:o},a={axis:e,exclusive:n,reverse:r};return l.a.runKernelFunc((t,s)=>{const a=Vt([e],o.rank);let i=o;null!=a&&(i=Kt(o,a));const u=Ht(1,o.rank)[0];let c=t.cumsum(i,u,n,r);if(s([o]),null!=a){const t=Gt(a);c=Kt(c,t)}return c},s,null,_.J,a)}}),io={kernelName:_.J,inputsToSave:["x"],gradFunc:(t,e,n)=>{const[r]=e,{axis:o,exclusive:s,reverse:a}=n;return{x:()=>{const e=Vt([o],r.rank);let n=ao(t,o,s,!a);return null!=e&&(n=Kt(n,e)),n}}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uo=Object(j.a)({depthwiseConv2dNativeBackpropFilter_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r){let o=t;3===t.rank&&(o=Et(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let s=e;3===s.rank&&(s=Et(e,[1,e.shape[0],e.shape[1],e.shape[2]]));const a={x:o,dy:s};return l.a.runKernelFunc(t=>t.depthwiseConv2DDerFilter(o,s,r),a,null,_.M)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const co=Object(j.a)({depthwiseConv2dNativeBackpropInput_:function(t,e,n,r){let o=e,s=!1;3===e.rank&&(s=!0,o=Et(e,[1,e.shape[0],e.shape[1],e.shape[2]]));const a={dy:o},i=l.a.runKernelFunc(t=>t.depthwiseConv2DDerInput(o,n,r),a,null,_.N);return s?Et(i,[i.shape[1],i.shape[2],i.shape[3]]):i}}),lo={kernelName:_.L,inputsToSave:["x","filter"],gradFunc:(t,e,n)=>{const{dilations:r,strides:o,pad:s,dimRoundingMode:a}=n,i=null==r?[1,1]:r;y.assert(Ot(i),()=>`Error in gradient of depthwiseConv2dNative: dilation rates greater than 1 are not yet supported. Got dilations '${i}'`);const[u,c]=e;y.assert(4===u.rank,()=>`Error in gradient of depthwiseConv2dNative: input must be rank 4, but got rank ${u.rank}.`),y.assert(4===c.rank,()=>`Error in gradient of depthwiseConv2dNative: filter must be rank 4, but got rank ${c.rank}.`),y.assert(u.shape[3]===c.shape[2],()=>`Error in gradient of depthwiseConv2d: number of input channels (${u.shape[3]}) must match the inChannels dimension in filter ${c.shape[2]}.`),y.assert(It(o,i),()=>`Error in gradient of depthwiseConv2d: Either strides or dilations must be  1. Got strides ${o} and dilations '${i}'.`),null!=a&&y.assert(y.isInt(s),()=>`Error in depthwiseConv2d: pad must be an integer when using, dimRoundingMode ${a} but got pad ${s}.`);const l=bt(u.shape,c.shape,o,i,s,a,!0);return{x:()=>co(u.shape,t,c,l),filter:()=>uo(u,t,c.shape,l)}}},ho={kernelName:_.O,inputsToSave:["x","filter"],gradFunc:(t,e,n)=>{const[r,o]=e,s={x:r,filter:o,dy:t},a={x:r,filter:o,dy:t};return{x:()=>l.a.runKernel(_.Q,s,n),filter:()=>l.a.runKernel(_.P,a,n)}}},po={kernelName:_.R,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e,o=Bt(n.shape,r.shape);return{a:()=>{const e=Tt(t,M(r,"float32")),s=_t(n.shape,o);return s.length>0?Et(tn(e,s),n.shape):e},b:()=>{let e=Xt(t,M(n,"float32"));const s=_t(r.shape,o);s.length>0&&(e=Et(tn(e,s),r.shape));const a=Je(r);return pn(Tt(e,M(a,"float32")))}}}},fo={kernelName:_.S,outputsToSave:[!0],gradFunc:(t,e)=>{const[n]=e,r=e=>e.eluDer(t,n),o={dy:t,y:n};return{x:()=>l.a.runKernelFunc(r,o,null,_.T)}}},go={kernelName:_.V,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e,r=Xt(On(pn(Je(n))),2/Math.sqrt(Math.PI));return{x:()=>Xt(t,r)}}},mo={kernelName:_.W,outputsToSave:[!0],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(t,n)}}},bo={kernelName:_.X,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(t,On(n))}}},xo={kernelName:_.bb,gradFunc:t=>({x:()=>ve(t)})},yo={kernelName:_.cb,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e,o=Bt(n.shape,r.shape);return{a:()=>{const e=Tt(t,M(r,"float32")),s=_t(n.shape,o);return s.length>0?Et(tn(e,s),n.shape):e},b:()=>{let e=Xt(t,M(n,"float32"));const s=_t(r.shape,o);s.length>0&&(e=Et(tn(e,s),r.shape));const a=Je(r);return pn(Tt(e,M(a,"float32")))}}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vo=Object(j.a)({rsqrt_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","rsqrt"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.rsqrt(e);return n([e]),r},n,null,_.hc)}}),wo={kernelName:_.eb,inputsToSave:["x","mean","variance","scale"],gradFunc:(t,e,n)=>{const{varianceEpsilon:r}=n,[o,s,a,i]=e,u=null==i?ie(1):i,c=_t(s.shape,o.shape),l=[];if(1===s.rank){for(let t=0;t<o.shape.length-1;++t)l.push(o.shape[t]);l.push(1)}const h=Ge(o,s),d=Xt(t,u),p=vo(dt(a,ie(r))),f=Xt(Xt(Xt(p,p),p),ie(-.5));return{x:()=>1===s.rank?Et(Xt(Xt(t,ln(Et(p,[1,1,1,s.shape[0]]),l)),u),o.shape):Et(Xt(Xt(t,p),u),o.shape),mean:()=>{let t=Xt(Xt(p,ie(-1)),d);return 1===s.rank&&(t=tn(t,c)),Et(t,s.shape)},variance:()=>{let t=Xt(Xt(f,h),d);return 1===s.rank&&(t=tn(t,c)),Et(t,s.shape)},scale:()=>{const e=Xt(h,p);let n=Xt(t,e);return 1===s.rank&&(n=tn(n,c)),Et(n,s.shape)},offset:()=>{let e=t;return 1===s.rank&&(e=tn(e,c)),Et(e,s.shape)}}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Co=Object(j.a)({unsortedSegmentSum_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n){const r=Object(B.a)(t,"x","unsortedSegmentSum"),o=Object(B.a)(e,"segmentIds","unsortedSegmentSum","int32");Object(y.assert)(Object(y.isInt)(n),()=>"numSegments must be of dtype int");const s={x:r,segmentIds:o},a={numSegments:n};return l.a.runKernelFunc((t,e)=>{const s=t.unsortedSegmentSum(r,o,n);return e([o]),s},s,null,_.Gc,a)}}),$o={kernelName:_.fb,inputsToSave:["x","indices"],gradFunc:(t,e,n)=>{const[r,o]=e,{axis:s}=n,a=Object(y.parseAxisParam)(s,r.shape)[0];return{x:()=>{const e=r.shape,n=o.size,i=e.slice(0,a),u=i.length,c=e.slice(s,e.length).slice(1),l=c.length,h=Oo(0,u),d=Oo(u+1,u+1+l),p=Io([i,[n],c]),f=Et(t,p),g=Et(o,[n]),m=Io([[u],h,d]),b=Kt(f,m);let x=Co(b,g,r.shape[a]);const y=Gt(m);return x=Kt(x,y),x},indices:()=>o}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oo(t,e){const n=[];for(let r=t;r<e;++r)n.push(r);return n}function Io(t){const e=[];for(let n=0;n<t.length;++n)for(let r=0;r<t[n].length;++r)e.push(t[n][r]);return e}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const So={kernelName:_.hb,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e;return{a:()=>ve(n),b:()=>ve(r)}}},Eo={kernelName:_.jb,gradFunc:t=>({x:()=>M(t,"float32")})},Ro={kernelName:_.lb,gradFunc:t=>({x:()=>ve(t)})},Ao={kernelName:_.mb,gradFunc:t=>({x:()=>ve(t)})},ko={kernelName:_.nb,gradFunc:t=>({x:()=>ve(t)})},To={kernelName:_.tb,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Tt(t,dt(n,1))}}},Fo={kernelName:_.sb,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Tt(t,M(n,"float32"))}}},No={kernelName:_.ub,inputsToSave:[],outputsToSave:[!0],gradFunc:(t,e,n)=>{const[r]=e,{axis:o}=n;return{logits:()=>{const e=On(r);return Ge(t,Xt(tn(t,o,!0),e))}}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Do=Object(j.a)({localResponseNormalizationBackprop_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r=5,o=1,s=1,a=.5){const i={x:t,y:e,dy:n},u={depthRadius:r,bias:o,alpha:s,beta:a};return l.a.runKernelFunc(i=>i.LRNGrad(n,t,e,r,o,s,a),i,null,_.pb,u)}}),_o={kernelName:_.ob,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(t,e,n)=>{const[r,o]=e,{depthRadius:s,bias:a,alpha:i,beta:u}=n;return{x:()=>Do(r,o,t,s,a,i,u)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function Bo(t,e,n,r,o){return e.rank<n.rank&&(e=Et(e,zt(e.shape,r))),t.rank<n.rank&&(t=Et(t,zt(t.shape,r))),{x:()=>{const r=Xt(t,M(jt(n,e),t.dtype));return null==o?r:Kt(r,o)}}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jo={kernelName:_.yb,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(t,e,n)=>{const r=n,{reductionIndices:o}=r,[s,a]=e,i=y.parseAxisParam(o,s.shape),u=Vt(i,s.rank),c=Bo(t,a,s,i,u);return{x:()=>{let t=c.x();return null!=u&&(t=Kt(t)),t}}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mo=Object(j.a)({less_:function(t,e){let n=Object(B.a)(t,"a","less"),r=Object(B.a)(e,"b","less");[n,r]=Object(ht.b)(n,r),Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc(t=>t.less(n,r),o,null,_.qb)}}),Po={kernelName:_.Eb,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e;return{a:()=>Xt(t,M(ze(n,r),"float32")),b:()=>Xt(t,M(Mo(n,r),"float32"))}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Lo=Object(j.a)({maxPool3dBackprop_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o,s=[1,1,1],a,i){const u=Object(B.a)(t,"dy","maxPool3dBackprop"),c=Object(B.a)(e,"input","maxPool3dBackprop"),h=Object(B.a)(n,"output","maxPool3dBackprop");let d=u,p=c,f=h,g=!1;4===c.rank&&(g=!0,d=Et(u,[1,u.shape[0],u.shape[1],u.shape[2],u.shape[3]]),p=Et(c,[1,c.shape[0],c.shape[1],c.shape[2],c.shape[3]]),f=Et(h,[1,h.shape[0],h.shape[1],h.shape[2],h.shape[3]])),y.assert(5===d.rank,()=>"Error in maxPool3dBackprop: dy must be rank 5 but got rank "+d.rank+"."),y.assert(5===p.rank,()=>"Error in maxPool3dBackprop: input must be rank 5 but got rank "+p.rank+"."),y.assert(5===f.rank,()=>"Error in maxPool3dBackprop: output must be rank 5 but got rank "+f.rank+"."),y.assert(It(o,s),()=>`Error in maxPool3dBackprop: Either strides or dilations must be 1. Got strides ${o} and dilations '${s}'`),null!=i&&y.assert(y.isInt(a),()=>`Error in maxPool3dBackprop: pad must be an integer when using, dimRoundingMode ${i} but got pad ${a}.`);const m={dy:d,input:p,output:f},b={filterSize:r,strides:o,dilations:s,pad:a,dimRoundingMode:i},x=l.a.runKernelFunc(t=>{const e=mt(p.shape,r,o,s,a,i);return t.maxPool3dBackprop(d,p,f,e)},m,null,_.Bb,b);return g?Et(x,[x.shape[1],x.shape[2],x.shape[3],x.shape[4]]):x}}),Wo={kernelName:_.Ab,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(t,e,n)=>{const[r,o]=e,{filterSize:s,strides:a,dilations:i,pad:u,dimRoundingMode:c}=n,l=null==i?[1,1,1]:i;return{x:()=>Lo(t,r,o,s,a,l,u,c)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zo=Object(j.a)({maxPoolBackprop_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o,s,a){const i=Object(B.a)(t,"dy","maxPoolBackprop"),u=Object(B.a)(e,"input","maxPoolBackprop"),c=Object(B.a)(n,"output","maxPoolBackprop");y.assert(u.rank===i.rank,()=>`Rank of input (${u.rank}) does not match rank of dy (${i.rank})`),y.assert(4===i.rank,()=>"Error in maxPoolBackprop: dy must be rank 4 but got rank "+i.rank+"."),y.assert(4===u.rank,()=>"Error in maxPoolBackprop: input must be rank 4 but got rank "+u.rank+"."),null!=a&&y.assert(y.isInt(s),()=>`Error in maxPoolBackprop: pad must be an integer when using, dimRoundingMode ${a} but got pad ${s}.`);const h={dy:i,input:u,output:c},d={filterSize:r,strides:o,pad:s,dimRoundingMode:a};return l.a.runKernelFunc(t=>{const e=gt(u.shape,r,o,1,s,a);return t.maxPoolBackprop(i,u,c,e)},h,null,_.Cb,d)}}),Uo={kernelName:_.zb,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(t,e,n)=>{const[r,o]=e,{filterSize:s,strides:a,pad:i}=n;return{x:()=>zo(t,r,o,s,a,i)}}},Vo={kernelName:_.Fb,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(t,e,n)=>{const r=n,{axis:o}=r,[s,a]=e,i=y.parseAxisParam(o,s.shape),u=Vt(i,s.rank),c=Bo(t,a,s,i,u);return{x:()=>{let t=c.x();return null!=u&&(t=Kt(t)),t}}}},Go={kernelName:_.Gb,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e;return{a:()=>Xt(t,M(Ue(n,r),"float32")),b:()=>Xt(t,M(dn(n,r),"float32"))}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ho=Object(j.a)({floor_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","floor"),n={x:e};return l.a.runKernelFunc(t=>t.floor(e),n,null,_.bb)}}),Ko={kernelName:_.Hb,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e,o=Bt(n.shape,r.shape);return{a:()=>{const e=_t(n.shape,o);return e.length>0?Et(tn(t,e),n.shape):t},b:()=>{const e=Xt(t,pn(Ho(Tt(n,r)))),s=_t(r.shape,o);return s.length>0?Et(tn(e,s),r.shape):e}}}},qo={kernelName:_.Ib,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e,o=Bt(n.shape,r.shape);return{a:()=>{const e=Xt(t,M(r,"float32")),s=_t(n.shape,o);return s.length>0?Et(tn(e,s),n.shape):e},b:()=>{const e=Xt(t,M(n,"float32")),s=_t(r.shape,o);return s.length>0?Et(tn(e,s),r.shape):e}}}},Xo={kernelName:_.Jb,gradFunc:t=>({x:()=>pn(t)})},Yo={kernelName:_.Ob,inputsToSave:["indices"],gradFunc:(t,e)=>{const n=e[0];return{indices:()=>re(n.shape,"float32")}}},Qo={kernelName:_.Pb,gradFunc:t=>({x:()=>ve(t)})},Zo={kernelName:_.Qb,inputsToSave:["x"],gradFunc:(t,e,n)=>{const r=e[0],{paddings:o}=n,s=o.map(t=>t[0]);return{x:()=>be(t,s,r.shape)}}},Jo={kernelName:_.Rb,inputsToSave:["a","b"],outputsToSave:[!0],gradFunc:(t,e)=>{const[n,r,o]=e,s=n,a=r,i=Bt(s.shape,a.shape);return{a:()=>{const e=M(a,"float32");let n=Xt(t,Xt(e,Qe(s,Ge(e,ie(1)))));const r=_t(s.shape,i);return r.length>0&&(n=tn(n,r)),Et(n,s.shape)},b:()=>{const e=dn(s,0),n=qe(e,Cn(s),ve(s));let r=Xt(t,Xt(o,n));const u=_t(a.shape,i);return u.length>0&&(r=tn(r,u)),Et(r,a.shape)}}}},ts={kernelName:_.Sb,inputsToSave:["x","alpha"],gradFunc:(t,e)=>{const[n,r]=e,o=dn(n,0);return{x:()=>qe(o,t,Xt(t,r)),alpha:()=>{let e=qe(o,ve(t),Xt(t,n));const s=_t(r.shape,t.shape);return s.length>0&&(e=tn(e,s)),Et(e,r.shape)}}}},es={kernelName:_.Wb,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Tt(t,pn(Je(n)))}}},ns={kernelName:_.Yb,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e,r=Xt(Ue(n,6),Tn(n));return{x:()=>Xt(t,M(r,"float32"))}}},rs={kernelName:_.Xb,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(t,M(Tn(n),"float32"))}}},os={kernelName:_.Zb,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Et(t,n.shape)}}},ss={kernelName:_.ac,inputsToSave:["images"],gradFunc:(t,e,n)=>{const[r]=e,o=e=>{const{alignCorners:o}=n;return e.resizeBilinearBackprop(t,r,o)},s={images:r};return{images:()=>l.a.runKernelFunc(o,s,null,_.bc,n)}}},as={kernelName:_.cc,inputsToSave:["images"],gradFunc:(t,e,n)=>{const[r]=e,o=e=>{const{alignCorners:o}=n;return e.resizeNearestNeighborBackprop(t,r,o)},s={images:r};return{images:()=>l.a.runKernelFunc(o,s,null,_.dc,n)}}},is={kernelName:_.ec,gradFunc:(t,e,n)=>{const{dims:r}=n,o=Object(y.parseAxisParam)(r,t.shape);return{x:()=>Oe(t,o)}}},us={kernelName:_.gc,gradFunc:t=>({x:()=>ve(t)})},cs={kernelName:_.hc,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>pn(Tt(t,Xt(Qe(n,1.5),2)))}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ls=Object(j.a)({logicalNot_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","logicalNot","bool"),n={x:e};return l.a.runKernelFunc(t=>t.logicalNot(e),n,null,_.wb)}}),hs={kernelName:_.ic,inputsToSave:["condition"],gradFunc:(t,e)=>{const[n]=e;return{condition:()=>M(ve(n),"float32"),t:()=>Xt(t,M(n,t.dtype)),e:()=>Xt(t,M(ls(n),t.dtype))}}},ds={kernelName:_.jc,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>{const e=dn(n,ie(0)),r=ie(qn),o=ie(Xn),s=Xt(t,o),a=Xt(Xt(t,r),On(M(n,"float32")));return qe(e,s,a)}}}},ps={kernelName:_.kc,outputsToSave:[!0],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(t,Xt(n,Ge(ie(1),n)))}}},fs={kernelName:_.lc,gradFunc:t=>({x:()=>ve(t)})};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gs=Object(j.a)({cos_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","cos"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.cos(e);return n([e]),r},n,null,_.G)}}),ms={kernelName:_.mc,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(gs(M(n,"float32")),t)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bs=Object(j.a)({cosh_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","cosh"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.cosh(e);return n([e]),r},n,null,_.H)}}),xs={kernelName:_.nc,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(bs(M(n,"float32")),t)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ys=Object(j.a)({pad_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n=0){const r=Object(B.a)(t,"x","pad");if(0===r.rank)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const o={paddings:e,constantValue:n},s={x:r};return l.a.runKernelFunc((t,o)=>(o([r]),t.pad(r,e,n)),s,null,_.Qb,o)}}),vs={kernelName:_.oc,inputsToSave:["x"],gradFunc:(t,e,n)=>{const[r]=e,{begin:o,size:s}=n,a=r.shape,[i,u]=ct(r,o,s),c=[];for(let e=0;e<t.rank;e++)c.push([i[e],a[e]-i[e]-u[e]]);return{x:()=>ys(t,c)}}},ws={kernelName:_.pc,outputsToSave:[!0],gradFunc:(t,e,n)=>{const[r]=e,{dim:o}=n,s=Xt(t,r);return{logits:()=>Ge(s,Xt(tn(s,[o],!0),r))}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cs=Object(j.a)({sigmoid_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","sigmoid"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.sigmoid(e);return n([r]),r},n,null,_.kc)}}),$s={kernelName:_.qc,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(t,Cs(n))}}},Os={kernelName:_.rc,gradFunc:(t,e,n)=>{const{blockShape:r,paddings:o}=n;return{x:()=>Qt(t,r,o)}}},Is={kernelName:_.sc,gradFunc:(t,e,n)=>{const{axis:r}=n;return{x:()=>de(t,r)}}},Ss={kernelName:_.tc,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Tt(t,Xt(Ze(M(n,"float32")),2))}}},Es={kernelName:_.uc,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(t,Xt(M(n,"float32"),2))}}},Rs={kernelName:_.vc,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e,o=ie(2);return{a:()=>Xt(t,Xt(o,Ge(n,r))),b:()=>Xt(t,Xt(o,Ge(r,n)))}}},As={kernelName:_.wc,gradFunc:t=>({x:()=>ve(t)})},ks={kernelName:_.yc,inputsToSave:["a","b"],gradFunc:(t,e)=>{const[n,r]=e,o=Bt(n.shape,r.shape);return{a:()=>{let e=t;const r=_t(n.shape,o);return r.length>0&&(e=tn(e,r)),Et(e,n.shape)},b:()=>{let e=t;const n=_t(r.shape,o);return n.length>0&&(e=tn(e,n)),Et(pn(e),r.shape)}}}},Ts={kernelName:_.zc,inputsToSave:["x"],gradFunc:(t,e,n)=>{const[r]=e,o=r.shape.slice(),{axis:s}=n;Object(y.parseAxisParam)(s,r.shape).forEach(t=>{o[t]=1});const a=Et(t,o),i=Xt(a,bn(r.shape,"float32"));return{x:()=>i}}},Fs={kernelName:_.Ac,inputsToSave:["x"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Tt(t,Je(gs(n)))}}},Ns={kernelName:_.Bc,outputsToSave:[!0],gradFunc:(t,e)=>{const[n]=e;return{x:()=>Xt(Ge(ie(1),Je(n)),t)}}},Ds={kernelName:_.Cc,inputsToSave:["x"],gradFunc:(t,e,n)=>{const[r]=e,{reps:o}=n;return{x:()=>{let e=ve(r);if(1===r.rank)for(let n=0;n<o[0];++n)e=dt(e,be(t,[n*r.shape[0]],[r.shape[0]]));else if(2===r.rank)for(let n=0;n<o[0];++n)for(let s=0;s<o[1];++s)e=dt(e,be(t,[n*r.shape[0],s*r.shape[1]],[r.shape[0],r.shape[1]]));else if(3===r.rank)for(let n=0;n<o[0];++n)for(let s=0;s<o[1];++s)for(let a=0;a<o[2];++a)e=dt(e,be(t,[n*r.shape[0],s*r.shape[1],a*r.shape[2]],[r.shape[0],r.shape[1],r.shape[2]]));else{if(4!==r.rank)throw new Error("Gradient for tile operation is not implemented for rank-"+r.rank+" tensors yet.");for(let n=0;n<o[0];++n)for(let s=0;s<o[1];++s)for(let a=0;a<o[2];++a)for(let i=0;i<o[3];++i)e=dt(e,be(t,[n*r.shape[0],s*r.shape[1],a*r.shape[2],i*r.shape[3]],[r.shape[0],r.shape[1],r.shape[2],r.shape[3]]))}return e}}}},_s={kernelName:_.Ec,gradFunc:(t,e,n)=>{const r=n,{perm:o}=r,s=Gt(o);return{x:()=>Kt(t,s)}}},Bs={kernelName:_.Fc,gradFunc:(t,e,n)=>{const r=n,{axis:o}=r;return{value:()=>fe(t,o)}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const js=Object(j.a)({gather_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n=0){const r=Object(B.a)(t,"x","gather"),o=Object(B.a)(e,"indices","gather","int32"),s={x:r,indices:o},a={axis:n};return l.a.runKernelFunc((t,e)=>{const s=Object(y.parseAxisParam)(n,r.shape)[0],a=fr(r,o,s),i=t.gather(r,Et(o,[o.size]),s);return e([r,o]),Et(i,a.outputShape)},s,null,_.fb,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ms=Object(j.a)({maximum_:function(t,e){let n=Object(B.a)(t,"a","maximum"),r=Object(B.a)(e,"b","maximum");[n,r]=Object(ht.b)(n,r),"bool"===n.dtype&&(n=M(n,"int32"),r=M(r,"int32")),Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.maximum(n,r);return e([n,r]),o},o,null,_.Eb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const Ps=[Sr,Er,Rr,Ar,kr,Tr,Fr,Nr,Dr,_r,Br,jr,Pr,Wr,zr,Ur,Vr,Gr,Hr,Kr,qr,Zr,Qr,eo,ro,so,io,lo,ho,po,fo,go,mo,bo,yo,xo,wo,$o,So,Eo,Ro,Ao,ko,To,Fo,No,_o,jo,jo,Po,Wo,Uo,Vo,Go,Ko,qo,Xo,Yo,Qo,Zo,Zo,Jo,ts,es,ns,rs,os,ss,as,is,us,cs,hs,ds,ps,fs,ms,xs,vs,ws,$s,Os,Os,Is,Is,Ss,Rs,Es,As,ks,Ts,Fs,Ns,Ds,_s,Bs,{kernelName:_.Gc,inputsToSave:["segmentIds"],gradFunc:(t,e)=>{const[n]=e;return{x:()=>function(t,e){const n=Ms(e,ve(e)),r=js(t,n);let o=ze(e,ie(0,"int32"));const s=r.rank-o.rank;for(let t=0;t<s;++t)o=pe(o,t+1);o=Ve(o,bn(r.shape,"bool"));const a=ve(r);return qe(o,r,a)}(t,n)}}},{kernelName:_.Hc,gradFunc:t=>({x:()=>ve(t)})}];
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */for(const t of Ps)Object(En.d)(t);
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.abs=function(){return this.throwIfDisposed(),Xe(this)};const Ls=Object(j.a)({acos_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","acos"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.acos(e);return n([e]),r},n,null,_.b)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.acos=function(){return this.throwIfDisposed(),Ls(this)};const Ws=Object(j.a)({acosh_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","acosh"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.acosh(e);return n([e]),r},n,null,_.c)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.acosh=function(){return this.throwIfDisposed(),Ws(this)};const zs=Object(j.a)({mod_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){let n=Object(B.a)(t,"a","mod"),r=Object(B.a)(e,"b","mod");[n,r]=Object(ht.b)(n,r);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.mod(n,r);return e([n,r]),o},o,null,_.Hb)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Us=Object(j.a)({addStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","addStrict"),r=Object(B.a)(e,"b","addStrict");return y.assertShapesMatch(n.shape,r.shape,"Error in addStrict: "),dt(n,r)}}),Vs=Object(j.a)({divStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","div"),r=Object(B.a)(e,"b","div");return y.assertShapesMatch(n.shape,r.shape,"Error in divideStrict: "),Tt(n,r)}}),Gs=Object(j.a)({maximumStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","maximumStrict"),r=Object(B.a)(e,"b","maximumStrict");return y.assertShapesMatch(n.shape,r.shape,"Error in maximumStrict: "),Ms(n,r)}}),Hs=Object(j.a)({minimumStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","minimumStrict"),r=Object(B.a)(e,"b","minimumStrict");return y.assertShapesMatch(n.shape,r.shape,"Error in minimumStrict: "),wn(n,r)}}),Ks=Object(j.a)({modStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","modStrict"),r=Object(B.a)(e,"b","modStrict");return y.assertShapesMatch(n.shape,r.shape,"Error in modStrict: "),zs(n,r)}}),qs=Object(j.a)({mulStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","mul"),r=Object(B.a)(e,"b","mul");return y.assertShapesMatch(n.shape,r.shape,"Error in multiplyStrict: "),Xt(n,r)}}),Xs=Object(j.a)({powStrict_:function(t,e){return rn("strict variants of ops have been deprecated and will be removed in future"),y.assertShapesMatch(t.shape,e.shape,"Error in powStrict: "),Qe(t,e)}}),Ys=Object(j.a)({squaredDifferenceStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","squaredDifferenceStrict"),r=Object(B.a)(e,"b","squaredDifferenceStrict");return y.assertShapesMatch(n.shape,r.shape,"Error in squaredDifferenceStrict: "),$n(n,r)}}),Qs=Object(j.a)({subStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","subStrict"),r=Object(B.a)(e,"b","subStrict");return y.assertShapesMatch(n.shape,r.shape,"Error in subStrict: "),Ge(n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.addStrict=function(t){return this.throwIfDisposed(),Us(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.add=function(t){return this.throwIfDisposed(),dt(this,t)};const Zs=Object(j.a)({all_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=null,n=!1){let r=Object(B.a)(t,"x","all","bool");const o={x:r},s={axis:e,keepDims:n};return l.a.runKernelFunc(t=>{const o=Object(y.parseAxisParam)(e,r.shape);let s=o;const a=Vt(s,r.rank);null!=a&&(r=Kt(r,a),s=Ht(s.length,r.rank));const i=t.all(r,s);if(n){const t=zt(i.shape,o);return Et(i,t)}return i},o,null,_.f,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.all=function(t,e){return this.throwIfDisposed(),Zs(this,t,e)};const Js=Object(j.a)({any_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=null,n=!1){let r=Object(B.a)(t,"x","any","bool");const o={x:r},s={axis:e,keepDims:n};return l.a.runKernelFunc(t=>{const o=Object(y.parseAxisParam)(e,r.shape);let s=o;const a=Vt(s,r.rank);null!=a&&(r=Kt(r,a),s=Ht(s.length,r.rank));const i=t.any(r,s);if(n){const t=zt(i.shape,o);return Et(i,t)}return i},o,null,_.g,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.any=function(t,e){return this.throwIfDisposed(),Js(this,t,e)};const ta=Object(j.a)({argMax_:
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=0){let n=Object(B.a)(t,"x","argMax");const r={x:n},o={axis:e};return l.a.runKernelFunc((t,r)=>{r([n]);let o=y.parseAxisParam(e,n.shape);const s=Vt(o,n.rank);return null!=s&&(n=Kt(n,s),o=Ht(o.length,n.rank)),t.argMax(n,o[0])},r,null,_.h,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.argMax=function(t){return this.throwIfDisposed(),ta(this,t)};const ea=Object(j.a)({argMin_:
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=0){let n=Object(B.a)(t,"x","argMin");const r={x:n},o={axis:e};return l.a.runKernelFunc((t,r)=>{r([n]),null==e&&(e=0);let o=y.parseAxisParam(e,n.shape);const s=Vt(o,n.rank);return null!=s&&(n=Kt(n,s),o=Ht(o.length,n.rank)),t.argMin(n,o[0])},r,null,_.i,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.argMin=function(t){return this.throwIfDisposed(),ea(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.asScalar=function(){return this.throwIfDisposed(),Object(y.assert)(1===this.size,()=>"The array must have only 1 element."),Et(this,[])},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.asType=function(t){return this.throwIfDisposed(),M(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.as1D=function(){return this.throwIfDisposed(),Et(this,[this.size])},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.as2D=function(t,e){return this.throwIfDisposed(),Et(this,[t,e])},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.as3D=function(t,e,n){return this.throwIfDisposed(),Et(this,[t,e,n])},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.as4D=function(t,e,n,r){return this.throwIfDisposed(),Et(this,[t,e,n,r])},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.as5D=function(t,e,n,r,o){return this.throwIfDisposed(),Et(this,[t,e,n,r,o])};const na=Object(j.a)({asin_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","asin"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.asin(e);return n([e]),r},n,null,_.j)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.asin=function(){return this.throwIfDisposed(),na(this)};const ra=Object(j.a)({asinh_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","asinh"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.asinh(e);return n([e]),r},n,null,_.k)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.asinh=function(){return this.throwIfDisposed(),ra(this)};const oa=Object(j.a)({atan_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","atan"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.atan(e);return n([e]),r},n,null,_.l)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.atan=function(){return this.throwIfDisposed(),oa(this)};const sa=Object(j.a)({atan2_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){let n=Object(B.a)(t,"a","atan2"),r=Object(B.a)(e,"b","atan2");[n,r]=Object(ht.b)(n,r);const o={a:n,b:r};return l.a.runKernelFunc((t,e)=>{const o=t.atan2(n,r);return e([n,r]),o},o,null,_.m)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.atan2=function(t){return this.throwIfDisposed(),sa(this,t)};const aa=Object(j.a)({atanh_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","atanh"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.atanh(e);return n([e]),r},n,null,_.n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ia(t){return null==t?null:0===t.rank?Et(t,[t.size]):1===t.rank?t:2===t.rank?Et(t,[1,1,t.shape[0],t.shape[1]]):3===t.rank?Et(t,[1,t.shape[0],t.shape[1],t.shape[2]]):t}N.a.prototype.atanh=function(){return this.throwIfDisposed(),aa(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.avgPool=function(t,e,n,r){return this.throwIfDisposed(),Yt(this,t,e,n,r)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.batchToSpaceND=function(t,e){return this.throwIfDisposed(),Qt(this,t,e)};const ua=Object(j.a)({batchNorm_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o,s){null==s&&(s=.001);const a=Object(B.a)(t,"x","batchNorm"),i=Object(B.a)(e,"mean","batchNorm"),u=Object(B.a)(n,"variance","batchNorm");let c,h;null!=o&&(c=Object(B.a)(o,"scale","batchNorm")),null!=r&&(h=Object(B.a)(r,"offset","batchNorm")),y.assert(i.rank===u.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),y.assert(null==h||i.rank===h.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),y.assert(null==c||i.rank===c.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const d=function(t){let e;return e=0===t.rank||1===t.rank?Et(t,[1,1,1,t.size]):2===t.rank?Et(t,[1,1,t.shape[0],t.shape[1]]):3===t.rank?Et(t,[1,t.shape[0],t.shape[1],t.shape[2]]):t,e}(a),p={x:d,scale:c,offset:h,mean:i,variance:u},f={varianceEpsilon:s},g=l.a.runKernelFunc((t,e)=>(e([d,i,u,c]),t.batchNorm(d,ia(i),ia(u),ia(h),ia(c),s)),p,null,_.eb,f);return Et(g,a.shape)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.batchNorm=function(t,e,n,r,o){return this.throwIfDisposed(),ua(this,t,e,n,r,o)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.broadcastTo=function(t){return this.throwIfDisposed(),Ke(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.cast=function(t){return this.throwIfDisposed(),M(this,t)};const ca=Object(j.a)({ceil_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","ceil"),n={x:e};return l.a.runKernelFunc(t=>t.ceil(e),n,null,_.w)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.ceil=function(){return this.throwIfDisposed(),ca(this)};const la=Object(j.a)({clipByValue_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n){const r=Object(B.a)(t,"x","clipByValue");y.assert(e<=n,()=>`Error in clip: min (${e}) must be less than or equal to max (${n}).`);const o={x:r},s={clipValueMin:e,clipValueMax:n};return l.a.runKernelFunc((t,o)=>{const s=t.clip(r,e,n);return o([r]),s},o,null,_.x,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.clipByValue=function(t,e){return this.throwIfDisposed(),la(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.concat=function(t,e){return this.throwIfDisposed(),t instanceof N.a&&(t=[t]),de([this,...t],e)};const ha=Object(j.a)({conv1d_:function(t,e,n,r,o="NWC",s=1,a){const i=Object(B.a)(t,"x","conv1d"),u=Object(B.a)(e,"filter","conv1d");let c=i,l=!1;2===i.rank&&(l=!0,c=Et(i,[1,i.shape[0],i.shape[1]])),y.assert(3===c.rank,()=>`Error in conv1d: input must be rank 3, but got rank ${c.rank}.`),y.assert(3===u.rank,()=>"Error in conv1d: filter must be rank 3, but got rank "+u.rank+"."),null!=a&&y.assert(y.isInt(r),()=>`Error in conv1d: pad must be an integer when using, dimRoundingMode ${a} but got pad ${r}.`),y.assert(c.shape[2]===u.shape[1],()=>`Error in conv1d: depth of input (${c.shape[2]}) must match input depth for filter ${u.shape[1]}.`),y.assert(It(n,s),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${s}'`),y.assert("NWC"===o,()=>`Error in conv1d: got dataFormat of ${o} but only NWC is currently supported.`);const h=Et(u,[1,u.shape[0],u.shape[1],u.shape[2]]),d=Et(c,[c.shape[0],1,c.shape[1],c.shape[2]]),p=Rt(d,h,[1,n],r,"NHWC",[1,s],a);return Et(p,l?[p.shape[2],p.shape[3]]:[p.shape[0],p.shape[2],p.shape[3]])}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.conv1d=function(t,e,n,r,o,s){return this.throwIfDisposed(),ha(this,t,e,n,r,o,s)};const da=Object(j.a)({conv2dTranspose_:function(t,e,n,r,o,s){const a=Object(B.a)(t,"x","conv2dTranspose"),i=Object(B.a)(e,"filter","conv2dTranspose");return Yr(n,a,i,r,o,"NHWC",s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.conv2dTranspose=function(t,e,n,r,o){return this.throwIfDisposed(),da(this,t,e,n,r,o)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.conv2d=function(t,e,n,r,o,s){return this.throwIfDisposed(),Rt(this,t,e,n,r,o,s)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.cos=function(){return this.throwIfDisposed(),gs(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.cosh=function(){return this.throwIfDisposed(),bs(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.cumsum=function(t,e,n){return this.throwIfDisposed(),ao(this,t,e,n)};const pa=Object(j.a)({depthToSpace_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n="NHWC"){const r=Object(B.a)(t,"x","depthToSpace"),o="NHWC"===n?r.shape[1]:r.shape[2],s="NHWC"===n?r.shape[2]:r.shape[3],a="NHWC"===n?r.shape[3]:r.shape[1];y.assert(o*e>=0,()=>`Negative dimension size caused by overflow when multiplying\n    ${o} and ${e}  for depthToSpace with input shape\n    ${r.shape}`),y.assert(s*e>=0,()=>`Negative dimension size caused by overflow when multiplying\n    ${s} and ${e} for depthToSpace with input shape\n        ${r.shape}`),y.assert(a%(e*e)==0,()=>`Dimension size must be evenly divisible by ${e*e} but is ${a} for depthToSpace with input shape ${r.shape}`);const i={x:r},u={blockSize:e,dataFormat:n};return l.a.runKernelFunc(t=>t.depthToSpace(r,e,n),i,null,_.K,u)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.depthToSpace=function(t,e){return this.throwIfDisposed(),pa(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.depthwiseConv2D=function(t,e,n,r,o,s){return rn("depthwiseConv2D is deprecated, use depthwiseConv2d instead"),this.throwIfDisposed(),At(this,t,e,n,r,o,s)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.depthwiseConv2d=function(t,e,n,r,o,s){return this.throwIfDisposed(),At(this,t,e,n,r,o,s)};const fa=Object(j.a)({dilation2d_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o=[1,1],s="NHWC"){const a=Object(B.a)(t,"x","dilation2d"),i=Object(B.a)(e,"filter","dilation2d");y.assert(3===a.rank||4===a.rank,()=>"Error in dilation2d: input must be rank 3 or 4, but got rank "+a.rank+"."),y.assert(3===i.rank,()=>"Error in dilation2d: filter must be rank 3, but got rank "+i.rank+"."),y.assert("NHWC"===s,()=>"Error in dilation2d: Only NHWC is currently supported, but got dataFormat of "+s);let u=a,c=!1;3===a.rank&&(u=Et(a,[1,a.shape[0],a.shape[1],a.shape[2]]),c=!0);const h={x:u,filter:i},d={strides:n,pad:r,dilations:o},p=l.a.runKernel(_.O,h,d);return c?Et(p,[p.shape[1],p.shape[2],p.shape[3]]):p}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.dilation2d=function(t,e,n,r,o){return this.throwIfDisposed(),fa(this,t,e,n,r,o)};const ga=Object(j.a)({divNoNan_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){let n=Object(B.a)(t,"a","div"),r=Object(B.a)(e,"b","div");[n,r]=Object(ht.b)(n,r);const o=Tt(n,r),s=ve(o),a=jt(r,s);return qe(a,s,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.divNoNan=function(t){return this.throwIfDisposed(),ga(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.divStrict=function(t){return this.throwIfDisposed(),Vs(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.div=function(t){return this.throwIfDisposed(),Tt(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.dot=function(t){return this.throwIfDisposed(),Nt(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.elu=function(){return this.throwIfDisposed(),Rn(this)};const ma=Object(j.a)({equalStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","equalStrict"),r=Object(B.a)(e,"b","equalStrict");return Object(y.assertShapesMatch)(n.shape,r.shape,"Error in equalStrict: "),jt(n,r)}}),ba=Object(j.a)({greaterEqualStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","greaterEqualStrict"),r=Object(B.a)(e,"b","greaterEqualStrict");return Object(y.assertShapesMatch)(n.shape,r.shape,"Error in greaterEqualStrict: "),ze(n,r)}}),xa=Object(j.a)({greaterStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","greaterStrict"),r=Object(B.a)(e,"b","greaterStrict");return Object(y.assertShapesMatch)(n.shape,r.shape,"Error in greaterStrict: "),dn(n,r)}}),ya=Object(j.a)({lessEqualStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","lessEqualStrict"),r=Object(B.a)(e,"b","lessEqualStrict");return Object(y.assertShapesMatch)(n.shape,r.shape,"Error in lessEqualStrict: "),Ue(n,r)}}),va=Object(j.a)({lessStrict_:function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","lessStrict"),r=Object(B.a)(e,"b","lessStrict");return Object(y.assertShapesMatch)(n.shape,r.shape,"Error in lessStrict: "),Mo(n,r)}}),wa=Object(j.a)({notEqualStrict_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){rn("strict variants of ops have been deprecated and will be removed in future");const n=Object(B.a)(t,"a","notEqualStrict"),r=Object(B.a)(e,"b","notEqualStrict");return Object(y.assertShapesMatch)(n.shape,r.shape,"Error in notEqualStrict: "),yn(n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.equalStrict=function(t){return this.throwIfDisposed(),ma(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.equal=function(t){return this.throwIfDisposed(),jt(this,t)};const Ca=Object(j.a)({erf_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){let e=Object(B.a)(t,"x","erf");y.assert("int32"===e.dtype||"float32"===e.dtype,()=>"Input dtype must be `int32` or `float32`."),"int32"===e.dtype&&(e=M(e,"float32"));const n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.erf(e);return n([e]),r},n,null,_.V)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.erf=function(){return this.throwIfDisposed(),Ca(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.exp=function(){return this.throwIfDisposed(),On(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.expandDims=function(t){return this.throwIfDisposed(),pe(this,t)};const $a=Object(j.a)({expm1_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","expm1"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.expm1(e);return n([e]),r},n,null,_.X)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.expm1=function(){return this.throwIfDisposed(),$a(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.fft=function(){return this.throwIfDisposed(),we(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.flatten=function(){return this.throwIfDisposed(),Et(this,[this.size])},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.floor=function(){return this.throwIfDisposed(),Ho(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.floorDiv=function(t){return this.throwIfDisposed(),kt(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.gather=function(t,e){return this.throwIfDisposed(),js(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.greaterEqualStrict=function(t){return this.throwIfDisposed(),ba(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.greaterEqual=function(t){return this.throwIfDisposed(),ze(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.greaterStrict=function(t){return this.throwIfDisposed(),xa(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.greater=function(t){return this.throwIfDisposed(),dn(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.ifft=function(){return this.throwIfDisposed(),$e(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.irfft=function(){return this.throwIfDisposed(),Ie(this)};const Oa=Object(j.a)({isFinite_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","isFinite"),n={x:e};return l.a.runKernelFunc(t=>t.isFinite(e),n,null,_.lb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.isFinite=function(){return this.throwIfDisposed(),Oa(this)};const Ia=Object(j.a)({isInf_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","isInf"),n={x:e};return l.a.runKernelFunc(t=>t.isInf(e),n,null,_.mb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.isInf=function(){return this.throwIfDisposed(),Ia(this)};const Sa=Object(j.a)({isNaN_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","isNaN"),n={x:e};return l.a.runKernelFunc(t=>t.isNaN(e),n,null,_.nb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.isNaN=function(){return this.throwIfDisposed(),Sa(this)};const Ea=Object(j.a)({leakyRelu_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=.2){const n=Object(B.a)(t,"x","leakyRelu");return Ms(Xt(ie(e),n),n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.leakyRelu=function(t){return this.throwIfDisposed(),Ea(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.lessEqualStrict=function(t){return this.throwIfDisposed(),ya(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.lessEqual=function(t){return this.throwIfDisposed(),Ue(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.lessStrict=function(t){return this.throwIfDisposed(),va(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.less=function(t){return this.throwIfDisposed(),Mo(this,t)};const Ra=Object(j.a)({localResponseNormalization_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=5,n=1,r=1,o=.5){const s=Object(B.a)(t,"x","localResponseNormalization");y.assert(4===s.rank||3===s.rank,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got\n               rank ${s.rank}.`),y.assert(y.isInt(e),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${e}.`);let a=s,i=!1;3===s.rank&&(i=!0,a=Et(s,[1,s.shape[0],s.shape[1],s.shape[2]]));const u={x:a},c={depthRadius:e,bias:n,alpha:r,beta:o},h=l.a.runKernelFunc((t,s)=>{const i=t.localResponseNormalization4D(a,e,n,r,o);return s([a,i]),i},u,null,_.ob,c);return i?Et(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.localResponseNormalization=function(t,e,n,r){return this.throwIfDisposed(),Ra(this,t,e,n,r)};const Aa=Object(j.a)({softplus_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","softplus"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.softplus(e);return n([e]),r},n,null,_.qc)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ka=Object(j.a)({logSigmoid_:function(t){const e=Object(B.a)(t,"x","logSigmoid");return mn(t=>({value:pn(Aa(pn(t))),gradFunc:e=>Xt(e,Cs(pn(t)))}))(e)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.logSigmoid=function(){return this.throwIfDisposed(),ka(this)};const Ta=Object(j.a)({logSoftmax_:
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=-1){const n=Object(B.a)(t,"logits","logSoftmax");if(-1===e&&(e=n.rank-1),e!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${e}`);const r={logits:n},o={axis:e};return l.a.runKernelFunc((n,r)=>{const o=qt(t,e,!0),s=Ge(t,o),a=Ge(M(s,"float32"),Cn(tn(On(s),e,!0)));return r([a]),a},r,null,_.ub,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.logSoftmax=function(t){return this.throwIfDisposed(),Ta(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.logSumExp=function(t,e){return this.throwIfDisposed(),Sn(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.log=function(){return this.throwIfDisposed(),Cn(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.log1p=function(){return this.throwIfDisposed(),In(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.logicalAnd=function(t){return this.throwIfDisposed(),Ve(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.logicalNot=function(){return this.throwIfDisposed(),ls(this)};const Fa=Object(j.a)({logicalOr_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){const n=Object(B.a)(t,"a","logicalOr","bool"),r=Object(B.a)(e,"b","logicalOr","bool");Bt(n.shape,r.shape);const o={a:n,b:r};return l.a.runKernelFunc(t=>t.logicalOr(n,r),o,null,_.xb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.logicalOr=function(t){return this.throwIfDisposed(),Fa(this,t)};const Na=Object(j.a)({logicalXor_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e){const n=Object(B.a)(t,"a","logicalXor","bool"),r=Object(B.a)(e,"b","logicalXor","bool");return Bt(n.shape,r.shape),Ve(Fa(t,e),ls(Ve(t,e)))}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.logicalXor=function(t){return this.throwIfDisposed(),Na(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.matMul=function(t,e,n){return this.throwIfDisposed(),Ft(this,t,e,n)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.maxPool=function(t,e,n,r){return this.throwIfDisposed(),Zt(this,t,e,n,r)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.max=function(t,e){return this.throwIfDisposed(),qt(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.maximumStrict=function(t){return this.throwIfDisposed(),Gs(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.maximum=function(t){return this.throwIfDisposed(),Ms(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.mean=function(t,e){return this.throwIfDisposed(),xn(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.min=function(t,e){return this.throwIfDisposed(),Ye(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.minimumStrict=function(t){return this.throwIfDisposed(),Hs(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.minimum=function(t){return this.throwIfDisposed(),wn(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.modStrict=function(t){return this.throwIfDisposed(),Ks(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.mod=function(t){return this.throwIfDisposed(),zs(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.mulStrict=function(t){return this.throwIfDisposed(),qs(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.mul=function(t){return this.throwIfDisposed(),Xt(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.neg=function(){return this.throwIfDisposed(),pn(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.norm=function(t,e,n){return this.throwIfDisposed(),en(this,t,e,n)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.notEqualStrict=function(t){return this.throwIfDisposed(),wa(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.notEqual=function(t){return this.throwIfDisposed(),yn(this,t)};const Da=Object(j.a)({oneHot_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n=1,r=0){if(e<2)throw new Error("Error in oneHot: depth must be >=2, but it is "+e);const o=Object(B.a)(t,"indices","oneHot","int32"),s=[...o.shape,e],a={indices:o},i={depth:e,onValue:n,offValue:r};return l.a.runKernelFunc((t,a)=>(a([o]),Et(t.oneHot(Et(o,[o.size]),e,n,r),s)),a,null,_.Ob,i)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.oneHot=function(t,e=1,n=0){return this.throwIfDisposed(),Da(this,t,e,n)};const _a=Object(j.a)({onesLike_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","onesLike"),n={x:e};return l.a.runKernelFunc((t,n)=>{if("complex64"===e.dtype){const t=_a(se(e)),n=ve(Mt(e));return Object(pt.a)(t,n)}return t.onesLike(e)},n,null,_.Pb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.onesLike=function(){return this.throwIfDisposed(),_a(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.pad=function(t,e){return this.throwIfDisposed(),ys(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.pool=function(t,e,n,r,o){return this.throwIfDisposed(),te(this,t,e,n,r,o)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.powStrict=function(t){return this.throwIfDisposed(),Xs(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.pow=function(t){return this.throwIfDisposed(),Qe(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.prelu=function(t){return this.throwIfDisposed(),An(this,t)};const Ba=Object(j.a)({prod_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=null,n=!1){let r=Object(B.a)(t,"x","prod");const o={x:r},s={axis:e,keepDims:n};return l.a.runKernelFunc(t=>{"bool"===r.dtype&&(r=M(r,"int32"));const o=Object(y.parseAxisParam)(e,r.shape),s=Vt(o,r.rank);let a=o,i=r;null!=s&&(i=Kt(r,s),a=Ht(a.length,r.rank));let u=t.prod(i,a);if(n){const t=zt(u.shape,o);u=Et(u,t)}return u},o,null,_.Tb,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.prod=function(t,e){return this.throwIfDisposed(),Ba(this,t,e)};const ja=Object(j.a)({reciprocal_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","reciprocal"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.reciprocal(e);return n([e]),r},n,null,_.Wb)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.reciprocal=function(){return this.throwIfDisposed(),ja(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.relu=function(){return this.throwIfDisposed(),ae(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.relu6=function(){return this.throwIfDisposed(),kn(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.reshapeAs=function(t){return this.throwIfDisposed(),Et(this,t.shape)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.reshape=function(t){return this.throwIfDisposed(),Et(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.resizeBilinear=function(t,e){return this.throwIfDisposed(),Le(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.resizeNearestNeighbor=function(t,e){return this.throwIfDisposed(),We(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.reverse=function(t){return this.throwIfDisposed(),Oe(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.rfft=function(){return this.throwIfDisposed(),Ce(this)};const Ma=Object(j.a)({round_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","round"),n={x:e};return l.a.runKernelFunc(t=>t.round(e),n,null,_.gc)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.round=function(){return this.throwIfDisposed(),Ma(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.rsqrt=function(){return this.throwIfDisposed(),vo(this)};const Pa=Object(j.a)({selu_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","selu"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.selu(e);return n([e]),r},n,null,_.jc)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.selu=function(){return this.throwIfDisposed(),Pa(this)};const La=Object(j.a)({separableConv2d_:function(t,e,n,r,o,s=[1,1],a="NHWC"){const i=Object(B.a)(t,"x","separableConv2d"),u=Object(B.a)(e,"depthwiseFilter","separableConv2d"),c=Object(B.a)(n,"pointwiseFilter","separableConv2d");let l=i,h=!1;if(3===i.rank&&(h=!0,l=Et(i,[1,i.shape[0],i.shape[1],i.shape[2]])),"NCHW"===a)throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");y.assert(4===l.rank,()=>`Error in separableConv2d: input must be rank 4, but got rank ${l.rank}.`),y.assert(4===u.rank,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${u.rank}.`),y.assert(4===c.rank,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${u.rank}.`),y.assert(1===c.shape[0],()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${c.shape[0]}.`),y.assert(1===c.shape[1],()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${c.shape[1]}.`);const d=u.shape[2],p=u.shape[3];y.assert(c.shape[2]===d*p,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${d*p}, but got ${c.shape[2]}.`);const f=At(l,u,r,o,a,s),g=Rt(f,c,1,"valid",a);return h?Et(g,[g.shape[1],g.shape[2],g.shape[3]]):g}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.separableConv2d=function(t,e,n,r,o,s){return this.throwIfDisposed(),La(this,t,e,n,r,o,s)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.sigmoid=function(){return this.throwIfDisposed(),Cs(this)};const Wa=Object(j.a)({sign_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","sign"),n={x:e};return l.a.runKernelFunc(t=>t.sign(e),n,null,_.lc)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.sign=function(){return this.throwIfDisposed(),Wa(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.sin=function(){return this.throwIfDisposed(),no(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.sinh=function(){return this.throwIfDisposed(),oo(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.slice=function(t,e){return this.throwIfDisposed(),be(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.softmax=function(t){return this.throwIfDisposed(),ue(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.softplus=function(){return this.throwIfDisposed(),Aa(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.spaceToBatchND=function(t,e){return this.throwIfDisposed(),Jt(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.split=function(t,e){return this.throwIfDisposed(),ye(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.sqrt=function(){return this.throwIfDisposed(),Ze(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.square=function(){return this.throwIfDisposed(),Je(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.squaredDifference=function(t){return this.throwIfDisposed(),$n(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.squaredDifferenceStrict=function(t){return this.throwIfDisposed(),Ys(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.squeeze=function(t){return this.throwIfDisposed(),nn(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.stack=function(t,e){this.throwIfDisposed();const n=t instanceof N.a?[this,t]:[this,...t];return fe(n,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.step=function(t){return this.throwIfDisposed(),Tn(this,t)};const za=Object(j.a)({stridedSlice_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o=0,s=0,a=0,i=0,u=0){let c=Object(B.a)(t,"x","stridedSlice");const h={x:c},d={begin:e,end:n,strides:r,beginMask:o,endMask:s,ellipsisMask:a,newAxisMask:i,shrinkAxisMask:u};return l.a.runKernelFunc(t=>{null==r&&(r=new Array(e.length));const l=Y(a);if(l.length>1)throw new Error("Multiple ellipses in slice is not allowed.");if(0!==a&&0!==i)throw new Error("Using both ellipsisMask and newAxisMask is not yet supported.");if(0!==a&&0!==u)throw new Error("Using both ellipsisMask and shrinkAxisMask is not yet supported.");const h=c.rank-e.length,d=Y(i),p=c.shape.slice();d.forEach(t=>{e[t]=0,n[t]=1,p.splice(t,0,1)}),c=Et(c,p);const{begin:f,end:g,strides:m}=et(c.shape,l,h,e,n,r,o,s,a);e=f,n=g,r=m;const b=Y(u);b.forEach(t=>{n[t]=e[t]+1,r[t]=1});const x=Q(e,n,r),y=x.filter((t,e)=>-1===b.indexOf(e));if(r.every(t=>1===t))return Et(be(c,e,x),y);const v=t.stridedSlice(c,e,n,r);return Et(v,y)},h,null,_.xc,d)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.stridedSlice=function(t,e,n,r,o,s,a,i){return this.throwIfDisposed(),za(this,t,e,n,r,o,s,a,i)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.subStrict=function(t){return this.throwIfDisposed(),Qs(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.sub=function(t){return this.throwIfDisposed(),Ge(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.sum=function(t,e){return this.throwIfDisposed(),tn(this,t,e)};const Ua=Object(j.a)({tan_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","tan"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.tan(e);return n([e]),r},n,null,_.Ac)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.tan=function(){return this.throwIfDisposed(),Ua(this)};const Va=Object(j.a)({tanh_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const e=Object(B.a)(t,"x","tanh"),n={x:e};return l.a.runKernelFunc((t,n)=>{const r=t.tanh(e);return n([r]),r},n,null,_.Bc)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */N.a.prototype.tanh=function(){return this.throwIfDisposed(),Va(this)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.tile=function(t){return this.throwIfDisposed(),ln(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.toBool=function(){return this.throwIfDisposed(),M(this,"bool")},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.toFloat=function(){return this.throwIfDisposed(),M(this,"float32")},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.toInt=function(){return this.throwIfDisposed(),M(this,"int32")};const Ga=Object(j.a)({topk_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e=1,n=!0){const r=Object(B.a)(t,"x","topk");if(0===r.rank)throw new Error("topk() expects the input to be of rank 1 or higher");const o=r.shape[r.shape.length-1];if(e>o)throw new Error(`'k' passed to topk() must be <= the last dimension (${o}) but got `+e);const s={x:r},a={k:e,sorted:n},[i,u]=l.a.runKernelFunc(t=>t.topk(r,e,n),s,null,_.Dc,a);return{values:i,indices:u}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function Ha(t,e){if(!t)throw new Error(e)}function Ka(t){return"number"==typeof t}function qa(t){return t instanceof Array&&t.every(t=>Ka(t))}function Xa(t){return"tensor-float32"===t||"tensor-float16"===t||"tensor-int32"===t}function Ya(t){if("float32"===t||"tensor-float32"===t)return Float32Array;if("int32"===t||"tensor-int32"===t)return Int32Array;if("uint32"===t)return Uint32Array;if("float16"===t||"tensor-float16"===t)return Uint16Array;throw new Error("Type is not supported.")}function Qa(t){let e;return"float32"===t.dtype?e=t.rankType===lt.a.R0?i.float32:i["tensor-float32"]:"int32"===t.dtype&&(e=t.rankType===lt.a.R0?i.int32:i["tensor-int32"]),{type:e,dimensions:t.shape}}function Za(t){Ha(t.type in i,"The operand type is invalid."),Xa(t.type)?Ha(qa(t.dimensions),"The operand dimensions is invalid."):Ha(void 0===t.dimensions,"The operand dimensions is not required.")}function Ja(t,e){var n;Ha((n=t)instanceof Float32Array||n instanceof Int32Array||n instanceof Uint32Array||n instanceof Int16Array||n instanceof Uint16Array,"The value is not a typed array."),Ha(t instanceof Ya(e.type),"The type of value is invalid."),Xa(e.type)?Ha(t.length===ei(e.dimensions),`the value length ${t.length} is invalid, size of [${e.dimensions}] ${ei(e.dimensions)} is expected.`):Ha(1===t.length,`The value length ${t.length} is invalid, 1 is expected.`)}function ti(t,e){const n=function(t){if("float32"===t||"tensor-float32"===t)return"float32";if("int32"===t||"tensor-int32"===t)return"int32";throw new Error("The operand type is not supported by TF.js.")}(t.type);return Xa(t.type)?(Ja(e,t),he.a(e,t.dimensions,n)):"number"==typeof e?ie(e,n):(Ja(e,t),ie(e[0],n))}function ei(t){return void 0===t||qa(t)&&0===t.length?1:t.reduce((t,e)=>t*e)}N.a.prototype.topk=function(t,e){return this.throwIfDisposed(),Ga(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.transpose=function(t){return this.throwIfDisposed(),Kt(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.unsortedSegmentSum=function(t,e){return this.throwIfDisposed(),Co(this,t,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.unstack=function(t){return this.throwIfDisposed(),He(this,t)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.where=function(t,e){return this.throwIfDisposed(),qe(t,this,e)},
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
N.a.prototype.zerosLike=function(){return this.throwIfDisposed(),ve(this)};class ni extends c{constructor(t,e){super(),this.desc=t,this.value=e}static createScalar(t,e=i.float32){return void 0===e&&(e=i.float32),Ha(e in i,"The operand type is invalid."),new ni({type:e},t)}static createTensor(t,e){return Za(t),Ja(e,t),new ni(t,e)}}class ri extends c{constructor(t,e){super(),Ha("string"==typeof t,"The name parameter is invalid"),this.name=t,Za(e),this.desc=e}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oi={},si={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function ai(t){t in oi||(oi[t]=function(t){if(1!==t&&2!==t)throw new Error("Cannot get WebGL rendering context, WebGL is disabled.");const e=function(t){if("undefined"!=typeof OffscreenCanvas&&2===t)return new OffscreenCanvas(300,150);if("undefined"!=typeof document)return document.createElement("canvas");throw new Error("Cannot create a canvas in this context")}(t);if(e.addEventListener("webglcontextlost",e=>{e.preventDefault(),delete oi[t]},!1),1===t)return e.getContext("webgl",si)||e.getContext("experimental-webgl",si);return e.getContext("webgl2",si)}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(t));const e=oi[t];return e.isContextLost()?(delete oi[t],ai(t)):(e.disable(e.DEPTH_TEST),e.disable(e.STENCIL_TEST),e.disable(e.BLEND),e.disable(e.DITHER),e.disable(e.POLYGON_OFFSET_FILL),e.disable(e.SAMPLE_COVERAGE),e.enable(e.SCISSOR_TEST),e.enable(e.CULL_FACE),e.cullFace(e.BACK),oi[t])}var ii,ui,ci;function li(t,e){return[e,t]}function hi(t){const e=y.sizeFromShape(t),n=Math.ceil(e/4);return y.sizeToSquarishShape(n)}function di(t,e){return[Math.max(1,Math.ceil(e/2)),Math.max(1,Math.ceil(t/2))]}function pi(t,e){const n=t;let r,o,s,a,i,u,c,l,d,p;return 2===Object(h.b)().getNumber("WEBGL_VERSION")?(r=n.R32F,o=n.R16F,s=n.RGBA16F,a=n.RGBA32F,i=n.RED,c=4,l=1,d=n.HALF_FLOAT,p=n.FLOAT):(r=t.RGBA,o=t.RGBA,s=t.RGBA,a=n.RGBA,i=t.RGBA,c=4,l=4,d=null!=e?e.HALF_FLOAT_OES:null,p=t.FLOAT),u=t.RGBA,{internalFormatFloat:r,internalFormatHalfFloat:o,internalFormatPackedHalfFloat:s,internalFormatPackedFloat:a,textureFormatFloat:i,downloadTextureFormat:u,downloadUnpackNumChannels:c,defaultNumChannels:l,textureTypeHalfFloat:d,textureTypeFloat:p}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fi(t,e){const n=e();return Object(h.b)().getBool("DEBUG")&&function(t){const e=t.getError();if(e!==t.NO_ERROR)throw new Error("WebGL Error: "+function(t,e){switch(e){case t.NO_ERROR:return"NO_ERROR";case t.INVALID_ENUM:return"INVALID_ENUM";case t.INVALID_VALUE:return"INVALID_VALUE";case t.INVALID_OPERATION:return"INVALID_OPERATION";case t.INVALID_FRAMEBUFFER_OPERATION:return"INVALID_FRAMEBUFFER_OPERATION";case t.OUT_OF_MEMORY:return"OUT_OF_MEMORY";case t.CONTEXT_LOST_WEBGL:return"CONTEXT_LOST_WEBGL";default:return"Unknown error code "+e}}(t,e))}(t),n}!function(t){t[t.DENSE=0]="DENSE",t[t.SHARED_BATCH=1]="SHARED_BATCH"}(ii||(ii={})),function(t){t[t.RENDER=0]="RENDER",t[t.UPLOAD=1]="UPLOAD",t[t.PIXELS=2]="PIXELS",t[t.DOWNLOAD=3]="DOWNLOAD"}(ui||(ui={})),function(t){t[t.UNPACKED_FLOAT16=0]="UNPACKED_FLOAT16",t[t.UNPACKED_FLOAT32=1]="UNPACKED_FLOAT32",t[t.PACKED_4X1_UNSIGNED_BYTE=2]="PACKED_4X1_UNSIGNED_BYTE",t[t.PACKED_2X2_FLOAT32=3]="PACKED_2X2_FLOAT32",t[t.PACKED_2X2_FLOAT16=4]="PACKED_2X2_FLOAT16"}(ci||(ci={}));function gi(t){return!!(Object(h.b)().getBool("WEBGL_RENDER_FLOAT32_ENABLED")||0===t||5.96e-8<Math.abs(t)&&Math.abs(t)<65504)}function mi(t,e){return Ii(t,()=>t.getExtension(e),'Extension "'+e+'" not supported on this browser.')}function bi(t,e){const n=Ii(t,()=>t.createShader(t.FRAGMENT_SHADER),"Unable to create fragment WebGLShader.");if(fi(t,()=>t.shaderSource(n,e)),fi(t,()=>t.compileShader(n)),!1===t.getShaderParameter(n,t.COMPILE_STATUS))throw function(t,e){const n=xi.exec(e);if(null==n)return console.log("Couldn't parse line number in error: "+e),void console.log(t);const r=+n[1],o=t.split("\n"),s=o.length.toString().length+2,a=o.map((t,e)=>y.rightPad((e+1).toString(),s)+t);let i=0;for(let t=0;t<a.length;t++)i=Math.max(a[t].length,i);const u=a.slice(0,r-1),c=a.slice(r-1,r),l=a.slice(r);console.log(u.join("\n")),console.log(e.split("\n")[0]),console.log("%c "+y.rightPad(c[0],i),"border:1px solid red; background-color:#e3d2d2; color:#a61717"),console.log(l.join("\n"))}(e,t.getShaderInfoLog(n)),new Error("Failed to compile fragment shader.");return n}const xi=/ERROR: [0-9]+:([0-9]+):/g;function yi(t,e){if(fi(t,()=>t.validateProgram(e)),!1===t.getProgramParameter(e,t.VALIDATE_STATUS))throw console.log(t.getProgramInfoLog(e)),new Error("Shader program validation failed.")}function vi(t,e,n,r,o,s,a){const i=t.getAttribLocation(e,n);return-1!==i&&(fi(t,()=>t.bindBuffer(t.ARRAY_BUFFER,r)),fi(t,()=>t.vertexAttribPointer(i,o,t.FLOAT,!1,s,a)),fi(t,()=>t.enableVertexAttribArray(i)),!0)}function wi(t,e,n,r){fi(t,()=>function(t,e,n){Si(t,n),fi(t,()=>t.activeTexture(t.TEXTURE0+n)),fi(t,()=>t.bindTexture(t.TEXTURE_2D,e))}(t,e,r)),fi(t,()=>t.uniform1i(n,r))}function Ci(t,e,n){fi(t,()=>t.bindFramebuffer(t.FRAMEBUFFER,n)),fi(t,()=>t.framebufferTexture2D(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,e,0))}function $i(t,e){fi(t,()=>t.bindFramebuffer(t.FRAMEBUFFER,e)),fi(t,()=>t.framebufferTexture2D(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,null,0))}function Oi(t){const e=t.checkFramebufferStatus(t.FRAMEBUFFER);if(e!==t.FRAMEBUFFER_COMPLETE)throw new Error("Error binding framebuffer: "+function(t,e){switch(e){case t.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_ATTACHMENT";case t.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";case t.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return"FRAMEBUFFER_INCOMPLETE_DIMENSIONS";case t.FRAMEBUFFER_UNSUPPORTED:return"FRAMEBUFFER_UNSUPPORTED";default:return"unknown error "+e}}(t,e))}function Ii(t,e,n){const r=fi(t,()=>e());if(null==r)throw new Error(n);return r}function Si(t,e){const n=t.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,r=e+t.TEXTURE0;if(r<t.TEXTURE0||r>n){throw new Error(`textureUnit must be in ${`[gl.TEXTURE0, gl.TEXTURE${n}]`}.`)}}function Ei(t,e=2){return y.sizeFromShape(t.slice(0,t.length-e))}function Ri(t){if(0===t.length)throw Error("Cannot get rows and columns of an empty shape array.");return[t.length>1?t[t.length-2]:1,t[t.length-1]]}function Ai(t){let e=[1,1,1];return 0===t.length||1===t.length&&1===t[0]||(e=[Ei(t),...Ri(t)]),e}function ki(t){return t%2==0}function Ti(t,e){if(t=t.slice(-2),e=e.slice(-2),y.arraysEqual(t,e))return!0;if(!t.length||!e.length)return!0;if(0===t[0]||0===t[1]||0===e[0]||0===e[1])return!0;if(t.length!==e.length){const n=t.slice(-1)[0],r=e.slice(-1)[0];if(n===r)return!0;if(ki(n)&&ki(r)&&(1===t[0]||1===e[0]))return!0}return t[1]===e[1]&&ki(t[0])&&ki(e[0])}let Fi,Ni;function Di(t,e){return null!=t.getExtension(e)}function _i(t){try{if(null!=ai(t))return!0}catch(t){return!1}return!1}function Bi(t){if(0===t)return!1;const e=ai(t);if(1!==t){if(Di(e,"EXT_color_buffer_float"))return ji(e);const t="EXT_color_buffer_half_float";if(Di(e,t)){const n=e.getExtension(t);return function(t,e){const n=pi(t,e),r=t.createTexture();t.bindTexture(t.TEXTURE_2D,r);t.texImage2D(t.TEXTURE_2D,0,n.internalFormatHalfFloat,1,1,0,n.textureFormatFloat,n.textureTypeHalfFloat,null);const o=t.createFramebuffer();t.bindFramebuffer(t.FRAMEBUFFER,o),t.framebufferTexture2D(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,r,0);const s=t.checkFramebufferStatus(t.FRAMEBUFFER)===t.FRAMEBUFFER_COMPLETE;return t.bindTexture(t.TEXTURE_2D,null),t.bindFramebuffer(t.FRAMEBUFFER,null),t.deleteTexture(r),t.deleteFramebuffer(o),s}(e,n)}return!1}if(!Di(e,"OES_texture_float"))return!1;if(!Di(e,"WEBGL_color_buffer_float"))return!1;return ji(e)}function ji(t){const e=pi(t),n=t.createTexture();t.bindTexture(t.TEXTURE_2D,n);t.texImage2D(t.TEXTURE_2D,0,e.internalFormatFloat,1,1,0,e.textureFormatFloat,e.textureTypeFloat,null);const r=t.createFramebuffer();t.bindFramebuffer(t.FRAMEBUFFER,r),t.framebufferTexture2D(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,n,0);const o=t.checkFramebufferStatus(t.FRAMEBUFFER)===t.FRAMEBUFFER_COMPLETE;return t.bindTexture(t.TEXTURE_2D,null),t.bindFramebuffer(t.FRAMEBUFFER,null),t.deleteTexture(n),t.deleteFramebuffer(r),o}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const Mi=Object(h.b)();Mi.registerFlag("HAS_WEBGL",()=>Mi.getNumber("WEBGL_VERSION")>0),Mi.registerFlag("WEBGL_VERSION",()=>_i(2)?2:_i(1)?1:0),Mi.registerFlag("WEBGL_CHECK_NUMERICAL_PROBLEMS",()=>!1),Mi.registerFlag("WEBGL_BUFFER_SUPPORTED",()=>2===Mi.get("WEBGL_VERSION")),Mi.registerFlag("WEBGL_CPU_FORWARD",()=>!0),Mi.registerFlag("WEBGL_FORCE_F16_TEXTURES",()=>!1),Mi.registerFlag("WEBGL_PACK",()=>Mi.getBool("HAS_WEBGL")),Mi.registerFlag("WEBGL_PACK_NORMALIZATION",()=>Mi.getBool("WEBGL_PACK")),Mi.registerFlag("WEBGL_PACK_CLIP",()=>Mi.getBool("WEBGL_PACK")),Mi.registerFlag("WEBGL_PACK_DEPTHWISECONV",()=>!1),Mi.registerFlag("WEBGL_PACK_BINARY_OPERATIONS",()=>Mi.getBool("WEBGL_PACK")),Mi.registerFlag("WEBGL_PACK_UNARY_OPERATIONS",()=>Mi.getBool("WEBGL_PACK")),Mi.registerFlag("WEBGL_PACK_ARRAY_OPERATIONS",()=>Mi.getBool("WEBGL_PACK")),Mi.registerFlag("WEBGL_PACK_IMAGE_OPERATIONS",()=>Mi.getBool("WEBGL_PACK")),Mi.registerFlag("WEBGL_PACK_REDUCE",()=>Mi.getBool("WEBGL_PACK")),Mi.registerFlag("WEBGL_LAZILY_UNPACK",()=>Mi.getBool("WEBGL_PACK")),Mi.registerFlag("WEBGL_CONV_IM2COL",()=>Mi.getBool("WEBGL_PACK")),Mi.registerFlag("WEBGL_MAX_TEXTURE_SIZE",()=>function(t){if(null==Fi){const e=ai(t);Fi=e.getParameter(e.MAX_TEXTURE_SIZE)}return Fi}(Mi.getNumber("WEBGL_VERSION"))),Mi.registerFlag("WEBGL_MAX_TEXTURES_IN_SHADER",()=>function(t){if(null==Ni){const e=ai(t);Ni=e.getParameter(e.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,Ni)}(Mi.getNumber("WEBGL_VERSION"))),Mi.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION",()=>{const t=Mi.getNumber("WEBGL_VERSION");return 0===t?0:function(t){if(0===t)return 0;let e;const n=ai(t);return e=Di(n,"EXT_disjoint_timer_query_webgl2")&&2===t?2:Di(n,"EXT_disjoint_timer_query")?1:0,e}(t)}),Mi.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE",()=>Mi.getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0&&!xr.isMobile()),Mi.registerFlag("WEBGL_RENDER_FLOAT32_CAPABLE",()=>function(t){if(0===t)return!1;const e=ai(t);if(1===t){if(!Di(e,"OES_texture_float"))return!1}else if(!Di(e,"EXT_color_buffer_float"))return!1;return ji(e)}(Mi.getNumber("WEBGL_VERSION"))),Mi.registerFlag("WEBGL_RENDER_FLOAT32_ENABLED",()=>!Mi.getBool("WEBGL_FORCE_F16_TEXTURES")&&Mi.getBool("WEBGL_RENDER_FLOAT32_CAPABLE")),Mi.registerFlag("WEBGL_DOWNLOAD_FLOAT_ENABLED",()=>Bi(Mi.getNumber("WEBGL_VERSION"))),Mi.registerFlag("WEBGL_FENCE_API_ENABLED",()=>{return 2===(t=Mi.getNumber("WEBGL_VERSION"))&&null!=ai(t).fenceSync;var t}),Mi.registerFlag("WEBGL_SIZE_UPLOAD_UNIFORM",()=>Mi.getBool("WEBGL_RENDER_FLOAT32_ENABLED")?4:0),Mi.registerFlag("WEBGL_DELETE_TEXTURE_THRESHOLD",()=>-1,t=>{if(t<0&&-1!==t)throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${t}.`)});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Pi{constructor(t,e){this.outputShape=[],this.outputShape=t,this.variableNames=e.map((t,e)=>"T"+e);const n=[];this.variableNames.forEach(t=>{n.push(`float v${t} = get${t}AtOutCoords();`)});const r=this.variableNames.map(t=>"v"+t).join(" + ");this.userCode=`\n      void main() {\n        ${n.join("\n        ")}\n\n        float result = ${r};\n        setOutput(result);\n      }\n    `}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Li{constructor(t,e){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.variableNames=e.map((t,e)=>"T"+e);const n=[];this.variableNames.forEach(t=>{n.push(`vec4 v${t} = get${t}AtOutCoords();`)});const r=this.variableNames.map(t=>"v"+t).join(" + ");this.userCode=`\n      void main() {\n        ${n.join("\n        ")}\n\n        vec4 result = ${r};\n        setOutput(result);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Wi{constructor(t,e,n){this.variableNames=["A"];const{windowSize:r,batchSize:o,outSize:s}=t;n||this.variableNames.push("bestIndicesA"),this.outputShape=[o,s];const a="max"===e?">":"<",i=n?"inOffset + i;":"round(getBestIndicesA(batch, inOffset + i));";this.userCode=`\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int batch = coords[0];\n        int outIdx = coords[1];\n        int inOffset = outIdx * ${r};\n\n        int bestIndex = inOffset;\n        float bestValue = getA(batch, bestIndex);\n\n        for (int i = 0; i < ${r}; i++) {\n          int inIdx = ${i};\n          float candidate = getA(batch, inIdx);\n          if (candidate ${a} bestValue) {\n            bestValue = candidate;\n            bestIndex = inIdx;\n          }\n        }\n        setOutput(float(bestIndex));\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zi(t,e){return["x","y","z","w","u","v"].slice(0,e).map(e=>`${t}.${e}`)}function Ui(t,e){return 1===e?[t]:zi(t,e)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function Vi(){let t,e,n,r,o,s,a,i,u,c;return 2===Object(h.b)().getNumber("WEBGL_VERSION")?(t="#version 300 es",e="in",n="out",r="in",o="texture",s="outputColor",a="out vec4 outputColor;",i="\n      bool isnan_custom(float val) {\n        return (val > 0.0 || val < 0.0) ? false : val != 0.0;\n      }\n\n      bvec4 isnan_custom(vec4 val) {\n        return bvec4(isnan_custom(val.x),\n          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));\n      }\n\n      #define isnan(value) isnan_custom(value)\n    ",u="",c="\n      #define round(value) newRound(value)\n      int newRound(float value) {\n        return int(floor(value + 0.5));\n      }\n\n      ivec4 newRound(vec4 value) {\n        return ivec4(floor(value + vec4(0.5)));\n      }\n    "):(t="",e="attribute",n="varying",r="varying",o="texture2D",s="gl_FragColor",a="",i="\n      #define isnan(value) isnan_custom(value)\n      bool isnan_custom(float val) {\n        return (val > 0. || val < 1. || val == 0.) ? false : true;\n      }\n      bvec4 isnan_custom(vec4 val) {\n        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));\n      }\n    ",u="\n      uniform float INFINITY;\n\n      bool isinf(float val) {\n        return abs(val) == INFINITY;\n      }\n      bvec4 isinf(vec4 val) {\n        return equal(abs(val), vec4(INFINITY));\n      }\n    ",c="\n      int round(float value) {\n        return int(floor(value + 0.5));\n      }\n\n      ivec4 round(vec4 value) {\n        return ivec4(floor(value + vec4(0.5)));\n      }\n    "),{version:t,attribute:e,varyingVs:n,varyingFs:r,texture2D:o,output:s,defineOutput:a,defineSpecialNaN:i,defineSpecialInf:u,defineRound:c}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gi(t,e,n="index"){const r=y.computeStrides(e);return r.map((e,o)=>`${`int ${t[o]} = ${n} / ${e}`}; ${o===r.length-1?`int ${t[o+1]} = ${n} - ${t[o]} * ${e}`:`index -= ${t[o]} * ${e}`};`).join("")}function Hi(t){const e=y.computeStrides(t).map(t=>t.toString());return`\n  int getFlatIndex(ivec3 coords) {\n    return coords.x * ${e[0]} + coords.y * ${e[1]} + coords.z;\n  }\n`}const Ki="\n  const float FLOAT_MAX = 1.70141184e38;\n  const float FLOAT_MIN = 1.17549435e-38;\n\n  lowp vec4 encode_float(highp float v) {\n    if (isnan(v)) {\n      return vec4(255, 255, 255, 255);\n    }\n\n    highp float av = abs(v);\n\n    if(av < FLOAT_MIN) {\n      return vec4(0.0, 0.0, 0.0, 0.0);\n    } else if(v > FLOAT_MAX) {\n      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;\n    } else if(v < -FLOAT_MAX) {\n      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;\n    }\n\n    highp vec4 c = vec4(0,0,0,0);\n\n    highp float e = floor(log2(av));\n    highp float m = exp2(fract(log2(av))) - 1.0;\n\n    c[2] = floor(128.0 * m);\n    m -= c[2] / 128.0;\n    c[1] = floor(32768.0 * m);\n    m -= c[1] / 32768.0;\n    c[0] = floor(8388608.0 * m);\n\n    highp float ebias = e + 127.0;\n    c[3] = floor(ebias / 2.0);\n    ebias -= c[3] * 2.0;\n    c[2] += floor(ebias) * 128.0;\n\n    c[3] += 128.0 * step(0.0, -v);\n\n    return c / 255.0;\n  }\n",{getBroadcastDims:qi}=s;
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xi(t,e,n,r){const o=[];t.forEach(t=>{const e=y.sizeFromShape(t.shapeInfo.logicalShape);t.shapeInfo.isUniform?o.push(`uniform float ${t.name}${e>1?`[${e}]`:""};`):(o.push(`uniform sampler2D ${t.name};`),o.push(`uniform int offset${t.name};`))});const s=o.join("\n"),a=t.map(t=>function(t,e,n=!1){let r="";r+=n?Qi(t):Yi(t);const o=t.shapeInfo.logicalShape,s=e.logicalShape;o.length<=s.length&&(r+=n?function(t,e){const n=t.name,r=n.charAt(0).toUpperCase()+n.slice(1),o="get"+r+"AtOutCoords",s=t.shapeInfo.logicalShape.length,a=e.logicalShape.length,i=qi(t.shapeInfo.logicalShape,e.logicalShape),u=su(a),c=a-s;let l;const h=["x","y","z","w","u","v"];l=0===s?"":a<2&&i.length>=1?"coords = 0;":i.map(t=>`coords.${h[t+c]} = 0;`).join("\n");let d="";d=a<2&&s>0?"coords":t.shapeInfo.logicalShape.map((t,e)=>"coords."+h[e+c]).join(", ");let p="return outputValue;";const f=1===y.sizeFromShape(t.shapeInfo.logicalShape),g=1===y.sizeFromShape(e.logicalShape);if(1!==s||f||g){if(f&&!g)p=1===a?"\n        return vec4(outputValue.x, outputValue.x, 0., 0.);\n      ":"\n        return vec4(outputValue.x);\n      ";else if(i.length){const t=s-2,e=s-1;i.indexOf(t)>-1&&i.indexOf(e)>-1?p="return vec4(outputValue.x);":i.indexOf(t)>-1?p="return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);":i.indexOf(e)>-1&&(p="return vec4(outputValue.xx, outputValue.zz);")}}else p="\n      return vec4(outputValue.xy, outputValue.xy);\n    ";return`\n    vec4 ${o}() {\n      ${u} coords = getOutputCoords();\n      ${l}\n      vec4 outputValue = get${r}(${d});\n      ${p}\n    }\n  `}(t,e):function(t,e){const n=t.name,r=n.charAt(0).toUpperCase()+n.slice(1),o="get"+r+"AtOutCoords",s=e.texShape,a=t.shapeInfo.texShape,i=t.shapeInfo.logicalShape.length,u=e.logicalShape.length;if(!t.shapeInfo.isUniform&&i===u&&null==t.shapeInfo.flatOffset&&y.arraysEqual(a,s))return`\n      float ${o}() {\n        return sampleTexture(${n}, resultUV);\n      }\n    `;const c=su(u),l=qi(t.shapeInfo.logicalShape,e.logicalShape),h=u-i;let d;const p=["x","y","z","w","u","v"];d=0===i?"":u<2&&l.length>=1?"coords = 0;":l.map(t=>`coords.${p[t+h]} = 0;`).join("\n");let f="";f=u<2&&i>0?"coords":t.shapeInfo.logicalShape.map((t,e)=>"coords."+p[e+h]).join(", ");return`\n    float ${o}() {\n      ${c} coords = getOutputCoords();\n      ${d}\n      return get${r}(${f});\n    }\n  `}(t,e));return r}(t,e,r)).join("\n"),i=e.texShape,u=Vi(),c=function(t){return`\n    float sampleTexture(sampler2D textureSampler, vec2 uv) {\n      return ${t.texture2D}(textureSampler, uv).r;\n    }\n  `}(u);let l,h,d=function(t){return`${t.version}\n    precision highp float;\n    precision highp int;\n    precision highp sampler2D;\n    ${t.varyingFs} vec2 resultUV;\n    ${t.defineOutput}\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    struct ivec5\n    {\n      int x;\n      int y;\n      int z;\n      int w;\n      int u;\n    };\n\n    struct ivec6\n    {\n      int x;\n      int y;\n      int z;\n      int w;\n      int u;\n      int v;\n    };\n\n    uniform float NAN;\n    ${t.defineSpecialNaN}\n    ${t.defineSpecialInf}\n    ${t.defineRound}\n\n    int imod(int x, int y) {\n      return x - y * (x / y);\n    }\n\n    int idiv(int a, int b, float sign) {\n      int res = a / b;\n      int mod = imod(a, b);\n      if (sign < 0. && mod != 0) {\n        res -= 1;\n      }\n      return res;\n    }\n\n    //Based on the work of Dave Hoskins\n    //https://www.shadertoy.com/view/4djSRW\n    #define HASHSCALE1 443.8975\n    float random(float seed){\n      vec2 p = resultUV * seed;\n      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);\n      p3 += dot(p3, p3.yzx + 19.19);\n      return fract((p3.x + p3.y) * p3.z);\n    }\n\n    ${Zi}\n    ${Ji}\n    ${tu}\n  `}(u);e.isPacked?(l=function(t,e){switch(t.length){case 0:return nu();case 1:return function(t,e){const n=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)];if(1===n[0])return`\n      int getOutputCoords() {\n        return 2 * int(resultUV.x * ${n[1]}.0);\n      }\n    `;if(1===n[1])return`\n      int getOutputCoords() {\n        return 2 * int(resultUV.y * ${n[0]}.0);\n      }\n    `;return`\n    int getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(${n[0]}, ${n[1]}));\n      return 2 * (resTexRC.x * ${n[1]} + resTexRC.y);\n    }\n  `}(0,e);case 2:return function(t,e){const n=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)];if(y.arraysEqual(t,e))return`\n      ivec2 getOutputCoords() {\n        return 2 * ivec2(resultUV.yx * vec2(${n[0]}, ${n[1]}));\n      }\n    `;const r=Math.ceil(t[1]/2);return`\n    ivec2 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(${n[0]}, ${n[1]}));\n\n      int index = resTexRC.x * ${n[1]} + resTexRC.y;\n      int r = 2 * (index / ${r});\n      int c = imod(index, ${r}) * 2;\n\n      return ivec2(r, c);\n    }\n  `}(t,e);case 3:return function(t,e){const n=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)],r=Math.ceil(t[2]/2),o=r*Math.ceil(t[1]/2);return`\n    ivec3 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(${n[0]}, ${n[1]}));\n      int index = resTexRC.x * ${n[1]} + resTexRC.y;\n\n      int b = index / ${o};\n      index -= b * ${o};\n\n      int r = 2 * (index / ${r});\n      int c = imod(index, ${r}) * 2;\n\n      return ivec3(b, r, c);\n    }\n  `}(t,e);default:return function(t,e){const n=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)],r=Math.ceil(t[t.length-1]/2),o=r*Math.ceil(t[t.length-2]/2);let s=o,a="",i="b, r, c";for(let e=2;e<t.length-1;e++)s*=t[t.length-e-1],a=`\n      int b${e} = index / ${s};\n      index -= b${e} * ${s};\n    `+a,i=`b${e}, `+i;return`\n    ivec${t.length} getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(${n[0]}, ${n[1]}));\n      int index = resTexRC.x * ${n[1]} + resTexRC.y;\n\n      ${a}\n\n      int b = index / ${o};\n      index -= b * ${o};\n\n      int r = 2 * (index / ${r});\n      int c = imod(index, ${r}) * 2;\n\n      return ivec${t.length}(${i});\n    }\n  `}(t,e)}}(e.logicalShape,i),h=function(t){return`\n    void setOutput(vec4 val) {\n      ${t.output} = val;\n    }\n  `}(u)):(l=function(t,e){switch(t.length){case 0:return nu();case 1:return function(t,e){if(1===e[0])return`\n      int getOutputCoords() {\n        return int(resultUV.x * ${e[1]}.0);\n      }\n    `;if(1===e[1])return`\n      int getOutputCoords() {\n        return int(resultUV.y * ${e[0]}.0);\n      }\n    `;return`\n    int getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(${e[0]}, ${e[1]}));\n      return resTexRC.x * ${e[1]} + resTexRC.y;\n    }\n  `}(0,e);case 2:return function(t,e){if(y.arraysEqual(t,e))return`\n      ivec2 getOutputCoords() {\n        return ivec2(resultUV.yx * vec2(${e[0]}, ${e[1]}));\n      }\n    `;if(1===t[1])return`\n      ivec2 getOutputCoords() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n                               vec2(${e[0]}, ${e[1]}));\n        int index = resTexRC.x * ${e[1]} + resTexRC.y;\n        return ivec2(index, 0);\n      }\n    `;if(1===t[0])return`\n      ivec2 getOutputCoords() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n                               vec2(${e[0]}, ${e[1]}));\n        int index = resTexRC.x * ${e[1]} + resTexRC.y;\n        return ivec2(0, index);\n      }\n    `;return`\n    ivec2 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(${e[0]}, ${e[1]}));\n      int index = resTexRC.x * ${e[1]} + resTexRC.y;\n      int r = index / ${t[1]};\n      int c = index - r * ${t[1]};\n      return ivec2(r, c);\n    }\n  `}(t,e);case 3:return function(t,e){const n=Gi(["r","c","d"],t);return`\n    ivec3 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(${e[0]}, ${e[1]}));\n      int index = resTexRC.x * ${e[1]} + resTexRC.y;\n      ${n}\n      return ivec3(r, c, d);\n    }\n  `}(t,e);case 4:return function(t,e){const n=Gi(["r","c","d","d2"],t);return`\n    ivec4 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n        vec2(${e[0]}, ${e[1]}));\n      int index = resTexRC.x * ${e[1]} + resTexRC.y;\n      ${n}\n      return ivec4(r, c, d, d2);\n    }\n  `}(t,e);case 5:return function(t,e){const n=Gi(["r","c","d","d2","d3"],t);return`\n    ivec5 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${e[0]},\n                             ${e[1]}));\n\n      int index = resTexRC.x * ${e[1]} + resTexRC.y;\n\n      ${n}\n\n      ivec5 outShape = ivec5(r, c, d, d2, d3);\n      return outShape;\n    }\n  `}(t,e);case 6:return function(t,e){const n=Gi(["r","c","d","d2","d3","d4"],t);return`\n    ivec6 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n        vec2(${e[0]}, ${e[1]}));\n      int index = resTexRC.x * ${e[1]} + resTexRC.y;\n\n      ${n}\n\n      ivec6 result = ivec6(r, c, d, d2, d3, d4);\n      return result;\n    }\n  `}(t,e);default:throw new Error(t.length+"-D output sampling is not yet supported")}}(e.logicalShape,i),h=function(t){return`\n    void setOutput(float val) {\n      ${t.output} = vec4(val, 0, 0, 0);\n    }\n  `}(u)),r&&(d+=eu);return[d,c,h,s,l,a,n].join("\n")}function Yi(t){const e=t.shapeInfo.logicalShape;switch(e.length){case 0:return function(t){const e=t.name,n="get"+e.charAt(0).toUpperCase()+e.slice(1);if(t.shapeInfo.isUniform)return`float ${n}() {return ${e};}`;const[r,o]=t.shapeInfo.texShape;if(1===r&&1===o)return`\n      float ${n}() {\n        return sampleTexture(${e}, halfCR);\n      }\n    `;const[s,a]=t.shapeInfo.texShape,i=ru(e);return`\n    float ${n}() {\n      vec2 uv = uvFromFlat(${s}, ${a}, ${i});\n      return sampleTexture(${e}, uv);\n    }\n  `}(t);case 1:return function(t){const e=t.name,n="get"+e.charAt(0).toUpperCase()+e.slice(1);if(t.shapeInfo.isUniform)return`\n      float ${n}(int index) {\n        ${ou(t)}\n      }\n    `;const r=t.shapeInfo.texShape,o=r[0],s=r[1];if(1===s&&1===o)return`\n      float ${n}(int index) {\n        return sampleTexture(${e}, halfCR);\n      }\n    `;const a=ru(e);if(1===s)return`\n      float ${n}(int index) {\n        vec2 uv = vec2(0.5, (float(index + ${a}) + 0.5) / ${o}.0);\n        return sampleTexture(${e}, uv);\n      }\n    `;if(1===o)return`\n      float ${n}(int index) {\n        vec2 uv = vec2((float(index + ${a}) + 0.5) / ${s}.0, 0.5);\n        return sampleTexture(${e}, uv);\n      }\n    `;return`\n    float ${n}(int index) {\n      vec2 uv = uvFromFlat(${o}, ${s}, index + ${a});\n      return sampleTexture(${e}, uv);\n    }\n  `}(t);case 2:return function(t){const e=t.shapeInfo.logicalShape,n=t.name,r="get"+n.charAt(0).toUpperCase()+n.slice(1),o=t.shapeInfo.texShape;if(null!=o&&y.arraysEqual(e,o)){const t=o[0],e=o[1];return`\n    float ${r}(int row, int col) {\n      vec2 uv = (vec2(col, row) + halfCR) / vec2(${e}.0, ${t}.0);\n      return sampleTexture(${n}, uv);\n    }\n  `}const{newShape:s,keptDims:a}=y.squeezeShape(e),i=s;if(i.length<e.length){const e=au(t,i),n=["row","col"];return`\n      ${Yi(e)}\n      float ${r}(int row, int col) {\n        return ${r}(${iu(n,a)});\n      }\n    `}if(t.shapeInfo.isUniform)return`\n      float ${r}(int row, int col) {\n        int index = round(dot(vec2(row, col), vec2(${e[1]}, 1)));\n        ${ou(t)}\n      }\n    `;const u=o[0],c=o[1],l=ru(n);if(1===c)return`\n    float ${r}(int row, int col) {\n      float index = dot(vec3(row, col, ${l}), vec3(${e[1]}, 1, 1));\n      vec2 uv = vec2(0.5, (index + 0.5) / ${u}.0);\n      return sampleTexture(${n}, uv);\n    }\n  `;if(1===u)return`\n    float ${r}(int row, int col) {\n      float index = dot(vec3(row, col, ${l}), vec3(${e[1]}, 1, 1));\n      vec2 uv = vec2((index + 0.5) / ${c}.0, 0.5);\n      return sampleTexture(${n}, uv);\n    }\n  `;return`\n  float ${r}(int row, int col) {\n    // Explicitly use integer operations as dot() only works on floats.\n    int index = row * ${e[1]} + col + ${l};\n    vec2 uv = uvFromFlat(${u}, ${c}, index);\n    return sampleTexture(${n}, uv);\n  }\n`}(t);case 3:return function(t){const e=t.shapeInfo.logicalShape,n=t.name,r="get"+n.charAt(0).toUpperCase()+n.slice(1),o=e[1]*e[2],s=e[2],{newShape:a,keptDims:i}=y.squeezeShape(e),u=a;if(u.length<e.length){const e=au(t,u),n=["row","col","depth"];return`\n        ${Yi(e)}\n        float ${r}(int row, int col, int depth) {\n          return ${r}(${iu(n,i)});\n        }\n      `}if(t.shapeInfo.isUniform)return`\n      float ${r}(int row, int col, int depth) {\n        int index = round(dot(vec3(row, col, depth),\n                          vec3(${o}, ${s}, 1)));\n        ${ou(t)}\n      }\n    `;const c=t.shapeInfo.texShape,l=c[0],h=c[1],d=t.shapeInfo.flatOffset;if(h===o&&null==d)return`\n        float ${r}(int row, int col, int depth) {\n          float texR = float(row);\n          float texC = dot(vec2(col, depth), vec2(${s}, 1));\n          vec2 uv = (vec2(texC, texR) + halfCR) /\n                     vec2(${h}.0, ${l}.0);\n          return sampleTexture(${n}, uv);\n        }\n      `;if(h===s&&null==d)return`\n    float ${r}(int row, int col, int depth) {\n      float texR = dot(vec2(row, col), vec2(${e[1]}, 1));\n      float texC = float(depth);\n      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${h}.0, ${l}.0);\n      return sampleTexture(${n}, uv);\n    }\n  `;const p=ru(n);return`\n      float ${r}(int row, int col, int depth) {\n        // Explicitly use integer operations as dot() only works on floats.\n        int index = row * ${o} + col * ${s} + depth + ${p};\n        vec2 uv = uvFromFlat(${l}, ${h}, index);\n        return sampleTexture(${n}, uv);\n      }\n  `}(t);case 4:return function(t){const e=t.shapeInfo.logicalShape,n=t.name,r="get"+n.charAt(0).toUpperCase()+n.slice(1),o=e[3],s=e[2]*o,a=e[1]*s,{newShape:i,keptDims:u}=y.squeezeShape(e);if(i.length<e.length){const e=au(t,i),n=["row","col","depth","depth2"];return`\n      ${Yi(e)}\n      float ${r}(int row, int col, int depth, int depth2) {\n        return ${r}(${iu(n,u)});\n      }\n    `}if(t.shapeInfo.isUniform)return`\n      float ${r}(int row, int col, int depth, int depth2) {\n        int index = round(dot(vec4(row, col, depth, depth2),\n                          vec4(${a}, ${s}, ${o}, 1)));\n        ${ou(t)}\n      }\n    `;const c=t.shapeInfo.flatOffset,l=t.shapeInfo.texShape,h=l[0],d=l[1];if(d===a&&null==c)return`\n      float ${r}(int row, int col, int depth, int depth2) {\n        float texR = float(row);\n        float texC =\n            dot(vec3(col, depth, depth2),\n                vec3(${s}, ${o}, 1));\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                   vec2(${d}.0, ${h}.0);\n        return sampleTexture(${n}, uv);\n      }\n    `;if(d===o&&null==c)return`\n      float ${r}(int row, int col, int depth, int depth2) {\n        float texR = dot(vec3(row, col, depth),\n                         vec3(${e[1]*e[2]}, ${e[2]}, 1));\n        float texC = float(depth2);\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                  vec2(${d}.0, ${h}.0);\n        return sampleTexture(${n}, uv);\n      }\n    `;const p=ru(n);return`\n    float ${r}(int row, int col, int depth, int depth2) {\n      // Explicitly use integer operations as dot() only works on floats.\n      int index = row * ${a} + col * ${s} +\n          depth * ${o} + depth2;\n      vec2 uv = uvFromFlat(${h}, ${d}, index + ${p});\n      return sampleTexture(${n}, uv);\n    }\n  `}(t);case 5:return function(t){const e=t.shapeInfo.logicalShape,n=t.name,r="get"+n.charAt(0).toUpperCase()+n.slice(1),o=e[4],s=e[3]*o,a=e[2]*s,i=e[1]*a,{newShape:u,keptDims:c}=y.squeezeShape(e);if(u.length<e.length){const e=au(t,u),n=["row","col","depth","depth2","depth3"];return`\n      ${Yi(e)}\n      float ${r}(int row, int col, int depth, int depth2, int depth3) {\n        return ${r}(${iu(n,c)});\n      }\n    `}if(t.shapeInfo.isUniform)return`\n      float ${r}(int row, int col, int depth, int depth2, int depth3) {\n        float index = dot(\n          vec4(row, col, depth, depth2),\n          vec4(${i}, ${a}, ${s}, ${o})) +\n          depth3;\n        ${ou(t)}\n      }\n    `;const l=t.shapeInfo.flatOffset,h=t.shapeInfo.texShape,d=h[0],p=h[1];if(p===i&&null==l)return`\n      float ${r}(int row, int col, int depth, int depth2, int depth3) {\n        int texR = row;\n        float texC = dot(vec4(col, depth, depth2, depth3),\n                         vec4(${a}, ${s}, ${o}, 1));\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                   vec2(${p}.0, ${d}.0);\n        return sampleTexture(${n}, uv);\n      }\n    `;if(p===o&&null==l)return`\n      float ${r}(int row, int col, int depth, int depth2, int depth3) {\n        float texR = dot(\n          vec4(row, col, depth, depth2),\n          vec4(${e[1]*e[2]*e[3]},\n               ${e[2]*e[3]}, ${e[3]}, 1));\n        int texC = depth3;\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                  vec2(${p}.0, ${d}.0);\n        return sampleTexture(${n}, uv);\n      }\n    `;const f=ru(n);return`\n    float ${r}(int row, int col, int depth, int depth2, int depth3) {\n      // Explicitly use integer operations as dot() only works on floats.\n      int index = row * ${i} + col * ${a} + depth * ${s} +\n          depth2 * ${o} + depth3 + ${f};\n      vec2 uv = uvFromFlat(${d}, ${p}, index);\n      return sampleTexture(${n}, uv);\n    }\n  `}(t);case 6:return function(t){const e=t.shapeInfo.logicalShape,n=t.name,r="get"+n.charAt(0).toUpperCase()+n.slice(1),{newShape:o,keptDims:s}=y.squeezeShape(e);if(o.length<e.length){const e=au(t,o),n=["row","col","depth","depth2","depth3","depth4"];return`\n      ${Yi(e)}\n      float ${r}(int row, int col, int depth,\n                    int depth2, int depth3, int depth4) {\n        return ${r}(${iu(n,s)});\n      }\n    `}const a=e[5],i=e[4]*a,u=e[3]*i,c=e[2]*u,l=e[1]*c;if(t.shapeInfo.isUniform)return`\n      float ${r}(int row, int col, int depth,\n                  int depth2, int depth3, int depth4) {\n        int index = round(dot(\n          vec4(row, col, depth, depth2),\n          vec4(${l}, ${c}, ${u}, ${i})) +\n          dot(\n            vec2(depth3, depth4),\n            vec2(${a}, 1)));\n        ${ou(t)}\n      }\n    `;const h=t.shapeInfo.flatOffset,d=t.shapeInfo.texShape,p=d[0],f=d[1];if(f===l&&null==h)return`\n      float ${r}(int row, int col, int depth,\n                    int depth2, int depth3, int depth4) {\n        int texR = row;\n        float texC = dot(vec4(col, depth, depth2, depth3),\n          vec4(${c}, ${u}, ${i}, ${a})) +\n               float(depth4);\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                   vec2(${f}.0, ${p}.0);\n        return sampleTexture(${n}, uv);\n      }\n    `;if(f===a&&null==h)return`\n      float ${r}(int row, int col, int depth,\n                    int depth2, int depth3, int depth4) {\n        float texR = dot(vec4(row, col, depth, depth2),\n          vec4(${e[1]*e[2]*e[3]*e[4]},\n               ${e[2]*e[3]*e[4]},\n               ${e[3]*e[4]},\n               ${e[4]})) + float(depth3);\n        int texC = depth4;\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                  vec2(${f}.0, ${p}.0);\n        return sampleTexture(${n}, uv);\n      }\n    `;const g=ru(n);return`\n    float ${r}(int row, int col, int depth,\n                  int depth2, int depth3, int depth4) {\n      // Explicitly use integer operations as dot() only works on floats.\n      int index = row * ${l} + col * ${c} + depth * ${u} +\n          depth2 * ${i} + depth3 * ${a} + depth4 + ${g};\n      vec2 uv = uvFromFlat(${p}, ${f}, index);\n      return sampleTexture(${n}, uv);\n    }\n  `}(t);default:throw new Error(e.length+"-D input sampling is not yet supported")}}function Qi(t){switch(t.shapeInfo.logicalShape.length){case 0:return function(t){const e=t.name,n="get"+e.charAt(0).toUpperCase()+e.slice(1),r=Vi();return`\n    vec4 ${n}() {\n      return ${r.texture2D}(${e}, halfCR);\n    }\n  `}(t);case 1:return function(t){const e=t.name,n="get"+e.charAt(0).toUpperCase()+e.slice(1),r=t.shapeInfo.texShape,o=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)],s=Vi();return`\n    vec4 ${n}(int index) {\n      vec2 uv = packedUVfrom1D(\n        ${o[0]}, ${o[1]}, index);\n      return ${s.texture2D}(${e}, uv);\n    }\n  `}(t);case 2:return function(t){const e=t.shapeInfo.logicalShape,n=t.name,r="get"+n.charAt(0).toUpperCase()+n.slice(1),o=t.shapeInfo.texShape,s=o[0],a=o[1],i=Vi();if(null!=o&&y.arraysEqual(e,o))return`\n      vec4 ${r}(int row, int col) {\n        vec2 uv = (vec2(col, row) + halfCR) / vec2(${a}.0, ${s}.0);\n\n        return ${i.texture2D}(${n}, uv);\n      }\n    `;const u=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)],c=Math.ceil(e[1]/2);return`\n    vec4 ${r}(int row, int col) {\n      vec2 uv = packedUVfrom2D(${c}, ${u[0]}, ${u[1]}, row, col);\n      return ${i.texture2D}(${n}, uv);\n    }\n  `}(t);case 3:return function(t){const e=t.shapeInfo.logicalShape,n=t.name,r="get"+n.charAt(0).toUpperCase()+n.slice(1),o=t.shapeInfo.texShape,s=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)];if(1===e[0]){const n=e.slice(1),o=[1,2],s=au(t,n),a=["b","row","col"];return`\n        ${Qi(s)}\n        vec4 ${r}(int b, int row, int col) {\n          return ${r}(${iu(a,o)});\n        }\n      `}const a=s[0],i=s[1],u=Math.ceil(e[2]/2),c=u*Math.ceil(e[1]/2),l=Vi();return`\n    vec4 ${r}(int b, int row, int col) {\n      vec2 uv = packedUVfrom3D(\n        ${a}, ${i}, ${c}, ${u}, b, row, col);\n      return ${l.texture2D}(${n}, uv);\n    }\n  `}(t);default:return function(t){const e=t.shapeInfo.logicalShape,n=e.length,r=t.name,o="get"+r.charAt(0).toUpperCase()+r.slice(1),s=t.shapeInfo.texShape,a=[Math.ceil(s[0]/2),Math.ceil(s[1]/2)],i=a[0],u=a[1],c=Math.ceil(e[n-1]/2);let l=c*Math.ceil(e[n-2]/2),h="int b, int row, int col",d=`b * ${l} + (row / 2) * ${c} + (col / 2)`;for(let t=2;t<n-1;t++)h=`int b${t}, `+h,l*=e[n-t-1],d=`b${t} * ${l} + `+d;const p=Vi();return`\n    vec4 ${o}(${h}) {\n      int index = ${d};\n      int texR = index / ${u};\n      int texC = index - texR * ${u};\n      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${u}, ${i});\n      return ${p.texture2D}(${r}, uv);\n    }\n  `}(t)}}const Zi="\nvec2 uvFromFlat(int texNumR, int texNumC, int index) {\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\nvec2 packedUVfrom1D(int texNumR, int texNumC, int index) {\n  int texelIndex = index / 2;\n  int texR = texelIndex / texNumC;\n  int texC = texelIndex - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n",Ji="\nvec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,\n  int texNumC, int row, int col) {\n  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);\n  int texR = texelIndex / texNumC;\n  int texC = texelIndex - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n",tu="\nvec2 packedUVfrom3D(int texNumR, int texNumC,\n    int texelsInBatch, int texelsInLogicalRow, int b,\n    int row, int col) {\n  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n",eu="\n  float getChannel(vec4 frag, vec2 innerDims) {\n    vec2 modCoord = mod(innerDims, 2.);\n    return modCoord.x == 0. ?\n      (modCoord.y == 0. ? frag.r : frag.g) :\n      (modCoord.y == 0. ? frag.b : frag.a);\n  }\n  float getChannel(vec4 frag, int dim) {\n    float modCoord = mod(float(dim), 2.);\n    return modCoord == 0. ? frag.r : frag.g;\n  }\n";function nu(){return"\n    int getOutputCoords() {\n      return 0;\n    }\n  "}function ru(t){return"offset"+t}function ou(t){const e=t.name,n=y.sizeFromShape(t.shapeInfo.logicalShape);return n<2?`return ${e};`:`\n    for (int i = 0; i < ${n}; i++) {\n      if (i == index) {\n        return ${e}[i];\n      }\n    }\n  `}function su(t){if(t<=1)return"int";if(2===t)return"ivec2";if(3===t)return"ivec3";if(4===t)return"ivec4";if(5===t)return"ivec5";if(6===t)return"ivec6";throw Error(`GPU for rank ${t} is not yet supported`)}function au(t,e){const n=JSON.parse(JSON.stringify(t));return n.shapeInfo.logicalShape=e,n}function iu(t,e){return e.map(e=>t[e]).join(", ")}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class uu{constructor(t,e,n,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,y.assert(t.length>2,()=>`Packed arg${n.charAt(0).toUpperCase()+n.slice(1)} supports only inputs with rank above 2.`);const o=t[t.length-1],s=Math.ceil(o/e);this.outputShape=t.slice(0,-1),s>1&&this.outputShape.push(s),r||this.variableNames.push("bestIndicesA");const a=this.outputShape,i=a.length,u=su(i),c=Ui("coords",i);let l,h;if(1===s){h=i+1;const t=su(h);l=`\n        ${t} sourceLocR = ${t}(${c.join()}, 0);\n        ++${c[i-1]};\n        ${t} sourceLocG = ${t}(${c.join()}, 0);\n        ++${c[i-2]};\n        ${t} sourceLocA = ${t}(${c.join()}, 0);\n        --${c[i-1]};\n        ${t} sourceLocB = ${t}(${c.join()}, 0);\n        --${c[i-2]};`}else h=i,l=`\n        ${u} sourceLocR = coords;\n        ++${c[i-1]};\n        ${u} sourceLocG = coords;\n        ++${c[i-2]};\n        ${u} sourceLocA = coords;\n        --${c[i-1]};\n        ${u} sourceLocB = coords;\n        --${c[i-2]};`;const d=["x","y","z","w","u","v"].slice(0,h),p="."+d[h-1],f=d.map(t=>"int "+t),g=Ui("sourceLocR",h-1).concat("inIdx.r"),m=Ui("sourceLocG",h-1).concat("inIdx.g"),b=Ui("sourceLocB",h-1).concat("inIdx.b"),x=Ui("sourceLocA",h-1).concat("inIdx.a"),v="max"===n?"greaterThan":"lessThan",w=r?"":`\n          inIdx = round(vec4(getBestIndicesAChannel(${g.join()}),\n                             getBestIndicesAChannel(${m.join()}),\n                             getBestIndicesAChannel(${b.join()}),\n                             getBestIndicesAChannel(${x.join()})));`,C=`vec4(\n            getAChannel(${g.join()}),\n            hasNextCol ? getAChannel(${m.join()}) : 0.,\n            hasNextRow ? getAChannel(${b.join()}) : 0.,\n            hasNextRow && hasNextCol ? getAChannel(${x.join()}) : 0.)`,$=r?"":`\n      float getBestIndicesAChannel(${f.join()}) {\n        return getChannel(getBestIndicesA(${d.join()}),\n                                          vec2(${d.slice(-2).join()}));\n      }`;this.userCode=`\n      float getAChannel(${f.join()}) {\n        return getChannel(getA(${d.join()}),\n                               vec2(${d.slice(-2).join()}));\n      }\n      ${$}\n      void main() {\n        ${u} coords = getOutputCoords();\n        bool hasNextCol = ${c[i-1]} < ${a[i-1]-1};\n        bool hasNextRow = ${c[i-2]} < ${a[i-2]-1};\n        ${l}\n        ivec4 srcIdx = ivec4(sourceLocR${p}, sourceLocG${p},\n          sourceLocB${p}, sourceLocA${p}) * ${e};\n        ivec4 inIdx = srcIdx;\n        vec4 bestIndex = vec4(inIdx);\n        vec4 bestValue = ${C};\n\n        for (int i = 0; i < ${e}; i++) {\n          inIdx = srcIdx;\n          ${w}\n          vec4 candidate = ${C};\n          bvec4 nan = isnan(candidate);\n          bvec4 replace = bvec4(\n            vec4(${v}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));\n\n          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,\n                           replace.y  ? candidate.y : bestValue.y,\n                           replace.z  ? candidate.z : bestValue.z,\n                           replace.w  ? candidate.w : bestValue.w);\n          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));\n          srcIdx++;\n        }\n        setOutput(bestIndex);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class cu{constructor(t){this.variableNames=["dy"],this.outputShape=t.inShape;const e=t.filterHeight,n=t.filterWidth,r=t.strideHeight,o=t.strideWidth,s=t.dilationHeight,a=t.dilationWidth,i=t.effectiveFilterHeight,u=t.effectiveFilterWidth,c=i-1-t.padInfo.top,l=u-1-t.padInfo.left,h=1/(e*n);this.userCode=`\n      const ivec2 pads = ivec2(${c}, ${l});\n      const float avgMultiplier = float(${h});\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n\n        ivec2 dyRCCorner = coords.yz - pads;\n        int dyRCorner = dyRCCorner.x;\n        int dyCCorner = dyRCCorner.y;\n\n        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < ${i};\n            wR += ${s}) {\n          float dyR = float(dyRCorner + wR) / ${r}.0;\n\n          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          for (int wC = 0; wC < ${u};\n            wC+= ${a}) {\n            float dyC = float(dyCCorner + wC) / ${o}.0;\n\n            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||\n                fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            float dyValue = getDy(b, idyR, idyC, d);\n\n            dotProd += dyValue * avgMultiplier;\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}class lu{constructor(t){this.variableNames=["dy"],this.outputShape=t.inShape;const e=t.filterDepth,n=t.filterHeight,r=t.filterWidth,o=t.strideDepth,s=t.strideHeight,a=t.strideWidth,i=t.dilationDepth,u=t.dilationHeight,c=t.dilationWidth,l=t.effectiveFilterDepth,h=t.effectiveFilterHeight,d=t.effectiveFilterWidth,p=l-1-t.padInfo.front,f=h-1-t.padInfo.top,g=d-1-t.padInfo.left,m=1/(e*n*r);this.userCode=`\n      const ivec3 pads = ivec3(${p}, ${f}, ${g});\n      const float avgMultiplier = float(${m});\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int ch = coords.u;\n\n        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;\n        int dyDCorner = dyCorner.x;\n        int dyRCorner = dyCorner.y;\n        int dyCCorner = dyCorner.z;\n\n        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get\n        // dx(xD, xR, xC, ch).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n\n        for (int wD = 0; wD < ${l};\n            wD += ${i}) {\n          float dyD = float(dyDCorner + wD) / ${o}.0;\n\n          if (dyD < 0.0 || dyD >= ${t.outDepth}.0 || fract(dyD) > 0.0) {\n            continue;\n          }\n          int idyD = int(dyD);\n\n          for (int wR = 0; wR < ${h};\n              wR += ${u}) {\n            float dyR = float(dyRCorner + wR) / ${s}.0;\n\n            if (dyR < 0.0 || dyR >= ${t.outHeight}.0 ||\n                fract(dyR) > 0.0) {\n              continue;\n            }\n            int idyR = int(dyR);\n\n            for (int wC = 0; wC < ${d};\n                wC += ${c}) {\n              float dyC = float(dyCCorner + wC) / ${a}.0;\n\n              if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||\n                  fract(dyC) > 0.0) {\n                continue;\n              }\n              int idyC = int(dyC);\n\n              float dyValue = getDy(batch, idyD, idyR, idyC, ch);\n\n              dotProd += dyValue * avgMultiplier;\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class hu{constructor(t,e,n,r,o,a){this.outputShape=[],this.variableNames=["x","mean","variance"],s.assertAndGetBroadcastShape(t,e),s.assertAndGetBroadcastShape(t,n);let i="0.0";null!=r&&(s.assertAndGetBroadcastShape(t,r),this.variableNames.push("offset"),i="getOffsetAtOutCoords()");let u="1.0";null!=o&&(s.assertAndGetBroadcastShape(t,o),this.variableNames.push("scale"),u="getScaleAtOutCoords()"),this.outputShape=t,this.userCode=`\n      void main() {\n        float x = getXAtOutCoords();\n        float mean = getMeanAtOutCoords();\n        float variance = getVarianceAtOutCoords();\n        float offset = ${i};\n        float scale = ${u};\n        float inv = scale * inversesqrt(variance + float(${a}));\n        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class du{constructor(t,e,n,r,o,a){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=["x","mean","variance"],s.assertAndGetBroadcastShape(t,e),s.assertAndGetBroadcastShape(t,n);let i="vec4(0.0)";null!=r&&(s.assertAndGetBroadcastShape(t,r),this.variableNames.push("offset"),i="getOffsetAtOutCoords()");let u="vec4(1.0)";null!=o&&(s.assertAndGetBroadcastShape(t,o),this.variableNames.push("scale"),u="getScaleAtOutCoords()"),this.outputShape=t,this.userCode=`\n      void main() {\n        vec4 offset = ${i};\n        vec4 scale = ${u};\n\n        vec4 x = getXAtOutCoords();\n        vec4 mean = getMeanAtOutCoords();\n        vec4 variance = getVarianceAtOutCoords();\n\n        vec4 inv = scale * inversesqrt(variance + vec4(${a}));\n\n        setOutput((x - mean) * inv + offset);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pu="return areal * breal - aimag * bimag;",fu="return areal * bimag + aimag * breal;";class gu{constructor(t,e,n){this.variableNames=["AReal","AImag","BReal","BImag"],this.outputShape=s.assertAndGetBroadcastShape(e,n),this.userCode=`\n      float binaryOpComplex(\n          float areal, float aimag, float breal, float bimag) {\n        ${t}\n      }\n\n      void main() {\n        float areal = getARealAtOutCoords();\n        float aimag = getAImagAtOutCoords();\n        float breal = getBRealAtOutCoords();\n        float bimag = getBImagAtOutCoords();\n        setOutput(binaryOpComplex(areal, aimag, breal, bimag));\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mu="return a + b;",bu="return a - b;",xu="return a * b;",yu="return (a < 0.) ? b * a : a;";class vu{constructor(t,e,n){this.variableNames=["A","B"],this.outputShape=s.assertAndGetBroadcastShape(e,n),this.userCode=`\n      float binaryOperation(float a, float b) {\n        ${t}\n      }\n\n      void main() {\n        float a = getAAtOutCoords();\n        float b = getBAtOutCoords();\n        setOutput(binaryOperation(a, b));\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wu="\n  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));\n  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);\n";class Cu{constructor(t,e,n,r=!1){this.variableNames=["A","B"],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=s.assertAndGetBroadcastShape(e,n);const o=this.outputShape.length;let a="";if(r)if(0===o||1===y.sizeFromShape(this.outputShape))a="\n          result.y = 0.;\n          result.z = 0.;\n          result.w = 0.;\n        ";else{if(a=`\n          ${su(o)} coords = getOutputCoords();\n        `,1===o)a+=`\n            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;\n            result.z = 0.;\n            result.w = 0.;\n          `;else{const t=Ui("coords",o);a+=`\n            bool nextRowOutOfBounds =\n              (${t[o-2]} + 1) >= ${this.outputShape[o-2]};\n            bool nextColOutOfBounds =\n              (${t[o-1]} + 1) >= ${this.outputShape[o-1]};\n            result.y = nextColOutOfBounds ? 0. : result.y;\n            result.z = nextRowOutOfBounds ? 0. : result.z;\n            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;\n          `}}this.userCode=`\n      vec4 binaryOperation(vec4 a, vec4 b) {\n        ${t}\n      }\n\n      void main() {\n        vec4 a = getAAtOutCoords();\n        vec4 b = getBAtOutCoords();\n\n        vec4 result = binaryOperation(a, b);\n        ${a}\n\n        setOutput(result);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $u{constructor(t){this.variableNames=["A"],this.outputShape=t,this.userCode="\n      uniform float minVal;\n      uniform float maxVal;\n\n      void main() {\n        float value = getAAtOutCoords();\n        if (isnan(value)) {\n          setOutput(value);\n          return;\n        }\n\n        setOutput(clamp(value, minVal, maxVal));\n      }\n    "}getCustomSetupFunc(t,e){return(n,r)=>{null==this.minLoc&&(this.minLoc=n.getUniformLocationNoThrow(r,"minVal"),this.maxLoc=n.getUniformLocationNoThrow(r,"maxVal")),n.gl.uniform1f(this.minLoc,t),n.gl.uniform1f(this.maxLoc,e)}}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ou{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.userCode="\n      uniform float minVal;\n      uniform float maxVal;\n\n      void main() {\n        vec4 value = getAAtOutCoords();\n\n        if (any(isnan(value))) {\n          setOutput(value);\n          return;\n        }\n\n        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));\n      }\n    "}getCustomSetupFunc(t,e){return(n,r)=>{null==this.minLoc&&(this.minLoc=n.getUniformLocationNoThrow(r,"minVal"),this.maxLoc=n.getUniformLocationNoThrow(r,"maxVal")),n.gl.uniform1f(this.minLoc,t),n.gl.uniform1f(this.maxLoc,e)}}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Iu{constructor(t){this.variableNames=["real","imag"],this.outputShape=t,this.userCode="\n      void main() {\n        float re = abs(getRealAtOutCoords());\n        float im = abs(getImagAtOutCoords());\n        float mx = max(re, im);\n\n        // sadly the length function in glsl is not underflow-safe\n        // (at least not on Intel GPUs). So the safe solution is\n        // to ensure underflow-safety in all cases.\n        setOutput(\n          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))\n        );\n      }\n    "}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Su{constructor(t){this.outputShape=[],this.outputShape=s.computeOutShape(t,1),this.variableNames=t.map((t,e)=>"T"+e);const e=new Array(t.length-1);e[0]=t[0][1];for(let n=1;n<e.length;n++)e[n]=e[n-1]+t[n][1];const n=[`if (yC < ${e[0]}) setOutput(getT0(yR, yC));`];for(let t=1;t<e.length;t++){const r=e[t-1];n.push(`else if (yC < ${e[t]}) setOutput(getT${t}(yR, yC-${r}));`)}const r=e.length,o=e[e.length-1];n.push(`else setOutput(getT${r}(yR, yC-${o}));`),this.userCode=`\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int yR = coords.x;\n        int yC = coords.y;\n\n        ${n.join("\n        ")}\n      }\n    `}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Eu{constructor(t,e){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=s.computeOutShape(t,e);const n=this.outputShape,r=n.length,o=su(r),a=Ui("coords",r),i=["x","y","z","w","u","v"].slice(0,r);this.variableNames=t.map((t,e)=>"T"+e);const u=new Array(t.length-1);u[0]=t[0][e];for(let n=1;n<u.length;n++)u[n]=u[n-1]+t[n][e];const c=i[e],l=i.slice(-2),h=i.join();let d=`if (${c} < ${u[0]}) {\n        return getChannel(\n            getT0(${h}), vec2(${l.join()}));\n        }`;for(let t=1;t<u.length;t++){const e=u[t-1];d+=`\n        if (${c} < ${u[t]}  && ${c} >= ${u[t-1]}) {\n          return getChannel(\n            getT${t}(${Ru(i,c,e)}),\n            vec2(${Ru(l,c,e)}));\n        }`}const p=u.length,f=u[u.length-1];d+=`\n        return getChannel(\n          getT${p}(${Ru(i,c,f)}),\n          vec2(${Ru(l,c,f)}));`,this.userCode=`\n      float getValue(${i.map(t=>"int "+t)}) {\n        ${d}\n      }\n\n      void main() {\n        ${o} coords = getOutputCoords();\n        vec4 result = vec4(getValue(${a}), 0., 0., 0.);\n\n        ${a[r-1]} = ${a[r-1]} + 1;\n        if (${a[r-1]} < ${n[r-1]}) {\n          result.g = getValue(${a});\n        }\n\n        ${a[r-2]} = ${a[r-2]} + 1;\n        if (${a[r-2]} < ${n[r-2]}) {\n          result.a = getValue(${a});\n        }\n\n        ${a[r-1]} = ${a[r-1]} - 1;\n        if (${a[r-2]} < ${n[r-2]} &&\n            ${a[r-1]} < ${n[r-1]}) {\n          result.b = getValue(${a});\n        }\n        setOutput(result);\n      }\n    `}}function Ru(t,e,n){const r=t.indexOf(e);return t.map((t,e)=>e===r?`${t} - ${n}`:t).join()}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Au{constructor(t){this.variableNames=["x","dy"],this.outputShape=t.filterShape;const e=t.strideHeight,n=t.strideWidth,r=t.padInfo.top,o=t.padInfo.left,s="channelsLast"===t.dataFormat;this.userCode=`\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int wR = coords.x;\n        int wC = coords.y;\n        int d1 = coords.z;\n        int d2 = coords.w;\n\n        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n\n        for (int b = 0; b < ${t.batchSize}; b++) {\n          for (int yR = 0; yR < ${t.outHeight}; yR++) {\n            int xR = wR + yR * ${e} - ${r};\n\n            if (xR < 0 || xR >= ${t.inHeight}) {\n              continue;\n            }\n\n            for (int yC = 0; yC < ${t.outWidth}; yC++) {\n              int xC = wC + yC * ${n} - ${o};\n\n              if (xC < 0 || xC >= ${t.inWidth}) {\n                continue;\n              }\n\n              if (${s}) {\n                float dyValue = getDy(b, yR, yC, d2);\n                float xValue = getX(b, xR, xC, d1);\n                dotProd += (xValue * dyValue);\n              } else {\n                float dyValue = getDy(b, d2, yR, yC);\n                float xValue = getX(b, d1, xR, xC);\n                dotProd += (xValue * dyValue);\n              }\n\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}class ku{constructor(t){this.variableNames=["dy","W"],this.outputShape=t.inShape;const e=t.filterHeight,n=t.filterWidth,r=t.strideHeight,o=t.strideWidth,s="channelsLast"===t.dataFormat,a=e-1-t.padInfo.top,i=n-1-t.padInfo.left,u=s?1:2,c=s?2:3,l=s?3:1;this.userCode=`\n      const ivec2 pads = ivec2(${a}, ${i});\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords[0];\n        int d1 = coords[${l}];\n\n        ivec2 dyCorner = ivec2(coords[${u}], coords[${c}]) - pads;\n        int dyRCorner = dyCorner.x;\n        int dyCCorner = dyCorner.y;\n\n        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < ${e}; wR++) {\n          float dyR = float(dyRCorner + wR) / ${r}.0;\n\n          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          int wRPerm = ${e} - 1 - wR;\n\n          for (int wC = 0; wC < ${n}; wC++) {\n            float dyC = float(dyCCorner + wC) / ${o}.0;\n\n            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||\n                fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            int wCPerm = ${n} - 1 - wC;\n\n            for (int d2 = 0; d2 < ${t.outChannels}; d2++) {\n\n              if (${s}) {\n                float xValue = getDy(batch, idyR, idyC, d2);\n                float wValue = getW(wRPerm, wCPerm, d1, d2);\n                dotProd += xValue * wValue;\n              } else {\n                float xValue = getDy(batch, d2, idyR, idyC);\n                float wValue = getW(wRPerm, wCPerm, d1, d2);\n                dotProd += xValue * wValue;\n              }\n\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}class Tu{constructor(t){this.variableNames=["x","dy"],this.outputShape=t.filterShape;const e=t.strideDepth,n=t.strideHeight,r=t.strideWidth,o=t.padInfo.front,s=t.padInfo.top,a=t.padInfo.left;this.userCode=`\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int wF = coords.x;\n        int wR = coords.y;\n        int wC = coords.z;\n        int d1 = coords.w;\n        int d2 = coords.u;\n\n        float dotProd = 0.0;\n\n        for (int b = 0; b < ${t.batchSize}; b++) {\n          for (int yF = 0; yF < ${t.outDepth}; yF++) {\n            int xF = wF + yF * ${e} - ${o};\n\n            if (xF < 0 || xF >= ${t.inDepth}) {\n              continue;\n            }\n\n            for (int yR = 0; yR < ${t.outHeight}; yR++) {\n              int xR = wR + yR * ${n} - ${s};\n\n              if (xR < 0 || xR >= ${t.inHeight}) {\n                continue;\n              }\n\n              for (int yC = 0; yC < ${t.outWidth}; yC++) {\n                int xC = wC + yC * ${r} - ${a};\n\n                if (xC < 0 || xC >= ${t.inWidth}) {\n                  continue;\n                }\n\n                float dyValue = getDy(b, yF, yR, yC, d2);\n                float xValue = getX(b, xF, xR, xC, d1);\n                dotProd += (xValue * dyValue);\n              }\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}class Fu{constructor(t){this.variableNames=["dy","W"],this.outputShape=t.inShape;const e=t.filterDepth,n=t.filterHeight,r=t.filterWidth,o=t.strideDepth,s=t.strideHeight,a=t.strideWidth,i=e-1-t.padInfo.front,u=n-1-t.padInfo.top,c=r-1-t.padInfo.left;this.userCode=`\n      const ivec3 pads = ivec3(${i}, ${u}, ${c});\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int d1 = coords.u;\n\n\n        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;\n        int dyFCorner = dyCorner.x;\n        int dyRCorner = dyCorner.y;\n        int dyCCorner = dyCorner.z;\n\n        float dotProd = 0.0;\n        for (int wF = 0; wF < ${e}; wF++) {\n          float dyF = float(dyFCorner + wF) / ${o}.0;\n\n          if (dyF < 0.0 || dyF >= ${t.outDepth}.0 || fract(dyF) > 0.0) {\n            continue;\n          }\n          int idyF = int(dyF);\n\n          int wFPerm = ${e} - 1 - wF;\n\n          for (int wR = 0; wR < ${n}; wR++) {\n            float dyR = float(dyRCorner + wR) / ${s}.0;\n\n            if (dyR < 0.0 || dyR >= ${t.outHeight}.0 ||\n              fract(dyR) > 0.0) {\n              continue;\n            }\n            int idyR = int(dyR);\n\n            int wRPerm = ${n} - 1 - wR;\n\n            for (int wC = 0; wC < ${r}; wC++) {\n              float dyC = float(dyCCorner + wC) / ${a}.0;\n\n              if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||\n                  fract(dyC) > 0.0) {\n                continue;\n              }\n              int idyC = int(dyC);\n\n              int wCPerm = ${r} - 1 - wC;\n\n              for (int d2 = 0; d2 < ${t.outChannels}; d2++) {\n                float xValue = getDy(batch, idyF, idyR, idyC, d2);\n                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);\n                dotProd += xValue * wValue;\n              }\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Nu{constructor(t){this.variableNames=["x","dy"],this.outputShape=t.filterShape;const e=t.strideHeight,n=t.strideWidth,r=t.padInfo.top,o=t.padInfo.left,s=t.outChannels/t.inChannels;this.userCode=`\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int wR = coords.x;\n        int wC = coords.y;\n        int d1 = coords.z;\n        int dm = coords.w;\n        int d2 = d1 * ${s} + dm;\n\n        float dotProd = 0.0;\n\n        // TO DO: Vec4 over the batch size\n        for (int b = 0; b < ${t.batchSize}; b++) {\n          for (int yR = 0; yR < ${t.outHeight}; yR++) {\n            int xR = wR + yR * ${e} - ${r};\n\n            if (xR < 0 || xR >= ${t.inHeight}) {\n              continue;\n            }\n\n            for (int yC = 0; yC < ${t.outWidth}; yC++) {\n              int xC = wC + yC * ${n} - ${o};\n\n              if (xC < 0 || xC >= ${t.inWidth}) {\n                continue;\n              }\n\n              float dyValue = getDy(b, yR, yC, d2);\n              float xValue = getX(b, xR, xC, d1);\n              dotProd += (xValue * dyValue);\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}class Du{constructor(t){this.variableNames=["dy","W"],this.outputShape=t.inShape;const e=t.filterHeight,n=t.filterWidth,r=t.strideHeight,o=t.strideWidth,s=e-1-t.padInfo.top,a=n-1-t.padInfo.left,i=t.outChannels/t.inChannels;this.userCode=`\n      const ivec2 pads = ivec2(${s}, ${a});\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords[0];\n        int d1 = coords[3];\n        ivec2 dyCorner = coords.yz - pads;\n        int dyRCorner = dyCorner.x;\n        int dyCCorner = dyCorner.y;\n\n        float dotProd = 0.0;\n\n        for (int wR = 0; wR < ${e}; wR++) {\n          float dyR = float(dyRCorner + wR) / ${r}.0;\n\n          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          int wRPerm = ${e} - 1 - wR;\n\n          for (int wC = 0; wC < ${n}; wC++) {\n            float dyC = float(dyCCorner + wC) / ${o}.0;\n\n            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||\n                fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            int wCPerm = ${n} - 1 - wC;\n\n            // TO DO: Vec4 over the channelMul\n            for (int dm = 0; dm < ${i}; dm++) {\n              int d2 = d1 * ${i} + dm;\n              float xValue = getDy(batch, idyR, idyC, d2);\n              float wValue = getW(wRPerm, wCPerm, d1, dm);\n              dotProd += xValue * wValue;\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class _u{constructor(t,e=!1,n=null,r=!1){this.variableNames=["x","W"],this.outputShape=t.outShape;const o=t.padInfo.top,s=t.padInfo.left,a=t.strideHeight,i=t.strideWidth,u=t.dilationHeight,c=t.dilationWidth,l=t.filterHeight,h=t.filterWidth,d=4*Math.floor(t.inChannels/4),p=t.inChannels%4,f="channelsLast"===t.dataFormat,g=f?1:2,m=f?2:3,b=f?3:1;let x="",y="";n&&(x=r?`float activation(float a) {\n          float b = getPreluActivationWeightsAtOutCoords();\n          ${n}\n        }`:`\n          float activation(float x) {\n            ${n}\n          }\n        `,y="result = activation(result);");const v=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),r&&this.variableNames.push("preluActivationWeights"),this.userCode=`\n      ${x}\n\n      const ivec2 strides = ivec2(${a}, ${i});\n      const ivec2 pads = ivec2(${o}, ${s});\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords[0];\n        int d2 = coords[${b}];\n\n        ivec2 xRCCorner =\n            ivec2(coords[${g}], coords[${m}]) * strides - pads;\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < ${l}; wR++) {\n          int xR = xRCorner + wR * ${u};\n\n          if (xR < 0 || xR >= ${t.inHeight}) {\n            continue;\n          }\n\n          for (int wC = 0; wC < ${h}; wC++) {\n            int xC = xCCorner + wC * ${c};\n\n            if (xC < 0 || xC >= ${t.inWidth}) {\n              continue;\n            }\n\n            for (int d1 = 0; d1 < ${d}; d1 += 4) {\n              vec4 wValues = vec4(\n                getW(wR, wC, d1, d2),\n                getW(wR, wC, d1 + 1, d2),\n                getW(wR, wC, d1 + 2, d2),\n                getW(wR, wC, d1 + 3, d2)\n              );\n\n              if (${f}) {\n                vec4 xValues = vec4(\n                  getX(batch, xR, xC, d1),\n                  getX(batch, xR, xC, d1 + 1),\n                  getX(batch, xR, xC, d1 + 2),\n                  getX(batch, xR, xC, d1 + 3)\n                );\n                dotProd += dot(xValues, wValues);\n              } else {\n                vec4 xValues = vec4(\n                  getX(batch, d1, xR, xC),\n                  getX(batch, d1 + 1, xR, xC),\n                  getX(batch, d1 + 2, xR, xC),\n                  getX(batch, d1 + 3, xR, xC)\n                );\n                dotProd += dot(xValues, wValues);\n              }\n            }\n\n            if (${1===p}) {\n\n              if (${f}) {\n                dotProd +=\n                    getX(batch, xR, xC, ${d}) *\n                    getW(wR, wC, ${d}, d2);\n              } else {\n                dotProd +=\n                    getX(batch, ${d}, xR, xC) *\n                    getW(wR, wC, ${d}, d2);\n              }\n\n            } else if (${2===p}) {\n              vec2 wValues = vec2(\n                getW(wR, wC, ${d}, d2),\n                getW(wR, wC, ${d} + 1, d2)\n              );\n\n              if (${f}) {\n                vec2 xValues = vec2(\n                  getX(batch, xR, xC, ${d}),\n                  getX(batch, xR, xC, ${d} + 1)\n                );\n                dotProd += dot(xValues, wValues);\n              } else {\n                vec2 xValues = vec2(\n                  getX(batch, ${d}, xR, xC),\n                  getX(batch, ${d} + 1, xR, xC)\n                );\n                dotProd += dot(xValues, wValues);\n              }\n\n            } else if (${3===p}) {\n              vec3 wValues = vec3(\n                getW(wR, wC, ${d}, d2),\n                getW(wR, wC, ${d} + 1, d2),\n                getW(wR, wC, ${d} + 2, d2)\n              );\n\n              if (${f}) {\n                vec3 xValues = vec3(\n                  getX(batch, xR, xC, ${d}),\n                  getX(batch, xR, xC, ${d} + 1),\n                  getX(batch, xR, xC, ${d} + 2)\n                );\n                dotProd += dot(xValues, wValues);\n              } else {\n                vec3 xValues = vec3(\n                  getX(batch, ${d}, xR, xC),\n                  getX(batch, ${d} + 1, xR, xC),\n                  getX(batch, ${d} + 2, xR, xC)\n                );\n                dotProd += dot(xValues, wValues);\n              }\n\n            }\n          }\n        }\n\n        float result = dotProd;\n        ${v}\n        ${y}\n        setOutput(result);\n      }\n    `}}class Bu{constructor(t){this.variableNames=["x","W"],this.outputShape=t.outShape;const e=t.padInfo.front,n=t.padInfo.top,r=t.padInfo.left,o=t.strideDepth,s=t.strideHeight,a=t.strideWidth,i=t.dilationDepth,u=t.dilationHeight,c=t.dilationWidth,l=t.filterDepth,h=t.filterHeight,d=t.filterWidth,p=4*Math.floor(t.inChannels/4),f=t.inChannels%4;this.userCode=`\n      const ivec3 strides = ivec3(${o}, ${s}, ${a});\n      const ivec3 pads = ivec3(${e}, ${n}, ${r});\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int d2 = coords.u;\n\n        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;\n        int xFCorner = xFRCCorner.x;\n        int xRCorner = xFRCCorner.y;\n        int xCCorner = xFRCCorner.z;\n\n        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get\n        // y(yF, yR, yC, d2). ? = to be determined. : = across all\n        // values in that axis.\n        float dotProd = 0.0;\n        for (int wF = 0; wF < ${l}; wF++) {\n          int xF = xFCorner + wF * ${i};\n\n          if (xF < 0 || xF >= ${t.inDepth}) {\n            continue;\n          }\n\n          for (int wR = 0; wR < ${h}; wR++) {\n            int xR = xRCorner + wR * ${u};\n\n            if (xR < 0 || xR >= ${t.inHeight}) {\n              continue;\n            }\n\n            for (int wC = 0; wC < ${d}; wC++) {\n              int xC = xCCorner + wC * ${c};\n\n              if (xC < 0 || xC >= ${t.inWidth}) {\n                continue;\n              }\n\n              for (int d1 = 0; d1 < ${p}; d1 += 4) {\n                vec4 xValues = vec4(\n                  getX(batch, xF, xR, xC, d1),\n                  getX(batch, xF, xR, xC, d1 + 1),\n                  getX(batch, xF, xR, xC, d1 + 2),\n                  getX(batch, xF, xR, xC, d1 + 3)\n                );\n                vec4 wValues = vec4(\n                  getW(wF, wR, wC, d1, d2),\n                  getW(wF, wR, wC, d1 + 1, d2),\n                  getW(wF, wR, wC, d1 + 2, d2),\n                  getW(wF, wR, wC, d1 + 3, d2)\n                );\n\n                dotProd += dot(xValues, wValues);\n              }\n\n              if (${1===f}) {\n                dotProd +=\n                  getX(batch, xF, xR, xC, ${p}) *\n                  getW(wF, wR, wC, ${p}, d2);\n              } else if (${2===f}) {\n                vec2 xValues = vec2(\n                  getX(batch, xF, xR, xC, ${p}),\n                  getX(batch, xF, xR, xC, ${p} + 1)\n                );\n                vec2 wValues = vec2(\n                  getW(wF, wR, wC, ${p}, d2),\n                  getW(wF, wR, wC, ${p} + 1, d2)\n                );\n                dotProd += dot(xValues, wValues);\n              } else if (${3===f}) {\n                vec3 xValues = vec3(\n                  getX(batch, xF, xR, xC, ${p}),\n                  getX(batch, xF, xR, xC, ${p} + 1),\n                  getX(batch, xF, xR, xC, ${p} + 2)\n                );\n                vec3 wValues = vec3(\n                  getW(wF, wR, wC, ${p}, d2),\n                  getW(wF, wR, wC, ${p} + 1, d2),\n                  getW(wF, wR, wC, ${p} + 2, d2)\n                );\n                dotProd += dot(xValues, wValues);\n              }\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ju{constructor(t,e=!1,n=null,r=!1){this.variableNames=["x","W"],this.outputShape=t.outShape;const o=t.inHeight,s=t.inWidth,a=t.padInfo.top,i=t.padInfo.left,u=t.strideHeight,c=t.strideWidth,l=t.dilationHeight,h=t.dilationWidth,d=t.filterHeight,p=t.filterWidth,f=t.outChannels/t.inChannels;let g="",m="";n&&(g=r?`float activation(float a) {\n          float b = getPreluActivationWeightsAtOutCoords();\n          ${n}\n        }`:`\n          float activation(float x) {\n            ${n}\n          }\n        `,m="result = activation(result);");const b=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),r&&this.variableNames.push("preluActivationWeights"),this.userCode=`\n      ${g}\n\n      const ivec2 strides = ivec2(${u}, ${c});\n      const ivec2 pads = ivec2(${a}, ${i});\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords.x;\n        ivec2 xRCCorner = coords.yz * strides - pads;\n        int d2 = coords.w;\n        int d1 = d2 / ${f};\n        int q = d2 - d1 * ${f};\n\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.\n        for (int wR = 0; wR < ${d}; wR++) {\n          int xR = xRCorner + wR * ${l};\n\n          if (xR < 0 || xR >= ${o}) {\n            continue;\n          }\n\n          for (int wC = 0; wC < ${p}; wC++) {\n            int xC = xCCorner + wC * ${h};\n\n            if (xC < 0 || xC >= ${s}) {\n              continue;\n            }\n\n            float xVal = getX(batch, xR, xC, d1);\n            float wVal = getW(wR, wC, d1, q);\n            dotProd += xVal * wVal;\n          }\n        }\n\n        float result = dotProd;\n        ${b}\n        ${m}\n        setOutput(result);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Mu{constructor(t,e=!1,n=null,r=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t.outShape;const o=t.inHeight,s=t.inWidth,a=t.padInfo.top,i=t.padInfo.left,u=t.strideHeight,c=t.strideWidth,l=t.dilationHeight,h=t.dilationWidth,d=t.filterHeight,p=t.filterWidth,f=p;let g="int xR; int xC; int xCOffset;";for(let t=0;t<d;t++)for(let e=0;e<p;e++)g+=`\n          vec4 xTexelR${t}C${2*e} = vec4(0.);\n          vec4 wR${t}C${e} = vec4(0.);\n          vec4 xR${t}C${e} = vec4(0.);`;for(let t=0;t<d;t++)for(let e=0;e<f;e++){const n=2*e;if(g+=`\n          xR = xRCorner + ${t*l};\n          xC = xCCorner + ${n*h};\n        `,1===c){if(n<p&&(g+=i%2==1?`\n                xCOffset = xC + 1;\n                if(xR >= 0 && xR < ${o} && xCOffset >= 0 && xCOffset < ${s}) {\n                  xTexelR${t}C${n} = getX(batch, xR, xCOffset, d1);\n\n                  // Need to manually clear unused channels in case\n                  // we're reading from recycled texture.\n                  if(xCOffset + 1 >= ${s}) {\n                    xTexelR${t}C${n}.zw = vec2(0.);\n                  }\n                } else {\n                  xTexelR${t}C${n} = vec4(0.);\n                }\n\n                xCOffset = xC + 1 - 2;\n                if(xR >= 0 && xR < ${o} && xCOffset >= 0 && xCOffset < ${s}) {\n                  vec4 previous = getX(batch, xR, xCOffset, d1);\n\n                  // Need to manually clear unused channels in case\n                  // we're reading from recycled texture.\n                  if(xCOffset + 1 >= ${s}) {\n                    previous.zw = vec2(0.);\n                  }\n\n                  xR${t}C${n} = vec4(previous.zw, xTexelR${t}C${n}.xy);\n                } else {\n                  xR${t}C${n} = vec4(0, 0, xTexelR${t}C${n}.xy);\n                }\n              `:`\n                if(xR >= 0 && xR < ${o} && xC >= 0 && xC < ${s}) {\n                  xTexelR${t}C${n} = getX(batch, xR, xC, d1);\n                } else {\n                  xTexelR${t}C${n} = vec4(0.);\n                }\n\n                xR${t}C${n} = xTexelR${t}C${n};\n              `,n+1<p)){const e=i%2==0?y.nearestLargerEven(h):h;h%2==0&&i%2==1||h%2!=0&&i%2!=1?(g+=`\n                  xCOffset = xC + ${i%2} + ${e};\n\n                  if(xR >= 0 && xR < ${o} &&\n                    xCOffset >= 0 && xCOffset < ${s}) {\n                    xTexelR${t}C${n+2} = getX(batch, xR, xCOffset, d1);\n                  }\n                `,h>1&&(g+=`\n                    xCOffset -= 2;\n                    if(xR >= 0 && xR < ${o} &&\n                      xCOffset >= 0 && xCOffset < ${s}) {\n                      xTexelR${t}C${n} = getX(batch, xR, xCOffset, d1);\n                    } else {\n                      xTexelR${t}C${n} = vec4(0.);\n                    }\n                  `),g+=`\n                  xR${t}C${n+1} = vec4(\n                    xTexelR${t}C${n}.zw, xTexelR${t}C${n+2}.xy);\n                `):g+=`\n                  xCOffset = xC + ${e};\n\n                  if(xR >= 0 && xR < ${o} &&\n                    xCOffset >= 0 && xCOffset < ${s}) {\n                    xTexelR${t}C${n+2} = getX(batch, xR, xCOffset, d1);\n                  }\n\n                  xR${t}C${n+1} = xTexelR${t}C${n+2};\n                `}}else n<p&&(g+=`\n              if(xR >= 0 && xR < ${o}) {\n            `,i%2==1?(g+=`\n                xCOffset = xC + 1 - ${c};\n                if(xCOffset >= 0 && xCOffset < ${s}) {\n                  xTexelR${t}C${n} = getX(batch, xR, xCOffset, d1);\n                } else {\n                  xTexelR${t}C${n} = vec4(0.);\n                }\n\n                if(xC + 1 >= 0 && xC + 1 < ${s}) {\n                  xTexelR${t}C${n+2} = getX(batch, xR, xC + 1, d1);\n                } else {\n                  xTexelR${t}C${n+2} = vec4(0.);\n                }\n\n                xR${t}C${n} = vec4(\n                  xTexelR${t}C${n}.zw, xTexelR${t}C${n+2}.zw);\n              `,n+1<p&&(g+=`\n                  vec4 final = vec4(0.);\n                  xCOffset = xC + 1 + ${c};\n                  if(xCOffset >= 0 && xCOffset < ${s}) {\n                    final = getX(batch, xR, xCOffset, d1);\n                  }\n                  xR${t}C${n+1} = vec4(xTexelR${t}C${n+2}.xy, final.xy);\n                `)):(g+=`\n                if(xC >= 0 && xC < ${s}) {\n                  xTexelR${t}C${n} = getX(batch, xR, xC, d1);\n                } else {\n                  xTexelR${t}C${n} = vec4(0.);\n                }\n\n                xCOffset = xC + ${c};\n                if(xCOffset >= 0 && xCOffset < ${s}) {\n                  xTexelR${t}C${n+2} = getX(batch, xR, xCOffset, d1);\n                } else {\n                  xTexelR${t}C${n+2} = vec4(0.);\n                }\n\n                xR${t}C${n} = vec4(\n                  xTexelR${t}C${n}.xy, xTexelR${t}C${n+2}.xy);\n              `,n+1<p&&(g+=`\n                  xR${t}C${n+1} = vec4(\n                    xTexelR${t}C${n}.zw, xTexelR${t}C${n+2}.zw);\n                `)),g+="}");n<p&&(g+=`\n            vec4 wTexelR${t}C${n} = getW(${t}, ${n}, d1, q);\n            wR${t}C${n} = vec4(wTexelR${t}C${n}.xz, wTexelR${t}C${n}.xz);\n          `,n+1<p&&(g+=`\n              vec4 wTexelR${t}C${n+1} = getW(${t}, ${n+1}, d1, q);\n              wR${t}C${n+1} =\n                vec4(wTexelR${t}C${n+1}.xz, wTexelR${t}C${n+1}.xz);`))}for(let t=0;t<d;t++)for(let e=0;e<p;e++)g+=`dotProd += xR${t}C${e} * wR${t}C${e};`;let m="",b="";n&&(m=r?`vec4 activation(vec4 a) {\n          vec4 b = getPreluActivationWeightsAtOutCoords();\n          ${n}\n        }`:`vec4 activation(vec4 x) {\n          ${n}\n        }`,b="result = activation(result);");const x=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),r&&this.variableNames.push("preluActivationWeights"),this.userCode=`\n      ${m}\n\n      const ivec2 strides = ivec2(${u}, ${c});\n      const ivec2 pads = ivec2(${a}, ${i});\n\n      void main() {\n\n        ivec4 coords = getOutputCoords();\n        int batch = coords.x;\n        ivec2 xRCCorner = coords.yz * strides - pads;\n        int d2 = coords.w;\n        int d1 = d2;\n        int q = 0;\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        vec4 dotProd = vec4(0.);\n\n        ${g}\n\n        vec4 result = dotProd;\n        ${x}\n        ${b}\n        setOutput(result);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Pu{constructor(t,e,n,r,o){this.variableNames=["Image","Boxes","BoxInd"],this.outputShape=[];const[s,a,i,u]=t,[c]=e,[l,h]=n;this.outputShape=[c,l,h,u];const d="bilinear"===r?1:0,[p,f]=[a-1+".0",i-1+".0"],[g,m,b]=l>1?[""+(a-1)/(l-1),"(y2-y1) * height_ratio",`y1*${p} + float(y)*(height_scale)`]:["0.0","0.0","0.5 * (y1+y2) * "+p],[x,y,v]=h>1?[""+(i-1)/(h-1),"(x2-x1) * width_ratio",`x1*${f} + float(x)*(width_scale)`]:["0.0","0.0","0.5 * (x1+x2) * "+f];this.userCode=`\n      const float height_ratio = float(${g});\n      const float width_ratio = float(${x});\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int y = coords[1];\n        int x = coords[2];\n        int d = coords[3];\n\n        // get box vals\n        float y1 = getBoxes(b,0);\n        float x1 = getBoxes(b,1);\n        float y2 = getBoxes(b,2);\n        float x2 = getBoxes(b,3);\n\n        // get image in batch index\n        int bInd = round(getBoxInd(b));\n        if(bInd < 0 || bInd >= ${s}) {\n          return;\n        }\n\n        float height_scale = ${m};\n        float width_scale = ${y};\n\n        float in_y = ${b};\n        if( in_y < 0.0 || in_y > ${p} ) {\n          setOutput(float(${o}));\n          return;\n        }\n        float in_x = ${v};\n        if( in_x < 0.0 || in_x > ${f} ) {\n          setOutput(float(${o}));\n          return;\n        }\n\n        vec2 sourceFracIndexCR = vec2(in_x,in_y);\n        if(${d} == 1) {\n          // Compute the four integer indices.\n          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);\n          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));\n\n          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);\n          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);\n          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);\n          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);\n\n          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);\n\n          float top = topLeft + (topRight - topLeft) * fracCR.x;\n          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;\n          float newValue = top + (bottom - top) * fracCR.y;\n          setOutput(newValue);\n        } else {\n          // Compute the coordinators of nearest neighbor point.\n          ivec2 sourceNearestCR = ivec2(floor(\n            sourceFracIndexCR + vec2(0.5,0.5)));\n          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);\n          setOutput(newValue);\n        }\n      }\n    `}}class Lu{constructor(t,e,n){this.variableNames=["x"],this.outputShape=t;const r=t.length,o=e?"0.0":`getX(${Wu(r,"coords")})`,s=t[t.length-1];let a="",i="";e?(a=n?"end != "+(s-1):"end != 0",i=n?"end + 1":"end - 1"):(a=n?"end + pow2 < "+s:"end >= pow2",i=n?"end + pow2":"end - pow2"),this.userCode=`\n      uniform float index;\n      void main() {\n        ${su(r)} coords = getOutputCoords();\n        int end = ${zu(r,"coords")};\n        float val = ${o};\n        int pow2 = int(pow(2.0, index));\n        if (${a}) {\n          int idx = ${i};\n          ${zu(r,"coords")} = idx;\n          val += getX(${Wu(r,"coords")});\n        }\n        setOutput(val);\n      }\n    `}getCustomSetupFunc(t){return(e,n)=>{null==this.index&&(this.index=e.getUniformLocation(n,"index")),e.gl.uniform1f(this.index,t)}}}function Wu(t,e){if(1===t)return""+e;if(2===t)return`${e}.x, ${e}.y`;if(3===t)return`${e}.x, ${e}.y, ${e}.z`;if(4===t)return`${e}.x, ${e}.y, ${e}.z, ${e}.w`;throw Error(`Cumulative sum for rank ${t} is not yet supported`)}function zu(t,e){if(1===t)return""+e;if(2===t)return e+".y";if(3===t)return e+".z";if(4===t)return e+".w";throw Error(`Cumulative sum for rank ${t} is not yet supported`)}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Uu{constructor(t){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=ii.DENSE;const e=hi(t),n=Vi();this.outputShape=t,this.userCode=`\n      ivec3 outCoordsFromFlatIndex(int index) {\n        ${Gi(["r","c","d"],t)}\n        return ivec3(r, c, d);\n      }\n\n      void main() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n          vec2(${e[0]}, ${e[1]}));\n        int index = 4 * (resTexRC.x * ${e[1]} + resTexRC.y);\n\n        vec4 result = vec4(0.);\n\n        for (int i=0; i<4; i++) {\n          int flatIndex = index + i;\n          ivec3 rc = outCoordsFromFlatIndex(flatIndex);\n          result[i] = getA(rc.x, rc.y, rc.z);\n        }\n\n        ${n.output} = result;\n      }\n    `}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Vu{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=ii.DENSE;const e=hi(t),n=Vi();this.outputShape=t,this.userCode=`\n      ivec3 outCoordsFromFlatIndex(int index) {\n        ${Gi(["r","c","d"],t)}\n        return ivec3(r, c, d);\n      }\n\n      void main() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n          vec2(${e[0]}, ${e[1]}));\n        int index = 4 * (resTexRC.x * ${e[1]} + resTexRC.y);\n\n        vec4 result = vec4(0.);\n\n        for (int i=0; i<4; i++) {\n          int flatIndex = index + i;\n          ivec3 rc = outCoordsFromFlatIndex(flatIndex);\n          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));\n        }\n\n        ${n.output} = result;\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Gu{constructor(t,e,n){this.variableNames=["x"],this.outputShape=[],this.outputShape=t,this.blockSize=e,this.dataFormat=n,this.userCode=`\n    void main() {\n      ivec4 coords = getOutputCoords();\n      int b = coords[0];\n      int h = ${this.getHeightCoordString()};\n      int w = ${this.getWidthCoordString()};\n      int d = ${this.getDepthCoordString()};\n\n      int in_h = h / ${e};\n      int offset_h = imod(h, ${e});\n      int in_w = w / ${e};\n      int offset_w = imod(w, ${e});\n      int offset_d = (offset_h * ${e} + offset_w) *\n        ${this.getOutputDepthSize()};\n      int in_d = d + offset_d;\n\n      float result = ${this.getInputSamplingString()};\n      setOutput(result);\n    }\n  `}getHeightCoordString(){return"NHWC"===this.dataFormat?"coords[1]":"coords[2]"}getWidthCoordString(){return"NHWC"===this.dataFormat?"coords[2]":"coords[3]"}getDepthCoordString(){return"NHWC"===this.dataFormat?"coords[3]":"coords[1]"}getOutputDepthSize(){return"NHWC"===this.dataFormat?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return"NHWC"===this.dataFormat?"getX(b, in_h, in_w, in_d)":"getX(b, in_d, in_h, in_w)"}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Hu{constructor(t){this.variableNames=["X"],this.outputShape=[t,t],this.userCode="\n      void main() {\n          ivec2 coords = getOutputCoords();\n          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;\n          setOutput(val);\n      }\n    "}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ku{constructor(t){this.variableNames=["A"],this.outTexUsage=ui.DOWNLOAD;const e=Vi();this.outputShape=t,this.userCode=`\n      ${Ki}\n\n      void main() {\n        float x = getAAtOutCoords();\n        ${e.output} = encode_float(x);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class qu{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=ui.DOWNLOAD;const e=Vi();this.outputShape=t,this.userCode=`\n      ${Ki}\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));\n        ${e.output} = encode_float(x);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Xu{constructor(t,e,n=!1){this.variableNames=["A"];const r=Vi(),[o,s]=e;this.outputShape=t;let a="result";n&&(a="floor(result * 255. + 0.5)"),this.userCode=`\n      ${Hi(t)}\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n\n        int flatIndex = getFlatIndex(coords);\n        int offset = imod(flatIndex, 4);\n\n        flatIndex = idiv(flatIndex, 4, 1.);\n\n        int r = flatIndex / ${s};\n        int c = imod(flatIndex, ${s});\n        vec2 uv = (vec2(c, r) + halfCR) / vec2(${s}.0, ${o}.0);\n        vec4 values = ${r.texture2D}(A, uv);\n\n        float result;\n\n        if(offset == 0) {\n          result = values[0];\n        } else if(offset == 1) {\n          result = values[1];\n        } else if(offset == 2) {\n          result = values[2];\n        } else {\n          result = values[3];\n        }\n\n        ${r.output} = vec4(${a}, 0., 0., 0.);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Yu{constructor(t,e,n=!1){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0;const r=Vi(),[o,s]=e;this.outputShape=t;let a="",i="result";n&&(i="floor(result * 255. + 0.5)");for(let e=0;e<=1;e++)for(let n=0;n<=1;n++){const i=2*e+n;a+=`\n          localCoords = coords;\n          if(localCoords[2] + ${n} < ${t[2]}) {\n            localCoords[2] += ${n};\n            if(localCoords[1] + ${e} < ${t[1]}) {\n              localCoords[1] += ${e};\n\n              flatIndex = getFlatIndex(localCoords);\n              offset = imod(flatIndex, 4);\n\n              flatIndex = idiv(flatIndex, 4, 1.);\n\n              r = flatIndex / ${s};\n              c = imod(flatIndex, ${s});\n              uv = (vec2(c, r) + halfCR) / vec2(${s}.0, ${o}.0);\n              values = ${r.texture2D}(A, uv);\n\n              if(offset == 0) {\n                result[${i}] = values[0];\n              } else if(offset == 1) {\n                result[${i}] = values[1];\n              } else if(offset == 2) {\n                result[${i}] = values[2];\n              } else {\n                result[${i}] = values[3];\n              }\n            }\n          }\n        `}this.userCode=`\n      ${Hi(t)}\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n\n        vec4 result = vec4(0.);\n        int flatIndex, r, c, offset;\n        ivec3 localCoords;\n        vec2 uv;\n        vec4 values;\n\n        ${a}\n\n        ${r.output} = ${i};\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qu="return real * expR - imag * expI;",Zu="return real * expI + imag * expR;";class Ju{constructor(t,e,n){this.variableNames=["real","imag"];const r=e[1];this.outputShape=e;const o=n?"2.0 * "+Math.PI:"-2.0 * "+Math.PI,s=n?r+".0":"1.0";this.userCode=`\n      const float exponentMultiplier = ${o};\n\n      float unaryOpComplex(float real, float expR, float imag, float expI) {\n        ${t}\n      }\n\n      float mulMatDFT(int batch, int index) {\n        float indexRatio = float(index) / float(${r});\n        float exponentMultiplierTimesIndexRatio =\n            exponentMultiplier * indexRatio;\n\n        float result = 0.0;\n\n        for (int i = 0; i < ${r}; i++) {\n          // x = (-2|2 * PI / N) * index * i;\n          float x = exponentMultiplierTimesIndexRatio * float(i);\n          float expR = cos(x);\n          float expI = sin(x);\n          float real = getReal(batch, i);\n          float imag = getImag(batch, i);\n\n          result +=\n              unaryOpComplex(real, expR, imag, expI) / ${s};\n        }\n\n        return result;\n      }\n\n      void main() {\n        ivec2 coords = getOutputCoords();\n        setOutput(mulMatDFT(coords[0], coords[1]));\n      }\n    `}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class tc{constructor(t,e){this.outputShape=[],this.variableNames=["x"],this.outputShape=t,this.userCode="\n      uniform float value;\n      void main() {\n        // Input can be obtained from uniform value.\n        setOutput(value);\n      }\n    "}getCustomSetupFunc(t){return(e,n)=>{null==this.valueLoc&&(this.valueLoc=e.getUniformLocationNoThrow(n,"value")),e.gl.uniform1f(this.valueLoc,t)}}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ec{constructor(t,e,n){this.variableNames=["A","indices"];const r=t.slice();r[n]=e,this.outputShape=r,this.rank=r.length;const o=su(this.rank),s=function(t,e){const n=t.length;if(n>4)throw Error(`Gather for rank ${n} is not yet supported`);if(1===n)return"int(getIndices(resRC))";const r=["resRC.x","resRC.y","resRC.z","resRC.w"],o=[];for(let n=0;n<t.length;n++)n===e?o.push(`int(getIndices(${r[n]}))`):o.push(""+r[n]);return o.join()}(t,n);this.userCode=`\n      void main() {\n        ${o} resRC = getOutputCoords();\n        setOutput(getA(${s}));\n      }\n    `}}class nc{constructor(t,e,n){this.sliceDim=t,this.strides=e,this.variableNames=["x","indices"],this.outputShape=n;const r=su(e.length),o=su(n.length),s=this.sliceDim>1?"strides[j]":"strides";this.userCode=`\n        ${r} strides = ${r}(${this.strides});\n         void main() {\n          ${o} coords = getOutputCoords();\n          int flattenIndex = 0;\n          for (int j = 0; j < ${this.sliceDim}; j++) {\n            int index = round(getIndices(coords[0], j));\n            flattenIndex += index * ${s};\n          }\n          setOutput(getX(flattenIndex, coords[1]));\n        }\n      `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rc(t){const e=Vi();return function(t,e){const n=Ii(t,()=>t.createShader(t.VERTEX_SHADER),"Unable to create vertex WebGLShader.");if(fi(t,()=>t.shaderSource(n,e)),fi(t,()=>t.compileShader(n)),!1===t.getShaderParameter(n,t.COMPILE_STATUS))throw console.log(t.getShaderInfoLog(n)),new Error("Failed to compile vertex shader.");return n}(t,`${e.version}\n    precision highp float;\n    ${e.attribute} vec3 clipSpacePos;\n    ${e.attribute} vec2 uv;\n    ${e.varyingVs} vec2 resultUV;\n\n    void main() {\n      gl_Position = vec4(clipSpacePos, 1);\n      resultUV = uv;\n    }`)}function oc(t){return function(t,e){const n=Ii(t,()=>t.createBuffer(),"Unable to create WebGLBuffer");return fi(t,()=>t.bindBuffer(t.ARRAY_BUFFER,n)),fi(t,()=>t.bufferData(t.ARRAY_BUFFER,e,t.STATIC_DRAW)),n}(t,new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]))}function sc(t){return function(t,e){const n=Ii(t,()=>t.createBuffer(),"Unable to create WebGLBuffer");return fi(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,n)),fi(t,()=>t.bufferData(t.ELEMENT_ARRAY_BUFFER,e,t.STATIC_DRAW)),n}(t,new Uint16Array([0,1,2,2,1,3]))}function ac(t,e,n,r,o,s){!function(t,e){const n=Object(h.b)().getNumber("WEBGL_MAX_TEXTURE_SIZE");if(t<=0||e<=0){throw new Error("Requested texture size "+`[${t}x${e}]`+" is invalid.")}if(t>n||e>n){throw new Error("Requested texture size "+`[${t}x${e}]`+" greater than WebGL maximum on this browser / GPU "+`[${n}x${n}]`+".")}}(e,n);const a=function(t){return Ii(t,()=>t.createTexture(),"Unable to create WebGLTexture.")}(t),i=t.TEXTURE_2D;return fi(t,()=>t.bindTexture(i,a)),fi(t,()=>t.texParameteri(i,t.TEXTURE_WRAP_S,t.CLAMP_TO_EDGE)),fi(t,()=>t.texParameteri(i,t.TEXTURE_WRAP_T,t.CLAMP_TO_EDGE)),fi(t,()=>t.texParameteri(i,t.TEXTURE_MIN_FILTER,t.NEAREST)),fi(t,()=>t.texParameteri(i,t.TEXTURE_MAG_FILTER,t.NEAREST)),fi(t,()=>t.texImage2D(i,0,r,e,n,0,o,s,null)),fi(t,()=>t.bindTexture(t.TEXTURE_2D,null)),a}function ic(t){return t.internalFormatFloat}function uc(t){return t.internalFormatHalfFloat}function cc(t){return t.downloadTextureFormat}function lc(t){return t.internalFormatPackedFloat}function hc(t){return t.internalFormatPackedHalfFloat}function dc(t,e,n,r,o,s,a,i){const u=t,c=new Float32Array(function(t,e){const[n,r]=di(t,e);return n*r*4}(s,a));return u.bindBuffer(u.PIXEL_PACK_BUFFER,e),u.getBufferSubData(u.PIXEL_PACK_BUFFER,0,c),u.bindBuffer(u.PIXEL_PACK_BUFFER,null),c}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class pc{constructor(t){this.outputTexture=null,this.program=null,this.disposed=!1,this.vertexAttrsAreBound=!1,this.itemsToPoll=[];const e=Object(h.b)().getNumber("WEBGL_VERSION");null!=t?(this.gl=t,function(t,e){oi[t]=e}(e,t)):this.gl=ai(e);let n="WEBGL_color_buffer_float";if(1===Object(h.b)().getNumber("WEBGL_VERSION")){const t="OES_texture_float",e="OES_texture_half_float";if(this.textureFloatExtension=mi(this.gl,t),Di(this.gl,e))this.textureHalfFloatExtension=mi(this.gl,e);else if(Object(h.b)().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.");if(this.colorBufferFloatExtension=this.gl.getExtension(n),Di(this.gl,"EXT_color_buffer_half_float"))this.colorBufferHalfFloatExtension=mi(this.gl,"EXT_color_buffer_half_float");else if(Object(h.b)().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.")}else if(n="EXT_color_buffer_float",Di(this.gl,n))this.colorBufferFloatExtension=this.gl.getExtension(n);else{if(!Di(this.gl,"EXT_color_buffer_half_float"))throw new Error("GL context does not support color renderable floats");this.colorBufferHalfFloatExtension=this.gl.getExtension("EXT_color_buffer_half_float")}this.vertexBuffer=oc(this.gl),this.indexBuffer=sc(this.gl),this.framebuffer=function(t){return Ii(t,()=>t.createFramebuffer(),"Unable to create WebGLFramebuffer.")}(this.gl),this.textureConfig=pi(this.gl,this.textureHalfFloatExtension)}get debug(){return Object(h.b)().getBool("DEBUG")}dispose(){if(this.disposed)return;null!=this.program&&console.warn("Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing."),null!=this.outputTexture&&console.warn("Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.");const t=this.gl;fi(t,()=>t.finish()),fi(t,()=>t.bindFramebuffer(t.FRAMEBUFFER,null)),fi(t,()=>t.deleteFramebuffer(this.framebuffer)),fi(t,()=>t.bindBuffer(t.ARRAY_BUFFER,null)),fi(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,null)),fi(t,()=>t.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(t,e){return this.throwIfDisposed(),function(t,e,n,r){const[o,s]=li(e,n);return ac(t,o,s,ic(r),r.textureFormatFloat,t.FLOAT)}(this.gl,t,e,this.textureConfig)}createFloat16MatrixTexture(t,e){return this.throwIfDisposed(),function(t,e,n,r){const[o,s]=li(e,n);return ac(t,o,s,uc(r),r.textureFormatFloat,r.textureTypeHalfFloat)}(this.gl,t,e,this.textureConfig)}createUnsignedBytesMatrixTexture(t,e){return this.throwIfDisposed(),function(t,e,n,r){const[o,s]=li(e,n);return ac(t,o,s,cc(r),t.RGBA,t.UNSIGNED_BYTE)}(this.gl,t,e,this.textureConfig)}uploadPixelDataToTexture(t,e){this.throwIfDisposed(),function(t,e,n){fi(t,()=>t.bindTexture(t.TEXTURE_2D,e)),n.data instanceof Uint8Array?fi(t,()=>t.texImage2D(t.TEXTURE_2D,0,t.RGBA,n.width,n.height,0,t.RGBA,t.UNSIGNED_BYTE,n.data)):fi(t,()=>t.texImage2D(t.TEXTURE_2D,0,t.RGBA,t.RGBA,t.UNSIGNED_BYTE,n)),fi(t,()=>t.bindTexture(t.TEXTURE_2D,null))}(this.gl,t,e)}uploadDenseMatrixToTexture(t,e,n,r){this.throwIfDisposed(),function(t,e,n,r,o,s){let a,i,u;fi(t,()=>t.bindTexture(t.TEXTURE_2D,e)),o instanceof Uint8Array?(a=new Uint8Array(n*r*4),i=t.UNSIGNED_BYTE,u=t.RGBA):(a=new Float32Array(n*r*4),i=t.FLOAT,u=s.internalFormatPackedFloat),a.set(o),fi(t,()=>t.texImage2D(t.TEXTURE_2D,0,u,n,r,0,t.RGBA,i,a)),fi(t,()=>t.bindTexture(t.TEXTURE_2D,null))}(this.gl,t,e,n,r,this.textureConfig)}createFloat16PackedMatrixTexture(t,e){return this.throwIfDisposed(),function(t,e,n,r){const[o,s]=di(e,n);return ac(t,o,s,hc(r),t.RGBA,r.textureTypeHalfFloat)}(this.gl,t,e,this.textureConfig)}createPackedMatrixTexture(t,e){return this.throwIfDisposed(),function(t,e,n,r){const[o,s]=di(e,n);return ac(t,o,s,lc(r),t.RGBA,t.FLOAT)}(this.gl,t,e,this.textureConfig)}deleteMatrixTexture(t){this.throwIfDisposed(),this.outputTexture===t&&($i(this.gl,this.framebuffer),this.outputTexture=null),fi(this.gl,()=>this.gl.deleteTexture(t))}downloadByteEncodedFloatMatrixFromOutputTexture(t,e,n){return this.downloadMatrixDriver(t,()=>function(t,e,n,r){const[o,s]=li(e,n),a=new Uint8Array(e*n*4);return fi(t,()=>t.readPixels(0,0,o,s,r.downloadTextureFormat,t.UNSIGNED_BYTE,a)),new Float32Array(a.buffer)}(this.gl,e,n,this.textureConfig))}downloadPackedMatrixFromBuffer(t,e,n,r,o,s){return dc(this.gl,t,0,0,0,o,s,this.textureConfig)}downloadFloat32MatrixFromBuffer(t,e){return function(t,e,n){const r=t,o=new Float32Array(n);return r.bindBuffer(r.PIXEL_PACK_BUFFER,e),r.getBufferSubData(r.PIXEL_PACK_BUFFER,0,o),r.bindBuffer(r.PIXEL_PACK_BUFFER,null),o}(this.gl,t,e)}createBufferFromTexture(t,e,n){this.bindTextureToFrameBuffer(t);const r=function(t,e,n,r){const o=t.createBuffer();fi(t,()=>t.bindBuffer(t.PIXEL_PACK_BUFFER,o));const s=16*e*n;return fi(t,()=>t.bufferData(t.PIXEL_PACK_BUFFER,s,t.STREAM_READ)),fi(t,()=>t.readPixels(0,0,n,e,t.RGBA,t.FLOAT,0)),fi(t,()=>t.bindBuffer(t.PIXEL_PACK_BUFFER,null)),o}(this.gl,e,n,this.textureConfig);return this.unbindTextureToFrameBuffer(),r}createAndWaitForFence(){const t=this.createFence(this.gl);return this.pollFence(t)}createFence(t){let e,n;if(Object(h.b)().getBool("WEBGL_FENCE_API_ENABLED")){const r=t,o=r.fenceSync(r.SYNC_GPU_COMMANDS_COMPLETE,0);t.flush(),n=()=>{const t=r.clientWaitSync(o,0,0);return t===r.ALREADY_SIGNALED||t===r.CONDITION_SATISFIED},e=o}else Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0?(e=this.beginQuery(),this.endQuery(),n=()=>this.isQueryAvailable(e,Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))):n=()=>!0;return{query:e,isFencePassed:n}}downloadMatrixFromPackedTexture(t,e,n){return this.downloadMatrixDriver(t,()=>function(t,e,n){const r=new Float32Array(e*n*4);return fi(t,()=>t.readPixels(0,0,n,e,t.RGBA,t.FLOAT,r)),r}(this.gl,e,n))}createProgram(t){this.throwIfDisposed();const e=this.gl,n=bi(e,t),r=rc(e),o=function(t){return Ii(t,()=>t.createProgram(),"Unable to create WebGLProgram.")}(e);return fi(e,()=>e.attachShader(o,r)),fi(e,()=>e.attachShader(o,n)),function(t,e){if(fi(t,()=>t.linkProgram(e)),!1===t.getProgramParameter(e,t.LINK_STATUS))throw console.log(t.getProgramInfoLog(e)),new Error("Failed to link vertex and fragment shaders.")}(e,o),this.debug&&yi(e,o),this.vertexAttrsAreBound||(this.setProgram(o),this.vertexAttrsAreBound=function(t,e,n){return fi(t,()=>t.bindBuffer(t.ARRAY_BUFFER,n)),vi(t,e,"clipSpacePos",n,3,20,0)&&vi(t,e,"uv",n,2,20,12)}(e,this.program,this.vertexBuffer)),o}deleteProgram(t){this.throwIfDisposed(),t===this.program&&(this.program=null),null!=t&&fi(this.gl,()=>this.gl.deleteProgram(t))}setProgram(t){this.throwIfDisposed(),this.program=t,null!=this.program&&this.debug&&yi(this.gl,this.program),fi(this.gl,()=>this.gl.useProgram(t))}getUniformLocation(t,e,n=!0){return this.throwIfDisposed(),n?function(t,e,n){return Ii(t,()=>t.getUniformLocation(e,n),'uniform "'+n+'" not present in program.')}(this.gl,t,e):function(t,e,n){return t.getUniformLocation(e,n)}(this.gl,t,e)}getAttributeLocation(t,e){return this.throwIfDisposed(),fi(this.gl,()=>this.gl.getAttribLocation(t,e))}getUniformLocationNoThrow(t,e){return this.throwIfDisposed(),this.gl.getUniformLocation(t,e)}setInputMatrixTexture(t,e,n){this.throwIfDisposed(),this.throwIfNoProgram(),wi(this.gl,t,e,n)}setOutputMatrixTexture(t,e,n){this.setOutputMatrixTextureDriver(t,n,e)}setOutputPackedMatrixTexture(t,e,n){this.throwIfDisposed();const[r,o]=di(e,n);this.setOutputMatrixTextureDriver(t,r,o)}setOutputMatrixWriteRegion(t,e,n,r){this.setOutputMatrixWriteRegionDriver(n,t,r,e)}setOutputPackedMatrixWriteRegion(t,e,n,r){throw new Error("setOutputPackedMatrixWriteRegion not implemented.")}debugValidate(){null!=this.program&&yi(this.gl,this.program),Oi(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();const t=this.gl;this.debug&&this.debugValidate(),fi(t,()=>t.drawElements(t.TRIANGLES,6,t.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),fi(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return null==this.disjointQueryTimerExtension&&(this.disjointQueryTimerExtension=mi(this.gl,2===Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")?"EXT_disjoint_timer_query_webgl2":"EXT_disjoint_timer_query")),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if(2===Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")){const t=this.gl,e=this.getQueryTimerExtensionWebGL2(),n=t.createQuery();return t.beginQuery(e.TIME_ELAPSED_EXT,n),n}const t=this.getQueryTimerExtensionWebGL1(),e=t.createQueryEXT();return t.beginQueryEXT(t.TIME_ELAPSED_EXT,e),e}endQuery(){if(2===Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")){const t=this.gl,e=this.getQueryTimerExtensionWebGL2();return void t.endQuery(e.TIME_ELAPSED_EXT)}const t=this.getQueryTimerExtensionWebGL1();t.endQueryEXT(t.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(t){return await y.repeatedTry(()=>this.disposed||this.isQueryAvailable(t,Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))),this.getQueryTime(t,Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))}getQueryTime(t,e){if(0===e)return null;if(2===e){const e=this.gl;return e.getQueryParameter(t,e.QUERY_RESULT)/1e6}{const e=this.getQueryTimerExtensionWebGL1();return e.getQueryObjectEXT(t,e.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(t,e){if(0===e)return!0;if(2===e){const e=this.gl,n=this.getQueryTimerExtensionWebGL2(),r=e.getQueryParameter(t,e.QUERY_RESULT_AVAILABLE);return null==this.disjoint&&(this.disjoint=this.gl.getParameter(n.GPU_DISJOINT_EXT)),r&&!this.disjoint}{const e=this.getQueryTimerExtensionWebGL1(),n=e.getQueryObjectEXT(t,e.QUERY_RESULT_AVAILABLE_EXT);return null==this.disjoint&&(this.disjoint=this.gl.getParameter(e.GPU_DISJOINT_EXT)),n&&!this.disjoint}}pollFence(t){return new Promise(e=>{this.addItemToPoll(()=>t.isFencePassed(),()=>e())})}pollItems(){const t=function(t){let e=0;for(;e<t.length;++e){if(!t[e]())break}return e-1}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(this.itemsToPoll.map(t=>t.isDoneFn));for(let e=0;e<=t;++e){const{resolveFn:t}=this.itemsToPoll[e];t()}this.itemsToPoll=this.itemsToPoll.slice(t+1)}addItemToPoll(t,e){this.itemsToPoll.push({isDoneFn:t,resolveFn:e}),this.itemsToPoll.length>1||y.repeatedTry(()=>(this.pollItems(),0===this.itemsToPoll.length))}bindTextureToFrameBuffer(t){this.throwIfDisposed(),Ci(this.gl,t,this.framebuffer),this.debug&&Oi(this.gl)}unbindTextureToFrameBuffer(){null!=this.outputTexture?(Ci(this.gl,this.outputTexture,this.framebuffer),this.debug&&Oi(this.gl)):$i(this.gl,this.framebuffer)}downloadMatrixDriver(t,e){this.bindTextureToFrameBuffer(t);const n=e();return this.unbindTextureToFrameBuffer(),n}setOutputMatrixTextureDriver(t,e,n){this.throwIfDisposed();const r=this.gl;Ci(r,t,this.framebuffer),this.debug&&Oi(r),this.outputTexture=t,fi(r,()=>r.viewport(0,0,e,n)),fi(r,()=>r.scissor(0,0,e,n))}setOutputMatrixWriteRegionDriver(t,e,n,r){this.throwIfDisposed(),fi(this.gl,()=>this.gl.scissor(t,e,n,r))}throwIfDisposed(){if(this.disposed)throw new Error("Attempted to use disposed GPGPUContext.")}throwIfNoProgram(){if(null==this.program)throw new Error("No GPU program is currently set.")}}function fc(t,e){if(t.length!==e.length)throw Error(`Binary was compiled with ${t.length} inputs, but was executed with ${e.length} inputs`);t.forEach((t,n)=>{const r=t.logicalShape,o=e[n],s=o.shape;if(!y.arraysEqual(r,s))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${r} and ${s} must match`);if(t.isUniform&&o.isUniform)return;const a=t.texShape,i=o.isUniform?null:o.texData.texShape;if(!y.arraysEqual(a,i))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${a} and ${i} must match`)})}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class gc{constructor(t,e,n){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t;const{filterWidth:r,inChannels:o,strideWidth:s,strideHeight:a,padInfo:i,outWidth:u,dilationWidth:c,dilationHeight:l,dataFormat:h}=n,{left:d,top:p}=i,f=o*r,g=Vi(),m="channelsLast"===h,b=m?0:1,x=m?1:2;let y="";for(let n=0;n<=1;n++)for(let r=0;r<=1;r++)y+=`\n          blockIndex = rc.y + ${r};\n          pos = rc.x + ${n};\n\n          if(blockIndex < ${t[1]} && pos < ${t[0]}) {\n            offsetY = int(blockIndex / (${u})) * ${a} - ${p};\n            d0 = offsetY + ${l} * (pos / ${f});\n\n            if(d0 < ${e[b]} && d0 >= 0) {\n\n              offsetX = int(mod(float(blockIndex), ${u}.) * ${s}. - ${d}.);\n              d1 = offsetX + ${c} * (int(mod(float(pos), ${f}.) / ${o}.));\n\n              if(d1 < ${e[x]} && d1 >= 0) {\n\n                ch = int(mod(float(pos), ${o}.));\n\n                if (${m}) {\n                  innerDims = vec2(d1, ch);\n                  result[${2*n+r}] = getChannel(\n                    getA(d0, int(innerDims.x),\n                    int(innerDims.y)), innerDims);\n                } else {\n                  innerDims = vec2(d0, d1);\n                  result[${2*n+r}] = getChannel(\n                    getA(ch, int(innerDims.x),\n                    int(innerDims.y)), innerDims);\n                }\n              }\n            }\n          }\n        `;this.userCode=`\n      void main() {\n        ivec2 rc = getOutputCoords();\n\n        vec4 result = vec4(0);\n\n        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;\n        vec2 innerDims;\n\n        ${y}\n\n        ${g.output} = result;\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class mc{constructor(t,e,n,r,o){this.variableNames=["x"],this.outputShape=[];const s=e,a=t[3]-1;let i;this.outputShape=t;const u=`float(${n}) + float(${r}) * sum`;i=.5===o?`inversesqrt(${u})`:1===o?`1.0/(${u})`:`exp(log(${u}) * float(-${o}));`,this.userCode=`\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int r = coords[1];\n        int c = coords[2];\n        int d = coords[3];\n        float x = getX(b, r, c, d);\n        float sum = 0.0;\n        for (int j = -${s}; j <= ${s}; j++) {\n          int idx = d + j;\n          if (idx >= 0 && idx <=  ${a}) {\n            float z = getX(b, r, c, idx);\n            sum += z * z;\n          }\n        }\n        float val = x * ${i};\n        setOutput(val);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class bc{constructor(t,e,n,r,o){this.variableNames=["inputImage","outputImage","dy"],this.outputShape=[],this.outputShape=t,this.depth=t[3],this.depthRadius=e,this.bias=n,this.alpha=r,this.beta=o,this.userCode=`\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int r = coords[1];\n        int c = coords[2];\n\n        float result = 0.0;\n        for (int d = 0; d < ${this.depth}; ++d) {\n          int depthBegin = int(max(0.0, float(d - ${e})));\n          int depthEnd = int(min(float(${this.depth}),\n              float(d + ${e} + 1)));\n\n          const int MIN_DEPTH_BEGIN = 0;\n          const int MAX_DEPTH_END = ${this.depth};\n\n          float norm = 0.0;\n          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {\n            if (k < depthBegin){\n              continue;\n            }\n            else if (k >= depthBegin && k < depthEnd) {\n              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);\n            }\n            else {\n              break;\n            }\n          }\n\n          norm = float(${r}) * norm + float(${n});\n\n          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){\n            if (k < depthBegin){\n              continue;\n            }\n            else if (k >= depthBegin && k < depthEnd){\n              float dyi = -2.0 * float(${r})\n                * float(${o})\n                * getInputImage(b ,r ,c, k) * getOutputImage(b, r, c, d)\n                / norm;\n              if (k == d) {\n                dyi += pow(norm, -1.0 * ${o});\n              }\n              if (k == coords[3]) {\n                dyi *= getDy(b, r, c, d);\n                result += dyi;\n              }\n            }\n            else {\n              break;\n            }\n          }\n      }\n      setOutput(result);\n      }\n    `}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class xc{constructor(t,e,n,r,o){this.variableNames=["x"],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;const s=e,a=t[3]-1;let i;this.outputShape=t;const u=`float(${n}) + float(${r}) * sum`;i=.5===o?`inversesqrt(${u})`:1===o?`1.0/(${u})`:`exp(log(${u}) * float(-${o}));`,this.userCode=`\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords.x;\n        int r = coords.y;\n        int c = coords.z;\n        int d = coords.w;\n\n        bool hasNextCol = d < ${this.outputShape[3]};\n        bool hasNextRow = c < ${this.outputShape[2]};\n\n        vec4 sum = vec4(0.);\n        vec4 xFragAtOutputCoords = getX(b, r, c, d);\n\n        vec4 xAtOutputCoords = vec4(\n          getChannel(xFragAtOutputCoords, vec2(c, d)),\n          hasNextCol ?\n            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,\n          hasNextRow ?\n            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0\n        );\n\n        int firstChannel = d - ${s};\n        vec2 cache = vec2(0.);\n        if(firstChannel >= 0){\n          vec4 firstChannelFrag = getX(b, r, c, firstChannel);\n          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));\n            if(hasNextRow){\n              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));\n            }\n        }\n\n        ivec2 depth = ivec2(d, d + 1);\n        for (int j = - ${s}; j <= ${s}; j++) {\n          ivec2 idx = depth + j;\n          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));\n          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${a}));\n\n          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;\n          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;\n\n          if(depthInRange || depthPlusOneInRange){\n            vec4 z = vec4(0.);\n            vec4 xFragAtCurrentDepth;\n            z.xz = cache.xy;\n            if(depthPlusOneInRange && hasNextCol){\n              xFragAtCurrentDepth = idx.y != d ?\n                getX(b, r, c, idx.y) : xFragAtOutputCoords;\n              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));\n              if(hasNextRow){\n                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));\n              }\n            }\n            cache.xy = z.yw;\n            sum += z * z;\n          }\n        }\n        vec4 result = xAtOutputCoords * ${i};\n        setOutput(result);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class yc{constructor(t){this.variableNames=["dy","maxPos"],this.outputShape=t.inShape;const e=t.strideHeight,n=t.strideWidth,r=t.dilationHeight,o=t.effectiveFilterHeight,s=t.effectiveFilterWidth,a=o-1-t.padInfo.top,i=s-1-t.padInfo.left,u=o*s-1;this.userCode=`\n      const ivec2 pads = ivec2(${a}, ${i});\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n\n        ivec2 dyRCCorner = coords.yz - pads;\n        int dyRCorner = dyRCCorner.x;\n        int dyCCorner = dyRCCorner.y;\n\n        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < ${o};\n          wR += ${r}) {\n          float dyR = float(dyRCorner + wR) / ${e}.0;\n\n          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          for (int wC = 0; wC < ${s}; wC++) {\n            float dyC = float(dyCCorner + wC) / ${n}.0;\n\n            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||\n                fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            float dyValue = getDy(b, idyR, idyC, d);\n            int maxPosValue = ${u} - int(getMaxPos(b, idyR, idyC, d));\n\n            // Get the current value, check it against the value from the\n            // position matrix.\n            int curPosValue = wR * ${s} + wC;\n            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);\n\n            dotProd += dyValue * mask;\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}class vc{constructor(t){this.variableNames=["dy","maxPos"],this.outputShape=t.inShape;const e=t.strideDepth,n=t.strideHeight,r=t.strideWidth,o=t.dilationDepth,s=t.dilationHeight,a=t.dilationWidth,i=t.effectiveFilterDepth,u=t.effectiveFilterHeight,c=t.effectiveFilterWidth,l=i-1-t.padInfo.front,h=u-1-t.padInfo.top,d=c-1-t.padInfo.left,p=i*u*c-1;this.userCode=`\n      const ivec3 pads = ivec3(${l}, ${h}, ${d});\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int ch = coords.u;\n\n        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;\n        int dyDCorner = dyCorner.x;\n        int dyRCorner = dyCorner.y;\n        int dyCCorner = dyCorner.z;\n\n        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get\n        // dx(xD, xR, xC, ch).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n\n        for (int wD = 0; wD < ${i};\n           wD += ${o}) {\n          float dyD = float(dyDCorner + wD) / ${e}.0;\n\n          if (dyD < 0.0 || dyD >= ${t.outDepth}.0 || fract(dyD) > 0.0) {\n            continue;\n          }\n          int idyD = int(dyD);\n\n          for (int wR = 0; wR < ${u};\n              wR += ${s}) {\n            float dyR = float(dyRCorner + wR) / ${n}.0;\n\n            if (dyR < 0.0 || dyR >= ${t.outHeight}.0 ||\n                fract(dyR) > 0.0) {\n              continue;\n            }\n            int idyR = int(dyR);\n\n            for (int wC = 0; wC < ${c};\n                wC += ${a}) {\n              float dyC = float(dyCCorner + wC) / ${r}.0;\n\n              if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||\n                  fract(dyC) > 0.0) {\n                continue;\n              }\n              int idyC = int(dyC);\n\n              float dyValue = getDy(batch, idyD, idyR, idyC, ch);\n              int maxPosValue = ${p} -\n                  int(getMaxPos(batch, idyD, idyR, idyC, ch));\n\n              // Get the current value, check it against the value from the\n              // position matrix.\n              int curPosValue =\n                  wD * ${u} * ${c} +\n                  wR * ${c} + wC;\n              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);\n\n              dotProd += dyValue * mask;\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wc{constructor(t,e,n=!1,r=!1,o=!1,s=null,a=!1){this.variableNames=["matrixA","matrixB"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e;const i=n?t[1]:t[2],u=Math.ceil(i/2),c=n?"i * 2, rc.y":"rc.y, i * 2",l=r?"rc.z, i * 2":"i * 2, rc.z",h=n?["a.xxyy","a.zzww"]:["a.xxzz","a.yyww"],d=r?["b.xzxz","b.ywyw"]:["b.xyxy","b.zwzw"];let p="",f="";s&&(p=a?`vec4 activation(vec4 a) {\n          vec4 b = getPreluActivationWeightsAtOutCoords();\n          ${s}\n        }`:`vec4 activation(vec4 x) {\n          ${s}\n        }`,f="result = activation(result);");const g=o?"result += getBiasAtOutCoords();":"";o&&this.variableNames.push("bias"),a&&this.variableNames.push("preluActivationWeights"),this.userCode=`\n      ${p}\n\n      const float sharedDimension = ${u}.0;\n\n      vec4 dot2x2ARowBCol(ivec3 rc) {\n        vec4 result = vec4(0);\n        for (int i = 0; i < ${u}; i++) {\n          vec4 a = getMatrixA(rc.x, ${c});\n          vec4 b = getMatrixB(rc.x, ${l});\n\n          // These swizzled products need to be separately added.\n          // See: https://github.com/tensorflow/tfjs/issues/1735\n          result += (${h[0]} * ${d[0]});\n          result += (${h[1]} * ${d[1]});\n        }\n        return result;\n      }\n\n      void main() {\n        ivec3 rc = getOutputCoords();\n        vec4 result = dot2x2ARowBCol(rc);\n\n        ${g}\n\n        ${f}\n\n        setOutput(result);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Cc{constructor(t,e,n){this.variableNames=["probs"],this.outputShape=[t,n],this.userCode=`\n      uniform float seed;\n\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int batch = coords[0];\n\n        float r = random(seed);\n        float cdf = 0.0;\n\n        for (int i = 0; i < ${e-1}; i++) {\n          cdf += getProbs(batch, i);\n\n          if (r < cdf) {\n            setOutput(float(i));\n            return;\n          }\n        }\n\n        // If no other event happened, last event happened.\n        setOutput(float(${e-1}));\n      }\n    `}getCustomSetupFunc(t){return(e,n)=>{null==this.seedLoc&&(this.seedLoc=e.getUniformLocation(n,"seed")),e.gl.uniform1f(this.seedLoc,t)}}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $c{constructor(t,e,n,r){this.variableNames=["indices"],this.outputShape=[t,e],this.userCode=`\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int index = round(getIndices(coords.x));\n        setOutput(mix(float(${r}), float(${n}),\n                      float(index == coords.y)));\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Oc{constructor(t){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=t;const e=t.length;if(0===e)this.userCode="\n        void main() {\n          setOutput(vec4(getA(), 0., 0., 0.));\n        }\n      ";else{const n=Ui("rc",e),r=su(e),o=function(t,e,n){if(1===t)return"rc > "+e[0];let r="";for(let o=t-2;o<t;o++)r+=`${n[o]} >= ${e[o]}`,o<t-1&&(r+="||");return r}(e,t,n),s=function(t,e,n,r){if(1===t)return"";const o=r.slice(-2);return`\n    int r = ${o[0]};\n    int c = ${o[1]};\n    int rp1 = r + 1;\n    int cp1 = c + 1;\n\n    bool cEdge = cp1 >= ${e};\n    bool rEdge = rp1 >= ${n};\n  `}(e,t[t.length-1],t[t.length-2],n),a=function(t,e){const n=t.length,r=function(t,e){const n=[];for(let r=0;r<=1;r++)for(let o=0;o<=1;o++){let s=`${0===r?"r":"rp1"}, ${0===o?"c":"cp1"}`;for(let n=2;n<t;n++)s=e[e.length-1-n]+","+s;n.push(s)}return n}(n,e);if(1===n)return`getA(rc),\n            rc + 1 >= ${t[0]} ? 0. : getA(rc + 1),\n            0, 0`;return`getA(${r[0]}),\n          cEdge ? 0. : getA(${r[1]}),\n          rEdge ? 0. : getA(${r[2]}),\n          rEdge || cEdge ? 0. : getA(${r[3]})`}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(t,n);this.userCode=`\n        void main() {\n          ${r} rc = getOutputCoords();\n\n          if(${o}) {\n            setOutput(vec4(0));\n          } else {\n            ${s}\n\n            setOutput(vec4(${a}));\n          }\n        }\n      `}}}class Ic{constructor(t,e,n){this.variableNames=["x"],this.outputShape=e.map((e,n)=>e[0]+t[n]+e[1]);const r=t.length,o=su(r),s=e.map(t=>t[0]).join(","),a=e.map((e,n)=>e[0]+t[n]).join(","),i=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,r);this.userCode=1!==r?`\n      ${o} start = ${o}(${s});\n      ${o} end = ${o}(${a});\n\n      void main() {\n        ${o} outC = getOutputCoords();\n        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {\n          setOutput(float(${n}));\n        } else {\n          ${o} coords = outC - start;\n          setOutput(getX(${i}));\n        }\n      }\n    `:`\n        int start = ${s};\n        int end = ${a};\n\n        void main() {\n          int outC = getOutputCoords();\n          if (outC < start || outC >= end) {\n            setOutput(float(${n}));\n          } else {\n            setOutput(getX(outC - start));\n          }\n        }\n      `}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Sc{constructor(t,e,n){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e.map((e,n)=>e[0]+t[n]+e[1]);const r=t.length,o=su(r),s=e.map(t=>t[0]).join(","),a=e.map((e,n)=>e[0]+t[n]).join(","),i=Ui("rc",r),u=Ui("source",r),c=`${i[r-1]} < ${this.outputShape[r-1]}`,l=1===r?"source":`vec2(${u.slice(-2).join()})`,h=[o+" rc = outputLoc;",`${i[r-1]} += 1;\n       if(${c}) {\n      `,1===r?"":`}\n       rc = outputLoc;\n       ${i[r-2]} += 1;\n       if(${i[r-2]} < ${this.outputShape[r-2]}) {`,1===r?"":`  ${i[r-1]} += 1;\n         if(${c}) {`],d=1===r?"rc < start || rc >= end":"any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))";let p="";for(let t=0,e=1===r?2:4;t<e;t++)p+=`\n        ${h[t]}\n        if (${d}) {\n          result[${t}] = float(${n});\n        } else {\n          ${o} source = rc - start;\n          result[${t}] = getChannel(getX(${u.join()}), ${l});\n        }\n      `;p+=1===r?"} ":"}}",this.userCode=`\n      const ${o} start = ${o}(${s});\n      const ${o} end = ${o}(${a});\n\n      void main() {\n        ${o} outputLoc = getOutputCoords();\n        vec4 result = vec4(0.);\n        ${p}\n        setOutput(result);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ec{constructor(t,e,n,r=!1,o=!1){if(this.variableNames=["x"],"avg"===e&&n)throw new Error("Cannot compute positions for average pool.");const s=t.filterWidth,a=t.strideHeight,i=t.strideWidth,u=t.dilationHeight,c=t.dilationWidth,l=t.effectiveFilterHeight,h=t.effectiveFilterWidth,d=t.padInfo.top,p=t.padInfo.left;this.outputShape=t.outShape;const f="avg"===e,g=`((batch  * ${t.inHeight} + xR) * ${t.inWidth} + xC) * ${t.inChannels} + d`,m=`(xR * ${t.inWidth} + xC) * ${t.inChannels} + d`;let b="0.0";if(f||(b="-1.0 / 1e-20"),n){const e=">=";return void(this.userCode=`\n        const ivec2 strides = ivec2(${a}, ${i});\n        const ivec2 pads = ivec2(${d}, ${p});\n\n        void main() {\n          ivec4 coords = getOutputCoords();\n          int batch = coords[0];\n          int d = coords[3];\n\n          ivec2 xRCCorner = coords.yz * strides - pads;\n          int xRCorner = xRCCorner.x;\n          int xCCorner = xRCCorner.y;\n\n          // max/min x(?, ?, d) to get y(yR, yC, d).\n          // ? = to be determined\n          float minMaxValue = 0.0;\n          float minMaxValueFound = 0.0;\n          int minMaxPosition = 0;\n          float avgValue = 0.0;\n\n          for (int wR = 0; wR < ${l};\n              wR += ${u}) {\n            int xR = xRCorner + wR;\n\n            if (xR < 0 || xR >= ${t.inHeight}) {\n              continue;\n            }\n\n            for (int wC = 0; wC < ${h};\n                wC += ${c}) {\n              int xC = xCCorner + wC;\n\n              if (xC < 0 || xC >= ${t.inWidth}) {\n                continue;\n              }\n\n              float value = getX(batch, xR, xC, d);\n\n              // If a min / max value has already been found, use it. If not,\n              // use the current value.\n              float currMinMaxValue = mix(\n                  value, minMaxValue, minMaxValueFound);\n              if (value ${e} currMinMaxValue) {\n                minMaxValue = value;\n                minMaxValueFound = 1.0;\n                minMaxPosition = ${r?o?g:m:`wR * ${h} + wC`};\n              }\n            }\n          }\n          setOutput(float(minMaxPosition));\n        }\n      `)}let x=`${e}(${e}(${e}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;"avg"===e&&(x="avgValue / count");const y=4*Math.floor(s/4),v=s%4,w=`\n      if (${f}) {\n        avgValue += dot(values, ones);\n      } else {\n        minMaxValue = max(values, minMaxValue);\n      }\n    `;this.userCode=`\n      const ivec2 strides = ivec2(${a}, ${i});\n      const ivec2 pads = ivec2(${d}, ${p});\n      const float initializationValue = ${b};\n      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);\n\n      float count = 0.0;\n\n      float getValue(int batch, int xR, int xC, int d) {\n        if (xC < 0 || xC >= ${t.inWidth}) {\n          return initializationValue;\n        }\n        count += 1.0;\n        return getX(batch, xR, xC, d);\n      }\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int batch = coords[0];\n        int d = coords[3];\n\n        ivec2 xRCCorner = coords.yz * strides - pads;\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        // max/min x(?, ?, d) to get y(yR, yC, d).\n        // ? = to be determined\n        vec4 minMaxValue = vec4(${b});\n        float avgValue = 0.0;\n        count = 0.0;\n\n        for (int wR = 0; wR < ${l};\n            wR += ${u}) {\n          int xR = xRCorner + wR;\n\n          if (xR < 0 || xR >= ${t.inHeight}) {\n            continue;\n          }\n\n          for (int wC = 0; wC < ${y}; wC += 4) {\n            int xC = xCCorner + wC * ${c};\n\n            vec4 values = vec4(\n              getValue(batch, xR, xC, d),\n              getValue(batch, xR, xC + ${c}, d),\n              getValue(batch, xR, xC + 2 * ${c}, d),\n              getValue(batch, xR, xC + 3 * ${c}, d)\n            );\n\n            ${w}\n          }\n\n          int xC = xCCorner + ${y};\n          if (${1===v}) {\n            vec4 values = vec4(\n              getValue(batch, xR, xC, d),\n              initializationValue,\n              initializationValue,\n              initializationValue\n            );\n\n            ${w}\n          } else if (${2===v}) {\n            vec4 values = vec4(\n              getValue(batch, xR, xC, d),\n              getValue(batch, xR, xC + ${c}, d),\n              initializationValue,\n              initializationValue\n            );\n\n            ${w}\n          } else if (${3===v}) {\n            vec4 values = vec4(\n              getValue(batch, xR, xC, d),\n              getValue(batch, xR, xC + ${c}, d),\n              getValue(batch, xR, xC + 2 * ${c}, d),\n              initializationValue\n            );\n\n            ${w}\n          }\n        }\n        setOutput(${x});\n      }\n    `}}class Rc{constructor(t,e,n,r=!1,o=!1){if(this.variableNames=["x"],"avg"===e&&n)throw new Error("Cannot compute positions for average pool.");const s=t.filterWidth,a=t.strideDepth,i=t.strideHeight,u=t.strideWidth,c=t.dilationDepth,l=t.dilationHeight,h=t.dilationWidth,d=t.effectiveFilterDepth,p=t.effectiveFilterHeight,f=t.effectiveFilterWidth,g=t.padInfo.front,m=t.padInfo.top,b=t.padInfo.left;this.outputShape=t.outShape;const x="avg"===e;let y="0.0";if(x||(y="-1.0 / 1e-20"),n){const e=">=";return void(this.userCode=`\n        const ivec3 strides =\n            ivec3(${a}, ${i}, ${u});\n        const ivec3 pads = ivec3(${g}, ${m}, ${b});\n\n        void main() {\n          ivec5 coords = getOutputCoords();\n          int batch = coords.x;\n          int ch = coords.u;\n\n          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;\n          int xDCorner = xCorner.x;\n          int xRCorner = xCorner.y;\n          int xCCorner = xCorner.z;\n\n          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).\n          // ? = to be determined\n          float minMaxValue = 0.0;\n          float minMaxValueFound = 0.0;\n          int minMaxPosition = 0;\n\n          for (int wD = 0; wD < ${d};\n              wD += ${c}) {\n            int xD = xDCorner + wD;\n\n            if (xD < 0 || xD >= ${t.inDepth}) {\n              continue;\n            }\n\n            for (int wR = 0; wR < ${p};\n                wR += ${l}) {\n              int xR = xRCorner + wR;\n\n              if (xR < 0 || xR >= ${t.inHeight}) {\n                continue;\n              }\n\n              for (int wC = 0; wC < ${f};\n                  wC += ${h}) {\n                int xC = xCCorner + wC;\n\n                if (xC < 0 || xC >= ${t.inWidth}) {\n                  continue;\n                }\n\n                float value = getX(batch, xD, xR, xC, ch);\n\n                // If a min / max value has already been found, use it. If not,\n                // use the current value.\n                float currMinMaxValue = mix(\n                    value, minMaxValue, minMaxValueFound);\n                if (value ${e} currMinMaxValue) {\n                  minMaxValue = value;\n                  minMaxValueFound = 1.0;\n                  minMaxPosition = ${r?o?`(((batch * ${t.inDepth} + xD) * ${t.inHeight} + xR) * ${t.inWidth} + xC) * ${t.inChannels} + ch`:`((xD * ${t.inHeight} + xR) * ${t.inWidth} + xC) * ${t.inChannels} + ch`:`wD * ${p} * ${f} +\n                      wR * ${f} + wC`};\n                }\n              }\n            }\n          }\n          setOutput(float(minMaxPosition));\n        }\n      `)}let v=`${e}(${e}(${e}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;"avg"===e&&(v="avgValue / count");const w=4*Math.floor(s/4),C=s%4,$=`\n      if (${x}) {\n        avgValue += dot(values, ones);\n      } else {\n        minMaxValue = max(values, minMaxValue);\n      }\n    `;this.userCode=`\n      const ivec3 strides =\n        ivec3(${a}, ${i}, ${u});\n      const ivec3 pads = ivec3(${g}, ${m}, ${b});\n      const float initializationValue = ${y};\n      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);\n\n      float count = 0.0;\n\n      float getValue(int batch, int xD, int xR, int xC, int ch) {\n        if (xC < 0 || xC >= ${t.inWidth}) {\n          return initializationValue;\n        }\n        count += 1.0;\n        return getX(batch, xD, xR, xC, ch);\n      }\n\n      void main() {\n        ivec5 coords = getOutputCoords();\n        int batch = coords.x;\n        int ch = coords.u;\n\n        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;\n        int xDCorner = xCorner.x;\n        int xRCorner = xCorner.y;\n        int xCCorner = xCorner.z;\n\n        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).\n        // ? = to be determined\n        vec4 minMaxValue = vec4(${y});\n        float avgValue = 0.0;\n        count = 0.0;\n\n        for (int wD = 0; wD < ${d};\n            wD += ${c}) {\n          int xD = xDCorner + wD;\n\n          if (xD < 0 || xD >= ${t.inDepth}) {\n            continue;\n          }\n\n          for (int wR = 0; wR < ${p};\n            wR += ${l}) {\n            int xR = xRCorner + wR;\n\n            if (xR < 0 || xR >= ${t.inHeight}) {\n              continue;\n            }\n\n            for (int wC = 0; wC < ${w}; wC += 4) {\n              int xC = xCCorner + wC * ${h};\n\n              vec4 values = vec4(\n                getValue(batch, xD, xR, xC, ch),\n                getValue(batch, xD, xR, xC + ${h}, ch),\n                getValue(batch, xD, xR, xC + 2 * ${h}, ch),\n                getValue(batch, xD, xR, xC + 3 * ${h}, ch)\n              );\n\n              ${$}\n            }\n\n            int xC = xCCorner + ${w};\n            if (${1===C}) {\n              vec4 values = vec4(\n                getValue(batch, xD, xR, xC, ch),\n                initializationValue,\n                initializationValue,\n                initializationValue\n              );\n\n              ${$}\n            } else if (${2===C}) {\n              vec4 values = vec4(\n                getValue(batch, xD, xR, xC, ch),\n                getValue(batch, xD, xR, xC + ${h}, ch),\n                initializationValue,\n                initializationValue\n              );\n\n              ${$}\n            } else if (${3===C}) {\n              vec4 values = vec4(\n                getValue(batch, xD, xR, xC, ch),\n                getValue(batch, xD, xR, xC + ${h}, ch),\n                getValue(batch, xD, xR, xC + 2 * ${h}, ch),\n                initializationValue\n              );\n\n              ${$}\n            }\n          }\n          setOutput(${v});\n        }\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ac{constructor(t,e){this.variableNames=["x"];const{windowSize:n,batchSize:r,inSize:o,outSize:s}=t;this.outputShape=[r,s];let a="0.0",i="";"prod"===e?a="1.0":"min"===e?(a="1.0 / 1e-20",i="min"):"max"===e&&(a="-1.0 / 1e-20",i="max");let u=`${e}(${e}(${e}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;"sum"===e?u="sumValue":"prod"===e?u="prodValue":"all"===e?u="allValue":"any"===e&&(u="anyValue");const c=4*Math.floor(n/4),l=n%4;let h=`\n      if (${"sum"===e}) {\n        sumValue += dot(values, ones);\n      } else if (${"prod"===e}) {\n        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);\n        prodValue *= tmp[0] * tmp[1];\n      } else {\n        minMaxValue = ${i}(values, minMaxValue);\n      }\n    `,d="vec4";"all"===e?(a="1.0",h="\n        bool reducedAllValue = all(values);\n        float floatedReducedAllValue = float(reducedAllValue);\n        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);\n      ",d="bvec4"):"any"===e&&(a="0.0",h="\n        bool reducedAnyValue = any(values);\n        float floatedReducedAnyValue = float(reducedAnyValue);\n        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);\n      ",d="bvec4");let p="";o%n>0&&(p=`\n        if (inIdx < 0 || inIdx >= ${o}) {\n          return initializationValue;\n        }\n      `),this.userCode=`\n      const float initializationValue = ${a};\n      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);\n\n      float getValue(int batch, int inIdx) {\n        ${p}\n        return getX(batch, inIdx);\n      }\n\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int batch = coords[0];\n        int outIdx = coords[1];\n        int inOffset = outIdx * ${n};\n\n        vec4 minMaxValue = vec4(${a});\n        float prodValue = 1.0;\n        float sumValue = 0.0;\n        float allValue = 1.0;\n        float anyValue = 0.0;\n\n        for (int i = 0; i < ${c}; i += 4) {\n          int inIdx = inOffset + i;\n          ${d} values = ${d}(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            getValue(batch, inIdx + 2),\n            getValue(batch, inIdx + 3)\n          );\n\n          ${h}\n        }\n\n        int inIdx = inOffset + ${c};\n        if (${1===l}) {\n          ${d} values = ${d}(\n            getValue(batch, inIdx),\n            initializationValue,\n            initializationValue,\n            initializationValue\n          );\n\n          ${h}\n        } else if (${2===l}) {\n          ${d} values = ${d}(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            initializationValue,\n            initializationValue\n          );\n\n          ${h}\n        } else if (${3===l}) {\n          ${d} values = ${d}(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            getValue(batch, inIdx + 2),\n            initializationValue\n          );\n\n          ${h}\n        }\n        setOutput(${u});\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class kc{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t;let n="";for(let t=0;t<4;t++){let e="thisRC = rc;";t%2==1&&(e+="thisRC.z += 1;"),t>1&&(e+="thisRC.y += 1;"),n+=`\n        ${e}\n        ${t>0?"if(thisRC.y < rows && thisRC.z < cols){":""}\n          int flatIndex = getFlatIndex(thisRC);\n\n          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);\n          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));\n\n          result[${t}] =\n            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);\n        ${t>0?"}":""}\n      `}var r;
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */this.userCode=`\n      ${r=e,`\n    ivec3 inputCoordsFromReshapedOutCoords(int index) {\n      ${Gi(["r","c","d"],r)}\n      return ivec3(r, c, d);\n    }\n  `}\n      ${Hi(t)}\n\n      void main() {\n        ivec3 rc = getOutputCoords();\n\n        vec4 result = vec4(0.);\n\n        ivec3 thisRC;\n        int rows = ${t[1]};\n        int cols = ${t[2]};\n\n        ${n}\n\n        setOutput(result);\n      }\n    `}}class Tc{constructor(t,e,n){this.variableNames=["dy"],this.outputShape=[],this.outputShape=e.shape;const[,r,o]=e.shape,[,s,a]=t.shape,i=[n&&s>1?r-1:r,n&&a>1?o-1:o],u=[n&&s>1?s-1:s,n&&a>1?a-1:a],c=i[0]/u[0],l=i[1]/u[1],h=1/c,d=1/l,p=2*Math.ceil(h)+2,f=2*Math.ceil(d)+2;this.userCode=`\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        int r = coords[1];\n        int c = coords[2];\n\n        float accumulator = 0.0;\n\n        const float heightScale = float(${c});\n        const float widthScale = float(${l});\n\n        const float invHeightScale = float(${h});\n        const float invWidthScale = float(${d});\n\n        const int winHeight = int(${p});\n        const int winWidth = int(${f});\n\n        // Compute bounds for where in dy we will look\n        float startRLerp = floor(float(r) * invHeightScale);\n        int startDyR = int(startRLerp - float(winHeight / 2));\n\n        float startCLerp = floor(float(c) * invWidthScale);\n        int startDyC = int(startCLerp - float(winWidth / 2));\n\n        // Loop over dy\n        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {\n          int dyR = dyROffset + startDyR;\n\n          // Guard against the window exceeding the bounds of dy\n          if (dyR < 0 || dyR >= ${s}) {\n            continue;\n          }\n\n          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {\n            int dyC = dyCOffset + startDyC;\n\n            // Guard against the window exceeding the bounds of dy\n            if (dyC < 0 || dyC >= ${a}) {\n              continue;\n            }\n\n            float dxR = float(dyR) * heightScale;\n            int topDxRIndex = int(floor(dxR));\n            int bottomDxRIndex = int(min(ceil(dxR), ${r-1}.0));\n            float dxRLerp = dxR - float(topDxRIndex);\n            float inverseDxRLerp = 1.0 - dxRLerp;\n\n            float dxC = float(dyC) * widthScale;\n            int leftDxCIndex = int(floor(dxC));\n            int rightDxCIndex = int(min(ceil(dxC), ${o-1}.0));\n            float dxCLerp = dxC - float(leftDxCIndex);\n            float inverseDxCLerp = 1.0 - dxCLerp;\n\n            if (r == topDxRIndex && c == leftDxCIndex) {\n              // topLeft\n              accumulator +=\n                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;\n            }\n\n            if (r == topDxRIndex && c == rightDxCIndex) {\n              // topRight\n              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;\n            }\n\n            if (r == bottomDxRIndex && c == leftDxCIndex) {\n              // bottomLeft\n              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;\n            }\n\n            if (r == bottomDxRIndex && c == rightDxCIndex) {\n              // bottomRight\n              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;\n            }\n          }\n        }\n        // End loop over dy\n\n        setOutput(accumulator);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Fc{constructor(t,e,n,r){this.variableNames=["A"],this.outputShape=[];const[o,s,a,i]=t;this.outputShape=[o,e,n,i];const u=[r&&e>1?s-1:s,r&&n>1?a-1:a],c=[r&&e>1?e-1:e,r&&n>1?n-1:n];this.userCode=`\n      const vec2 effectiveInputOverOutputRatioRC = vec2(\n          ${u[0]/c[0]},\n          ${u[1]/c[1]});\n      const vec2 inputShapeRC = vec2(${s}.0, ${a}.0);\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        ivec2 yRC = coords.yz;\n\n        // Fractional source index.\n        vec2 sourceFracIndexRC = vec2(yRC) * effectiveInputOverOutputRatioRC;\n\n        // Compute the four integer indices.\n        ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);\n        ivec2 sourceCeilRC = ivec2(\n          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));\n\n        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);\n        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);\n        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);\n        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);\n\n        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);\n\n        float top = topLeft + (topRight - topLeft) * fracRC.y;\n        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;\n        float newValue = top + (bottom - top) * fracRC.x;\n\n        setOutput(newValue);\n      }\n    `}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Nc{constructor(t,e,n,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[o,s,a,i]=t;this.outputShape=[o,e,n,i];const u=[r&&e>1?s-1:s,r&&n>1?a-1:a],c=[r&&e>1?e-1:e,r&&n>1?n-1:n];this.userCode=`\n      const vec3 effectiveInputOverOutputRatioRC = vec3(\n          ${u[0]/c[0]},\n          ${u[1]/c[1]},\n          ${u[1]/c[1]});\n      const vec3 inputShapeRC = vec3(${s}.0, ${a}.0,\n                                     ${a}.0);\n\n      float getAValue(int b, int r, int c, int d) {\n        return getChannel(getA(b, r, c, d), vec2(c, d));\n      }\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        // Calculate values for next column in yRC.z.\n        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);\n\n        // Fractional source index.\n        vec3 sourceFracIndexRC = vec3(yRC) * effectiveInputOverOutputRatioRC;\n\n        // Compute the four integer indices.\n        ivec3 sourceFloorRC = ivec3(sourceFracIndexRC);\n        ivec3 sourceCeilRC = ivec3(\n          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));\n\n        // Should we calculate next column and row elements in 2x2 packed cell.\n        bool hasNextCol = d < ${i-1};\n        bool hasNextRow = coords.z < ${n-1};\n\n        // In parallel, construct four corners for all four components in\n        // packed 2x2 cell.\n        vec4 topLeft = vec4(\n          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),\n          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)\n                     : 0.0,\n          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)\n                     : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);\n\n        vec4 bottomLeft = vec4(\n          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),\n          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)\n                     : 0.0,\n          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)\n                     : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);\n\n        vec4 topRight = vec4(\n          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),\n          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)\n                     : 0.0,\n          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)\n                     : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);\n\n        vec4 bottomRight = vec4(\n          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),\n          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)\n                     : 0.0,\n          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)\n                     : 0.0,\n          (hasNextRow && hasNextCol) ?\n            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);\n\n        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);\n\n        vec4 top = mix(topLeft, topRight, fracRC.yyzz);\n        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);\n        vec4 newValue = mix(top, bottom, fracRC.x);\n\n        setOutput(newValue);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Dc{constructor(t,e,n){this.variableNames=["dy"],this.outputShape=[],this.outputShape=e.shape;const[,r,o]=e.shape,[,s,a]=t.shape,i=[n&&s>1?r-1:r,n&&a>1?o-1:o],u=[n&&s>1?s-1:s,n&&a>1?a-1:a],c=i[0]/u[0],l=i[1]/u[1],h=1/c,d=1/l,p=2*Math.ceil(h)+2,f=2*Math.ceil(d)+2;this.userCode=`\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        int r = coords[1];\n        int c = coords[2];\n\n        float accumulator = 0.0;\n\n        const float heightScale = float(${c});\n        const float widthScale = float(${l});\n\n        const float invHeightScale = float(${h});\n        const float invWidthScale = float(${d});\n\n        const int winHeight = int(${p});\n        const int winWidth = int(${f});\n\n        // Compute bounds for where in dy we will look\n        float startRLerp = floor(float(r) * invHeightScale);\n        int startDyR = int(floor(startRLerp - float(winHeight / 2)));\n\n        float startCLerp = floor(float(c) * invWidthScale);\n        int startDyC = int(floor(startCLerp - float(winWidth / 2)));\n\n        // Loop over dy\n        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {\n          int dyR = dyROffset + startDyR;\n\n          // Guard against the window exceeding the bounds of dy\n          if (dyR < 0 || dyR >= ${s}) {\n            continue;\n          }\n\n          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {\n            int dyC = dyCOffset + startDyC;\n\n            // Guard against the window exceeding the bounds of dy\n            if (dyC < 0 || dyC >= ${a}) {\n              continue;\n            }\n\n            float sourceFracRow =\n              float(${i[0]}) *\n                (float(dyR) / float(${u[0]}));\n\n            float sourceFracCol =\n                float(${i[1]}) *\n                  (float(dyC) / float(${u[1]}));\n\n            int sourceNearestRow = int(min(\n                float(int(${r}) - 1),\n                ${n} ? float(round(sourceFracRow)) :\n                                  float(floor(sourceFracRow))));\n\n            int sourceNearestCol = int(min(\n                float(int(${o}) - 1),\n                ${n} ? float(round(sourceFracCol)) :\n                                  float(floor(sourceFracCol))));\n\n            if (r == sourceNearestRow && c == sourceNearestCol) {\n              accumulator += getDy(b, dyR, dyC, d);\n            }\n          }\n        }\n        // End loop over dy\n\n        setOutput(accumulator);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class _c{constructor(t,e,n,r){this.variableNames=["A"],this.outputShape=[];const[o,s,a,i]=t;this.outputShape=[o,e,n,i];const u=[r&&e>1?s-1:s,r&&n>1?a-1:a],c=[r&&e>1?e-1:e,r&&n>1?n-1:n],l=r?"0.5":"0.0";this.userCode=`\n      const vec2 effectiveInputOverOutputRatioRC = vec2(\n          ${u[0]/c[0]},\n          ${u[1]/c[1]});\n      const vec2 inputShapeRC = vec2(${s}.0, ${a}.0);\n\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        ivec2 yRC = coords.yz;\n\n        // Fractional source index.\n        vec2 sourceFracIndexRC = vec2(yRC) * effectiveInputOverOutputRatioRC;\n\n        // Compute the coordinators of nearest neighbor point.\n        ivec2 sourceNearestRC = ivec2(\n          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${l})));\n\n        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);\n\n        setOutput(newValue);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Bc{constructor(t,e){this.variableNames=["x"];const n=t.length;if(n>4)throw new Error(`WebGL backend: Reverse of rank-${n} tensor is not yet supported`);if(this.outputShape=t,1===n)return void(this.userCode=`\n        void main() {\n          int coord = getOutputCoords();\n          setOutput(getX(${t[0]} - coord - 1));\n        }\n      `);const r=t.map((n,r)=>(n=>-1!==e.indexOf(n)&&1!==t[n]?`${t[n]} - coords[${n}] - 1`:`coords[${n}]`)(r)).join(","),o=su(n);this.userCode=`\n      void main() {\n        ${o} coords = getOutputCoords();\n        setOutput(getX(${r}));\n      }\n    `}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class jc{constructor(t,e){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0;const n=t.length;if(n>4)throw new Error(`WebGL backend: Reverse of rank-${n} tensor is not yet supported`);this.outputShape=t;const r=Ui("rc",n),o=`${r[n-1]} + 1 < ${this.outputShape[n-1]}`,s=`${r[n-2]} + 1 < ${this.outputShape[n-2]}`,a=su(n);function i(n){const r=t.map((r,o)=>function(n,r){return-1!==e.indexOf(n)&&1!==t[n]?`${t[n]} - ${r[n]} - 1`:""+r[n]}(o,n));return`getChannel(getX(${r.join(",")}), vec2(${r.slice(-2).join(",")}))`}this.userCode=1===n?`\n        void main(){\n          int rc = getOutputCoords();\n          vec4 result = vec4(0.);\n          result.r = getChannel(getX(${t[0]} - rc - 1),\n            ${t[0]} - rc - 1);\n          if(${o}){\n              result.g = getChannel(getX(${t[0]} - (rc  + 1) - 1),\n                ${t[0]} - (rc  + 1) - 1);\n          }\n          setOutput(result);\n        }\n      `:`\n        void main() {\n          ${a} rc = getOutputCoords();\n          vec4 result = vec4(0.);\n          result.r = ${function(t){return i(t)}(r.slice())};\n          if(${o}){\n            result.g = ${function(t){return t[n-1]="("+t[n-1]+" + 1)",i(t)}(r.slice())};\n          }\n          if(${s}) {\n            result.b = ${function(t){return t[n-2]="("+t[n-2]+" + 1)",i(t)}(r.slice())};\n            if(${o}) {\n              result.a = ${function(t){return t[n-1]="("+t[n-1]+" + 1)",t[n-2]="("+t[n-2]+" + 1)",i(t)}(r.slice())};\n            }\n          }\n          setOutput(result);\n        }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Mc{constructor(t,e,n,r,o,s,a=!0){this.variableNames=["updates","indices","defaultValue"],this.outputShape=s;const i=su(o.length),u=su(s.length);let c="";1===n?c="i":2===n&&(c="i, j");const l=`getIndices(${c})`;let h="";1===r?h="i":2===r&&(h="i, coords[1]");const d=`getUpdates(${h})`,p=e>1?"strides[j]":"strides";this.userCode=`\n        ${i} strides = ${i}(${o});\n\n        void main() {\n          ${u} coords = getOutputCoords();\n          float sum = 0.0;\n          bool found = false;\n          for (int i = 0; i < ${t}; i++) {\n            int flattenedIndex = 0;\n            for (int j = 0; j < ${e}; j++) {\n              int index = round(${l});\n              flattenedIndex += index * ${p};\n            }\n            if (flattenedIndex == coords[0]) {\n              sum += ${d};\n              found = true;\n            }\n          }\n          setOutput(mix(getDefaultValue(), sum, float(found)));\n        }\n      `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Pc{constructor(t,e){this.variableNames=["x","segmentIds"];const n=t.windowSize,r=t.batchSize,o=t.inSize,s=t.numSegments,a=s*Math.ceil(o/n);this.outputShape=[r,a];const i=4*Math.floor(n/4),u=n%4,c="\n        sumValue += dot(values, segFilter);\n    ";let l="";o%n>0&&(l=`\n        if (inIdx < 0 || inIdx >= ${o}) {\n          return initializationValue;\n        }\n      `);let h="";o%n>0&&(h=`\n        if (inIdx < 0 || inIdx >= ${o}) {\n          return -1.0;\n        }\n      `),this.userCode=`\n      const float initializationValue = 0.0;\n\n      float getValue(int batch, int inIdx) {\n        ${l}\n        return getX(batch, inIdx);\n      }\n\n      float getSegmentIdAtIndex(int inIdx) {\n        ${h}\n        return getSegmentIds(inIdx);\n      }\n\n      void main() {\n        ivec2 coords = getOutputCoords();\n        int batch = coords[0];\n        int outIdx = coords[1];\n        int inOffset = int(floor(float(outIdx) / float(\n          ${s})) * float(${n}));\n        int currentSeg = int(mod(float(outIdx), float(${s})));\n\n        float sumValue = 0.0;\n\n        for (int i = 0; i < ${i}; i += 4) {\n          int inIdx = inOffset + i;\n          vec4 values = vec4(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            getValue(batch, inIdx + 2),\n            getValue(batch, inIdx + 3)\n          );\n\n          vec4 segFilter = vec4(\n            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0\n          );\n\n          ${c}\n        }\n\n        int inIdx = inOffset + ${i};\n        if (${1===u}) {\n          vec4 values = vec4(\n            getValue(batch, inIdx),\n            initializationValue,\n            initializationValue,\n            initializationValue\n          );\n\n          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));\n\n          vec4 segFilter = vec4(\n            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,\n            0,\n            0,\n            0\n          );\n\n          ${c}\n        } else if (${2===u}) {\n          vec4 values = vec4(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            initializationValue,\n            initializationValue\n          );\n\n          vec4 segFilter = vec4(\n            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,\n              0,\n              0\n          );\n\n          ${c}\n        } else if (${3===u}) {\n          vec4 values = vec4(\n            getValue(batch, inIdx),\n            getValue(batch, inIdx + 1),\n            getValue(batch, inIdx + 2),\n            initializationValue\n          );\n\n          vec4 segFilter = vec4(\n            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,\n            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,\n            0\n          );\n\n          ${c}\n        }\n        setOutput(sumValue);\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Lc{constructor(t,e,n){let r,o;if(this.variableNames=["c","a","b"],this.outputShape=e,n>4)throw Error(`Where for rank ${n} is not yet supported`);if(1===n)o="resRC",r="resRC";else{const n=["resRC.x","resRC.y","resRC.z","resRC.w"],s=[],a=[];for(let r=0;r<e.length;r++)a.push(""+n[r]),r<t&&s.push(""+n[r]);r=s.join(),o=a.join()}const s=su(n);this.userCode=`\n      void main() {\n        ${s} resRC = getOutputCoords();\n        float cVal = getC(${r});\n        if (cVal >= 1.0) {\n          setOutput(getA(${o}));\n        } else {\n          setOutput(getB(${o}));\n        }\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Wc{constructor(t){this.variableNames=["source"],this.outputShape=t,this.rank=t.length;const e=su(this.rank),n=`uniform int start[${this.rank}];`,r=function(t){if(1===t)return"sourceLoc";if(t<=6)return zc.slice(0,t).map(t=>"sourceLoc."+t).join(",");throw Error(`Slicing for rank ${t} is not yet supported`)}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(this.rank);let o;o=`\n        ${e} sourceLoc;\n        ${e} coords = getOutputCoords();\n        ${t.map((t,e)=>`sourceLoc.${zc[e]} = start[${e}] + coords.${zc[e]};`).join("\n")}\n      `,this.userCode=`\n      ${n}\n      void main() {\n        ${o}\n        setOutput(getSource(${r}));\n      }\n    `}getCustomSetupFunc(t){if(t.length!==this.rank)throw Error(`The rank (${this.rank}) of the program must match the length of start (${t.length})`);return(e,n)=>{null==this.startLoc&&(this.startLoc=e.getUniformLocationNoThrow(n,"start"),null==this.startLoc)||e.gl.uniform1iv(this.startLoc,t)}}}const zc=["x","y","z","w","u","v"];class Uc{constructor(t){this.variableNames=["source"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.rank=t.length;const e=su(this.rank),n=Ui("coords",this.rank),r=Ui("sourceLoc",this.rank),o=1===this.rank?"sourceLoc":`vec2(${r.slice(-2).join()})`,s=`getChannel(getSource(${r.join()}), ${o})`,a=`\n      result.x = ${s};\n      if (++${n[this.rank-1]} < ${t[this.rank-1]}) {\n        ++${r[this.rank-1]};\n        result.y = ${s};\n        --${r[this.rank-1]};\n      }\n    `,i=1===this.rank?"":`\n      --${n[this.rank-1]};\n      if (++${n[this.rank-2]} < ${t[this.rank-2]}) {\n        ++${r[this.rank-2]};\n        result.z = ${s};\n        if (++${n[this.rank-1]} < ${t[this.rank-1]}) {\n          ++${r[this.rank-1]};\n          result.w = ${s};\n        }\n      }\n    `,u=this.rank<=4?`sourceLoc = coords +\n            ${e}(${t.map((t,e)=>`start[${e}]`).join()});`:t.map((t,e)=>`${r[e]} = ${n[e]} + start[${e}];`).join("\n");this.userCode=`\n      uniform int start[${this.rank}];\n      void main() {\n        ${e} coords = getOutputCoords();\n        ${e} sourceLoc;\n        ${u}\n        vec4 result = vec4(0.);\n        ${a}\n        ${i}\n        setOutput(result);\n      }\n    `}getCustomSetupFunc(t){if(t.length!==this.rank)throw Error(`The rank (${this.rank}) of the program must match the length of start (${t.length})`);return(e,n)=>{null==this.startLoc&&(this.startLoc=e.getUniformLocationNoThrow(n,"start"),null==this.startLoc)||e.gl.uniform1iv(this.startLoc,t)}}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Vc{constructor(t,e,n){this.variableNames=["x"],this.outputShape=n;const r=n.length,o=su(n.length),s=su(n.length);let a="";if(1===r)a="coords * strides + begin";else{let t=0;a=n.map((e,r)=>(t++,1===n.length?`coords * strides[${r}] + begin[${r}]`:`coords[${t-1}] * strides[${r}] + begin[${r}]`)).join(",")}this.userCode=`\n      ${o} begin = ${o}(${t});\n      ${o} strides = ${o}(${e});\n\n      void main() {\n        ${s} coords = getOutputCoords();\n        setOutput(getX(${a}));\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Gc{constructor(t){this.gpgpu=t,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.logEnabled=!1,this.usedTextures={}}acquireTexture(t,e,n){const r=Kc(e,n),o=qc(t,r,n);o in this.freeTextures||(this.freeTextures[o]=[]),o in this.usedTextures||(this.usedTextures[o]=[]);const s=Hc(t,r,this.gpgpu.gl,this.gpgpu.textureConfig,n);if(this.freeTextures[o].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=s,this.log();const t=this.freeTextures[o].shift();return this.usedTextures[o].push(t),t}let a;return r===ci.PACKED_2X2_FLOAT32?a=this.gpgpu.createPackedMatrixTexture(t[0],t[1]):r===ci.PACKED_2X2_FLOAT16?a=this.gpgpu.createFloat16PackedMatrixTexture(t[0],t[1]):r===ci.UNPACKED_FLOAT32?a=this.gpgpu.createFloat32MatrixTexture(t[0],t[1]):r===ci.UNPACKED_FLOAT16?a=this.gpgpu.createFloat16MatrixTexture(t[0],t[1]):r===ci.PACKED_4X1_UNSIGNED_BYTE&&(a=this.gpgpu.createUnsignedBytesMatrixTexture(t[0],t[1])),this.usedTextures[o].push(a),this.numUsedTextures++,this._numBytesAllocated+=s,this.log(),a}releaseTexture(t,e,n,r){if(null==this.freeTextures)return;const o=Kc(n,r),s=qc(e,o,r);s in this.freeTextures||(this.freeTextures[s]=[]);const a=Hc(e,o,this.gpgpu.gl,this.gpgpu.textureConfig,r),i=Object(h.b)().get("WEBGL_DELETE_TEXTURE_THRESHOLD");-1!==i&&this._numBytesAllocated>i?(this.gpgpu.deleteMatrixTexture(t),this._numBytesAllocated-=a):(this.freeTextures[s].push(t),this.numFreeTextures++,this._numBytesFree+=a),this.numUsedTextures--;const u=this.usedTextures[s],c=u.indexOf(t);if(c<0)throw new Error("Cannot release a texture that was never provided by this texture manager");u.splice(c,1),this.log()}log(){if(!this.logEnabled)return;const t=this.numFreeTextures+this.numUsedTextures;console.log("Free/Used",`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${t})`);const e=this._numBytesFree/this._numBytesAllocated;console.log("Bytes allocated: "+this._numBytesAllocated),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*e)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(null!=this.freeTextures){for(const t in this.freeTextures)this.freeTextures[t].forEach(t=>{this.gpgpu.deleteMatrixTexture(t)});for(const t in this.usedTextures)this.usedTextures[t].forEach(t=>{this.gpgpu.deleteMatrixTexture(t)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}}function Hc(t,e,n,r,o){const s=function(t,e){switch(t){case ci.PACKED_2X2_FLOAT32:return lc(e);case ci.PACKED_2X2_FLOAT16:return hc(e);case ci.UNPACKED_FLOAT32:return ic(e);case ci.UNPACKED_FLOAT16:return uc(e);case ci.PACKED_4X1_UNSIGNED_BYTE:return cc(e);default:throw new Error("Unknown physical texture type "+t)}}(e,r);let a;if(o){const[e,n]=di(t[0],t[1]);a=e*n}else{const[e,n]=li(t[0],t[1]);a=e*n}return a*function(t,e){const n=t;if(e===n.R32F)return 4;if(e===n.R16F)return 2;if(e===n.RGBA32F)return 16;if(e===t.RGBA)return 16;if(e===n.RGBA16F)return 8;throw new Error("Unknown internal format "+e)}(n,s)}function Kc(t,e){if(t===ui.UPLOAD)return ci.PACKED_2X2_FLOAT32;if(t===ui.RENDER||null==t)return function(t){return Object(h.b)().getBool("WEBGL_RENDER_FLOAT32_ENABLED")?t?ci.PACKED_2X2_FLOAT32:ci.UNPACKED_FLOAT32:t?ci.PACKED_2X2_FLOAT16:ci.UNPACKED_FLOAT16}(e);if(t===ui.DOWNLOAD||t===ui.PIXELS)return ci.PACKED_4X1_UNSIGNED_BYTE;throw new Error("Unknown logical texture type "+t)}function qc(t,e,n){return`${t[0]}_${t[1]}_${e}_${n}`}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Xc{constructor(t,e){this.variableNames=["A"];const n=new Array(t.length);for(let r=0;r<n.length;r++)n[r]=t[r]*e[r];this.outputShape=n,this.rank=n.length;const r=su(this.rank),o=function(t){const e=t.length;if(e>5)throw Error(`Tile for rank ${e} is not yet supported`);if(1===e)return`imod(resRC, ${t[0]})`;const n=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u"],r=[];for(let e=0;e<t.length;e++)r.push(`imod(${n[e]}, ${t[e]})`);return r.join()}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(t);this.userCode=`\n      void main() {\n        ${r} resRC = getOutputCoords();\n        setOutput(getA(${o}));\n      }\n    `}}class Yc{constructor(t,e){this.variableNames=["A"],this.outputShape=t,this.userCode=`\n      float unaryOperation(float x) {\n        ${e}\n      }\n\n      void main() {\n        float x = getAAtOutCoords();\n        float y = unaryOperation(x);\n\n        setOutput(y);\n      }\n    `}}const Qc="return abs(x);",Zc="if (isnan(x)) return x;\n  return (x < 0.0) ? 0.0 : x;\n",Jc="if (isnan(x)) return x;\n  return (x < 0.0) ? 0.0 : min(6.0, x);\n",tl="return (x >= 0.0) ? x : (exp(x) - 1.0);",el=`\n  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.\n  // see: https://arxiv.org/abs/1706.02515\n  float scaleAlpha = ${s.SELU_SCALEALPHA};\n  float scale = ${s.SELU_SCALE};\n  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);\n`;const nl="return -x;",rl="return ceil(x);",ol="return floor(x);",sl="return exp(x);",al="return exp(x) - 1.0;",il=`\n  // Error function is calculated approximately with elementary function.\n  // See "Handbook of Mathematical Functions with Formulas,\n  // Graphs, and Mathematical Tables", Abramowitz and Stegun.\n  float p = ${s.ERF_P};\n  float a1 = ${s.ERF_A1};\n  float a2 = ${s.ERF_A2};\n  float a3 = ${s.ERF_A3};\n  float a4 = ${s.ERF_A4};\n  float a5 = ${s.ERF_A5};\n\n  float sign = sign(x);\n  x = abs(x);\n  float t = 1.0 / (1.0 + p * x);\n  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));\n`,ul="return x;",cl="\n  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));\n  bvec4 isNaN = isnan(x);\n\n  result.r = isNaN.r ? x.r : result.r;\n  result.g = isNaN.g ? x.g : result.g;\n  result.b = isNaN.b ? x.b : result.b;\n  result.a = isNaN.a ? x.a : result.a;\n\n  return result;\n",ll="\n  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));\n  bvec4 isNaN = isnan(x);\n\n  result.r = isNaN.r ? x.r : result.r;\n  result.g = isNaN.g ? x.g : result.g;\n  result.b = isNaN.b ? x.b : result.b;\n  result.a = isNaN.a ? x.a : result.a;\n\n  return result;\n",hl="\n  vec4 result;\n\n  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);\n  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);\n  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);\n  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);\n\n  return result;\n";class dl{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.userCode=`\n      vec4 unaryOperation(vec4 x) {\n        ${e}\n      }\n\n      void main() {\n        vec4 x = getAAtOutCoords();\n        vec4 y = unaryOperation(x);\n\n        setOutput(y);\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class pl{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=t;const e=t.length,n=Ui("rc",e),r=su(e),o=function(t,e){if(1===t)return"rc";let n="";for(let r=0;r<t;r++)n+=e[r],r<t-1&&(n+=",");return n}(e,n),s=n.slice(-2),a=e<=1?"rc":`vec2(${s.join(",")})`;this.userCode=`\n      void main() {\n        ${r} rc = getOutputCoords();\n        vec4 packedInput = getA(${o});\n\n        setOutput(getChannel(packedInput, ${a}));\n      }\n    `}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const{segment_util:fl}=s,gl=a.split,ml=a.tile,bl=a.topkImpl,xl=a.whereImpl,yl={};function vl(t,e=!1){if("linear"===t)return"return x;";if("relu"===t)return e?cl:Zc;if("elu"===t)return e?hl:tl;if("relu6"===t)return e?ll:Jc;if("prelu"===t)return e?wu:yu;throw new Error(`Activation ${t} has not been implemented for the WebGL backend.`)}class wl extends Or{constructor(t){if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.warnedAboutMemory=!1,this.warnedAboutCPUBackend=!1,this.pendingDeletes=0,this.disposed=!1,!Object(h.b)().getBool("HAS_WEBGL"))throw new Error("WebGL is not supported on this device");if(null==t){const t=ai(Object(h.b)().getNumber("WEBGL_VERSION"));this.binaryCache=((e=Object(h.b)().getNumber("WEBGL_VERSION"))in yl||(yl[e]={}),yl[e]),this.gpgpu=new pc(t),this.canvas=t.canvas,this.gpgpuCreatedLocally=!0}else this.gpgpu=t,this.binaryCache={},this.gpgpuCreatedLocally=!1,this.canvas=t.gl.canvas;var e;this.textureManager=new Gc(this.gpgpu),this.numMBBeforeWarning=null==Object(h.b)().global.screen?1024:Object(h.b)().global.screen.height*Object(h.b)().global.screen.width*window.devicePixelRatio*600/1024/1024,this.texData=new $r(this,on())}numDataIds(){return this.texData.numDataIds()+(this.cpuBackend?this.cpuBackend.numDataIds():0)-this.pendingDeletes}write(t,e,n){if((Object(h.b)().getBool("WEBGL_CHECK_NUMERICAL_PROBLEMS")||Object(h.b)().getBool("DEBUG"))&&this.checkNumericalProblems(t),"complex64"===n&&null!=t)throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");const r={};return this.texData.set(r,{shape:e,dtype:n,values:t,usage:ui.UPLOAD}),r}move(t,e,n,r){if(Object(h.b)().getBool("DEBUG")&&this.checkNumericalProblems(e),"complex64"===r)throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");this.texData.set(t,{shape:n,dtype:r,values:e,usage:ui.UPLOAD})}readSync(t){const e=this.texData.get(t),{values:n,dtype:r,complexTensors:o,slice:a,shape:i,isPacked:u}=e;if(null!=a){let e;e=u?new dl(i,ul):new Yc(i,ul);const n=this.runWebGLProgram(e,[{dataId:t,shape:i,dtype:r}],r),o=this.readSync(n.dataId);return this.disposeData(n.dataId),o}if(null!=n)return this.convertAndCacheOnCPU(t);if("string"===r)return n;const c=null!=this.activeTimers;let l,h;if(c&&(l=y.now()),"complex64"===r){const t=o.real.dataSync(),e=o.imag.dataSync();h=s.mergeRealAndImagArrays(t,e)}else h=this.getValuesFromTexture(t);return c&&(this.downloadWaitMs+=y.now()-l),this.convertAndCacheOnCPU(t,h)}async read(t){if(this.pendingRead.has(t)){const e=this.pendingRead.get(t);return new Promise(t=>e.push(t))}const e=this.texData.get(t),{values:n,shape:r,slice:o,dtype:a,complexTensors:i,isPacked:u}=e;if(null!=o){let e;e=u?new dl(r,ul):new Yc(r,ul);const n=this.runWebGLProgram(e,[{dataId:t,shape:r,dtype:a}],a),o=this.read(n.dataId);return this.disposeData(n.dataId),o}if(null!=n)return this.convertAndCacheOnCPU(t);if(!Object(h.b)().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")&&2===Object(h.b)().getNumber("WEBGL_VERSION"))throw new Error("tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.");let c,l,d=null;if("complex64"!==a&&Object(h.b)().get("WEBGL_BUFFER_SUPPORTED")){c=this.decode(t);const e=this.texData.get(c.dataId);d=this.gpgpu.createBufferFromTexture(e.texture,...hi(r))}if(this.pendingRead.set(t,[]),"complex64"!==a&&await this.gpgpu.createAndWaitForFence(),"complex64"===a){const t=await Promise.all([i.real.data(),i.imag.data()]),e=t[0],n=t[1];l=s.mergeRealAndImagArrays(e,n)}else if(null==d)l=this.getValuesFromTexture(t);else{const t=y.sizeFromShape(r);l=this.gpgpu.downloadFloat32MatrixFromBuffer(d,t)}null!=c&&this.disposeData(c.dataId);const p=this.convertAndCacheOnCPU(t,l),f=this.pendingRead.get(t);return this.pendingRead.delete(t),f.forEach(t=>t(p)),this.pendingDisposal.has(t)&&(this.pendingDisposal.delete(t),this.disposeData(t),this.pendingDeletes--),p}checkNumericalProblems(t){if(null!=t)for(let e=0;e<t.length;e++){const n=t[e];if(!gi(n)){if(Object(h.b)().getBool("WEBGL_RENDER_FLOAT32_CAPABLE"))throw Error(`The value ${n} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`);throw Error(`The value ${n} cannot be represented on this device.`)}}}getValuesFromTexture(t){const{shape:e,dtype:n,isPacked:r}=this.texData.get(t),o=y.sizeFromShape(e);if(Object(h.b)().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")){const n=this.decode(t),r=this.texData.get(n.dataId),s=this.gpgpu.downloadMatrixFromPackedTexture(r.texture,...hi(e)).subarray(0,o);return this.disposeData(n.dataId),s}const s=Object(h.b)().getBool("WEBGL_PACK")&&!0===r,a=s?Ai(e):e,i=s?new qu(a):new Ku(a),u=this.runWebGLProgram(i,[{shape:a,dtype:n,dataId:t}],"float32"),c=this.texData.get(u.dataId),l=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(c.texture,c.texShape[0],c.texShape[1]).subarray(0,o);return this.disposeData(u.dataId),l}async time(t){const e=this.activeTimers,n=[];let r=!1;null==this.programTimersStack?(this.programTimersStack=n,r=!0):this.activeTimers.push(n),this.activeTimers=n,t();const o=y.flatten(this.activeTimers.map(t=>t.query)).filter(t=>null!=t),s=y.flatten(this.activeTimers.map(t=>t.name)).filter(t=>null!=t);this.activeTimers=e,r&&(this.programTimersStack=null);const a={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};if(Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0){const t=await Promise.all(o);a.kernelMs=y.sum(t),a.getExtraProfileInfo=()=>t.map((t,e)=>({name:s[e],ms:t})).map(t=>`${t.name}: ${t.ms}`).join(", ")}else a.kernelMs={error:"WebGL query timers are not supported in this environment."};return this.uploadWaitMs=0,this.downloadWaitMs=0,a}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?this.gpgpu.beginQuery():{startMs:y.now(),endMs:null}}endTimer(t){return Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?(this.gpgpu.endQuery(),t):(t.endMs=y.now(),t)}async getQueryTime(t){if(Object(h.b)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0)return this.gpgpu.waitForQueryAndGetTime(t);const e=t;return e.endMs-e.startMs}disposeData(t){if(this.pendingDisposal.has(t))return;if(this.pendingRead.has(t))return this.pendingDisposal.add(t),void this.pendingDeletes++;if(!this.texData.has(t))return;this.releaseGPUData(t);const{complexTensors:e}=this.texData.get(t);null!=e&&(e.real.dispose(),e.imag.dispose()),this.texData.delete(t)}releaseGPUData(t){const{texture:e,dtype:n,texShape:r,usage:o,isPacked:s,slice:a}=this.texData.get(t),i=a&&a.origDataId||t,u=this.dataRefCount.get(i);u>1?this.dataRefCount.set(i,u-1):(this.dataRefCount.delete(i),null!=e&&(this.numBytesInGPU-=this.computeBytes(r,n),this.textureManager.releaseTexture(e,r,o,s)));const c=this.texData.get(t);c.texture=null,c.texShape=null,c.isPacked=!1,c.slice=null}getTexture(t){return this.uploadToGPU(t),this.texData.get(t).texture}getDataInfo(t){return this.texData.get(t)}getCPUBackend(){return Object(h.b)().getBool("WEBGL_CPU_FORWARD")?(null==this.cpuBackend&&(this.cpuBackend=on().findBackend("cpu")),this.cpuBackend):null}shouldExecuteOnCPU(t,e=128){const n=this.getCPUBackend();return this.warnedAboutCPUBackend||null!=n||Object(h.b)().getBool("IS_TEST")||(console.warn("Your application contains ops that are small enough to be executed on the CPU backend, however the CPU backend cannot be found. Consider importing the CPU backend (@tensorflow/tfjs-backend-cpu) for better performance."),this.warnedAboutCPUBackend=!0),null!=n&&t.every(t=>null==this.texData.get(t.dataId).texture&&y.sizeFromShape(t.shape)<e)}getGPGPUContext(){return this.gpgpu}complex(t,e){const n=this.makeOutput(t.shape,"complex64");return this.texData.get(n.dataId).complexTensors={real:on().keep(t.clone()),imag:on().keep(e.clone())},n}real(t){return this.texData.get(t.dataId).complexTensors.real.clone()}imag(t){return this.texData.get(t.dataId).complexTensors.imag.clone()}slice(t,e,n){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.slice(t,e,n);if(0===y.sizeFromShape(n))return Object(he.a)([],n,t.dtype);const{isPacked:o}=this.texData.get(t.dataId),s=r.isSliceContinous(t.shape,e,n);if(o||!s){const r=Object(h.b)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new Uc(n):new Wc(n),o=r.getCustomSetupFunc(e);return this.compileAndRun(r,[t],null,o)}return this.uploadToGPU(t.dataId),this.shallowSlice(t,e,n)}shallowSlice(t,e,n){const o=this.texData.get(t.dataId),s=this.makeOutput(n,t.dtype),a=this.texData.get(s.dataId);Object.assign(a,o),a.shape=n,a.dtype=t.dtype;let i=r.computeFlatOffset(e,t.strides);o.slice&&(i+=o.slice.flatOffset),a.slice={flatOffset:i,origDataId:o.slice&&o.slice.origDataId||t.dataId};const u=this.dataRefCount.get(a.slice.origDataId)||1;return this.dataRefCount.set(a.slice.origDataId,u+1),s}stridedSlice(t,e,n,o){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.stridedSlice(t,e,n,o);const s=r.computeOutShape(e,n,o);if(s.some(t=>0===t))return Object(he.a)([],s);const a=new Vc(e,o,s);return this.compileAndRun(a,[t])}reverse(t,e){const n=Object(h.b)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new jc(t.shape,e):new Bc(t.shape,e);return this.compileAndRun(n,[t])}concat(t,e){if("complex64"===t[0].dtype){const n=t.map(t=>se(t)),r=t.map(t=>Mt(t));return Object(pt.a)(this.concat(n,e),this.concat(r,e))}if(this.shouldExecuteOnCPU(t))return this.cpuBackend.concat(t,e);if(1===t.length)return t[0];if(t.length>Object(h.b)().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER")){const n=Math.floor(t.length/2),r=this.concat(t.slice(0,n),e),o=this.concat(t.slice(n),e);return this.concat([r,o],e)}if(Object(h.b)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")&&t[0].rank>1){const n=new Eu(t.map(t=>t.shape),e);return this.compileAndRun(n,t)}const n=s.computeOutShape(t.map(t=>t.shape),e),r=t.map(t=>t.as2D(-1,y.sizeFromShape(t.shape.slice(e)))),o=new Su(r.map(t=>t.shape));return this.compileAndRun(o,r).reshape(n)}neg(t){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.neg(t);if(Object(h.b)().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,nl,t.dtype);const e=new Yc(t.shape,nl);return this.compileAndRun(e,[t])}batchMatMul(t,e,n,r){const o=n?t.shape[2]:t.shape[1],s=r?e.shape[1]:e.shape[2],a=n?t.shape[1]:t.shape[2],[i,,]=t.shape;if((1===o||1===s)&&a>1e3){n&&(t=Kt(t,[0,2,1])),r&&(e=Kt(e,[0,2,1]));const o=1===s?t:t.as3D(i,a,1),u=1===s?2:1,c=1===s?e.as3D(i,1,a):e;return this.multiply(o,c).sum(u,!0)}const u=Object(lt.c)(t.dtype,e.dtype),c=new wc(t.shape,[i,o,s],n,r);return this.compileAndRun(c,[t,e],u)}fusedBatchMatMul({a:t,b:e,transposeA:n,transposeB:r,bias:o,activation:s,preluActivationWeights:a}){const i=n?t.shape[2]:t.shape[1],u=r?e.shape[1]:e.shape[2],[c,,]=t.shape,l=Object(lt.c)(t.dtype,e.dtype),h=null!=o,d=null!=a,p=s?vl(s,!0):null,f=new wc(t.shape,[c,i,u],n,r,h,p,d),g=[t,e];return o&&g.push(o),a&&g.push(a),this.compileAndRun(f,g,l)}multiply(t,e){if("complex64"===t.dtype){const n=this.texData.get(t.dataId),r=this.texData.get(e.dataId),o=new gu(pu,t.shape,e.shape),s=new gu(fu,t.shape,e.shape),a=[this.makeComplexComponentTensorInfo(t,n.complexTensors.real),this.makeComplexComponentTensorInfo(t,n.complexTensors.imag),this.makeComplexComponentTensorInfo(e,r.complexTensors.real),this.makeComplexComponentTensorInfo(e,r.complexTensors.imag)],i=this.compileAndRun(o,a),u=this.compileAndRun(s,a),c=this.complex(i,u);return i.dispose(),u.dispose(),c}if(this.shouldExecuteOnCPU([t,e]))return this.cpuBackend.multiply(t,e);if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,xu,t.dtype);const n=new vu(xu,t.shape,e.shape);return this.compileAndRun(n,[t,e],t.dtype)}batchNorm(t,e,n,r,o,s){const a=[t,e,n];let i=null;null!=r&&(i=r.shape,a.push(r));let u=null;if(null!=o&&(u=o.shape,a.push(o)),Object(h.b)().getBool("WEBGL_PACK_NORMALIZATION")){const r=new du(t.shape,e.shape,n.shape,i,u,s);return this.compileAndRun(r,a)}const c=new hu(t.shape,e.shape,n.shape,i,u,s);return this.compileAndRun(c,a)}localResponseNormalization4D(t,e,n,r,o){const s=Object(h.b)().getBool("WEBGL_PACK_NORMALIZATION")?new xc(t.shape,e,n,r,o):new mc(t.shape,e,n,r,o);return this.compileAndRun(s,[t])}LRNGrad(t,e,n,r,o,s,a){const i=new bc(e.shape,r,o,s,a);return this.compileAndRun(i,[e,n,t])}tile(t,e){if("string"===t.dtype){const n=this.readSync(t.dataId).map(t=>y.decodeString(t)),r=D(t.shape,t.dtype,n);return ml(r,e)}const n=new Xc(t.shape,e);return this.compileAndRun(n,[t])}pad(t,e,n){const r=Object(h.b)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new Sc(t.shape,e,n):new Ic(t.shape,e,n);return this.compileAndRun(r,[t])}gather(t,e,n){if(this.shouldExecuteOnCPU([t,e]))return this.cpuBackend.gather(t,e,n);const r=new ec(t.shape,e.size,n);return this.compileAndRun(r,[t,e])}batchToSpaceND(t,e,n){y.assert(t.rank<=4,()=>"batchToSpaceND for rank > 4 with a WebGL backend not implemented yet");const r=e.reduce((t,e)=>t*e),o=s.getReshaped(t.shape,e,r),a=s.getPermuted(o.length,e.length),i=s.getReshapedPermuted(t.shape,e,r),u=s.getSliceBeginCoords(n,e.length),c=s.getSliceSize(i,n,e.length);return Kt(t.reshape(o),a).reshape(i).slice(u,c)}spaceToBatchND(t,e,n){y.assert(t.rank<=4,()=>"spaceToBatchND for rank > 4 with a WebGL backend not implemented yet");const r=e.reduce((t,e)=>t*e),o=[[0,0]];o.push(...n);for(let n=1+e.length;n<t.shape.length;++n)o.push([0,0]);const a=t.pad(o),i=s.getReshaped(a.shape,e,r,!1),u=s.getPermuted(i.length,e.length,!1),c=s.getReshapedPermuted(a.shape,e,r,!1),l=Kt(a.reshape(i),u);return Et(l,c)}reduce(t,e,n){const r=t.shape[0],o=t.shape[1],a=s.computeOptimalWindowSize(o),i=Math.ceil(o/a),u=new Ac({windowSize:a,inSize:o,batchSize:r,outSize:i},e),c=this.compileAndRun(u,[t],n);return 1===c.shape[1]?c:this.reduce(c,e,n)}argReduce(t,e,n=null){let r=t.shape[0],o=t.shape[1];null!=n&&(r=n.shape[0],o=n.shape[1]);const a=s.computeOptimalWindowSize(o),i={windowSize:a,inSize:o,batchSize:r,outSize:Math.ceil(o/a)},u=new Wi(i,e,null==n),c=[t];null!=n&&c.push(n);const l=this.compileAndRun(u,c,"int32");return 1===l.shape[1]?l:this.argReduce(t,e,l)}argReducePacked(t,e,n=null){const r=null!=n?n.shape:t.shape,o=r[r.length-1],a=s.computeOptimalWindowSize(o),i=new uu(r,a,e,null==n),u=null==n?[t]:[t,n],c=this.compileAndRun(i,u,"int32");return c.rank===t.rank?this.argReducePacked(t,e,c):c}sum(t,e){s.assertAxesAreInnerMostDims("sum",e,t.rank);const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=y.sizeFromShape(r),a=t.as2D(-1,o),i=lt.b(t.dtype);return this.reduce(a,"sum",i).reshape(n)}prod(t,e){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.prod(t,e);const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=y.sizeFromShape(r),a=t.as2D(-1,o),i=lt.b(t.dtype);return this.reduce(a,"prod",i).reshape(n)}unsortedSegmentSum(t,e,n){let r=0;const o=s.getAxesPermutation([r],t.rank);let a=t;null!=o&&(a=Kt(t,o),r=s.getInnerMostAxes(1,t.rank)[0]);const i=fl.computeOutShape(a.shape,r,n),u=y.sizeFromShape([a.shape[r]]),c=a.as2D(-1,u),l=lt.b(t.dtype);let h=this.segOpCompute(c,"unsortedSegmentSum",e,l,n).reshape(i);return null!=o&&(h=Kt(h,s.getUndoAxesPermutation(o))),h}segOpCompute(t,e,n,r,o){const s=t.shape[0],a=t.shape[1],i=fl.segOpComputeOptimalWindowSize(a,o),u=new Pc({windowSize:i,inSize:a,batchSize:s,numSegments:o},e),c=this.compileAndRun(u,[t,n],r);return c.shape[1]===o?c:(n=oe(0,o).tile([a/i]),this.segOpCompute(c,e,n,r,o))}argMinMaxReduce(t,e,n){const r=[e];if(s.assertAxesAreInnerMostDims("arg"+n.charAt(0).toUpperCase()+n.slice(1),r,t.rank),!Object(h.b)().getBool("WEBGL_PACK_REDUCE")||t.rank<=2){const[e,o]=s.computeOutAndReduceShapes(t.shape,r),a=y.sizeFromShape(o),i=t.as2D(-1,a);return this.argReduce(i,n).reshape(e)}return this.argReducePacked(t,n)}argMin(t,e){return this.argMinMaxReduce(t,e,"min")}argMax(t,e){return this.argMinMaxReduce(t,e,"max")}cumsum(t,e,n,r){if(e!==t.rank-1)throw new Error(`WebGL cumsum shader expects an inner-most axis=${t.rank-1} but got axis=`+e);const o=t.shape[e];let s=t;for(let e=0;e<=Math.ceil(Math.log2(o))-1;e++){const n=new Lu(t.shape,!1,r),o=n.getCustomSetupFunc(e),a=s;s=this.compileAndRun(n,[s],s.dtype,o),a.dispose()}if(n){const e=new Lu(t.shape,n,r),o=s;s=this.compileAndRun(e,[s]),o.dispose()}return s}equal(t,e){if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,"\n  return vec4(equal(a, b));\n","bool");const n=new vu("return float(a == b);",t.shape,e.shape);return this.compileAndRun(n,[t,e],"bool")}notEqual(t,e){if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,"\n  return vec4(notEqual(a, b));\n","bool");const n=new vu("return float(a != b);",t.shape,e.shape);return this.compileAndRun(n,[t,e],"bool")}less(t,e){if(this.shouldExecuteOnCPU([t,e]))return this.cpuBackend.less(t,e);if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,"\n  return vec4(lessThan(a, b));\n","bool");const n=new vu("return float(a < b);",t.shape,e.shape);return this.compileAndRun(n,[t,e],"bool")}lessEqual(t,e){if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,"\n  return vec4(lessThanEqual(a, b));\n","bool");const n=new vu("return float(a <= b);",t.shape,e.shape);return this.compileAndRun(n,[t,e],"bool")}greater(t,e){if(this.shouldExecuteOnCPU([t,e]))return this.cpuBackend.greater(t,e);if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,"\n  return vec4(greaterThan(a, b));\n","bool");const n=new vu("return float(a > b);",t.shape,e.shape);return this.compileAndRun(n,[t,e],"bool")}greaterEqual(t,e){if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,"\n  return vec4(greaterThanEqual(a, b));\n","bool");const n=new vu("return float(a >= b);",t.shape,e.shape);return this.compileAndRun(n,[t,e],"bool")}logicalNot(t){const e=new Yc(t.shape,"return float(!(x >= 1.0));");return this.compileAndRun(e,[t])}logicalAnd(t,e){if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,"\n  return vec4(\n    vec4(greaterThanEqual(a, vec4(1.0))) *\n    vec4(greaterThanEqual(b, vec4(1.0))));\n","bool");const n=new vu("return float(a >= 1.0 && b >= 1.0);",t.shape,e.shape);return this.compileAndRun(n,[t,e],"bool")}logicalOr(t,e){if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,"\n  return min(\n    vec4(greaterThanEqual(a, vec4(1.0))) +\n    vec4(greaterThanEqual(b, vec4(1.0))),\n    vec4(1.0));\n","bool");const n=new vu("return float(a >= 1.0 || b >= 1.0);",t.shape,e.shape);return this.compileAndRun(n,[t,e],"bool")}select(t,e,n){const r=new Lc(t.rank,e.shape,e.rank);return this.compileAndRun(r,[t,e,n],Object(lt.c)(e.dtype,n.dtype))}where(t){s.warn("tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead");const e=t.dataSync();return xl(t.shape,e)}topk(t,e,n){const r=t.dataSync();return bl(r,t.shape,t.dtype,e,n)}min(t,e){s.assertAxesAreInnerMostDims("min",e,t.rank);const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=y.sizeFromShape(r),a=t.as2D(-1,o);return this.reduce(a,"min",a.dtype).reshape(n)}minimum(t,e){if(this.shouldExecuteOnCPU([t,e]))return this.cpuBackend.minimum(t,e);const n=Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Cu("\n  vec4 result = vec4(min(a, b));\n  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));\n  \n  result.r = isNaN.r > 0. ? NAN : result.r;\n  result.g = isNaN.g > 0. ? NAN : result.g;\n  result.b = isNaN.b > 0. ? NAN : result.b;\n  result.a = isNaN.a > 0. ? NAN : result.a;\n\n  return result;\n",t.shape,e.shape):new vu("\n  if (isnan(a)) return a;\n  if (isnan(b)) return b;\n\n  return min(a, b);\n",t.shape,e.shape);return this.compileAndRun(n,[t,e])}mod(t,e){const n=Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Cu("\n  vec4 result = mod(a, b);\n  vec4 isNaN = vec4(equal(b, vec4(0.0)));\n  \n  result.r = isNaN.r > 0. ? NAN : result.r;\n  result.g = isNaN.g > 0. ? NAN : result.g;\n  result.b = isNaN.b > 0. ? NAN : result.b;\n  result.a = isNaN.a > 0. ? NAN : result.a;\n\n  return result;\n",t.shape,e.shape):new vu("if (b == 0.0) return NAN;\n  return mod(a, b);",t.shape,e.shape);return this.compileAndRun(n,[t,e])}maximum(t,e){if(this.shouldExecuteOnCPU([t,e]))return this.cpuBackend.maximum(t,e);const n=Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Cu("\n  vec4 result = vec4(max(a, b));\n  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));\n  \n  result.r = isNaN.r > 0. ? NAN : result.r;\n  result.g = isNaN.g > 0. ? NAN : result.g;\n  result.b = isNaN.b > 0. ? NAN : result.b;\n  result.a = isNaN.a > 0. ? NAN : result.a;\n\n  return result;\n",t.shape,e.shape):new vu("\n  if (isnan(a)) return a;\n  if (isnan(b)) return b;\n\n  return max(a, b);\n",t.shape,e.shape);return this.compileAndRun(n,[t,e])}all(t,e){s.assertAxesAreInnerMostDims("all",e,t.rank);const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=y.sizeFromShape(r),a=t.as2D(-1,o);return this.reduce(a,"all",a.dtype).reshape(n)}any(t,e){s.assertAxesAreInnerMostDims("any",e,t.rank);const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=y.sizeFromShape(r),a=t.as2D(-1,o);return this.reduce(a,"any",a.dtype).reshape(n)}floorDiv(t,e){if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,"\n  ivec4 ia = round(a);\n  ivec4 ib = round(b);\n  bvec4 cond = notEqual(ib, ivec4(0));\n  ivec4 result = ivec4(0);\n  vec4 s = sign(a) * sign(b);\n\n  // Windows (D3D) wants guaranteed non-zero int division at compile-time.\n  if (cond[0]) {\n    result[0] = idiv(ia[0], ib[0], s[0]);\n  }\n  if (cond[1]) {\n    result[1] = idiv(ia[1], ib[1], s[1]);\n  }\n  if (cond[2]) {\n    result[2] = idiv(ia[2], ib[2], s[2]);\n  }\n  if (cond[3]) {\n    result[3] = idiv(ia[3], ib[3], s[3]);\n  }\n  return vec4(result);\n","int32");const n=new vu("\n  float s = sign(a) * sign(b);\n  int ia = round(a);\n  int ib = round(b);\n  if (ib != 0) {\n    // Windows (D3D) wants guaranteed non-zero int division at compile-time.\n    return float(idiv(ia, ib, s));\n  } else {\n    return NAN;\n  }\n",t.shape,e.shape);return this.compileAndRun(n,[t,e],"int32")}add(t,e){if("complex64"===t.dtype&&"complex64"===e.dtype)return this.complexSeparableBinaryOp(t,e,mu);if(this.shouldExecuteOnCPU([t,e]))return this.cpuBackend.add(t,e);const n=Object(lt.c)(t.dtype,e.dtype);if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,mu,n);const r=new vu(mu,t.shape,e.shape);return this.compileAndRun(r,[t,e],n)}packedUnaryOp(t,e,n){const r=new dl(t.shape,e);return this.compileAndRun(r,[t],n)}packedBinaryOp(t,e,n,r,o=!1){const s=new Cu(n,t.shape,e.shape,o);return this.compileAndRun(s,[t,e],r)}complexSeparableBinaryOp(t,e,n){const r=this.texData.get(t.dataId),o=this.texData.get(e.dataId),[s,a]=[[r.complexTensors.real,o.complexTensors.real],[r.complexTensors.imag,o.complexTensors.imag]].map(r=>{const[o,s]=r,a=this.makeComplexComponentTensorInfo(t,o),i=this.makeComplexComponentTensorInfo(e,s),u=new vu(n,t.shape,e.shape);return this.compileAndRun(u,[a,i],Object(lt.c)(o.dtype,s.dtype))}),i=this.complex(s,a);return s.dispose(),a.dispose(),i}makeComplexComponentTensorInfo(t,e){return{dataId:e.dataId,dtype:e.dtype,shape:t.shape}}addN(t){if(1===t.length)return t[0];if(t.length>Object(h.b)().get("WEBGL_MAX_TEXTURES_IN_SHADER")){const e=Math.floor(t.length/2),n=this.addN(t.slice(0,e)),r=this.addN(t.slice(e));return this.addN([n,r])}const e=t.map(t=>t.dtype).reduce((t,e)=>Object(lt.c)(t,e)),n=t.map(t=>t.shape),r=Object(h.b)().getBool("WEBGL_PACK")?new Li(t[0].shape,n):new Pi(t[0].shape,n);return this.compileAndRun(r,t,e)}subtract(t,e){if("complex64"===t.dtype&&"complex64"===e.dtype)return this.complexSeparableBinaryOp(t,e,bu);if(this.shouldExecuteOnCPU([t,e]))return this.cpuBackend.subtract(t,e);const n=Object(lt.c)(t.dtype,e.dtype);if(Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS"))return this.packedBinaryOp(t,e,bu,t.dtype);const r=new vu(bu,t.shape,e.shape);return this.compileAndRun(r,[t,e],n)}pow(t,e){const n=Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Cu("\n  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.\n  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));\n  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);\n  vec4 result = multiplier * pow(abs(a), b);\n\n  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS\n  bvec4 isExpZero = equal(b, vec4(0.0));\n  result.r = isExpZero.r ? 1.0 : result.r;\n  result.g = isExpZero.g ? 1.0 : result.g;\n  result.b = isExpZero.b ? 1.0 : result.b;\n  result.a = isExpZero.a ? 1.0 : result.a;\n\n  vec4 isNaN = vec4(lessThan(a, vec4(0.0))) * vec4(lessThan(floor(b), b));\n  \n  result.r = isNaN.r > 0. ? NAN : result.r;\n  result.g = isNaN.g > 0. ? NAN : result.g;\n  result.b = isNaN.b > 0. ? NAN : result.b;\n  result.a = isNaN.a > 0. ? NAN : result.a;\n\n  return result;\n",t.shape,e.shape):new vu("\nif(a < 0.0 && floor(b) < b){\n  return NAN;\n}\nif (b == 0.0) {\n  return 1.0;\n}\nreturn (round(mod(b, 2.0)) != 1) ?\n    pow(abs(a), b) : sign(a) * pow(abs(a), b);\n",t.shape,e.shape),r=Object(lt.c)(t.dtype,e.dtype);return this.compileAndRun(n,[t,e],r)}ceil(t){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.ceil(t);if(Object(h.b)().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,rl,t.dtype);const e=new Yc(t.shape,rl);return this.compileAndRun(e,[t])}floor(t){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.floor(t);if(Object(h.b)().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,ol,t.dtype);const e=new Yc(t.shape,ol);return this.compileAndRun(e,[t])}sign(t){const e=new Yc(t.shape,"\n  if (isnan(x)) { return 0.0; }\n  return sign(x);\n");return this.compileAndRun(e,[t])}isNaN(t){const e=new Yc(t.shape,"return float(isnan(x));");return this.compileAndRun(e,[t],"bool")}isInf(t){const e=new Yc(t.shape,"return float(isinf(x));");return this.compileAndRun(e,[t],"bool")}isFinite(t){const e=new Yc(t.shape,"return float(!isnan(x) && !isinf(x));");return this.compileAndRun(e,[t],"bool")}round(t){const e=new Yc(t.shape,"\n  // OpenGL ES does not support round function.\n  // The algorithm is based on banker's rounding.\n  float base = floor(x);\n  if ((x - base) < 0.5) {\n    return floor(x);\n  } else if ((x - base) > 0.5) {\n    return ceil(x);\n  } else {\n    if (mod(base, 2.0) == 0.0) {\n      return base;\n    } else {\n      return base + 1.0;\n    }\n  }\n");return this.compileAndRun(e,[t])}exp(t){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.exp(t);if(Object(h.b)().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,sl,t.dtype);const e=new Yc(t.shape,sl);return this.compileAndRun(e,[t])}expm1(t){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.expm1(t);if(Object(h.b)().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,al,t.dtype);const e=new Yc(t.shape,al);return this.compileAndRun(e,[t])}softmax(t,e){const n=y.parseAxisParam([e],t.shape),r=qt(t,n),o=s.expandShapeToKeepDim(r.shape,n),a=this.subtract(t,r.reshape(o)),i=this.exp(a),u=this.sum(i,n).reshape(o);return Tt(i,u)}log(t){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.log(t);if(Object(h.b)().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,"\n  vec4 result = log(x);\n  vec4 isNaN = vec4(lessThan(x, vec4(0.0)));\n  result.r = isNaN.r == 1.0 ? NAN : result.r;\n  result.g = isNaN.g == 1.0 ? NAN : result.g;\n  result.b = isNaN.b == 1.0 ? NAN : result.b;\n  result.a = isNaN.a == 1.0 ? NAN : result.a;\n\n  return result;\n",t.dtype);const e=new Yc(t.shape,"if (x < 0.0) return NAN;\n  return log(x);");return this.compileAndRun(e,[t])}log1p(t){const e=new Yc(t.shape,"return log(1.0 + x);");return this.compileAndRun(e,[t])}sqrt(t){const e=new Yc(t.shape,"return sqrt(x);");return this.compileAndRun(e,[t])}rsqrt(t){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.rsqrt(t);const e=new Yc(t.shape,"return inversesqrt(x);");return this.compileAndRun(e,[t])}reciprocal(t){const e=new Yc(t.shape,"return 1.0 / x;");return this.compileAndRun(e,[t])}relu(t){let e;return e=Object(h.b)().getBool("WEBGL_PACK")?new dl(t.shape,cl):new Yc(t.shape,Zc),this.compileAndRun(e,[t])}relu6(t){let e;return e=Object(h.b)().getBool("WEBGL_PACK")?new dl(t.shape,ll):new Yc(t.shape,Jc),this.compileAndRun(e,[t])}prelu(t,e){const n=Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Cu(wu,t.shape,e.shape):new vu(yu,t.shape,e.shape);return this.compileAndRun(n,[t,e])}elu(t){if(Object(h.b)().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,hl,t.dtype);const e=new Yc(t.shape,tl);return this.compileAndRun(e,[t])}eluDer(t,e){const n=Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Cu("\n  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));\n  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));\n",t.shape,e.shape):new vu("return (b >= 1.0) ? a : a * (b + 1.0);",t.shape,e.shape);return this.compileAndRun(n,[t,e])}selu(t){const e=new Yc(t.shape,el);return this.compileAndRun(e,[t])}int(t){const e=new Yc(t.shape,"return float(int(x));");return this.compileAndRun(e,[t],"int32")}clip(t,e,n){let r;r=Object(h.b)().getBool("WEBGL_PACK_CLIP")?new Ou(t.shape):new $u(t.shape);const o=r.getCustomSetupFunc(e,n);return this.compileAndRun(r,[t],null,o)}abs(t){if(this.shouldExecuteOnCPU([t]))return this.cpuBackend.abs(t);if(Object(h.b)().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,Qc,t.dtype);const e=new Yc(t.shape,Qc);return this.compileAndRun(e,[t])}complexAbs(t){const e=this.texData.get(t.dataId),n=new Iu(t.shape),r=[this.makeComplexComponentTensorInfo(t,e.complexTensors.real),this.makeComplexComponentTensorInfo(t,e.complexTensors.imag)];return this.compileAndRun(n,r)}sigmoid(t){const e=new Yc(t.shape,"return 1.0 / (1.0 + exp(-1.0 * x));");return this.compileAndRun(e,[t])}softplus(t){const e=new Yc(t.shape,"\n  float epsilon = 1.1920928955078125e-7;\n  float threshold = log(epsilon) + 2.0;\n\n  bool too_large = x > -threshold;\n  bool too_small = x < threshold;\n\n  float result;\n  float exp_x = exp(x);\n\n  if (too_large){\n    result = x;\n  }\n  else if (too_small){\n    result = exp_x;\n  }\n  else{\n    result = log(exp_x + 1.0);\n  }\n  return result;\n");return this.compileAndRun(e,[t])}sin(t){const e=new Yc(t.shape,"if (isnan(x)) return x;\n  return sin(x);\n");return this.compileAndRun(e,[t])}cos(t){const e=new Yc(t.shape,"if (isnan(x)) return x;\n  return cos(x);\n");return this.compileAndRun(e,[t])}tan(t){const e=new Yc(t.shape,"return tan(x);");return this.compileAndRun(e,[t])}asin(t){const e=new Yc(t.shape,"if (isnan(x)) return x;\n  if (abs(x) > 1.) {\n    return NAN;\n  }\n  return asin(x);\n");return this.compileAndRun(e,[t])}acos(t){const e=new Yc(t.shape,"if (isnan(x)) return x;\n  if (abs(x) > 1.) {\n    return NAN;\n  }\n  return acos(x);\n");return this.compileAndRun(e,[t])}atan(t){const e=new Yc(t.shape,"if (isnan(x)) return x;\n  return atan(x);\n");return this.compileAndRun(e,[t])}atan2(t,e){const n=Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Cu("\n  vec4 result = atan(a, b);\n  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));\n  \n  result.r = isNaN.r > 0. ? NAN : result.r;\n  result.g = isNaN.g > 0. ? NAN : result.g;\n  result.b = isNaN.b > 0. ? NAN : result.b;\n  result.a = isNaN.a > 0. ? NAN : result.a;\n\n  return result;\n",t.shape,e.shape):new vu("\n  if (isnan(a)) return a;\n  if (isnan(b)) return b;\n\n  return atan(a, b);\n",t.shape,e.shape);return this.compileAndRun(n,[t,e])}sinh(t){const e=new Yc(t.shape,"\n  float e2x = exp(x);\n  return (e2x - 1.0 / e2x) / 2.0;\n");return this.compileAndRun(e,[t])}cosh(t){const e=new Yc(t.shape,"\n  float e2x = exp(-x);\n  return (e2x + 1.0 / e2x) / 2.0;\n");return this.compileAndRun(e,[t])}tanh(t){const e=new Yc(t.shape,"\n  float e2x = exp(-2.0 * abs(x));\n  return sign(x) * (1.0 - e2x) / (1.0 + e2x);\n");return this.compileAndRun(e,[t])}asinh(t){const e=new Yc(t.shape,"if (isnan(x)) return x;return log(x + sqrt(x * x + 1.0));");return this.compileAndRun(e,[t])}acosh(t){const e=new Yc(t.shape,"if (isnan(x)) return x;\n  if (x < 1.0) return NAN;\n  return log(x + sqrt(x * x - 1.0));");return this.compileAndRun(e,[t])}atanh(t){const e=new Yc(t.shape,"if (isnan(x)) return x;\n  if ((x < -1.0) || (x > 1.0)) return NAN;\n  return (log(1.0 + x) - log(1.0 - x)) / 2.0;");return this.compileAndRun(e,[t])}erf(t){const e=new Yc(t.shape,il);return this.compileAndRun(e,[t])}step(t,e){const n=new Yc(t.shape,function(t=0){return`if (isnan(x)) return x;\n    return x > 0.0 ? 1.0 : float(${t});\n  `}(e));return this.compileAndRun(n,[t])}conv2dByMatMul(t,e,n,r,o,s){const a=t.shape,i=this.texData.get(t.dataId),u=n.inChannels,c=a[0]*a[1]*a[2],l=n.outChannels,d="channelsLast"===n.dataFormat,p=(1===c||1===l)&&u>1e3,f=a[2]%2!=0&&!!i.isPacked;if(p||!Object(h.b)().getBool("WEBGL_LAZILY_UNPACK")||!Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")||!f){const i=d?a[0]*a[1]*a[2]:a[0]*a[2]*a[3],u=this.reshape(t,[1,i,n.inChannels]),c=this.reshape(e,[1,n.inChannels,n.outChannels]);return this.reshape(this.fusedBatchMatMul({a:u,b:c,transposeA:!1,transposeB:!1,bias:r,activation:o,preluActivationWeights:s}),n.outShape)}const g=d?a[0]*a[1]*(a[2]+1):a[0]*a[2]*(a[3]+1),m={dataId:t.dataId,shape:[1,g,n.inChannels],dtype:t.dtype},b=i.shape;i.shape=i.shape.slice(),i.shape[i.shape.length-2]++,y.assert(Ti(i.shape,m.shape),()=>`packed reshape ${i.shape} to ${m.shape} isn't free`);const x=this.reshape(e,[1,n.inChannels,n.outChannels]),v=this.fusedBatchMatMul({a:m,b:x,transposeA:!1,transposeB:!1,bias:r,activation:o,preluActivationWeights:s}),w=this.texData.get(v.dataId);return y.assert(w.isPacked,()=>"batchMatMul result is expected to be packed"),i.shape=b,w.shape=n.outShape,on().makeTensorFromDataId(v.dataId,n.outShape,v.dtype)}conv2dWithIm2Row(t,e,n,r,o,s){const{filterWidth:a,filterHeight:i,inChannels:u,outWidth:c,outHeight:l,dataFormat:h}=n,d="channelsLast"===h,p=a*i*u,f=l*c,g=[p,f],m=t.squeeze([0]),b=e.reshape([1,p,-1]),x=new gc(g,m.shape,n),y=this.compileAndRun(x,[m]).reshape([1,g[0],g[1]]),v=null!=r,w=null!=s,C=o?vl(o,!0):null,$=new wc(y.shape,[1,f,n.outChannels],!0,!1,v,C,w),O=[y,b];r&&O.push(r),w&&O.push(s);const I=this.compileAndRun($,O);return d?I.reshape([1,l,c,n.outChannels]):I.reshape([1,n.outChannels,l,c])}fusedConv2d({input:t,filter:e,convInfo:n,bias:r,activation:o,preluActivationWeights:s}){if(1===n.filterHeight&&1===n.filterWidth&&1===n.dilationHeight&&1===n.dilationWidth&&1===n.strideHeight&&1===n.strideWidth&&("SAME"===n.padInfo.type||"VALID"===n.padInfo.type))return this.conv2dByMatMul(t,e,n,r,o,s);if(Object(h.b)().getBool("WEBGL_CONV_IM2COL")&&1===t.shape[0])return this.conv2dWithIm2Row(t,e,n,r,o,s);const a=null!=r,i=null!=s,u=o?vl(o,!1):null,c=new _u(n,a,u,i),l=[t,e];return r&&l.push(r),s&&l.push(s),this.compileAndRun(c,l)}conv2d(t,e,n){if(1===n.filterHeight&&1===n.filterWidth&&1===n.dilationHeight&&1===n.dilationWidth&&1===n.strideHeight&&1===n.strideWidth&&("SAME"===n.padInfo.type||"VALID"===n.padInfo.type))return this.conv2dByMatMul(t,e,n);if(Object(h.b)().getBool("WEBGL_CONV_IM2COL")&&1===t.shape[0])return this.conv2dWithIm2Row(t,e,n);const r=new _u(n);return this.compileAndRun(r,[t,e])}conv2dDerInput(t,e,n){const r=new ku(n);return this.compileAndRun(r,[t,e])}conv2dDerFilter(t,e,n){const r=new Au(n);return this.compileAndRun(r,[t,e])}fusedDepthwiseConv2D({input:t,filter:e,convInfo:n,bias:r,activation:o,preluActivationWeights:s}){const a=Object(h.b)().getBool("WEBGL_PACK_DEPTHWISECONV")&&n.strideWidth<=2&&n.outChannels/n.inChannels==1,i=o?vl(o,a):null,u=[t,e],c=null!=r,l=null!=s;let d;return c&&u.push(r),l&&u.push(s),a?(d=new Mu(n,c,i,l),this.compileAndRun(d,u)):(d=new ju(n,c,i,l),this.compileAndRun(d,u))}depthwiseConv2D(t,e,n){let r;return Object(h.b)().getBool("WEBGL_PACK_DEPTHWISECONV")&&n.strideWidth<=2&&n.outChannels/n.inChannels==1?(r=new Mu(n),this.compileAndRun(r,[t,e])):(r=new ju(n),this.compileAndRun(r,[t,e]))}depthwiseConv2DDerInput(t,e,n){const r=new Du(n);return this.compileAndRun(r,[t,e])}depthwiseConv2DDerFilter(t,e,n){const r=new Nu(n);return this.compileAndRun(r,[t,e])}conv3d(t,e,n){const r=new Bu(n);return this.compileAndRun(r,[t,e])}conv3dDerInput(t,e,n){const r=new Fu(n);return this.compileAndRun(r,[t,e])}conv3dDerFilter(t,e,n){const r=new Tu(n);return this.compileAndRun(r,[t,e])}maxPool(t,e){const n=new Ec(e,"max",!1);return this.compileAndRun(n,[t])}avgPool(t,e){const n=new Ec(e,"avg",!1);return this.compileAndRun(n,[t],"float32")}maxPoolBackprop(t,e,n,r){const o=new Ec(r,"max",!0),s=this.compileAndRun(o,[e]),a=new yc(r),i=this.compileAndRun(a,[t,s],e.dtype);return s.dispose(),i}avgPoolBackprop(t,e,n){const r=new cu(n);return this.compileAndRun(r,[t],e.dtype)}cast(t,e){return s.castTensor(t,e,this)}unstack(t,e){const n=t.shape[e],r=new Array(t.rank-1);let o=0;for(let n=0;n<t.rank;n++)n!==e&&(r[o++]=t.shape[n]);const s=new Array(t.rank).fill(0),a=t.shape.slice();a[e]=1;const i=new Array(n);for(let n=0;n<i.length;n++)s[e]=n,i[n]=this.slice(t,s,a).reshape(r);return i}avgPool3d(t,e){const n=new Rc(e,"avg",!1);return this.compileAndRun(n,[t],"float32")}avgPool3dBackprop(t,e,n){const r=new lu(n);return this.compileAndRun(r,[t],e.dtype)}maxPool3d(t,e){const n=new Rc(e,"max",!1);return this.compileAndRun(n,[t],"float32")}maxPool3dBackprop(t,e,n,r){const o=new Rc(r,"max",!0),s=this.compileAndRun(o,[e]),a=new vc(r),i=this.compileAndRun(a,[t,s],e.dtype);return s.dispose(),i}reshape(t,e){const n=this.texData.get(t.dataId);if(n.isPacked&&!Ti(t.shape,e)&&(null===n.texture||!Ti(n.shape,e))){const n=this.packedReshape(t,e);return on().makeTensorFromDataId(n.dataId,n.shape,n.dtype)}return s.reshapeTensor(t,e)}resizeBilinear(t,e,n,r){const o=Object(h.b)().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new Nc(t.shape,e,n,r):new Fc(t.shape,e,n,r);return this.compileAndRun(o,[t],"float32")}resizeBilinearBackprop(t,e,n){const r=new Tc(t,e,n);return this.compileAndRun(r,[t])}resizeNearestNeighbor(t,e,n,r){const o=new _c(t.shape,e,n,r);return this.compileAndRun(o,[t])}resizeNearestNeighborBackprop(t,e,n){const r=new Dc(t,e,n);return this.compileAndRun(r,[t])}multinomial(t,e,n,r){const o=e?t:ue(t),s=o.shape[0],a=o.shape[1],i=new Cc(s,a,n),u=i.getCustomSetupFunc(r);return this.compileAndRun(i,[o],"int32",u)}oneHot(t,e,n,r){const o=new $c(t.size,e,n,r);return this.compileAndRun(o,[t])}diag(t){const e=new Hu(t.size);return this.compileAndRun(e,[t])}cropAndResize(t,e,n,r,o,s){const a=new Pu(t.shape,e.shape,r,o,s);return this.compileAndRun(a,[t,e,n],"float32")}depthToSpace(t,e,n){y.assert(e>1,()=>"blockSize should be > 1 for depthToSpace, but was: "+e);const r=t.shape[0],o="NHWC"===n?t.shape[1]:t.shape[2],s="NHWC"===n?t.shape[2]:t.shape[3],a="NHWC"===n?t.shape[3]:t.shape[1],i=o*e,u=s*e,c=a/(e*e),l=new Gu("NHWC"===n?[r,i,u,c]:[r,c,i,u],e,n);return this.compileAndRun(l,[t])}split(t,e,n){return gl(t,e,n)}scatterND(t,e,n){const{sliceRank:r,numUpdates:o,sliceSize:a,strides:i,outputSize:u}=s.calculateShapes(e,t,n),c=[u/a,a],l=t.reshape([o,r]),h=e.reshape([o,a]);if(0===u)return s.reshapeTensor(Object(he.a)([]),n);const d=ie(0),p=new Mc(o,r,l.rank,h.rank,i,c);return this.compileAndRun(p,[h,l,d]).reshape(n)}sparseToDense(t,e,n,r){const{sliceRank:o,numUpdates:a,strides:i,outputSize:u}=s.calculateShapes(e,t,n),c=new Mc(a,o,t.rank,e.rank,i,[u,1],!1);return this.compileAndRun(c,[e,t,r]).reshape(n)}fft(t){return this.fftImpl(t,!1)}ifft(t){return this.fftImpl(t,!0)}fftImpl(t,e){const n=this.texData.get(t.dataId),r=new Ju(Qu,t.shape,e),o=new Ju(Zu,t.shape,e),s=[this.makeComplexComponentTensorInfo(t,n.complexTensors.real),this.makeComplexComponentTensorInfo(t,n.complexTensors.imag)],a=this.compileAndRun(r,s),i=this.compileAndRun(o,s),u=this.complex(a,i).as2D(t.shape[0],t.shape[1]);return a.dispose(),i.dispose(),u}gatherND(t,e){const n=e.shape,r=n[n.length-1],[o,a,i,u]=s.prepareAndValidate(t,e),c=e.reshape([a,r]),l=t.reshape([t.size/i,i]),h=new nc(r,u,[a,i]);return this.compileAndRun(h,[l,c]).reshape(o)}fill(t,e,n){if("string"===(n=n||y.inferDtype(e))){const r=y.getArrayFromDType(n,y.sizeFromShape(t));return r.fill(e),on().makeTensor(r,t,n,this)}{const r=new tc(t,e),o=r.getCustomSetupFunc(e);return this.compileAndRun(r,[],n,o)}}onesLike(t){if("string"===t.dtype)throw new Error("onesLike is not supported under string dtype");return this.fill(t.shape,1,t.dtype)}zerosLike(t){return this.fill(t.shape,"string"===t.dtype?"":0,t.dtype)}linspace(t,e,n){return s.linspaceImpl(t,e,n)}makeTensorInfo(t,e){const n=this.write(null,t,e);return this.texData.get(n).usage=null,{dataId:n,shape:t,dtype:e}}makeOutput(t,e){const{dataId:n}=this.makeTensorInfo(t,e);return on().makeTensorFromDataId(n,t,e,this)}unpackTensor(t){const e=new pl(t.shape);return this.runWebGLProgram(e,[t],t.dtype)}packTensor(t){const e=new Oc(t.shape);return this.runWebGLProgram(e,[t],t.dtype,null,!0)}packedReshape(t,e){const n=[Ei(t.shape),...Ri(t.shape)],r={dtype:t.dtype,shape:n,dataId:t.dataId},o=[Ei(e),...Ri(e)],s=new kc(o,n),a=this.runWebGLProgram(s,[r],t.dtype,null,!0);return{dataId:a.dataId,shape:e,dtype:a.dtype}}decode(t){const e=this.texData.get(t),{isPacked:n,shape:r,dtype:o}=e,s=Ai(r);let a;a=n?new Vu(s):new Uu(s);return{dtype:o,shape:r,dataId:this.runWebGLProgram(a,[{shape:s,dtype:o,dataId:t}],o,null,!0).dataId}}runWebGLProgram(t,e,n,r,o=!1){const s=this.makeTensorInfo(t.outputShape,n),a=this.texData.get(s.dataId);if(t.packedOutput&&(a.isPacked=!0),t.outPackingScheme===ii.DENSE){const e=hi(t.outputShape);a.texShape=e.map(t=>2*t)}if(null!=t.outTexUsage&&(a.usage=t.outTexUsage),0===y.sizeFromShape(s.shape))return a.values=y.getTypedArrayFromDType(s.dtype,0),s;const i=[],u=e.map(e=>{if("complex64"===e.dtype)throw new Error("GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.");let n=this.texData.get(e.dataId);if(null==n.texture){if(!t.packedInputs&&y.sizeFromShape(e.shape)<=Object(h.b)().getNumber("WEBGL_SIZE_UPLOAD_UNIFORM"))return{shape:e.shape,texData:null,isUniform:!0,uniformValues:n.values};t.packedInputs&&(n.isPacked=!0,n.shape=e.shape)}else if(!!n.isPacked!=!!t.packedInputs)e=n.isPacked?this.unpackTensor(e):this.packTensor(e),i.push(e),n=this.texData.get(e.dataId);else if(n.isPacked&&!Ti(n.shape,e.shape)){const t=e,r=e.shape;e.shape=n.shape,e=this.packedReshape(e,r),i.push(e),n=this.texData.get(e.dataId),t.shape=r}return this.uploadToGPU(e.dataId),{shape:e.shape,texData:n,isUniform:!1}});this.uploadToGPU(s.dataId);const c={shape:s.shape,texData:a,isUniform:!1},l=function(t,e,n){let r="";e.concat(n).forEach(t=>{const e=null!=t.texData&&null!=t.texData.slice&&t.texData.slice.flatOffset>0,n=t.isUniform?"uniform":t.texData.texShape;r+=`${t.shape}_${n}_${e}`});const o=t.userCode;let s=t.constructor.name;return s+="_"+r+"_"+o,s}(t,u,c),d=this.getAndSaveBinary(l,()=>function(t,e,n,r){const o=e.userCode,s=n.map((t,n)=>{const r={logicalShape:t.shape,texShape:t.isUniform?null:t.texData.texShape,isUniform:t.isUniform,isPacked:!t.isUniform&&t.texData.isPacked,flatOffset:null};return null!=t.texData&&null!=t.texData.slice&&t.texData.slice.flatOffset>0&&(r.flatOffset=t.texData.slice.flatOffset),{name:e.variableNames[n],shapeInfo:r}}),a=s.map(t=>t.shapeInfo),i={logicalShape:r.shape,texShape:r.texData.texShape,isUniform:!1,isPacked:r.texData.isPacked,flatOffset:null},u=Xi(s,i,o,e.packedInputs),c=t.createProgram(u);let l=null;const d=t.getUniformLocation(c,"NAN",!1);1===Object(h.b)().getNumber("WEBGL_VERSION")&&(l=t.getUniformLocation(c,"INFINITY",!1));const p={};for(let n=0;n<e.variableNames.length;n++){const r=e.variableNames[n],o=!1;p[r]=t.getUniformLocation(c,r,o),p["offset"+r]=t.getUniformLocation(c,"offset"+r,o)}return{program:e,source:u,webGLProgram:c,uniformLocations:p,inShapeInfos:a,outShapeInfo:i,infLoc:l,nanLoc:d}}(this.gpgpu,t,u,c)),p=null!=this.activeTimers;let f;if(p&&(f=this.startTimer()),function(t,e,n,r,o){fc(e.inShapeInfos,n),fc([e.outShapeInfo],[r]);const s=r.texData.texture,a=r.texData.texShape;r.texData.isPacked?t.setOutputPackedMatrixTexture(s,a[0],a[1]):t.setOutputMatrixTexture(s,a[0],a[1]),t.setProgram(e.webGLProgram),1===Object(h.b)().getNumber("WEBGL_VERSION")&&null!==e.infLoc&&t.gl.uniform1f(e.infLoc,1/0),null!==e.nanLoc&&t.gl.uniform1f(e.nanLoc,NaN),n.forEach((n,r)=>{const o=e.program.variableNames[r],s=e.uniformLocations[o],a=e.uniformLocations["offset"+o];if(null!=s)if(n.isUniform)if(y.sizeFromShape(n.shape)<2)t.gl.uniform1f(s,n.uniformValues[0]);else{let e=n.uniformValues;e instanceof Float32Array||(e=new Float32Array(e)),t.gl.uniform1fv(s,e)}else null!=n.texData.slice&&null!=a&&t.gl.uniform1i(a,n.texData.slice.flatOffset),t.setInputMatrixTexture(n.texData.texture,s,r)}),null!=o&&o(t,e.webGLProgram),t.executeProgram()}(this.gpgpu,d,u,c,r),i.forEach(t=>this.disposeData(t.dataId)),p&&(f=this.endTimer(f),this.activeTimers.push({name:t.constructor.name,query:this.getQueryTime(f)})),!Object(h.b)().getBool("WEBGL_LAZILY_UNPACK")&&a.isPacked&&!1===o){const t=this.unpackTensor(s);return this.disposeData(s.dataId),t}return s}compileAndRun(t,e,n,r,o=!1){n=n||e[0].dtype;const s=this.runWebGLProgram(t,e,n,r,o);return on().makeTensorFromDataId(s.dataId,s.shape,s.dtype)}getAndSaveBinary(t,e){return t in this.binaryCache||(this.binaryCache[t]=e()),this.binaryCache[t]}getTextureManager(){return this.textureManager}dispose(){if(!this.disposed){if(!Object(h.b)().getBool("IS_TEST")){Object.keys(this.binaryCache).forEach(t=>{this.gpgpu.deleteProgram(this.binaryCache[t].webGLProgram),delete this.binaryCache[t]})}this.textureManager.dispose(),null!=this.canvas&&"undefined"!=typeof HTMLCanvasElement&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),this.disposed=!0}}floatPrecision(){return null==this.floatPrecisionValue&&(this.floatPrecisionValue=sn(()=>{if(!Object(h.b)().get("WEBGL_RENDER_FLOAT32_ENABLED")){const t=Object(h.b)().getBool("DEBUG");Object(h.b)().set("DEBUG",!1);const e=this.abs(ie(1e-8)).dataSync()[0];if(Object(h.b)().set("DEBUG",t),e>0)return 32}return 16})),this.floatPrecisionValue}epsilon(){return 32===this.floatPrecision()?1e-7:1e-4}uploadToGPU(t){const e=this.texData.get(t),{shape:n,dtype:r,values:o,texture:s,usage:a,isPacked:i}=e;if(null!=s)return;const u=null!=this.activeTimers;let c;u&&(c=y.now());let l=e.texShape;if(null==l&&(l=function(t,e=!1){let n=Object(h.b)().getNumber("WEBGL_MAX_TEXTURE_SIZE");if(e&&(n*=2,1===(t=t.map((e,n)=>n>=t.length-2?y.nearestLargerEven(t[n]):t[n])).length&&(t=[2,t[0]])),2!==t.length){const e=y.squeezeShape(t);t=e.newShape}let r=y.sizeFromShape(t);if(t.length<=1&&r<=n)return[1,r];if(2===t.length&&t[0]<=n&&t[1]<=n)return t;if(3===t.length&&t[0]*t[1]<=n&&t[2]<=n)return[t[0]*t[1],t[2]];if(3===t.length&&t[0]<=n&&t[1]*t[2]<=n)return[t[0],t[1]*t[2]];if(4===t.length&&t[0]*t[1]*t[2]<=n&&t[3]<=n)return[t[0]*t[1]*t[2],t[3]];if(4===t.length&&t[0]<=n&&t[1]*t[2]*t[3]<=n)return[t[0],t[1]*t[2]*t[3]];if(e){const e=Ei(t);let n=2,o=2;return t.length&&([n,o]=Ri(t)),r=e*(n/2)*(o/2),y.sizeToSquarishShape(r).map(t=>2*t)}return y.sizeToSquarishShape(r)}(n,i),e.texShape=l),null!=o){const t=Ai(n);let s,a=l[1],h=l[0];const d=o instanceof Uint8Array;i?([a,h]=di(l[0],l[1]),s=new Yu(t,[h,a],d)):s=new Xu(t,[h,a],d);const p=this.makeTensorInfo([h,a],r);this.texData.get(p.dataId).usage=d?ui.PIXELS:ui.UPLOAD,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(p.dataId),a,h,o);const f=!0,g=this.runWebGLProgram(s,[p],r,null,f),m=this.texData.get(g.dataId);e.texture=m.texture,e.texShape=m.texShape,e.isPacked=m.isPacked,e.usage=m.usage,this.disposeData(p.dataId),this.texData.delete(g.dataId),e.values=null,u&&(this.uploadWaitMs+=y.now()-c)}else{const t=this.acquireTexture(l,a,r,i);e.texture=t}}convertAndCacheOnCPU(t,e){const n=this.texData.get(t),{dtype:r}=n;return this.releaseGPUData(t),null!=e&&(n.values=function(t,e){if("float32"===e||"complex64"===e)return t;if("int32"===e||"bool"===e){const n="int32"===e?new Int32Array(t.length):new Uint8Array(t.length);for(let e=0;e<n.length;++e)n[e]=Math.round(t[e]);return n}throw new Error("Unknown dtype "+e)}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(e,r)),n.values}acquireTexture(t,e,n,r){if(this.numBytesInGPU+=this.computeBytes(t,n),!this.warnedAboutMemory&&this.numBytesInGPU>1024*this.numMBBeforeWarning*1024){const t=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${t} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(t,e,r)}computeBytes(t,e){return t[0]*t[1]*y.bytesPerElement(e)}}
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
xr.isBrowser()&&cn("webgl",()=>new wl,2);
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const Cl={kernelName:_.R,backendName:"webgl",kernelFunc:({inputs:t,backend:e})=>{const{a:n,b:r}=t;
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
return function(t,e,n){let r=new vu("\nif (a == b) {\n  return 1.0;\n};\nreturn a / b;",t.shape,e.shape);return Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")&&(r=new Cu("\n  // vec4 one = vec4(equal(a, b));\n  // return one + (vec4(1.0) - one) * a / b;\n  vec4 result = a / b;\n  if(a.x == b.x) {\n    result.x = 1.;\n  }\n  if(a.y == b.y) {\n    result.y = 1.;\n  }\n  if(a.z == b.z) {\n    result.z = 1.;\n  }\n  if(a.w == b.w) {\n    result.w = 1.;\n  }\n\n  return result;\n",t.shape,e.shape,!0)),n.runWebGLProgram(r,[t,e],"float32")}(n,r,e)}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $l{constructor(t){this.variableNames=["Image"],this.outputShape=[];const e=t[2];this.outputShape=t,this.userCode=`\n        void main() {\n          ivec4 coords = getOutputCoords();\n          int x = coords[2];\n\n          int coordX = ${e} - x;\n          float outputValue;\n          if(coordX >= 0 && coordX < ${e}) {\n            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);\n          } else {\n            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);\n          }\n          setOutput(outputValue);\n        }\n    `}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ol={kernelName:_.ab,backendName:"webgl",kernelFunc:({inputs:t,backend:e})=>{const{image:n}=t,r=e,o=new $l(n.shape);return r.runWebGLProgram(o,[n],n.dtype)}};
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Il{constructor(t){this.variableNames=["A"];const e=Vi(),[n,r]=t;this.outputShape=t,this.userCode=`\n      void main() {\n        ivec3 coords = getOutputCoords();\n        int texR = coords[0];\n        int texC = coords[1];\n        int depth = coords[2];\n        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${r}.0, ${n}.0);\n\n        vec4 values = ${e.texture2D}(A, uv);\n        float value;\n        if (depth == 0) {\n          value = values.r;\n        } else if (depth == 1) {\n          value = values.g;\n        } else if (depth == 2) {\n          value = values.b;\n        } else if (depth == 3) {\n          value = values.a;\n        }\n\n        setOutput(floor(value * 255.0 + 0.5));\n      }\n    `}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Sl{constructor(t){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0;const e=Vi(),[n,r]=t;this.outputShape=t,this.userCode=`\n      void main() {\n        ivec3 coords = getOutputCoords();\n        int texR = coords[0];\n        int texC = coords[1];\n        int depth = coords[2];\n\n        vec4 result = vec4(0.);\n\n        for(int row=0; row<=1; row++) {\n          for(int col=0; col<=1; col++) {\n            texC = coords[1] + row;\n            depth = coords[2] + col;\n\n            vec2 uv = (vec2(texC, texR) + halfCR) /\n                       vec2(${r}.0, ${n}.0);\n            vec4 values = ${e.texture2D}(A, uv);\n            float value;\n            if (depth == 0) {\n              value = values.r;\n            } else if (depth == 1) {\n              value = values.g;\n            } else if (depth == 2) {\n              value = values.b;\n            } else if (depth == 3) {\n              value = values.a;\n            }\n\n            result[row * 2 + col] = floor(value * 255.0 + 0.5);\n          }\n        }\n\n        ${e.output} = result;\n      }\n    `}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const El={kernelName:_.db,backendName:"webgl",kernelFunc:function(t){const{inputs:e,backend:n,attrs:r}=t;let{pixels:o}=e;const{numChannels:s}=r,a="undefined"!=typeof HTMLVideoElement&&o instanceof HTMLVideoElement,i="undefined"!=typeof HTMLImageElement&&o instanceof HTMLImageElement,[u,c]=a?[o.videoWidth,o.videoHeight]:[o.width,o.height],l=[c,u],d=[c,u,s];(i||a)&&(null==Rl&&(Rl=document.createElement("canvas").getContext("2d")),Rl.canvas.width=u,Rl.canvas.height=c,Rl.drawImage(o,0,0,u,c),o=Rl.canvas);const p=n.makeTensorInfo(l,"int32");n.texData.get(p.dataId).usage=ui.PIXELS,n.gpgpu.uploadPixelDataToTexture(n.getTexture(p.dataId),o);const f=Object(h.b)().getBool("WEBGL_PACK")?new Sl(d):new Il(d),g=n.runWebGLProgram(f,[p],"int32");return n.disposeData(p.dataId),g}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */};let Rl;function Al(t,e,n,r){const o=y.getTypedArrayFromDType(r,y.sizeFromShape(n));for(let n=0;n<o.length;++n){const r=n*e;let s=t[r];for(let n=0;n<e;++n){const e=t[r+n];e>s&&(s=e)}o[n]=s}return o}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kl(t,e,n,r,o){const s=e.length,a=y.sizeFromShape(e),i=y.computeStrides(e),u=y.computeStrides(o),c=y.getTypedArrayFromDType(n,y.sizeFromShape(o));for(let e=0;e<a;++e){const n=y.indexToLoc(e,s,i),o=new Array(n.length);for(let t=0;t<o.length;t++)o[t]=n[r[t]];c[y.locToIndex(o,s,u)]=t[e]}return c}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const{maxImpl:Tl,transposeImpl:Fl}=u;
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nl(t,e,n,r){const o=function(t){const e=[];for(;0===e.length||1!==e[e.length-1].outSize;){const n=e.length?e[e.length-1].outSize:t[1],r=s.computeOptimalWindowSize(n);e.push({inSize:n,windowSize:r,outSize:Math.ceil(n/r)})}return e}(t.shape);let a=t;for(let s=0;s<o.length;s++){const{inSize:i,windowSize:u,outSize:c}=o[s],l=new Ac({windowSize:u,inSize:i,batchSize:t.shape[0],outSize:c},n),h=a;a=r.runWebGLProgram(l,[a],e),h.dataId!==t.dataId&&r.disposeData(h.dataId)}return a}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dl(t,e,n){const r=n.texData.get(t.dataId);return!r.isPacked||Ti(t.shape,e)||null!==r.texture&&Ti(r.shape,e)?{dataId:t.dataId,shape:e,dtype:t.dtype}:function(t,e,n){const r=[Ei(t.shape),...Ri(t.shape)],o={dtype:t.dtype,shape:r,dataId:t.dataId},s=[Ei(e),...Ri(e)],a=new kc(s,r),i=n.runWebGLProgram(a,[o],t.dtype,null,!0);return{dataId:i.dataId,shape:e,dtype:i.dtype}}(t,e,n)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class _l{constructor(t,e){this.variableNames=["A"];const n=new Array(t.length);for(let r=0;r<n.length;r++)n[r]=t[e[r]];this.outputShape=n,this.rank=n.length;const r=su(this.rank),o=function(t){const e=t.length;if(e>6)throw Error(`Transpose for rank ${e} is not yet supported`);const n=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u","resRC.v"],r=new Array(e);for(let e=0;e<t.length;e++)r[t[e]]=n[e];return r.join()}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(e);this.userCode=`\n    void main() {\n      ${r} resRC = getOutputCoords();\n      setOutput(getA(${o}));\n    }\n    `}}class Bl{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0;const n=new Array(t.length);for(let r=0;r<n.length;r++)n[r]=t[e[r]];if(this.outputShape=n,this.rank=n.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);const r=su(this.rank),o=zi("rc",this.rank),s=new Array(this.rank);for(let t=0;t<e.length;t++)s[e[t]]=o[t];const a=`vec2(${s.slice(-2).join()})`,i=`++${o[this.rank-1]} < ${n[this.rank-1]}`,u=`getChannel(getA(${s.join()}), ${a})`;this.userCode=`\n    void main() {\n      ${r} rc = getOutputCoords();\n      vec4 result = vec4(0.);\n      result[0] = ${u};\n      if(${i}) {\n        result[1] = ${u};\n      }\n      --${o[this.rank-1]};\n      if(++${o[this.rank-2]} < ${n[this.rank-2]}) {\n        result[2] = ${u};\n        if(${i}) {\n          result[3] = ${u};\n        }\n      }\n      setOutput(result);\n    }\n    `}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jl(t,e,n){const r=Object(h.b)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new Bl(t.shape,e):new _l(t.shape,e);return n.runWebGLProgram(r,[t],t.dtype)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ml={kernelName:_.yb,backendName:"webgl",kernelFunc:({inputs:t,attrs:e,backend:n})=>{const{x:r}=t,{reductionIndices:o,keepDims:a}=e,i=n,u=r.shape.length,c=y.parseAxisParam(o,r.shape);let l=c;const h=s.getAxesPermutation(l,u),d=null!=h,p=i.shouldExecuteOnCPU([r]);let f=r;if(d){if(p){const t=i.texData.get(f.dataId).values,e=new Array(u);for(let t=0;t<e.length;t++)e[t]=r.shape[h[t]];const n=Fl(t,r.shape,r.dtype,h,e);f=i.makeTensorInfo(e,r.dtype);i.texData.get(f.dataId).values=n}else f=jl(r,h,i);l=s.getInnerMostAxes(l.length,u)}s.assertAxesAreInnerMostDims("max",l,u);const[g,m]=s.computeOutAndReduceShapes(f.shape,l);let b,x=g;if(a&&(x=s.expandShapeToKeepDim(g,c)),p){const t=i.texData.get(f.dataId).values,e=Tl(t,y.sizeFromShape(m),x,r.dtype);b=i.makeTensorInfo(x,r.dtype);i.texData.get(b.dataId).values=e}else b=function(t,e,n,r){const o=y.sizeFromShape(e),s=Dl(t,[y.sizeFromShape(t.shape)/o,o],r),a=Nl(s,t.dtype,"max",r);return s.dataId!==t.dataId&&r.disposeData(s.dataId),Dl(a,n,r)}(f,m,x,i);return d&&i.disposeData(f.dataId),b}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const Pl={kernelName:_.Db,backendName:"webgl",kernelFunc:({inputs:t,attrs:e,backend:n})=>{const{x:r}=t,{filterSize:o,strides:a,pad:i,includeBatchInIndex:u}=e,c=n;y.assert(4===r.shape.length,()=>`Error in maxPool: input must be rank 4 but got rank ${r.shape.length}.`);const l=[1,1];y.assert(s.eitherStridesOrDilationsAreOne(a,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${a} and dilations '${l}'`);const h=s.computePool2DInfo(r.shape,o,a,l,i),[d,p]=function(t,e,n,r){let o=new Ec(n,"max",!1);const s=r.runWebGLProgram(o,[t],"float32");return o=new Ec(n,"max",!0,!0,e),[s,r.runWebGLProgram(o,[t],"float32")]}(r,u,h,c);return[d,p]}},Ll={kernelName:_.Kb,backendName:"webgl",kernelFunc:({inputs:t,backend:e,attrs:n})=>{s.warn("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{boxes:r,scores:o}=t,{maxOutputSize:i,iouThreshold:u,scoreThreshold:c}=n,l=e,h=l.readSync(r.dataId),d=l.readSync(o.dataId),p=i,f=u,g=c;return a.nonMaxSuppressionV3Impl(h,d,p,f,g)}},Wl=a.nonMaxSuppressionV4Impl,zl={kernelName:_.Lb,backendName:"webgl",kernelFunc:({inputs:t,backend:e,attrs:n})=>{s.warn("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{boxes:r,scores:o}=t,{maxOutputSize:a,iouThreshold:i,scoreThreshold:u,padToMaxOutputSize:c}=n,l=e,h=l.readSync(r.dataId),d=l.readSync(o.dataId),{selectedIndices:p,validOutputs:f}=Wl(h,d,a,i,u,c);return[p,f]}},Ul=a.nonMaxSuppressionV5Impl,Vl={kernelName:_.Mb,backendName:"webgl",kernelFunc:({inputs:t,backend:e,attrs:n})=>{s.warn("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{boxes:r,scores:o}=t,{maxOutputSize:a,iouThreshold:i,scoreThreshold:u,softNmsSigma:c}=n,l=e,h=l.readSync(r.dataId),d=l.readSync(o.dataId),p=a,f=i,g=u,m=c,{selectedIndices:b,selectedScores:x}=Ul(h,d,p,f,g,m);return[b,x]}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Gl{constructor(t,e,n,r){this.variableNames=["Image"],this.outputShape=[];const o=t[1],a=t[2],i=Math.sin(e).toFixed(3),u=Math.cos(e).toFixed(3);this.outputShape=t;const[c,l]=s.getImageCenter(r,o,a),h=c.toFixed(3),d=l.toFixed(3);let p="";p="number"==typeof n?`float outputValue = ${n.toFixed(2)};`:`\n        vec3 fill = vec3(${n.join(",")});\n        float outputValue = fill[coords[3]];`,this.userCode=`\n        void main() {\n          ivec4 coords = getOutputCoords();\n          int x = coords[2];\n          int y = coords[1];\n          float coordXFloat = (float(x) - ${h}) * ${u} - (float(y) - ${d}) * ${i};\n          float coordYFloat = (float(x) - ${h}) * ${i} + (float(y) - ${d}) * ${u};\n          int coordX = int(round(coordXFloat + ${h}));\n          int coordY = int(round(coordYFloat + ${d}));\n          ${p}\n          if(coordX >= 0 && coordX < ${a} && coordY >= 0 && coordY < ${o}) {\n            outputValue = getImage(coords[0], coordY, coordX, coords[3]);\n          }\n          setOutput(outputValue);\n        }\n    `}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hl=[Ml,Ol,El,Cl,Pl,Ll,zl,Vl,{kernelName:_.fc,backendName:"webgl",kernelFunc:({inputs:t,attrs:e,backend:n})=>{const{image:r}=t,{radians:o,fillValue:s,center:a}=e,i=n,u=new Gl(r.shape,o,s,a);return i.runWebGLProgram(u,[r],r.dtype)}},{kernelName:_.uc,backendName:"webgl",kernelFunc:({inputs:t,backend:e})=>{const{x:n}=t,r=e,o=new Yc(n.shape,"return x * x;");return r.runWebGLProgram(o,[n],n.dtype)}},{kernelName:_.vc,backendName:"webgl",kernelFunc:({inputs:t,backend:e})=>{const{a:n,b:r}=t,o=e,s=Object(h.b)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Cu("return (a - b) * (a - b);",n.shape,r.shape):new vu("return (a - b) * (a - b);",n.shape,r.shape);return o.compileAndRun(s,[n,r])}},{kernelName:_.Ec,backendName:"webgl",kernelFunc:({inputs:t,attrs:e,backend:n})=>{const{x:r}=t,{perm:o}=e,s=n,a=r.shape.length,i=new Array(a);for(let t=0;t<i.length;t++)i[t]=r.shape[o[t]];let u;if(s.shouldExecuteOnCPU([r])){const t=s.texData.get(r.dataId).values,e=Fl(t,r.shape,r.dtype,o,i);u=s.makeTensorInfo(i,r.dtype);s.texData.get(u.dataId).values=e}else u=jl(r,o,s);return u}}];
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */for(const t of Hl)Object(En.e)(t);
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Kl=n(23);
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ql(t,e){Array.isArray(t)||(t=[t]),t.forEach(t=>{null!=t&&y.assert("complex64"!==t.dtype,()=>e+" does not support complex64 tensors in the CPU backend.")})}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xl(t,e,n,r,o,s){const a=o.strideHeight,i=o.strideWidth,u=o.dilationHeight,c=o.dilationWidth,l=o.effectiveFilterHeight,h=o.effectiveFilterWidth,d=o.padInfo.top,p=o.padInfo.left,f="max"===s?Number.NEGATIVE_INFINITY:Number.POSITIVE_INFINITY,g=D(o.outShape,n),m=g.values,b=o.outShape[1]*o.outShape[2]*o.outShape[3],x=o.outShape[2]*o.outShape[3],y=o.outShape[3];for(let e=0;e<o.batchSize;++e){const n=e*b,g=e*r[0];for(let e=0;e<o.inChannels;++e)for(let b=0;b<o.outHeight;++b){const v=b*a-d,w=Math.max(0,v),C=Math.min(o.inHeight,l+v),$=n+b*x;for(let n=0;n<o.outWidth;++n){const a=n*i-p,l=Math.max(0,a),d=Math.min(o.inWidth,h+a);let b=f,x=0,v=0;for(let n=w;n<C;n+=u){const o=g+n*r[1];for(let n=l;n<d;n+=c){const a=t[o+n*r[2]+e];"max"===s&&a>b?b=a:"avg"===s&&(x+=a,v++)}if(isNaN(b))break}m[$+n*y+e]="avg"===s?x/v:b}}}return g}function Yl(t,e,n,r,o=!1,s=!1){const a=D(r.outShape,"int32"),i=r.strideHeight,u=r.strideWidth,c=r.dilationHeight,l=r.dilationWidth,h=r.effectiveFilterHeight,d=r.effectiveFilterWidth,p=r.padInfo.top,f=r.padInfo.left,g=D(e,n,t);for(let t=0;t<r.batchSize;++t)for(let e=0;e<r.inChannels;++e)for(let n=0;n<r.outHeight;++n){const m=n*i-p;let b=m;for(;b<0;)b+=c;const x=Math.min(r.inHeight,h+m);for(let i=0;i<r.outWidth;++i){const h=i*u-f;let p=h;for(;p<0;)p+=l;const y=Math.min(r.inWidth,d+h);let v=Number.NEGATIVE_INFINITY,w=-1;for(let n=b;n<x;n+=c){const a=n-m;for(let i=p;i<y;i+=l){const u=i-h,c=g.get(t,n,i,e);c>v&&(v=c,w=o?s?((t*r.inHeight+n)*r.inWidth+i)*r.inChannels+e:(n*r.inWidth+i)*r.inChannels+e:a*d+u)}}a.set(w,t,n,i,e)}}return a}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ql=a.nonMaxSuppressionV3Impl,Zl=a.split,Jl=a.tile,th=a.topkImpl,eh=a.whereImpl;function nh(t,e,n,r){if("linear"===n)return t.linear(e);if("relu"===n)return t.relu(e);if("elu"===n)return t.elu(e);if("relu6"===n)return t.relu6(e);if("prelu"===n)return t.prelu(e,r);throw new Error(`Activation ${n} has not been implemented for the CPU backend.`)}class rh extends Or{constructor(){super(),this.blockSize=48,this.firstUse=!0,this.data=new $r(this,on())}write(t,e,n){this.firstUse&&(this.firstUse=!1,Object(h.b)().get("IS_NODE")&&s.warn("\n============================\nHi there 👋. Looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, which binds to TensorFlow C++, by running npm i @tensorflow/tfjs-node, or npm i @tensorflow/tfjs-node-gpu if you have CUDA. Then call require('@tensorflow/tfjs-node'); (-gpu suffix for CUDA) at the start of your program. Visit https://github.com/tensorflow/tfjs-node for more details.\n============================"));const r={};return this.data.set(r,{values:t,dtype:n,refCount:1}),r}incRef(t){this.data.get(t).refCount++}decRef(t){if(this.data.has(t)){this.data.get(t).refCount--}}move(t,e,n,r){this.data.set(t,{values:e,dtype:r,refCount:1})}numDataIds(){return this.data.numDataIds()}async read(t){return this.readSync(t)}readSync(t){const{dtype:e,complexTensors:n}=this.data.get(t);if("complex64"===e){const t=this.readSync(n.real.dataId),e=this.readSync(n.imag.dataId);return s.mergeRealAndImagArrays(t,e)}return this.data.get(t).values}bufferSync(t){const e=this.readSync(t.dataId);let n=e;if("string"===t.dtype)try{n=e.map(t=>y.decodeString(t))}catch(t){throw new Error("Failed to decode encoded string bytes into utf-8")}return D(t.shape,t.dtype,n)}makeOutput(t,e,n){const r=this.write(t,e,n);return on().makeTensorFromDataId(r,e,n,this)}disposeData(t){if(this.data.has(t)){const{complexTensors:e}=this.data.get(t);null!=e&&(e.real.dispose(),e.imag.dispose()),this.data.delete(t)}}disposeIntermediateTensorInfo(t){const e=t.dataId;if(this.data.has(e)){const t=this.data.get(e);t.refCount--,t.refCount<1&&this.disposeData(e)}}async time(t){const e=y.now();t();return{kernelMs:y.now()-e}}memory(){return{unreliable:!0,reasons:["The reported memory is an upper bound. Due to automatic garbage collection, the true allocated memory may be less."]}}complex(t,e){const n=this.makeOutput(null,t.shape,"complex64");return this.data.get(n.dataId).complexTensors={real:on().keep(t.clone()),imag:on().keep(e.clone())},n}real(t){return this.data.get(t.dataId).complexTensors.real.clone()}imag(t){return this.data.get(t.dataId).complexTensors.imag.clone()}slice(t,e,n){ql(t,"slice");if(r.isSliceContinous(t.shape,e,n)){const o=r.computeFlatOffset(e,t.strides),s=y.sizeFromShape(n),a=this.readSync(t.dataId);return he.a(a.subarray(o,o+s),n,t.dtype)}const o=D(n,t.dtype),s=this.bufferSync(t);for(let t=0;t<o.size;++t){const n=o.indexToLoc(t).map((t,n)=>t+e[n]);o.values[t]=s.get(...n)}return o.toTensor()}stridedSlice(t,e,n,o){ql(t,"stridedSlice");const s=r.computeOutShape(e,n,o);if(s.some(t=>0===t))return he.a([],s);const a=D(s,t.dtype),i=this.bufferSync(t);for(let t=0;t<a.size;t++){const n=a.indexToLoc(t),r=new Array(n.length);for(let t=0;t<r.length;t++)r[t]=n[t]*o[t]+e[t];a.set(i.get(...r),...n)}return a.toTensor()}diag(t){const e=this.readSync(t.dataId),n=D([t.size,t.size],t.dtype),r=n.values;for(let n=0;n<e.length;n++)r[n*t.size+n]=e[n];return n.toTensor()}unstack(t,e){const n=t.shape[e],r=new Array(t.rank-1);let o=0;for(let n=0;n<t.rank;n++)n!==e&&(r[o++]=t.shape[n]);const s=new Array(t.rank).fill(0),a=t.shape.slice();a[e]=1;const i=new Array(n);for(let n=0;n<i.length;n++)s[e]=n,i[n]=this.slice(t,s,a).reshape(r);return i}reverse(t,e){ql(t,"reverse");const n=D(t.shape,t.dtype),r=this.bufferSync(t);for(let o=0;o<n.size;o++){const s=n.indexToLoc(o),a=s.slice();e.forEach(e=>a[e]=t.shape[e]-1-a[e]),n.set(r.get(...a),...s)}return n.toTensor()}concat(t,e){if("complex64"===t[0].dtype){const n=t.map(t=>se(t)),r=t.map(t=>Mt(t));return pt.a(this.concat(n,e),this.concat(r,e))}const n=t.map(t=>{const n=y.sizeFromShape(t.shape.slice(e));return t.as2D(-1,n)}),r=s.computeOutShape(n.map(t=>t.shape),1),o=D(r,t[0].dtype).values;if(1===n[0].shape[0]){let t=0;n.forEach(e=>{o.set(this.readSync(e.dataId),t),t+=e.size})}else{let t=0;n.forEach(e=>{const n=this.readSync(e.dataId);let s=0;for(let a=0;a<e.shape[0];++a){const i=a*r[1]+t;for(let t=0;t<e.shape[1];++t)o[i+t]=n[s++]}t+=e.shape[1]})}const a=s.computeOutShape(t.map(t=>t.shape),e);return he.a(o,a,t[0].dtype)}neg(t){return ql(t,"neg"),this.multiply(ie(-1),t)}add(t,e){return"complex64"===t.dtype||"complex64"===e.dtype?this.broadcastedBinaryComplexOp(t.cast("complex64"),e.cast("complex64"),(t,e,n,r)=>({real:t+n,imag:e+r})):this.broadcastedBinaryOp(t,e,Object(lt.c)(t.dtype,e.dtype),(t,e)=>t+e)}addN(t){ql(t,"addN");const e=t.map(t=>this.readSync(t.dataId)),n=D(t[0].shape,t[0].dtype),r=n.values;for(let n=0;n<t.length;n++){const t=e[n];for(let e=0;e<r.length;e++)r[e]+=t[e]}return n.toTensor()}softmax(t,e){const n=y.parseAxisParam([e],t.shape),r=qt(t,n),o=s.expandShapeToKeepDim(r.shape,n),a=this.subtract(t,r.reshape(o)),i=this.exp(a),u=this.sum(i,n).reshape(o);return Tt(i,u)}subtract(t,e){return"complex64"===t.dtype||"complex64"===e.dtype?this.broadcastedBinaryComplexOp(t.cast("complex64"),e.cast("complex64"),(t,e,n,r)=>({real:t-n,imag:e-r})):this.broadcastedBinaryOp(t,e,Object(lt.c)(t.dtype,e.dtype),(t,e)=>t-e)}pow(t,e){return ql([t,e],"pow"),this.broadcastedBinaryOp(t,e,t.dtype,(t,e)=>Math.pow(t,e))}batchMatMul(t,e,n,r){ql([t,e],"matMul");const o=n?t.shape[1]:t.shape[2],s=n?t.shape[2]:t.shape[1],a=r?e.shape[1]:e.shape[2],i=t.shape[0],u=this.readSync(t.dataId),c=this.readSync(e.dataId),[l,h,d]=n?[t.strides[0],1,t.strides[1]]:[t.strides[0],t.strides[1],1],[p,f,g]=r?[1,e.strides[1],e.strides[0]]:[e.strides[1],1,e.strides[0]],m=s*a,b=D([i,s,a],t.dtype),x=b.values,y=this.blockSize;for(let t=0;t<i;t++)for(let e=0;e<s;e+=y)for(let n=0;n<a;n+=y)for(let r=0;r<o;r+=y){const i=Math.min(e+y,s),b=Math.min(n+y,a),v=Math.min(r+y,o);for(let o=e;o<i;o++)for(let e=n;e<b;e++){let n=0;for(let s=r;s<v;s++)n+=u[t*l+o*h+s*d]*c[s*p+e*f+t*g];x[t*m+(o*a+e)]+=n}}return b.toTensor()}fusedBatchMatMul({a:t,b:e,transposeA:n,transposeB:r,bias:o,activation:s,preluActivationWeights:a}){let i=this.batchMatMul(t,e,n,r);return o&&(i=this.add(i,o)),s&&(i=nh(this,i,s,a)),i}multiply(t,e){return"complex64"===t.dtype||"complex64"===e.dtype?this.broadcastedBinaryComplexOp(t.cast("complex64"),e.cast("complex64"),(t,e,n,r)=>({real:t*n-e*r,imag:t*r+e*n})):this.broadcastedBinaryOp(t,e,Object(lt.c)(t.dtype,e.dtype),(t,e)=>t*e)}floorDiv(t,e){ql([t,e],"floorDiv");return this.broadcastedBinaryOp(t,e,"int32",(t,e)=>Math.floor(t/e))}sum(t,e){ql(t,"sum"),s.assertAxesAreInnerMostDims("sum",e,t.rank);const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=re(n,Object(lt.c)(t.dtype,"int32")),a=y.sizeFromShape(r),i=this.readSync(o.dataId),u=this.readSync(t.dataId);for(let t=0;t<i.length;++t){const e=t*a;let n=0;for(let t=0;t<a;++t)n+=u[e+t];i[t]=n}return o}prod(t,e){ql(t,"sum");const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=re(n,Object(lt.c)(t.dtype,"int32")),a=y.sizeFromShape(r),i=this.readSync(o.dataId),u=this.readSync(t.dataId);for(let t=0;t<i.length;++t){const e=t*a;let n=1;for(let t=0;t<a;++t)n*=u[e+t];i[t]=n}return o}unsortedSegmentSum(t,e,n){ql(t,"unsortedSegmentSum");const r=[],o=t.rank-e.rank;for(let t=0;t<o;++t)e=e.expandDims(t+1);for(let o=0;o<n;++o){const n=ie(o,"int32"),s=jt(n,e).asType("float32").mul(t).sum(0);r.push(s)}return fe(r)}argMin(t,e){ql(t,"argMin");const n=[e];s.assertAxesAreInnerMostDims("argMin",n,t.rank);const[r,o]=s.computeOutAndReduceShapes(t.shape,n),a=re(r,"int32"),i=y.sizeFromShape(o),u=this.readSync(a.dataId),c=this.readSync(t.dataId);for(let t=0;t<u.length;++t){const e=t*i;let n=c[e],r=0;for(let t=0;t<i;++t){const o=c[e+t];o<n&&(n=o,r=t)}u[t]=r}return a}argMax(t,e){ql(t,"argMax");const n=[e];s.assertAxesAreInnerMostDims("argMax",n,t.rank);const[r,o]=s.computeOutAndReduceShapes(t.shape,n),a=re(r,"int32"),i=y.sizeFromShape(o),u=this.readSync(a.dataId),c=this.readSync(t.dataId);for(let t=0;t<u.length;++t){const e=t*i;let n=c[e],r=0;for(let t=0;t<i;++t){const o=c[e+t];o>n&&(n=o,r=t)}u[t]=r}return a}cumsum(t,e,n,r){if(ql(t,"cumsum"),e!==t.rank-1)throw new Error(`backend.cumsum in CPU expects an inner-most axis=${t.rank-1} but got axis=`+e);const o=Object(lt.c)(t.dtype,"int32"),s=re(t.shape,o),a=this.readSync(s.dataId),i=this.readSync(t.dataId),u=t.shape[t.rank-1],c=r?(t,e)=>t+u-e-1:(t,e)=>t+e;for(let t=0;t<i.length;t+=u)for(let e=0;e<u;e++){const r=c(t,e);if(0===e)a[r]=n?0:i[r];else{const o=c(t,e-1);a[r]=n?i[o]+a[o]:i[r]+a[o]}}return s}equal(t,e){return ql([t,e],"equal"),this.broadcastedBinaryOp(t,e,"bool",(t,e)=>t===e?1:0)}notEqual(t,e){return ql([t,e],"notEqual"),this.broadcastedBinaryOp(t,e,"bool",(t,e)=>t!==e?1:0)}less(t,e){return ql([t,e],"less"),this.broadcastedBinaryOp(t,e,"bool",(t,e)=>t<e?1:0)}lessEqual(t,e){return ql([t,e],"lessEqual"),this.broadcastedBinaryOp(t,e,"bool",(t,e)=>t<=e?1:0)}greater(t,e){return ql([t,e],"greater"),this.broadcastedBinaryOp(t,e,"bool",(t,e)=>t>e?1:0)}greaterEqual(t,e){return ql([t,e],"greaterEqual"),this.broadcastedBinaryOp(t,e,"bool",(t,e)=>t>=e?1:0)}logicalNot(t){ql(t,"logicalNot");const e=this.readSync(t.dataId),n=new Uint8Array(e.length);for(let t=0;t<e.length;++t)n[t]=e[t]?0:1;return this.makeOutput(n,t.shape,"bool")}logicalAnd(t,e){return ql([t,e],"logicalAnd"),this.broadcastedBinaryOp(t,e,"bool",(t,e)=>t&&e)}logicalOr(t,e){return ql([t,e],"logicalOr"),this.broadcastedBinaryOp(t,e,"bool",(t,e)=>t||e)}select(t,e,n){ql([t,e,n],"select");const r=this.readSync(t.dataId),o=this.readSync(e.dataId),s=this.readSync(n.dataId),a=re(e.shape,Object(lt.c)(e.dtype,n.dtype)),i=this.readSync(a.dataId);let u=0;const c=0===t.rank||t.rank>1||1===e.rank?1:y.sizeFromShape(e.shape.slice(1));for(let t=0;t<r.length;t++)for(let e=0;e<c;e++)1===r[t]?i[u++]=o[t]:i[u++]=s[t];return a}where(t){ql([t],"where");const e=this.readSync(t.dataId);return eh(t.shape,e)}topk(t,e,n){ql(t,"topk");const r=this.readSync(t.dataId);return th(r,t.shape,t.dtype,e,n)}min(t,e){ql(t,"min"),s.assertAxesAreInnerMostDims("min",e,t.rank);const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=re(n,t.dtype),a=y.sizeFromShape(r),i=this.readSync(o.dataId),u=this.readSync(t.dataId);for(let t=0;t<i.length;++t){const e=t*a;let n=u[e];for(let t=0;t<a;++t){const r=u[e+t];r<n&&(n=r)}i[t]=n}return o}minimum(t,e){return ql([t,e],"minimum"),this.broadcastedBinaryOp(t,e,t.dtype,(t,e)=>Math.min(t,e))}mod(t,e){return ql([t,e],"mod"),this.broadcastedBinaryOp(t,e,t.dtype,(t,e)=>{const n=t%e;return t<0&&e<0||t>=0&&e>=0?n:(n+e)%e})}maximum(t,e){return ql([t,e],"maximum"),this.broadcastedBinaryOp(t,e,t.dtype,(t,e)=>Math.max(t,e))}all(t,e){ql(t,"all"),s.assertAxesAreInnerMostDims("all",e,t.rank);const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=re(n,t.dtype),a=y.sizeFromShape(r),i=this.readSync(o.dataId),u=this.readSync(t.dataId);for(let t=0;t<i.length;++t){const e=t*a;let n=u[e];for(let t=0;t<a;++t){const r=u[e+t];n=n&&r}i[t]=n}return o}any(t,e){ql(t,"any"),s.assertAxesAreInnerMostDims("any",e,t.rank);const[n,r]=s.computeOutAndReduceShapes(t.shape,e),o=re(n,t.dtype),a=y.sizeFromShape(r),i=this.readSync(o.dataId),u=this.readSync(t.dataId);for(let t=0;t<i.length;++t){const e=t*a;let n=u[e];for(let t=0;t<a;++t){const r=u[e+t];n=n||r}i[t]=n}return o}squaredDifference(t,e){return ql([t,e],"squaredDifference"),this.broadcastedBinaryOp(t,e,t.dtype,(t,e)=>{const n=t-e;return n*n})}ceil(t){ql(t,"ceil");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t)n[t]=Math.ceil(e[t]);return this.makeOutput(n,t.shape,"float32")}floor(t){ql(t,"floor");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t)n[t]=Math.floor(e[t]);return this.makeOutput(n,t.shape,"float32")}sign(t){ql(t,"x");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t)e[t]<0?n[t]=-1:e[t]>0?n[t]=1:n[t]=0;return this.makeOutput(n,t.shape,"float32")}isNaN(t){ql(t,"x");const e=this.readSync(t.dataId),n=new Uint8Array(e.length);for(let t=0;t<e.length;++t)Number.isNaN(e[t])&&(n[t]=1);return this.makeOutput(n,t.shape,"bool")}isInf(t){ql(t,"x");const e=this.readSync(t.dataId),n=new Uint8Array(e.length);for(let t=0;t<e.length;++t)Math.abs(e[t])===1/0&&(n[t]=1);return this.makeOutput(n,t.shape,"bool")}isFinite(t){ql(t,"x");const e=this.readSync(t.dataId),n=new Uint8Array(e.length);for(let t=0;t<e.length;++t)Number.isFinite(e[t])&&(n[t]=1);return this.makeOutput(n,t.shape,"bool")}round(t){ql(t,"round");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t){const r=Math.floor(e[t]);e[t]-r<.5?n[t]=Math.floor(e[t]):e[t]-r>.5?n[t]=Math.ceil(e[t]):n[t]=r%2==0?r:r+1}return this.makeOutput(n,t.shape,"float32")}exp(t){ql(t,"exp");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t)n[t]=Math.exp(e[t]);return this.makeOutput(n,t.shape,"float32")}expm1(t){ql(t,"expm1");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t)n[t]=Math.expm1(e[t]);return this.makeOutput(n,t.shape,"float32")}log(t){ql(t,"log");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t){const r=e[t];n[t]=Math.log(r)}return this.makeOutput(n,t.shape,"float32")}log1p(t){ql(t,"log1p");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t){const r=e[t];n[t]=Math.log1p(r)}return this.makeOutput(n,t.shape,"float32")}sqrt(t){ql(t,"sqrt");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t){const r=e[t];n[t]=Math.sqrt(r)}return this.makeOutput(n,t.shape,"float32")}rsqrt(t){ql(t,"rsqrt");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t){const r=e[t];n[t]=1/Math.sqrt(r)}return this.makeOutput(n,t.shape,"float32")}reciprocal(t){ql(t,"reciprocal");const e=this.readSync(t.dataId),n=new Float32Array(e.length);for(let t=0;t<e.length;++t)n[t]=1/e[t];return this.makeOutput(n,t.shape,"float32")}linear(t){return t}relu(t){ql(t,"relu");const e=re(t.shape,t.dtype),n=this.readSync(e.dataId),r=this.readSync(t.dataId);for(let t=0;t<r.length;++t)n[t]=Math.max(0,r[t]);return e}relu6(t){ql(t,"relu");const e=re(t.shape,t.dtype),n=this.readSync(e.dataId),r=this.readSync(t.dataId);for(let t=0;t<r.length;++t)n[t]=Math.min(Math.max(0,r[t]),6);return e}prelu(t,e){return ql([t,e],"prelu"),this.broadcastedBinaryOp(t,e,t.dtype,(t,e)=>t<0?e*t:t)}elu(t){ql(t,"elu");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t){const r=n[t];e[t]=r>=0?r:Math.exp(r)-1}return this.makeOutput(e,t.shape,"float32")}eluDer(t,e){ql([t,e],"eluDer");const n=new Float32Array(e.size),r=this.readSync(e.dataId),o=this.readSync(t.dataId);for(let t=0;t<r.length;++t){const e=r[t];n[t]=e>=1?o[t]:o[t]*(e+1)}return this.makeOutput(n,e.shape,"float32")}selu(t){ql(t,"selu");const e=s.SELU_SCALEALPHA,n=s.SELU_SCALE,r=new Float32Array(t.size),o=this.readSync(t.dataId);for(let t=0;t<o.length;++t){const s=o[t];r[t]=s>=0?n*s:e*(Math.exp(s)-1)}return this.makeOutput(r,t.shape,"float32")}clip(t,e,n){ql(t,"clip");const r=new Float32Array(t.size),o=this.readSync(t.dataId);for(let t=0;t<o.length;++t){const s=o[t];r[t]=s>n?n:s<e?e:s}return this.makeOutput(r,t.shape,t.dtype)}abs(t){const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.abs(n[t]);return this.makeOutput(e,t.shape,"float32")}complexAbs(t){const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let r=0;r<t.size;++r){const t=n[2*r],o=n[2*r+1];e[r]=Math.hypot(t,o)}return this.makeOutput(e,t.shape,"float32")}int(t){ql(t,"int");const e=new Int32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=n[t];return this.makeOutput(e,t.shape,"int32")}sigmoid(t){ql(t,"sigmoid");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=1/(1+Math.exp(-n[t]));return this.makeOutput(e,t.shape,"float32")}softplus(t){ql(t,"softplus");const e=Math.log(1.1920928955078125e-7)+2,n=new Float32Array(t.size),r=this.readSync(t.dataId);for(let t=0;t<r.length;++t){const o=r[t]>-e,s=r[t]<e,a=Math.exp(r[t]);let i;i=s?a:o?r[t]:Math.log(1+a),n[t]=i}return this.makeOutput(n,t.shape,"float32")}sin(t){ql(t,"sin");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.sin(n[t]);return this.makeOutput(e,t.shape,"float32")}tan(t){ql(t,"tan");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.tan(n[t]);return this.makeOutput(e,t.shape,"float32")}asin(t){ql(t,"asin");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.asin(n[t]);return this.makeOutput(e,t.shape,"float32")}acos(t){ql(t,"acos");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.acos(n[t]);return this.makeOutput(e,t.shape,"float32")}atan(t){ql(t,"atan");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.atan(n[t]);return this.makeOutput(e,t.shape,"float32")}atan2(t,e){return ql([t,e],"atan2"),this.broadcastedBinaryOp(t,e,t.dtype,(t,e)=>Math.atan2(t,e))}sinh(t){ql(t,"sinh");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.sinh(n[t]);return this.makeOutput(e,t.shape,"float32")}cosh(t){ql(t,"cosh");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.cosh(n[t]);return this.makeOutput(e,t.shape,"float32")}tanh(t){ql(t,"tanh");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=y.tanh(n[t]);return this.makeOutput(e,t.shape,"float32")}asinh(t){ql(t,"asinh");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.asinh(n[t]);return this.makeOutput(e,t.shape,"float32")}acosh(t){ql(t,"acosh");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.acosh(n[t]);return this.makeOutput(e,t.shape,"float32")}atanh(t){ql(t,"atanh");const e=new Float32Array(t.size),n=this.readSync(t.dataId);for(let t=0;t<n.length;++t)e[t]=Math.atanh(n[t]);return this.makeOutput(e,t.shape,"float32")}erf(t){ql(t,"erf");const e=new Float32Array(t.size),n=this.readSync(t.dataId),r=s.ERF_P,o=s.ERF_A1,a=s.ERF_A2,i=s.ERF_A3,u=s.ERF_A4,c=s.ERF_A5;for(let t=0;t<n.length;++t){const s=Math.sign(n[t]),l=Math.abs(n[t]),h=1/(1+r*l);e[t]=s*(1-((((c*h+u)*h+i)*h+a)*h+o)*h*Math.exp(-l*l))}return this.makeOutput(e,t.shape,"float32")}step(t,e=0){ql(t,"step");const n=new Float32Array(t.size),r=this.readSync(t.dataId);for(let t=0;t<r.length;++t){const o=r[t];isNaN(o)?n[t]=NaN:n[t]=o>0?1:e}return this.makeOutput(n,t.shape,"float32")}fusedConv2d({input:t,filter:e,convInfo:n,bias:r,activation:o,preluActivationWeights:s}){let a=this.conv2d(t,e,n);return r&&(a=this.add(a,r)),o&&(a=nh(this,a,o,s)),a}conv2d(t,e,n){ql([t,e],"conv2d");const r=n.filterHeight,o=n.filterWidth,s=n.dilationHeight,a=n.dilationWidth,i=n.padInfo.left,u=n.padInfo.top,c="channelsLast"===n.dataFormat,l=D(n.outShape,t.dtype),h=t.strides[0],d=c?t.strides[1]:t.strides[2],p=c?t.strides[2]:1,f=c?1:t.strides[1],g=l.strides[0],m=c?l.strides[1]:l.strides[2],b=c?l.strides[2]:1,x=c?1:l.strides[1],y=this.readSync(t.dataId),v=this.readSync(e.dataId),w=l.values;for(let t=0;t<n.batchSize;++t){const c=t*h,l=t*g;for(let t=0;t<n.outHeight;++t){const h=l+t*m,g=t*n.strideHeight-u;for(let t=0;t<r;t++){const r=g+t*s;if(r<0||r>=n.inHeight)continue;const u=t*e.strides[0],l=c+r*d;for(let t=0;t<n.outWidth;++t){const r=h+t*b,s=t*n.strideWidth-i;for(let t=0;t<o;t++){const o=s+t*a;if(o<0||o>=n.inWidth)continue;const i=l+o*p;let c=u+t*e.strides[1];for(let t=0;t<n.inChannels;++t){const e=y[i+t*f];for(let t=0;t<n.outChannels;++t)w[r+t*x]+=e*v[c+t];c+=n.outChannels}}}}}}return l.toTensor()}conv3d(t,e,n){const r=n.filterDepth,o=n.filterHeight,s=n.filterWidth,a=n.dilationDepth,i=n.dilationHeight,u=n.dilationWidth,c=n.padInfo.front,l=n.padInfo.left,h=n.padInfo.top,d=D(n.outShape,t.dtype),p=this.readSync(t.dataId),f=this.readSync(e.dataId),g=d.values;for(let m=0;m<n.batchSize;++m){const b=m*t.strides[0],x=m*d.strides[0];for(let m=0;m<n.outDepth;++m){const y=x+m*d.strides[1],v=m*n.strideDepth-c;for(let c=0;c<r;c++){const r=v+c*a;if(r<0||r>=n.inDepth)continue;const m=c*e.strides[0],x=b+r*t.strides[1];for(let r=0;r<n.outHeight;++r){const a=y+r*d.strides[2],c=r*n.strideHeight-h;for(let r=0;r<o;r++){const o=c+r*i;if(o<0||o>=n.inHeight)continue;const h=m+r*e.strides[1],d=x+o*t.strides[2];for(let t=0;t<n.outWidth;++t){const r=a+t*n.outChannels,o=t*n.strideWidth-l;for(let t=0;t<s;t++){const s=o+t*u;if(s<0||s>=n.inWidth)continue;const a=h+t*e.strides[2],i=d+s*n.inChannels;let c=a;for(let t=0;t<n.inChannels;++t){const e=p[i+t];for(let t=0;t<n.outChannels;++t)g[r+t]+=e*f[c+t];c+=n.outChannels}}}}}}}}return d.toTensor()}conv2dDerInput(t,e,n){ql([t,e],"conv2dDerInput");const r=D(n.inShape,"float32"),o=r.values,s=this.readSync(t.dataId),a=this.readSync(e.dataId),[i,u,c]=e.strides,{batchSize:l,filterHeight:h,filterWidth:d,inChannels:p,inHeight:f,inWidth:g,outChannels:m,outHeight:b,outWidth:x,strideHeight:y,strideWidth:v,dataFormat:w}=n,C=h-1-n.padInfo.top,$=d-1-n.padInfo.left,O="channelsLast"===w,I=r.strides[0],S=O?r.strides[1]:r.strides[2],E=O?r.strides[2]:1,R=O?1:r.strides[1],A=t.strides[0],k=O?t.strides[1]:t.strides[2],T=O?t.strides[2]:1,F=O?1:t.strides[1];for(let t=0;t<l;++t)for(let e=0;e<p;++e)for(let n=0;n<f;++n){const r=n-C,l=Math.max(0,Math.ceil(r/y)),p=Math.min(b,(h+r)/y);for(let f=0;f<g;++f){const g=f-$,b=Math.max(0,Math.ceil(g/v)),w=Math.min(x,(d+g)/v);let C=0;for(let n=l;n<p;++n){const o=n*y-r;for(let r=b;r<w;++r){const l=A*t+k*n+T*r,p=i*(h-1-o)+u*(d-1-(r*v-g))+c*e;for(let t=0;t<m;++t){C+=s[l+F*t]*a[p+t]}}}o[I*t+S*n+E*f+R*e]=C}}return r.toTensor()}conv3dDerInput(t,e,n){const r=D(n.inShape,"float32"),o=r.values,[s,a,i,u]=r.strides,c=this.readSync(t.dataId),[l,h,d,p]=t.strides,f=this.readSync(e.dataId),[g,m,b,x]=e.strides,{batchSize:y,filterDepth:v,filterHeight:w,filterWidth:C,inChannels:$,inDepth:O,inHeight:I,inWidth:S,outChannels:E,outDepth:R,outHeight:A,outWidth:k,strideDepth:T,strideHeight:F,strideWidth:N}=n,_=v-1-n.padInfo.front,B=w-1-n.padInfo.top,j=C-1-n.padInfo.left;for(let t=0;t<y;++t)for(let e=0;e<$;++e)for(let n=0;n<O;++n){const r=n-_,y=Math.max(0,Math.ceil(r/T)),$=Math.min(R,(v+r)/T);for(let O=0;O<I;++O){const I=O-B,R=Math.max(0,Math.ceil(I/F)),D=Math.min(A,(w+I)/F);for(let A=0;A<S;++A){const S=A-j,_=Math.max(0,Math.ceil(S/N)),B=Math.min(k,(C+S)/N);let M=0;for(let n=y;n<$;++n){const o=n*T-r;for(let r=R;r<D;++r){const s=r*F-I;for(let a=_;a<B;++a){const i=l*t+h*n+d*r+p*a,u=g*(v-1-o)+m*(w-1-s)+b*(C-1-(a*N-S))+x*e;for(let t=0;t<E;++t){M+=c[i+t]*f[u+t]}}}}o[s*t+a*n+i*O+u*A+e]=M}}}return r.toTensor()}conv2dDerFilter(t,e,n){ql([t,e],"conv2dDerFilter");const r=n.strideHeight,o=n.strideWidth,s=n.filterHeight,a=n.filterWidth,i="channelsLast"===n.dataFormat,u=D(n.filterShape,"float32"),c=n.padInfo.left,l=n.padInfo.top,h=this.bufferSync(t),d=this.bufferSync(e);for(let t=0;t<s;++t){const e=Math.max(0,Math.ceil((l-t)/r)),s=Math.min(n.outHeight,(n.inHeight+l-t)/r);for(let p=0;p<a;++p){const a=Math.max(0,Math.ceil((c-p)/o)),f=Math.min(n.outWidth,(n.inWidth+c-p)/o);for(let g=0;g<n.inChannels;++g)for(let m=0;m<n.outChannels;++m){let b=0;for(let u=0;u<n.batchSize;++u)for(let n=e;n<s;++n){const e=t+n*r-l;for(let t=a;t<f;++t){const r=p+t*o-c;b+=i?h.get(u,e,r,g)*d.get(u,n,t,m):h.get(u,g,e,r)*d.get(u,m,n,t)}}u.set(b,t,p,g,m)}}}return u.toTensor()}conv3dDerFilter(t,e,n){const r=n.strideDepth,o=n.strideHeight,s=n.strideWidth,a=n.filterDepth,i=n.filterHeight,u=n.filterWidth,c=D(n.filterShape,"float32"),l=c.values,[h,d,p,f]=c.strides,g=this.readSync(e.dataId),[m,b,x,y]=e.strides,v=this.readSync(t.dataId),[w,C,$,O]=t.strides,I=n.padInfo.front,S=n.padInfo.left,E=n.padInfo.top;for(let t=0;t<a;++t){const e=Math.max(0,Math.ceil((I-t)/r)),a=Math.min(n.outDepth,(n.inDepth+I-t)/r),c=t*h;for(let h=0;h<i;++h){const i=Math.max(0,Math.ceil((E-h)/o)),R=Math.min(n.outHeight,(n.inHeight+E-h)/o),A=h*d+c;for(let c=0;c<u;++c){const u=Math.max(0,Math.ceil((S-c)/s)),d=Math.min(n.outWidth,(n.inWidth+S-c)/s),k=c*p+A;for(let p=0;p<n.inChannels;++p){const A=p*f+k;for(let f=0;f<n.outChannels;++f){let k=0;for(let l=0;l<n.batchSize;++l){const n=l*w,A=l*m;for(let l=e;l<a;++l){const e=(t+l*r-I)*C+n,a=l*b+A;for(let t=i;t<R;++t){const n=(h+t*o-E)*$+e,r=t*x+a;for(let t=u;t<d;++t){const e=t*y+r;k+=v[(c+t*s-S)*O+n+p]*g[e+f]}}}}l[A+f]=k}}}}}return c.toTensor()}fusedDepthwiseConv2D({input:t,filter:e,convInfo:n,bias:r,activation:o,preluActivationWeights:s}){let a=this.depthwiseConv2D(t,e,n);return r&&(a=this.add(a,r)),o&&(a=nh(this,a,o,s)),a}depthwiseConv2D(t,e,n){ql([t,e],"depthwiseConv2D");const r=n.filterHeight,o=n.filterWidth,s=n.dilationHeight,a=n.dilationWidth,i=n.padInfo.left,u=n.padInfo.top,c=n.outChannels/n.inChannels,l=D(n.outShape,t.dtype),h=this.readSync(t.dataId),d=this.readSync(e.dataId),p=l.values;for(let f=0;f<n.batchSize;++f){const g=f*t.strides[0],m=f*l.strides[0];for(let f=0;f<n.outHeight;++f){const b=m+f*l.strides[1],x=f*n.strideHeight-i;for(let i=0;i<r;++i){const r=x+i*s;if(r<0||r>=n.inHeight)continue;const f=i*e.strides[0],m=g+r*t.strides[1];for(let t=0;t<n.outWidth;++t){const r=b+t*l.strides[2],s=t*n.strideWidth-u;for(let t=0;t<o;++t){const o=s+t*a;if(o<0||o>=n.inWidth)continue;const i=f+t*e.strides[1],u=m+o*n.inChannels;let l=r,g=i;for(let t=0;t<n.inChannels;++t){const e=h[u+t];for(let t=0;t<c;++t)p[l+t]+=e*d[g+t];l+=c,g+=c}}}}}}return l.toTensor()}depthwiseConv2DDerInput(t,e,n){ql([t,e],"depthwiseConv2DDerInput");const r=D(n.inShape,"float32"),o=r.values,[s,a,i]=r.strides,u=this.readSync(t.dataId),[c,l,h]=t.strides,d=this.readSync(e.dataId),[p,f,g]=e.strides,{batchSize:m,filterHeight:b,filterWidth:x,inChannels:y,inHeight:v,inWidth:w,outChannels:C,outHeight:$,outWidth:O,strideHeight:I,strideWidth:S}=n,E=b-1-n.padInfo.top,R=x-1-n.padInfo.left,A=C/y;for(let t=0;t<m;++t)for(let e=0;e<y;++e)for(let n=0;n<v;++n){const r=n-E,m=Math.max(0,Math.ceil(r/I)),y=Math.min($,(b+r)/I);for(let v=0;v<w;++v){const w=v-R,C=Math.max(0,Math.ceil(w/S)),$=Math.min(O,(x+w)/S);let E=0;for(let n=m;n<y;++n){const o=n*I-r;for(let r=C;r<$;++r){const s=c*t+l*n+h*r,a=p*(b-1-o)+f*(x-1-(r*S-w))+g*e;for(let t=0;t<A;++t){E+=u[s+(e*A+t)]*d[a+t]}}}o[s*t+a*n+i*v+e]=E}}return r.toTensor()}depthwiseConv2DDerFilter(t,e,n){ql([t,e],"depthwiseConv2DDerFilter");const r=n.strideHeight,o=n.strideWidth,s=n.filterHeight,a=n.filterWidth,i=D(n.filterShape,"float32"),u=n.padInfo.left,c=n.padInfo.top,l=n.outChannels/n.inChannels,h=this.bufferSync(t),d=this.bufferSync(e);for(let t=0;t<s;++t){const e=Math.max(0,Math.ceil((c-t)/r)),s=Math.min(n.outHeight,(n.inHeight+c-t)/r);for(let p=0;p<a;++p){const a=Math.max(0,Math.ceil((u-p)/o)),f=Math.min(n.outWidth,(n.inWidth+u-p)/o);for(let g=0;g<n.outChannels;++g){const m=Math.trunc(g/l),b=g%l;let x=0;for(let i=0;i<n.batchSize;++i)for(let n=e;n<s;++n){const e=t+n*r-c;for(let t=a;t<f;++t){const r=p+t*o-u;x+=h.get(i,e,r,m)*d.get(i,n,t,g)}}i.set(x,t,p,m,b)}}}return i.toTensor()}tile(t,e){return ql(t,"tile"),Jl(this.bufferSync(t),e)}gather(t,e,n){ql([t,e],"gather");const r=t.shape.slice(),o=this.readSync(e.dataId);r[n]=o.length;const s=D(r,t.dtype),a=this.bufferSync(t);for(let t=0;t<s.size;++t){const e=s.indexToLoc(t),r=e.slice();r[n]=o[e[n]];const i=a.locToIndex(r);s.values[t]=a.values[i]}return s.toTensor()}batchToSpaceND(t,e,n){ql([t],"batchToSpaceND");const r=e.reduce((t,e)=>t*e),o=s.getReshaped(t.shape,e,r),a=s.getPermuted(o.length,e.length),i=s.getReshapedPermuted(t.shape,e,r),u=s.getSliceBeginCoords(n,e.length),c=s.getSliceSize(i,n,e.length);return Kt(t.reshape(o),a).reshape(i).slice(u,c)}maxPool(t,e){ql(t,"maxPool");return Xl(this.readSync(t.dataId),t.shape,t.dtype,t.strides,e,"max").toTensor()}maxPoolBackprop(t,e,n,r){ql([e,n],"maxPoolBackprop");const o=this.readSync(e.dataId),s=D(r.outShape,e.dtype,Yl(o,e.shape,e.dtype,r).values),a=r.strideHeight,i=r.strideWidth,u=r.dilationHeight,c=r.dilationWidth,l=r.effectiveFilterHeight,h=r.effectiveFilterWidth,d=h-1-r.padInfo.left,p=l-1-r.padInfo.top,f=D(e.shape,"float32"),g=this.bufferSync(t);for(let t=0;t<r.batchSize;++t)for(let e=0;e<r.inChannels;++e)for(let n=0;n<r.inHeight;++n)for(let o=0;o<r.inWidth;++o){const m=n-p,b=o-d;let x=0;for(let n=0;n<l;n+=u){const o=(m+n)/a;if(!(o<0||o>=r.outHeight||Math.floor(o)!==o))for(let a=0;a<h;a+=c){const u=(b+a)/i;if(u<0||u>=r.outWidth||Math.floor(u)!==u)continue;const c=l*h-1-s.get(t,o,u,e)===n*h+a?1:0;if(0===c)continue;x+=g.get(t,o,u,e)*c}}f.set(x,t,n,o,e)}return f.toTensor()}avgPoolBackprop(t,e,n){ql([t,e],"avgPoolBackprop");const r=n.strideHeight,o=n.strideWidth,s=n.filterHeight,a=n.filterWidth,i=n.dilationHeight,u=n.dilationWidth,c=n.effectiveFilterHeight,l=n.effectiveFilterWidth,h=l-1-n.padInfo.left,d=c-1-n.padInfo.top,p=D(e.shape,"float32"),f=1/(s*a),g=this.bufferSync(t);for(let t=0;t<n.batchSize;++t)for(let e=0;e<n.inChannels;++e)for(let s=0;s<n.inHeight;++s)for(let a=0;a<n.inWidth;++a){const m=s-d,b=a-h;let x=0;for(let s=0;s<c;s+=i){const a=(m+s)/r;if(!(a<0||a>=n.outHeight||Math.floor(a)!==a))for(let r=0;r<l;r+=u){const s=(b+r)/o;if(s<0||s>=n.outWidth||Math.floor(s)!==s)continue;x+=g.get(t,a,s,e)}}p.set(x*f,t,s,a,e)}return p.toTensor()}pool3d(t,e,n){ql(t,"pool3d");const r=e.strideDepth,o=e.strideHeight,s=e.strideWidth,a=e.dilationDepth,i=e.dilationHeight,u=e.dilationWidth,c=e.effectiveFilterDepth,l=e.effectiveFilterHeight,h=e.effectiveFilterWidth,d=e.padInfo.front,p=e.padInfo.top,f=e.padInfo.left,g="max"===n?Number.NEGATIVE_INFINITY:Number.POSITIVE_INFINITY,m=this.readSync(t.dataId),b=D(e.outShape,t.dtype),x=b.values,y=e.outShape[1]*e.outShape[2]*e.outShape[3]*e.outShape[4],v=e.outShape[2]*e.outShape[3]*e.outShape[4],w=e.outShape[3]*e.outShape[4],C=e.outShape[4];for(let b=0;b<e.batchSize;++b){const $=b*y,O=b*t.strides[0];for(let b=0;b<e.inChannels;++b)for(let y=0;y<e.outDepth;++y){const I=y*r-d;let S=I;for(;S<0;)S+=a;const E=Math.min(e.inDepth,c+I),R=$+y*v;for(let r=0;r<e.outHeight;++r){const c=r*o-p;let d=c;for(;d<0;)d+=i;const y=Math.min(e.inHeight,l+c),v=R+r*w;for(let r=0;r<e.outWidth;++r){const o=r*s-f;let c=o;for(;c<0;)c+=u;const l=Math.min(e.inWidth,h+o),p=v+r*C;let w=g,$=0,I=0;for(let e=S;e<E;e+=a){const r=O+e*t.strides[1];for(let e=d;e<y;e+=i){const o=r+e*t.strides[2];for(let e=c;e<l;e+=u){const r=m[o+e*t.strides[3]+b];if("max"===n&&r>w?w=r:"avg"===n&&($+=r,I++),isNaN(w))break}if(isNaN(w))break}if(isNaN(w))break}x[p+b]="avg"===n?$/I:w}}}}return b.toTensor()}avgPool3d(t,e){return ql(t,"avgPool3d"),this.pool3d(t,e,"avg").toFloat()}avgPool3dBackprop(t,e,n){ql([t,e],"avgPool3dBackprop");const r=n.strideDepth,o=n.strideHeight,s=n.strideWidth,a=n.filterDepth,i=n.filterHeight,u=n.filterWidth,c=n.dilationDepth,l=n.dilationHeight,h=n.dilationWidth,d=n.effectiveFilterDepth,p=n.effectiveFilterHeight,f=n.effectiveFilterWidth,g=d-1-n.padInfo.front,m=f-1-n.padInfo.left,b=p-1-n.padInfo.top,x=D(e.shape,"float32"),y=1/(a*i*u),v=this.bufferSync(t);for(let t=0;t<n.batchSize;++t)for(let e=0;e<n.inChannels;++e)for(let a=0;a<n.inDepth;++a)for(let i=0;i<n.inHeight;++i)for(let u=0;u<n.inWidth;++u){const w=a-g,C=i-b,$=u-m;let O=0;for(let a=0;a<d;a+=c){const i=(w+a)/r;if(!(i<0||i>=n.outDepth||Math.floor(i)!==i))for(let r=0;r<p;r+=l){const a=(C+r)/o;if(!(a<0||a>=n.outHeight||Math.floor(a)!==a))for(let r=0;r<f;r+=h){const o=($+r)/s;if(o<0||o>=n.outWidth||Math.floor(o)!==o)continue;O+=v.get(t,i,a,o,e)}}}x.set(O*y,t,a,i,u,e)}return x.toTensor()}maxPool3d(t,e){return ql(t,"maxPool3d"),this.pool3d(t,e,"max").toFloat()}maxPool3dPositions(t,e){const n=D(e.outShape,"int32"),r=e.strideDepth,o=e.strideHeight,s=e.strideWidth,a=e.dilationDepth,i=e.dilationHeight,u=e.dilationWidth,c=e.effectiveFilterDepth,l=e.effectiveFilterHeight,h=e.effectiveFilterWidth,d=e.padInfo.front,p=e.padInfo.top,f=e.padInfo.left,g=this.bufferSync(t);for(let t=0;t<e.batchSize;++t)for(let m=0;m<e.inChannels;++m)for(let b=0;b<e.outDepth;++b){const x=b*r-d;let y=x;for(;y<0;)y+=a;const v=Math.min(e.inDepth,c+x);for(let r=0;r<e.outHeight;++r){const c=r*o-p;let d=c;for(;d<0;)d+=i;const w=Math.min(e.inHeight,l+c);for(let o=0;o<e.outWidth;++o){const p=o*s-f;let C=p;for(;C<0;)C+=u;const $=Math.min(e.inWidth,h+p);let O=Number.NEGATIVE_INFINITY,I=-1;for(let e=y;e<v;e+=a){const n=e-x;for(let r=d;r<w;r+=i){const o=r-c;for(let s=C;s<$;s+=u){const a=s-p,i=g.get(t,e,r,s,m);i>=O&&(O=i,I=n*l*h+o*l+a)}}}n.set(I,t,b,r,o,m)}}}return n.toTensor()}maxPool3dBackprop(t,e,n,r){ql([e,n],"maxPool3dBackprop");const o=this.maxPool3dPositions(e,r),s=r.strideDepth,a=r.strideHeight,i=r.strideWidth,u=r.dilationDepth,c=r.dilationHeight,l=r.dilationWidth,h=r.effectiveFilterDepth,d=r.effectiveFilterHeight,p=r.effectiveFilterWidth,f=h-1-r.padInfo.front,g=p-1-r.padInfo.left,m=d-1-r.padInfo.top,b=D(e.shape,"float32"),x=this.bufferSync(o),y=this.bufferSync(t);for(let t=0;t<r.batchSize;++t)for(let e=0;e<r.inChannels;++e)for(let n=0;n<r.inDepth;++n)for(let o=0;o<r.inHeight;++o)for(let v=0;v<r.inWidth;++v){const w=n-f,C=o-m,$=v-g;let O=0;for(let n=0;n<h;n+=u){const o=(w+n)/s;if(!(o<0||o>=r.outDepth||Math.floor(o)!==o))for(let s=0;s<d;s+=c){const u=(C+s)/a;if(!(u<0||u>=r.outHeight||Math.floor(u)!==u))for(let a=0;a<p;a+=l){const c=($+a)/i;if(c<0||c>=r.outWidth||Math.floor(c)!==c)continue;const l=h*d*p-1-x.get(t,o,u,c,e)===n*d*p+s*p+a?1:0;if(0===l)continue;O+=y.get(t,o,u,c,e)*l}}}b.set(O,t,n,o,v,e)}return b.toTensor()}cast(t,e){return s.castTensor(t,e,this)}avgPool(t,e){ql(t,"avgPool"),ql(t,"maxPool");return Xl(this.readSync(t.dataId),t.shape,t.dtype,t.strides,e,"avg").toTensor().toFloat()}resizeBilinear(t,e,n,r){ql(t,"resizeBilinear");const[o,s,a,i]=t.shape,u=this.readSync(t.dataId),c=new Float32Array(y.sizeFromShape([o,e,n,i])),l=[r&&e>1?s-1:s,r&&n>1?a-1:a],h=[r&&e>1?e-1:e,r&&n>1?n-1:n];let d=0;const p=l[0]/h[0],f=l[1]/h[1];for(let r=0;r<o;r++)for(let o=0;o<e;o++){const e=p*o,l=Math.floor(e),h=e-l,g=Math.min(s-1,Math.ceil(e)),m=r*t.strides[0]+l*t.strides[1],b=r*t.strides[0]+g*t.strides[1];for(let e=0;e<n;e++){const n=f*e,r=Math.floor(n),o=n-r,s=Math.min(a-1,Math.ceil(n)),l=m+r*t.strides[2],p=b+r*t.strides[2],g=m+s*t.strides[2],x=b+s*t.strides[2];for(let t=0;t<i;t++){const e=u[l+t],n=u[p+t],r=e+(u[g+t]-e)*o,s=r+(n+(u[x+t]-n)*o-r)*h;c[d++]=s}}}return he.a(c,[o,e,n,i])}resizeBilinearBackprop(t,e,n){ql([t,e],"resizeBilinearBackprop");const[r,o,s,a]=e.shape,[,i,u]=t.shape,c=new Float32Array(r*o*s*a),l=[n&&i>1?o-1:o,n&&u>1?s-1:s],h=[n&&i>1?i-1:i,n&&u>1?u-1:u],d=l[0]/h[0],p=l[1]/h[1],f=this.readSync(t.dataId);let g=0;for(let t=0;t<r;t++){const n=t*e.strides[0];for(let t=0;t<i;t++){const r=t*d,i=Math.floor(r),l=Math.min(Math.ceil(r),o-1),h=n+i*e.strides[1],m=n+l*e.strides[1],b=r-i,x=1-b;for(let t=0;t<u;t++){const n=t*p,r=Math.floor(n),o=Math.min(Math.ceil(n),s-1),i=n-r,u=1-i,l=h+r*e.strides[2],d=h+o*e.strides[2],y=m+r*e.strides[2],v=m+o*e.strides[2],w=x*u,C=x*i,$=b*u,O=b*i;for(let t=0;t<a;t++){const e=f[g++];c[l+t]+=e*w,c[d+t]+=e*C,c[y+t]+=e*$,c[v+t]+=e*O}}}}return me(c,[r,s,o,a],e.dtype)}resizeNearestNeighbor(t,e,n,r){ql(t,"resizeNearestNeighbor");const[o,s,a,i]=t.shape,u=this.readSync(t.dataId),c=new Float32Array(o*e*n*i),l=[r&&e>1?s-1:s,r&&n>1?a-1:a],h=[r&&e>1?e-1:e,r&&n>1?n-1:n],d=l[0]/h[0],p=l[1]/h[1];let f=0;for(let l=0;l<o;l++){const o=l*t.strides[0];for(let l=0;l<e;l++){const e=d*l,h=o+Math.min(s-1,r?Math.round(e):Math.floor(e))*t.strides[1];for(let e=0;e<n;e++){const n=p*e,o=h+Math.min(a-1,r?Math.round(n):Math.floor(n))*t.strides[2];for(let t=0;t<i;t++){const e=u[o+t];c[f++]=e}}}}return he.a(c,[o,e,n,i],t.dtype)}resizeNearestNeighborBackprop(t,e,n){ql([t,e],"resizeNearestNeighborBackprop");const[r,o,s,a]=e.shape,[,i,u]=t.shape,c=new Float32Array(r*o*s*a),l=this.readSync(t.dataId),h=[n&&i>1?o-1:o,n&&u>1?s-1:s],d=[n&&i>1?i-1:i,n&&u>1?u-1:u],p=h[0]/d[0],f=h[1]/d[1],g=1/p,m=1/f,b=2*Math.ceil(g)+2,x=2*Math.ceil(m)+2;for(let h=0;h<r;h++){const r=h*e.strides[0];for(let h=0;h<o;h++){const d=r+h*e.strides[1],y=Math.floor(h*g),v=Math.floor(y-b/2);for(let g=0;g<s;g++){const y=d+g*e.strides[2],w=Math.floor(g*m),C=Math.floor(w-x/2);for(let e=0;e<a;e++){let a=0;for(let c=0;c<b;c++){const d=c+v;if(d<0||d>=i)continue;const m=r+d*t.strides[1],b=d*p;if(h===Math.min(o-1,n?Math.round(b):Math.floor(b)))for(let r=0;r<x;r++){const o=r+C;if(o<0||o>=u)continue;const i=m+o*t.strides[2],c=o*f;g===Math.min(s-1,n?Math.round(c):Math.floor(c))&&(a+=l[i+e])}}c[y+e]=a}}}}return me(c,e.shape,e.dtype)}batchNorm(t,e,n,r,o,s){ql([t,e,n,o,r],"batchNorm");const a=this.readSync(t.dataId),i=this.readSync(e.dataId),u=this.readSync(n.dataId),c=o?this.readSync(o.dataId):new Float32Array([1]),l=r?this.readSync(r.dataId):new Float32Array([0]),h=new Float32Array(a.length),d=l.length,p=c.length,f=u.length,g=i.length;let m=0,b=0,x=0,y=0;for(let t=0;t<a.length;++t)h[t]=l[m++]+(a[t]-i[b++])*c[x++]/Math.sqrt(u[y++]+s),m>=d&&(m=0),b>=g&&(b=0),x>=p&&(x=0),y>=f&&(y=0);return me(h,t.shape)}localResponseNormalization4D(t,e,n,r,o){ql(t,"localResponseNormalization4D");const s=t.shape[3],a=s-1,i=this.readSync(t.dataId),u=t.size,c=new Float32Array(u);function l(t){const n=t%s;let r=t-n+Math.max(0,n-e);const o=t-n+Math.min(n+e,a);let u=0;for(;r<=o;r++){const t=i[r];u+=t*t}return u}for(let t=0;t<u;t++){const e=l(t),s=i[t]*Math.pow(n+r*e,-o);c[t]=s}return me(c,t.shape)}LRNGrad(t,e,n,r,o,s,a){ql(t,"LRNGrad");const i=t.shape[3],u=this.readSync(t.dataId),c=this.readSync(e.dataId),l=this.readSync(n.dataId),h=new Float32Array(t.size),d=t.size;for(let t=0;t<d;t++){const e=t%i,n=t-e+Math.max(0,e-r),d=t-e+Math.min(i,e+r+1);let p=0;for(let t=n;t<d;t++)p+=Math.pow(c[t],2);p=s*p+o;for(let e=n;e<d;e++){let n=-2*s*a*c[e]*l[t]/p;t===e&&(n+=Math.pow(p,-a)),n*=u[t],h[e]+=n}}return me(h,t.shape)}multinomial(t,e,n,r){ql(t,"multinomial");const o=e?t:ue(t),s=o.shape[0],a=o.shape[1],i=re([s,n],"int32"),u=this.readSync(i.dataId),c=this.readSync(o.dataId);for(let t=0;t<s;++t){const e=t*a,o=new Float32Array(a-1);o[0]=c[e];for(let t=1;t<o.length;++t)o[t]=o[t-1]+c[e+t];const s=Kl.alea(r.toString()),i=t*n;for(let t=0;t<n;++t){const e=s();u[i+t]=o.length;for(let n=0;n<o.length;n++)if(e<o[n]){u[i+t]=n;break}}}return i}oneHot(t,e,n,r){ql(t,"oneHot");const o=new Float32Array(t.size*e);o.fill(r);const s=this.readSync(t.dataId);for(let r=0;r<t.size;++r)s[r]>=0&&s[r]<e&&(o[r*e+s[r]]=n);return ge(o,[t.size,e],"int32")}nonMaxSuppression(t,e,n,r,o){ql(t,"nonMaxSuppression");const s=this.readSync(t.dataId),a=this.readSync(e.dataId);return Ql(s,a,n,r,o)}fft(t){return this.fftBatch(t,!1)}ifft(t){return this.fftBatch(t,!0)}fftBatch(t,e){const n=t.shape[0],r=t.shape[1],o=D(t.shape,"float32"),a=D(t.shape,"float32"),i=se(t).as2D(n,r),u=Mt(t).as2D(n,r);for(let t=0;t<n;t++){const n=i.slice([t,0],[1,r]),c=u.slice([t,0],[1,r]),l=pt.a(n,c),h=this.readSync(this.fftImpl(l,e).dataId);for(let e=0;e<r;e++){const n=s.getComplexWithIndex(h,e);o.values[t*r+e]=n.real,a.values[t*r+e]=n.imag}}return pt.a(o.toTensor(),a.toTensor()).as2D(n,r)}fftImpl(t,e){const n=t.as1D(),r=n.size;if(this.isExponentOf2(r)){let o=this.fftRadix2(n,r,e).as2D(t.shape[0],t.shape[1]);return e&&(o=pt.a(se(o).div(ie(r)),Mt(o).div(ie(r)))),o}{const n=this.readSync(t.dataId),o=this.fourierTransformByMatmul(n,r,e),a=s.splitRealAndImagArrays(o);return pt.a(a.real,a.imag).as2D(t.shape[0],t.shape[1])}}isExponentOf2(t){return 0==(t&t-1)}fftRadix2(t,e,n){if(1===e)return t;const r=this.readSync(t.dataId),o=e/2,a=s.complexWithEvenIndex(r);let i=pt.a(a.real,a.imag).as1D();const u=s.complexWithOddIndex(r);let c=pt.a(u.real,u.imag).as1D();i=this.fftRadix2(i,o,n),c=this.fftRadix2(c,o,n);const l=s.exponents(e,n),h=pt.a(l.real,l.imag).mul(c),d=i.add(h),p=i.sub(h),f=se(d).concat(se(p)),g=Mt(d).concat(Mt(p));return pt.a(f,g).as1D()}fourierTransformByMatmul(t,e,n){const r=new Float32Array(2*e);for(let o=0;o<e;o++){let a=0,i=0;for(let r=0;r<e;r++){const u=s.exponent(o*r,e,n),c=s.getComplexWithIndex(t,r);a+=c.real*u.real-c.imag*u.imag,i+=c.real*u.imag+c.imag*u.real}n&&(a/=e,i/=e),s.assignToTypedArray(r,a,i,o)}return r}depthToSpace(t,e,n){y.assert("NHWC"===n,()=>"Only NHWC dataFormat supported on CPU for depthToSpace. Got "+n),y.assert(e>1,()=>"blockSize should be > 1 for depthToSpace, but was: "+e);const r=t.shape[0],o=t.shape[1],s=t.shape[2],a=t.shape[3],i=o*e,u=s*e,c=a/(e*e),l=this.readSync(t.dataId),h=new Float32Array(r*i*u*c);let d=0;for(let t=0;t<r;++t)for(let n=0;n<i;++n){const r=Math.floor(n/e),i=n%e;for(let n=0;n<u;++n){const u=Math.floor(n/e),p=(i*e+n%e)*c;for(let e=0;e<c;++e){const n=e+p+a*(u+s*(r+o*t));h[d++]=l[n]}}}return me(h,[r,i,u,c])}broadcastedBinaryOp(t,e,n,r){const o=s.assertAndGetBroadcastShape(t.shape,e.shape),a=D(o,n),i=this.readSync(t.dataId),u=this.readSync(e.dataId),c=s.getBroadcastDims(t.shape,o),l=s.getBroadcastDims(e.shape,o),h=a.values;if(c.length+l.length===0)for(let t=0;t<h.length;++t)h[t]=r(i[t%i.length],u[t%u.length]);else{const n=this.bufferSync(t),o=this.bufferSync(e);for(let s=0;s<h.length;++s){const d=a.indexToLoc(s),p=d.slice(-t.rank);c.forEach(t=>p[t]=0);const f=n.locToIndex(p),g=d.slice(-e.rank);l.forEach(t=>g[t]=0);const m=o.locToIndex(g);h[s]=r(i[f],u[m])}}return a.toTensor()}broadcastedBinaryComplexOp(t,e,n){const r=s.assertAndGetBroadcastShape(t.shape,e.shape),o=D(r,"float32"),a=D(r,"float32"),i=this.readSync(t.dataId),u=this.readSync(e.dataId),c=s.getBroadcastDims(t.shape,r),l=s.getBroadcastDims(e.shape,r),h=o.values,d=a.values;if(c.length+l.length===0)for(let t=0;t<h.length;t++){const e=t%i.length,r=t%u.length,o=n(i[2*e],i[2*e+1],u[2*r],u[2*r+1]);h[t]=o.real,d[t]=o.imag}else{const r=this.bufferSync(this.data.get(t.dataId).complexTensors.real),s=this.bufferSync(this.data.get(e.dataId).complexTensors.real);for(let a=0;a<h.length;a++){const p=o.indexToLoc(a),f=p.slice(-t.rank);c.forEach(t=>f[t]=0);const g=r.locToIndex(f),m=p.slice(-e.rank);l.forEach(t=>m[t]=0);const b=s.locToIndex(m),x=n(i[2*g],i[2*g+1],u[2*b],u[2*b+1]);h[a]=x.real,d[a]=x.imag}}return this.complex(o.toTensor(),a.toTensor())}split(t,e,n){return Zl(t,e,n)}dispose(){}floatPrecision(){return 32}epsilon(){return super.epsilon()}cropAndResize(t,e,n,r,o,s){const[a,i,u,c]=t.shape,l=e.shape[0],[h,d]=r,p=D([l,h,d,c],"float32"),f=this.readSync(e.dataId),g=this.readSync(n.dataId),m=this.readSync(t.dataId),b=t.strides,x=p.strides;for(let t=0;t<l;t++){const e=4*t,n=f[e],r=f[e+1],l=f[e+2],y=f[e+3],v=g[t];if(v>=a)continue;const w=h>1?(l-n)*(i-1)/(h-1):0,C=d>1?(y-r)*(u-1)/(d-1):0;for(let e=0;e<h;e++){const a=h>1?n*(i-1)+e*w:.5*(n+l)*(i-1);if(a<0||a>i-1)for(let n=0;n<d;n++)for(let r=0;r<c;r++){const o=r+n*x[2]+e*x[1]+t*x[0];p.values[o]=s}else if("bilinear"===o){const n=Math.floor(a),o=Math.ceil(a),i=a-n;for(let a=0;a<d;a++){const l=d>1?r*(u-1)+a*C:.5*(r+y)*(u-1);if(l<0||l>u-1){for(let n=0;n<c;n++){const r=n+a*x[2]+e*x[1]+t*x[0];p.values[r]=s}continue}const h=Math.floor(l),f=Math.ceil(l),g=l-h;for(let r=0;r<c;r++){let s=r+h*b[2]+n*b[1]+v*b[0];const u=m[s];s=r+f*b[2]+n*b[1]+v*b[0];const c=m[s];s=r+h*b[2]+o*b[1]+v*b[0];const l=m[s];s=r+f*b[2]+o*b[1]+v*b[0];const d=u+(c-u)*g,y=l+(m[s]-l)*g;s=r+a*x[2]+e*x[1]+t*x[0],p.values[s]=d+(y-d)*i}}}else for(let n=0;n<d;++n){const o=d>1?r*(u-1)+n*C:.5*(r+y)*(u-1);if(o<0||o>u-1){for(let r=0;r<c;r++){const o=r+n*x[2]+e*x[1]+t*x[0];p.values[o]=s}continue}const i=Math.round(o),l=Math.round(a);for(let r=0;r<c;r++){const o=r+i*b[2]+l*b[1]+v*b[0],s=r+n*x[2]+e*x[1]+t*x[0];p.values[s]=m[o]}}}}return p.toTensor()}sparseToDense(t,e,n,r){const{sliceRank:o,numUpdates:a,sliceSize:i,strides:u,outputSize:c}=s.calculateShapes(e,t,n);return this.scatter(t,e,n,c,i,a,o,u,r,!1)}gatherND(t,e){const n=e.shape,r=n[n.length-1],[o,a,i,u]=s.prepareAndValidate(t,e);if(0===a)return he.a([],o,t.dtype);const c=new N.b([a,i],t.dtype),l=this.readSync(e.dataId),h=this.readSync(t.dataId);for(let e=0;e<a;e++){const n=[];let o=0;for(let t=0;t<r;t++){const s=l[e*r+t];o+=s*u[t],n.push(s)}if(o<0||o>=t.size/i)throw new Error(`Invalid indices: ${n} does not index into ${t.shape}`);for(let t=0;t<i;t++)c.values[e*i+t]=h[o*i+t]}return c.toTensor().reshape(o)}scatterND(t,e,n){const{sliceRank:r,numUpdates:o,sliceSize:a,strides:i,outputSize:u}=s.calculateShapes(e,t,n),c=ie(0);return this.scatter(t,e,n,u,a,o,r,i,c,!0)}fill(t,e,n){n=n||y.inferDtype(e);const r=y.getArrayFromDType(n,y.sizeFromShape(t));return r.fill(e),on().makeTensor(r,t,n,this)}onesLike(t){if("string"===t.dtype)throw new Error("onesLike is not supported for string tensors");return this.fill(t.shape,1,t.dtype)}zerosLike(t){const e=y.getArrayFromDType(t.dtype,y.sizeFromShape(t.shape));return this.makeOutput(e,t.shape,t.dtype)}linspace(t,e,n){return s.linspaceImpl(t,e,n)}scatter(t,e,n,r,o,s,a,i,u,c){const l=[r/o,o],h=this.readSync(t.dataId),d=this.readSync(e.dataId);if(0===r)return he.a([],n,e.dtype);const p=new N.b(l,e.dtype);p.values.fill(this.readSync(u.dataId)[0]);for(let t=0;t<s;t++){const s=[];let u=0;for(let e=0;e<a;e++){const n=h[t*a+e];s.push(n),u+=n*i[e]}if(u<0||u>=r/o)throw new Error(`Invalid indices: ${s} does not index into ${n}`);for(let n=0;n<o;n++)c?p.values[u*o+n]+=d[t*o+n]:p.values[u*o+n]=0===e.rank?d[0]:d[t*o+n]}return p.toTensor().reshape(n)}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */cn("cpu",()=>new rh,1);
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const oh={kernelName:_.G,backendName:"cpu",kernelFunc:({inputs:t,backend:e})=>{const{x:n}=t,r=e;ql(n,"cos");const o=r.data.get(n.dataId).values,s=y.sizeFromShape(n.shape),a=new Float32Array(s);for(let t=0;t<s;++t)a[t]=Math.cos(o[t]);return{dataId:r.write(a,n.shape,n.dtype),shape:n.shape,dtype:n.dtype}}},sh={kernelName:_.O,backendName:"cpu",kernelFunc:({inputs:t,backend:e,attrs:n})=>{const{x:r,filter:o}=t,{strides:a,pad:i,dilations:u}=n,c=e,l=c.data.get(r.dataId).values,h=r.shape.length,d=c.data.get(o.dataId).values,p=o.shape.length,{batchSize:f,inHeight:g,inWidth:m,inChannels:b,outHeight:x,outWidth:v,padInfo:w,strideHeight:C,strideWidth:$,filterHeight:O,filterWidth:I,dilationHeight:S,dilationWidth:E,outShape:R}=s.computeDilation2DInfo(r.shape,o.shape,a,i,"NHWC",u),A=y.sizeFromShape(R),k=R.length,T=y.getArrayFromDType(r.dtype,A);for(let t=0;t<f;++t)for(let e=0;e<x;++e){const n=e*C-w.top;for(let s=0;s<v;++s){const a=s*$-w.left;for(let i=0;i<b;++i){let u=Number.MIN_SAFE_INTEGER;for(let e=0;e<O;++e){const s=n+e*S;if(s>=0&&s<g)for(let n=0;n<I;++n){const c=a+n*E;if(c>=0&&c<m){const a=y.locToIndex([t,s,c,i],h,y.computeStrides(r.shape)),f=y.locToIndex([e,n,i],p,y.computeStrides(o.shape)),g=l[a]+d[f];g>u&&(u=g)}}}T[y.locToIndex([t,e,s,i],k,y.computeStrides(R))]=u}}}return{dataId:c.write(y.toTypedArray(T,r.dtype),R,r.dtype),shape:R,dtype:r.dtype}}},ah={kernelName:_.P,backendName:"cpu",kernelFunc:({inputs:t,backend:e,attrs:n})=>{const{x:r,filter:o,dy:a}=t,{strides:i,pad:u,dilations:c}=n,l=e,h=y.toNestedArray(r.shape,l.data.get(r.dataId).values),d=y.toNestedArray(o.shape,l.data.get(o.dataId).values),{batchSize:p,inHeight:f,inWidth:g,inChannels:m,outHeight:b,outWidth:x,padInfo:v,strideHeight:w,strideWidth:C,filterHeight:$,filterWidth:O,dilationHeight:I,dilationWidth:S,outShape:E}=s.computeDilation2DInfo(r.shape,o.shape,i,u,"NHWC",c);y.assert(a.rank===E.length,()=>`Error in ${_.P}, dy must have the same rank as output ${E.length}, but got `+a.rank);const R=y.toNestedArray(E,l.data.get(a.dataId).values),A=y.makeZerosNestedTypedArray(o.shape,o.dtype);for(let t=0;t<p;++t)for(let e=0;e<b;++e){const n=e*w-v.top;for(let r=0;r<x;++r){const o=r*C-v.left;for(let s=0;s<m;++s){let a=Number.MIN_SAFE_INTEGER,i=0,u=0;for(let e=0;e<$;++e){const r=n+e*I;if(r>=0&&r<f)for(let n=0;n<O;++n){const c=o+n*S;if(c>=0&&c<g){const o=h[t][r][c][s]+d[e][n][s];o>a&&(a=o,i=e,u=n)}}}A[i][u][s]+=R[t][e][r][s]}}}return{dataId:l.write(y.toTypedArray(A,r.dtype),o.shape,o.dtype),shape:o.shape,dtype:o.dtype}}},ih={kernelName:_.Q,backendName:"cpu",kernelFunc:({inputs:t,backend:e,attrs:n})=>{const{x:r,filter:o,dy:a}=t,{strides:i,pad:u,dilations:c}=n,l=e,h=y.toNestedArray(r.shape,l.data.get(r.dataId).values),d=y.toNestedArray(o.shape,l.data.get(o.dataId).values),{batchSize:p,inHeight:f,inWidth:g,inChannels:m,outHeight:b,outWidth:x,padInfo:v,strideHeight:w,strideWidth:C,filterHeight:$,filterWidth:O,dilationHeight:I,dilationWidth:S,outShape:E}=s.computeDilation2DInfo(r.shape,o.shape,i,u,"NHWC",c);y.assert(a.rank===E.length,()=>`Error in ${_.Q}, dy must have the same rank as output ${E.length}, but got `+a.rank);const R=y.toNestedArray(E,l.data.get(a.dataId).values),A=y.makeZerosNestedTypedArray(r.shape,r.dtype);for(let t=0;t<p;++t)for(let e=0;e<b;++e){const n=e*w-v.top;for(let r=0;r<x;++r){const o=r*C-v.left;for(let s=0;s<m;++s){let a=Number.MIN_SAFE_INTEGER,i=n<0?0:n,u=o<0?0:o;for(let e=0;e<$;++e){const r=n+e*I;if(r>=0&&r<f)for(let n=0;n<O;++n){const c=o+n*S;if(c>=0&&c<g){const o=h[t][r][c][s]+d[e][n][s];o>a&&(a=o,i=r,u=c)}}}A[t][i][u][s]+=R[t][e][r][s]}}}return{dataId:l.write(y.toTypedArray(A,r.dtype),r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function uh(t,e){return{kernelName:t,backendName:"cpu",kernelFunc:({inputs:n,backend:r})=>{const{a:o,b:s}=n,a=r;ql([o,s],t);const i=a.data.get(o.dataId).values,u=a.data.get(s.dataId).values,[c,l]=e(o.shape,s.shape,i,u,o.dtype);return{dataId:a.write(c,l,o.dtype),shape:l,dtype:o.dtype}}}}function ch(t){return(e,n,r,o,a)=>{const i=s.assertAndGetBroadcastShape(e,n),u=i.length,c=y.computeStrides(i),l=y.sizeFromShape(i),h=y.getTypedArrayFromDType(a,l),d=e.length,p=n.length,f=y.computeStrides(e),g=y.computeStrides(n),m=s.getBroadcastDims(e,i),b=s.getBroadcastDims(n,i);if(m.length+b.length===0)for(let e=0;e<h.length;++e)h[e]=t(r[e%r.length],o[e%o.length]);else for(let e=0;e<h.length;++e){const n=y.indexToLoc(e,u,c),s=n.slice(-d);m.forEach(t=>s[t]=0);const a=y.locToIndex(s,d,f),i=n.slice(-p);b.forEach(t=>i[t]=0);const l=y.locToIndex(i,p,g);h[e]=t(r[a],o[l])}return[h,i]}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lh=ch((t,e)=>t/e),hh=uh(_.R,lh),dh={kernelName:_.ab,backendName:"cpu",kernelFunc:({inputs:t,attrs:e,backend:n})=>{const{image:r}=t,o=n,s=y.getTypedArrayFromDType(r.dtype,y.sizeFromShape(r.shape)),[a,i,u,c]=r.shape,l=o.data.get(r.dataId).values;for(let t=0;t<a;t++){const e=t*u*i*c;for(let t=0;t<i;t++){const n=t*(u*c);for(let r=0;r<u;r++){const o=r*c;for(let i=0;i<c;i++){const h=[a,t,r,i][2],d=Math.round(u-h),p=e+n+o+i;let f=l[p];if(d>=0&&d<u){f=l[e+n+d*c+i]}s[p]=f}}}}return{dataId:o.write(s,r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ph={kernelName:_.jb,backendName:"cpu",kernelFunc:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const{inputs:e,backend:n}=t,{x:r}=e;return n.incRef(r.dataId),{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}},fh={kernelName:_.yb,backendName:"cpu",kernelFunc:({inputs:t,attrs:e,backend:n})=>{const{x:r}=t,{reductionIndices:o,keepDims:a}=e,i=n;let u=r.shape;const c=u.length,l=y.parseAxisParam(o,u);let h=l;const d=s.getAxesPermutation(h,c);let p=i.data.get(r.dataId).values;if(null!=d){const t=new Array(c);for(let e=0;e<t.length;e++)t[e]=u[d[e]];p=kl(p,u,r.dtype,d,t),h=s.getInnerMostAxes(h.length,c),u=t}ql(r,"max"),s.assertAxesAreInnerMostDims("max",h,c);const[f,g]=s.computeOutAndReduceShapes(u,h),m=Al(p,y.sizeFromShape(g),f,r.dtype),b=i.write(m,f,r.dtype);let x=f;if(a){x=s.expandShapeToKeepDim(f,l)}return{dataId:b,shape:x,dtype:r.dtype}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const gh={kernelName:_.Db,backendName:"cpu",kernelFunc:({inputs:t,attrs:e,backend:n})=>{const{x:r}=t,{filterSize:o,strides:a,pad:i,includeBatchInIndex:u}=e,c=n;ql(r,"MaxPoolWithArgmax");const l=c.data.get(r.dataId).values,h=s.computePool2DInfo(r.shape,o,a,[1,1],i),[d,p]=
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t,e,n,r,o){const s=Xl(t,0,n,y.computeStrides(e),o,"max"),a=Yl(t,e,n,o,!0,r);return[s.values,a.values]}(l,r.shape,r.dtype,u,h),f=c.write(d,h.outShape,r.dtype),g=c.write(p,h.outShape,r.dtype);return[{dataId:f,shape:h.outShape,dtype:r.dtype},{dataId:g,shape:h.outShape,dtype:"int32"}]}},mh=a.nonMaxSuppressionV4Impl,bh={kernelName:_.Lb,backendName:"cpu",kernelFunc:({inputs:t,backend:e,attrs:n})=>{const{boxes:r,scores:o}=t,{maxOutputSize:s,iouThreshold:a,scoreThreshold:i,padToMaxOutputSize:u}=n,c=e;ql(r,"NonMaxSuppressionPadded");const l=c.data.get(r.dataId).values,h=c.data.get(o.dataId).values,{selectedIndices:d,validOutputs:p}=mh(l,h,s,a,i,u);return[d,p]}},xh=a.nonMaxSuppressionV5Impl,yh={kernelName:_.Mb,backendName:"cpu",kernelFunc:({inputs:t,backend:e,attrs:n})=>{const{boxes:r,scores:o}=t,{maxOutputSize:s,iouThreshold:a,scoreThreshold:i,softNmsSigma:u}=n,c=e;ql(r,"NonMaxSuppressionWithScore");const l=c.data.get(r.dataId).values,h=c.data.get(o.dataId).values,d=s,p=a,f=i,g=u,{selectedIndices:m,selectedScores:b}=xh(l,h,d,p,f,g);return[m,b]}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vh={kernelName:_.Qb,backendName:"cpu",kernelFunc:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(t){const{inputs:e,backend:n,attrs:r}=t,{x:o}=e,{paddings:s,constantValue:a}=r;ql(o,"pad");const i=s.map((t,e)=>t[0]+o.shape[e]+t[1]),u=s.map(t=>t[0]),c=n.data.get(o.dataId).values,l=y.sizeFromShape(o.shape),h=o.shape.length,d=y.computeStrides(o.shape),p=y.sizeFromShape(i),f=i.length,g=y.computeStrides(i),m=y.getTypedArrayFromDType(o.dtype,p);0!==a&&m.fill(a);for(let t=0;t<l;t++){const e=y.indexToLoc(t,h,d).map((t,e)=>t+u[e]);m[y.locToIndex(e,f,g)]=c[t]}return{dataId:n.write(m,i,o.dtype),shape:i,dtype:o.dtype}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wh(t){const{inputs:e,backend:n,attrs:r}=t,{x:o}=e,{shape:s}=r;return n.incRef(o.dataId),{dataId:o.dataId,shape:s,dtype:o.dtype}}const Ch={kernelName:_.Zb,backendName:"cpu",kernelFunc:wh},$h={kernelName:_.fc,backendName:"cpu",kernelFunc:({inputs:t,attrs:e,backend:n})=>{const{image:r}=t,{radians:o,fillValue:a,center:i}=e,u=n,c=y.getTypedArrayFromDType(r.dtype,y.sizeFromShape(r.shape)),[l,h,d,p]=r.shape,[f,g]=s.getImageCenter(i,h,d),m=Math.sin(o),b=Math.cos(o),x=u.data.get(r.dataId).values;for(let t=0;t<l;t++){const e=t*d*h*p;for(let t=0;t<h;t++){const n=t*(d*p);for(let r=0;r<d;r++){const o=r*p;for(let s=0;s<p;s++){const i=[l,t,r,s],u=i[2],y=i[1];let v=(u-f)*b-(y-g)*m,w=(u-f)*m+(y-g)*b;v=Math.round(v+f),w=Math.round(w+g);let C=a;if("number"!=typeof a&&(C=3===s?255:a[s]),v>=0&&v<d&&w>=0&&w<h){C=x[e+w*(d*p)+v*p+s]}c[e+n+o+s]=C}}}}return{dataId:u.write(c,r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function Oh(t){const{inputs:e,attrs:n,backend:r}=t,{x:o}=e,{perm:s}=n;ql(o,"transpose");const a=o.shape.length,i=new Array(a);for(let t=0;t<i.length;t++)i[t]=o.shape[s[t]];const u=kl(r.data.get(o.dataId).values,o.shape,o.dtype,s,i);return{dataId:r.write(u,i,o.dtype),shape:i,dtype:o.dtype}}const Ih={kernelName:_.Ec,backendName:"cpu",kernelFunc:Oh};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sh={kernelName:_.rc,backendName:"cpu",kernelFunc:function(t){const{inputs:e,backend:n,attrs:r}=t,{x:o}=e,{blockShape:a,paddings:i}=r;ql([o],"spaceToBatchND");const u=y.sizeFromShape(a),c=[[0,0]];c.push(...i);for(let t=1+a.length;t<o.shape.length;++t)c.push([0,0]);const l=vh.kernelFunc({inputs:{x:o},backend:n,attrs:{paddings:c,constantValue:0}}),h=s.getReshaped(l.shape,a,u,!1),d=s.getPermuted(h.length,a.length,!1),p=s.getReshapedPermuted(l.shape,a,u,!1),f=wh({inputs:{x:l},backend:n,attrs:{shape:h}}),g=Oh({inputs:{x:f},backend:n,attrs:{perm:d}}),m=wh({inputs:{x:g},backend:n,attrs:{shape:p}});return n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(g),m}},Eh={kernelName:_.uc,backendName:"cpu",kernelFunc:({inputs:t,backend:e})=>{const{x:n}=t,r=e;ql(n,"square");const o=r.data.get(n.dataId).values,s=new Float32Array(o.length);for(let t=0;t<o.length;++t){const e=o[t];s[t]=e*e}return{dataId:r.write(s,n.shape,n.dtype),shape:n.shape,dtype:n.dtype}}},Rh=ch((t,e)=>{const n=t-e;return n*n}),Ah=uh(_.vc,Rh),kh=[oh,sh,ih,ah,hh,dh,ph,gh,fh,bh,yh,vh,Ch,$h,Sh,Eh,Ah,Ih];
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */for(const t of kh)Object(En.e)(t);
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Th{constructor(t){this.inputTensors_=new Map,this.outputBuffers_=new Map,Ha(void 0!==t,"Invalid argument"),this.compilation_=t}setInput(t,e){Ha("string"==typeof t&&this.compilation_.model.inputs.has(t),"The name parameter is invalid.");const n=this.compilation_.model.inputs.get(t);Ja(e,n.desc),this.inputTensors_.set(n,ti(n.desc,e))}setOutput(t,e){Ha("string"==typeof t&&this.compilation_.model.outputs.has(t),"The name parameter is invalid.");const n=this.compilation_.model.outputs.get(t);Ja(e,this.compilation_.outputDescriptors.get(n)),this.outputBuffers_.set(n,e)}async startCompute(){for(const t of this.compilation_.model.outputs.values()){const e=sn(()=>t.operation.run({inputTensors:this.inputTensors_,constantTenosrs:this.compilation_.constantTensors})),n=await e.data();an(e),this.outputBuffers_.get(t).set(n)}for(const t of this.inputTensors_.values())an(t)}}class Fh{constructor(t,e){this.constantTensors_=new Map,this.outputDescriptors_=new Map,Ha(void 0!==e,"Invalid arguments"),this.model_=e}get model(){return this.model_}get constantTensors(){return this.constantTensors_}get outputDescriptors(){return this.outputDescriptors_}async createExecution(){return new Th(this)}static async createAndCompile(t,e){const n=new Fh(t,e);return await n.compile(),n}async compile(){try{if(!await un("webgl")&&(console.warn("Failed to set tf.js webgl backend, fallback to cpu backend."),!await un("cpu")))throw new Error("Failed to set tf.js cpu backend.")}catch(t){if(!await un("cpu"))throw new Error("Failed to set tf.js cpu backend.")}await l.a.ready(),this.allocateConstants(),await this.inferOnce()}allocateConstants(){for(const t of this.model_.constants)this.constantTensors_.set(t,ti(t.desc,t.value))}async inferOnce(){const t=new Map;for(const e of this.model_.inputs.values()){const n=new(Ya(e.desc.type))(ei(e.desc.dimensions));t.set(e,ti(e.desc,n))}for(const e of this.model_.outputs.values()){const n=sn(()=>e.operation.run({inputTensors:t,constantTenosrs:this.constantTensors_}));await n.data(),this.outputDescriptors_.set(e,Qa(n)),an(n)}for(const e of t.values())an(e)}}class Nh extends c{constructor(t){super(),this.operation=t}}class Dh{constructor(t){this.inputs_=new Map,this.outputs_=new Map,this.constants_=[],Ha(void 0!==t,"Invalid argument"),Ha(0!==t.length,"The length of outputs parameter should not be 0."),Ha(t.every(t=>"string"==typeof t.name&&t.operand instanceof Nh),"The outputs parameter is invalid.");for(const e of t)this.outputs_.set(e.name,e.operand);this.initialize()}get inputs(){return this.inputs_}get outputs(){return this.outputs_}get constants(){return this.constants_}async createCompilation(t){return await Fh.createAndCompile(t,this)}initialize(){for(const t of this.outputs_.values())this.handleOperation(t.operation)}handleOperation(t){for(const e of t.inputs)e instanceof ri?this.inputs_.set(e.name,e):e instanceof ni?this.constants_.push(e):e instanceof Nh&&this.handleOperation(e.operation)}}var _h;!function(t){t.nchw="nchw",t.nhwc="nhwc"}(_h||(_h={}));class Bh{constructor(t){this.inputs=[],this.outputs=[],Ha(t.every(t=>t instanceof c),"The inputs parameter is invalid."),this.inputs=t,this.outputs.push(new Nh(this))}get output(){return this.outputs[0]}getTensor(t,e){if(t instanceof ni)return e.constantTenosrs.get(t);if(t instanceof ri)return e.inputTensors.get(t);if(t instanceof Nh)return t.operation.run(e);throw new Error("The operand is invalid.")}}class jh extends Bh{constructor(t,e){super([t,e])}run(t){const e=this.getTensor(this.inputs[0],t),n=this.getTensor(this.inputs[1],t);return this.runOp(e,n)}}class Mh extends jh{runOp(t,e){return dt(t,e)}}class Ph extends Bh{constructor(t,e=[-1,-1],n=[0,0,0,0],r=[1,1],o=[1,1],s=_h.nchw){super([t]),Ha(qa(e)&&2===e.length,"The padding parameter is invalid."),this.windowDimensions_=e,Ha(qa(n)&&4===n.length,"The padding parameter is invalid."),this.padding_=n,Ha(qa(r)&&2===r.length,"The strides parameter is invalid."),this.strides_=r,Ha(qa(o)&&2===o.length,"The dilations parameter is invalid."),this.dilations_=o,Ha(s in _h,"The layout parameter is invalid."),this.layout_=s}run(t){let e=this.getTensor(this.inputs[0],t);Ha(this.padding_.every(t=>t===this.padding_[0]),"The tf.conv2d only supports the same padding value.");const n=this.padding_[0],r=this.getPoolingType();this.layout_===_h.nchw&&(e=e.transpose([0,2,3,1]));const o=this.windowDimensions_;-1===o[0]&&-1===o[1]&&(o[0]=e.shape[1],o[1]=e.shape[2]);let s=te(e,this.windowDimensions_,r,n,this.dilations_,this.strides_);return this.layout_===_h.nchw&&(s=s.transpose([0,3,1,2])),s}}class Lh extends Ph{getPoolingType(){return"avg"}}class Wh extends Bh{constructor(t,e,n=[0,0,0,0],r=[1,1],o=[1,1],s=1,a=_h.nchw){super([t,e]),Ha(qa(n)&&4===n.length,"The padding parameter is invalid."),this.padding_=n,Ha(qa(r)&&2===r.length,"The strides parameter is invalid."),this.strides_=r,Ha(qa(o)&&2===o.length,"The dilations parameter is invalid."),this.dilations_=o,Ha(Ka(s),"The gourps parameter is invalid."),this.groups_=s,Ha(a in _h,"The layout parameter is invalid."),this.layout_=a}run(t){let e=this.getTensor(this.inputs[0],t),n=this.getTensor(this.inputs[1],t);Ha(this.padding_.every(t=>t===this.padding_[0]),"The tf.conv2d only supports the same padding value.");const r=this.padding_[0];let o,s;if(this.layout_===_h.nchw?(e=e.transpose([0,2,3,1]),o=e.shape[1],n=n.transpose([2,3,1,0])):o=e.shape[3],1===this.groups_)s=Rt(e,n,this.strides_,r,"NHWC",this.dilations_);else{if(this.groups_!==o)throw new Error("The tf.js convolution doesn't support groups parameter "+this.groups_);s=At(e,n,this.strides_,r,"NHWC",this.dilations_)}return this.layout_===_h.nchw&&(s=s.transpose([0,3,1,2])),s}}class zh extends Bh{constructor(t,e){super([t,e])}run(t){const e=this.getTensor(this.inputs[0],t),n=this.getTensor(this.inputs[1],t);return 1===e.rank||1===n.rank?Nt(e,n):Ft(e,n)}}class Uh extends Ph{getPoolingType(){return"max"}}class Vh extends jh{runOp(t,e){return Xt(t,e)}}class Gh extends Bh{constructor(t){super([t])}run(t){const e=this.getTensor(this.inputs[0],t);return ae(e)}}class Hh extends Bh{constructor(t,e){super([t]),Ha(qa(e),"The newShape parameter is invalid."),this.newShape_=e}run(t){const e=this.getTensor(this.inputs[0],t);return Et(e,this.newShape_)}}class Kh extends Bh{constructor(t){super([t])}run(t){const e=this.getTensor(this.inputs[0],t);if(2!==e.rank)throw new Error("The rank of x parameter should be 2.");return ue(e)}}class qh extends Bh{constructor(t,e){super([t]),e?(Ha(qa(e),"The permutation parameter is invalid."),this.permutation_=e):this.permutation_=void 0}run(t){const e=this.getTensor(this.inputs[0],t);return Kt(e,this.permutation_)}}class Xh{async createModel(t){return new Dh(t)}input(t,e){return new ri(t,e)}constant(t,e){return"number"==typeof t?ni.createScalar(t,e):ni.createTensor(t,e)}add(t,e){return new Mh(t,e).output}averagePool2d(t,e=[-1,-1],n=[0,0,0,0],r=[1,1],o=[1,1],s=_h.nchw){return new Lh(t,e,n,r,o,s).output}conv2d(t,e,n=[0,0,0,0],r=[1,1],o=[1,1],s=1,a=_h.nchw){return new Wh(t,e,n,r,o,s,a).output}matmul(t,e){return new zh(t,e).output}mul(t,e){return new Vh(t,e).output}maxPool2d(t,e=[-1,-1],n=[0,0,0,0],r=[1,1],o=[1,1],s=_h.nchw){return new Uh(t,e,n,r,o,s).output}relu(t){return new Gh(t).output}reshape(t,e){return new Hh(t,e).output}softmax(t){return new Kh(t).output}transpose(t,e){return new qh(t,e).output}}const Yh=class{constructor(){this.nnContext=null}getNeuralNetworkContext(){return this.nnContext||(this.nnContext=new Xh),this.nnContext}};null==navigator.ml&&(navigator.ml=new Yh)}]);
//# sourceMappingURL=webnn-polyfill.js.map