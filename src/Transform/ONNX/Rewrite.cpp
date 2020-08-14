//===----------- ONNXRewrite.cpp - ONNX High Level Optimizer --------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {

// Check whether an ArrayAttr contains non-zero values or not.
bool hasNonZeroInArrayAttr(ArrayAttr attrs) {
  bool allZeros = true;
  if (attrs) {
    for (auto attr : attrs.getValue()) {
      if (attr.cast<IntegerAttr>().getInt() > 0) {
        allZeros = false;
        break;
      }
    }
  }
  return !allZeros;
}

// Create an ArrayAttr of IntergerAttr(s) of zero values.
// This function is used for padding attribute in Conv.
ArrayAttr createArrayAttrOfZeros(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  int nElements = origAttrs.getValue().size();
  SmallVector<int64_t, 4> vals(nElements, 0);
  return rewriter.getI64ArrayAttr(vals);
}

DenseElementsAttr createDenseFloatAttrOfValue(
    PatternRewriter &rewriter, Value origValue, float constantValue) {
  Type elementType = origValue.getType().cast<TensorType>().getElementType();
  SmallVector<float, 1> wrapper(1, 0);
  wrapper[0] = constantValue;
  return DenseElementsAttr::get(
      RankedTensorType::get(wrapper.size(), elementType),
      llvm::makeArrayRef(wrapper));
}

// Pad a ArrayAttr with zeros.
//
// pads = [B1, B2, ... Bk, E1, E2, ..., Ek]
//
// becomes:
//
// pads = [0,... 0, B1, B2, ... Bk, 0,... 0, E1, E2, ..., Ek]
//         |_____|                  |_____|
//                 nZeros                    nZeros
//
// This function is used for padding attribute in Conv.
DenseElementsAttr insertZerosForNonPaddedDims(
    PatternRewriter &rewriter, ArrayAttr origAttrs, int extensionLength) {
  int nDims = (int)origAttrs.getValue().size() / 2;
  int nElements = (nDims + extensionLength) * 2;
  SmallVector<int64_t, 4> pads(nElements, 0);
  for (int i = 0; i < nDims; ++i) {
    int64_t beginPad = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();
    int64_t endPad =
        origAttrs.getValue()[nDims + i].cast<IntegerAttr>().getInt();
    pads[i + extensionLength] = beginPad;
    pads[nDims + extensionLength + i + extensionLength] = endPad;
  }

  mlir::Type elementType = rewriter.getIntegerType(64);
  llvm::ArrayRef<int64_t> tensorDims(pads.data(), pads.size());
  mlir::ShapedType tensorType =
      mlir::RankedTensorType::get(tensorDims, elementType);
  return rewriter.getI64TensorAttr(llvm::makeArrayRef(pads));
}


DenseElementsAttr createArrayAttrOneHotEncoder(
    PatternRewriter &rewriter, Value origValue, ArrayAttr attrs, IntegerAttr zeros) {
  Type elementType = origValue.getType().cast<TensorType>().getElementType();
  int outDim = attrs.getValue().size();

  // SmallVector<float, 2> res(outDim, 0);
  // for (int i = 0; i < outDim; ++i) {
  //   wrapper[i] = origAttrs.getValue()[i].cast<FloatAttr>().getValueAsDouble();
  // }  
  SmallVector<float, 1> res(outDim, 0);
  int i = 0;
  for (mlir::detail::InLineOpResult::use_iterator it=origValue.use_begin(); it != origValue.use_end(); ++it){
    if ((*it).getOperand(i).get() == attrs.getValue()){
      std::cout <<"in"<<std::endl;
    }
    i+=1;
  }
  // for (auto val : origValue) {
  //   SmallVector<float, 1> row(outDim, 0);
  //   for (int i = 0; i < outDim; ++i) {
  //     if (val.cast<elementType>() == attrs.getValue()[i].cast<FloatAttr>().getValueAsDouble()) {
  //       row[i] = 1;
  //       break;
  //     }
  //   }
  //   res.emplace_back(row);
  // }

  return DenseElementsAttr::get(
      RankedTensorType::get(res.size(), rewriter.getF32Type()),
      llvm::makeArrayRef(res));
}

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXRewrite.inc"

} // end anonymous namespace

/// on the ONNXConvOp.
void ONNXConvOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ConvOpPaddingPattern>(context);
}

/// on the ONNXConvOp.
void ONNXOneHotEncoderOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<OneHotEncoderPattern>(context);
}