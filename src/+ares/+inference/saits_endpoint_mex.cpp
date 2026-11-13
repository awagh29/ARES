// Native FP32 endpoint SAITS inference, using MATLAB's supported MEX C API.
// Compile with scripts/buildSaitsMex.m; only MATLAB and a C++ compiler are
// needed. MATLAB BLAS supplies SGEMM; MATLAB Coder is not required.
//
// Entry: [physical, normalized] = saits_endpoint_mex(model, rawSingle, mask)
// model is the exported row-in/column-out weight structure. rawSingle is a
// chronological [nSteps,31] physical-unit matrix; mask is logical of the same
// size. Missing placeholders are discarded before any model computation.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "mex.h"
#include "blas.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const mxArray* field(const mxArray* object, const char* name) {
    require(object && mxIsStruct(object) && mxGetNumberOfElements(object) == 1,
            "Expected a scalar model/parameter structure.");
    const mxArray* value = mxGetField(object, 0, name);
    require(value != nullptr, std::string("Missing model field: ") + name);
    return value;
}

double scalar(const mxArray* value) {
    require(value && mxIsNumeric(value) && !mxIsComplex(value) &&
            mxGetNumberOfElements(value) == 1, "Expected a real numeric scalar.");
    const double result = mxGetScalar(value);
    require(std::isfinite(result), "A model scalar is nonfinite.");
    return result;
}

mwSize positiveInteger(const mxArray* value) {
    const double result = scalar(value);
    require(result >= 1 && result == std::floor(result) && result <= 100000,
            "Invalid positive integer model dimension.");
    return static_cast<mwSize>(result);
}

struct View {
    mwSize rows;
    mwSize columns;
    const float* values;
};

View singleView(const mxArray* value) {
    require(value && mxIsSingle(value) && !mxIsComplex(value) &&
            !mxIsSparse(value) && mxGetNumberOfDimensions(value) == 2,
            "Model arrays and raw input must be real full single matrices.");
    return {mxGetM(value), mxGetN(value), mxGetSingles(value)};
}

View singleView(const mxArray* value, mwSize rows, mwSize columns) {
    const View result = singleView(value);
    require(result.rows == rows && result.columns == columns,
            "A model weight/bias matrix has the wrong shape.");
    return result;
}

struct Matrix {
    mwSize rows;
    mwSize columns;
    std::vector<float> values;

    Matrix(mwSize r, mwSize c) : rows(r), columns(c), values(r * c, 0.0f) {}

    float& at(mwSize row, mwSize column) { return values[row + column * rows]; }
    float at(mwSize row, mwSize column) const { return values[row + column * rows]; }
    View view() const { return {rows, columns, values.data()}; }
};

Matrix multiply(const Matrix& left, const View& right) {
    require(left.columns == right.rows, "Incompatible SAITS matrix product.");
    Matrix result(left.rows, right.columns);
    const ptrdiff_t m = static_cast<ptrdiff_t>(left.rows);
    const ptrdiff_t n = static_cast<ptrdiff_t>(right.columns);
    const ptrdiff_t k = static_cast<ptrdiff_t>(left.columns);
    const char noTranspose = 'N';
    const float one = 1.0f;
    const float zero = 0.0f;
    sgemm(&noTranspose, &noTranspose, &m, &n, &k, &one,
          left.values.data(), &m, right.values, &k, &zero,
          result.values.data(), &m);
    return result;
}

Matrix linear(const Matrix& input, const mxArray* parameters) {
    const View weight = singleView(field(parameters, "W"));
    require(weight.rows == input.columns, "Linear layer input width changed.");
    const View bias = singleView(field(parameters, "b"), 1, weight.columns);
    Matrix output = multiply(input, weight);
    for (mwSize column = 0; column < output.columns; ++column) {
        for (mwSize row = 0; row < output.rows; ++row) {
            output.at(row, column) += bias.values[column];
        }
    }
    return output;
}

Matrix layerNorm(const Matrix& input, const mxArray* parameters) {
    const View scale = singleView(field(parameters, "scale"), 1, input.columns);
    const View bias = singleView(field(parameters, "bias"), 1, input.columns);
    const float epsilon = static_cast<float>(scalar(field(parameters, "epsilon")));
    require(epsilon > 0.0f, "Layer-normalization epsilon must be positive.");
    Matrix output(input.rows, input.columns);
    for (mwSize row = 0; row < input.rows; ++row) {
        float mean = 0.0f;
        for (mwSize column = 0; column < input.columns; ++column) {
            mean += input.at(row, column);
        }
        mean /= static_cast<float>(input.columns);
        float variance = 0.0f;
        for (mwSize column = 0; column < input.columns; ++column) {
            const float centered = input.at(row, column) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(input.columns);
        const float denominator = std::sqrt(variance + epsilon);
        for (mwSize column = 0; column < input.columns; ++column) {
            output.at(row, column) = ((input.at(row, column) - mean) / denominator) *
                                     scale.values[column] + bias.values[column];
        }
    }
    return output;
}

void addInPlace(Matrix& output, const Matrix& increment) {
    require(output.rows == increment.rows && output.columns == increment.columns,
            "Residual matrix shapes differ.");
    for (mwSize index = 0; index < output.values.size(); ++index) {
        output.values[index] += increment.values[index];
    }
}

void addPosition(Matrix& hidden, const mxArray* positionEncoding) {
    const View position = singleView(positionEncoding, hidden.rows, hidden.columns);
    for (mwSize index = 0; index < hidden.values.size(); ++index) {
        hidden.values[index] += position.values[index];
    }
}

Matrix lastRow(const Matrix& input) {
    Matrix output(1, input.columns);
    for (mwSize column = 0; column < input.columns; ++column) {
        output.at(0, column) = input.at(input.rows - 1, column);
    }
    return output;
}

void reluInPlace(Matrix& value) {
    for (float& element : value.values) {
        element = std::max(element, 0.0f);
    }
}

Matrix attentionLayer(const Matrix& hidden, const mxArray* layer,
                      mwSize heads, mwSize headDimension, bool endpointOnly,
                      std::vector<float>* endpointAttention = nullptr) {
    const mwSize steps = hidden.rows;
    const mwSize width = hidden.columns;
    const Matrix normalized = layerNorm(hidden, field(layer, "attentionNorm"));
    const Matrix queryInput = endpointOnly ? lastRow(normalized) : normalized;
    const Matrix query = multiply(queryInput, singleView(field(layer, "queryW"), width, width));
    const Matrix key = multiply(normalized, singleView(field(layer, "keyW"), width, width));
    const Matrix value = multiply(normalized, singleView(field(layer, "valueW"), width, width));
    const mwSize querySteps = query.rows;
    Matrix attended(querySteps, width);
    const char noTranspose = 'N';
    const char transpose = 'T';
    const float scoreScale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(headDimension)));
    const float one = 1.0f;
    const float zero = 0.0f;
    const ptrdiff_t m = static_cast<ptrdiff_t>(querySteps);
    const ptrdiff_t n = static_cast<ptrdiff_t>(steps);
    const ptrdiff_t k = static_cast<ptrdiff_t>(headDimension);
    if (endpointAttention) {
        endpointAttention->assign(steps, 0.0f);
    }
    for (mwSize head = 0; head < heads; ++head) {
        Matrix probability(querySteps, steps);
        const float* queryHead = query.values.data() + head * headDimension * querySteps;
        const float* keyHead = key.values.data() + head * headDimension * steps;
        const float* valueHead = value.values.data() + head * headDimension * steps;
        sgemm(&noTranspose, &transpose, &m, &n, &k, &scoreScale,
              queryHead, &m, keyHead, &n, &zero, probability.values.data(), &m);
        for (mwSize row = 0; row < querySteps; ++row) {
            const mwSize diagonal = endpointOnly ? steps - 1 : row;
            probability.at(row, diagonal) = -std::numeric_limits<float>::infinity();
            float maximum = -std::numeric_limits<float>::infinity();
            for (mwSize column = 0; column < steps; ++column) {
                maximum = std::max(maximum, probability.at(row, column));
            }
            float total = 0.0f;
            for (mwSize column = 0; column < steps; ++column) {
                const float exponent = std::exp(probability.at(row, column) - maximum);
                probability.at(row, column) = exponent;
                total += exponent;
            }
            for (mwSize column = 0; column < steps; ++column) {
                probability.at(row, column) /= total;
            }
        }
        if (endpointAttention) {
            for (mwSize column = 0; column < steps; ++column) {
                (*endpointAttention)[column] += probability.at(querySteps - 1, column) /
                                               static_cast<float>(heads);
            }
        }
        const ptrdiff_t outputColumns = static_cast<ptrdiff_t>(headDimension);
        const ptrdiff_t attentionInner = static_cast<ptrdiff_t>(steps);
        sgemm(&noTranspose, &noTranspose, &m, &outputColumns, &attentionInner, &one,
              probability.values.data(), &m, valueHead, &n, &zero,
              attended.values.data() + head * headDimension * querySteps, &m);
    }
    Matrix output = endpointOnly ? lastRow(hidden) : hidden;
    addInPlace(output, multiply(attended, singleView(field(layer, "outputW"), width, width)));
    Matrix feedForward = linear(layerNorm(output, field(layer, "ffNorm")), field(layer, "ffFirst"));
    reluInPlace(feedForward);
    addInPlace(output, linear(feedForward, field(layer, "ffSecond")));
    return output;
}

const mxArray* layerAt(const mxArray* layers, mwSize index) {
    require(layers && mxIsCell(layers) && index < mxGetNumberOfElements(layers),
            "Attention blocks must use the exported cell-array schema.");
    const mxArray* layer = mxGetCell(layers, index);
    require(layer && mxIsStruct(layer), "An attention layer is missing.");
    return layer;
}

std::vector<float> infer(const mxArray* model, const mxArray* rawArgument,
                         const mxArray* maskArgument) {
    const mwSize steps = positiveInteger(field(model, "nSteps"));
    const mwSize features = positiveInteger(field(model, "nFeatures"));
    const mwSize targets = positiveInteger(field(model, "nTargets"));
    const mwSize width = positiveInteger(field(model, "dModel"));
    const mwSize heads = positiveInteger(field(model, "nHeads"));
    const mwSize headDimension = positiveInteger(field(model, "headDim"));
    require(steps >= 2 && features == 31 && targets == 14 && width == heads * headDimension,
            "Expected the exported 31-input/14-target SAITS model.");
    const View raw = singleView(rawArgument, steps, features);
    require(mxIsLogical(maskArgument) && !mxIsSparse(maskArgument) &&
            mxGetNumberOfDimensions(maskArgument) == 2 &&
            mxGetM(maskArgument) == steps && mxGetN(maskArgument) == features,
            "The observation mask must be logical and match the raw-feature matrix.");
    const mxLogical* observed = mxGetLogicals(maskArgument);
    const mxArray* meanArgument = field(model, "contextMean");
    const mxArray* stdArgument = field(model, "contextStd");
    require(mxIsDouble(meanArgument) && !mxIsComplex(meanArgument) &&
            mxGetNumberOfElements(meanArgument) == 17 &&
            mxIsDouble(stdArgument) && !mxIsComplex(stdArgument) &&
            mxGetNumberOfElements(stdArgument) == 17,
            "Context normalization statistics must be FP64 seventeen-element vectors.");
    const double* mean = mxGetDoubles(meanArgument);
    const double* standardDeviation = mxGetDoubles(stdArgument);
    for (mwSize column = 0; column < 17; ++column) {
        require(std::isfinite(mean[column]) && std::isfinite(standardDeviation[column]) &&
                standardDeviation[column] > 0.0, "Invalid context normalization statistics.");
    }
    const float distanceScale = static_cast<float>(scalar(field(model, "distanceScale")));
    require(distanceScale > 0.0f, "Distance scale must be positive.");
    Matrix filled(steps, features);
    Matrix mask(steps, features);
    Matrix embeddedInput(steps, 2 * features);
    for (mwSize column = 0; column < features; ++column) {
        for (mwSize row = 0; row < steps; ++row) {
            const mwSize index = row + column * steps;
            mask.values[index] = observed[index] ? 1.0f : 0.0f;
            float normalized = 0.0f;
            if (observed[index]) {
                const float physical = raw.values[index];
                require(std::isfinite(physical), "An observed physical input is nonfinite.");
                normalized = physical;
                if (column < 7) {
                    normalized /= distanceScale;
                } else if (column >= 14) {
                    normalized = static_cast<float>((static_cast<double>(physical) - mean[column - 14]) /
                                                    standardDeviation[column - 14]);
                }
                require(std::isfinite(normalized), "A normalized observed input is nonfinite.");
            }
            filled.values[index] = normalized;
            embeddedInput.at(row, column) = normalized;
            embeddedInput.at(row, features + column) = mask.values[index];
        }
    }
    Matrix hidden = linear(embeddedInput, field(model, "embeddingFirst"));
    addPosition(hidden, field(model, "positionEncoding"));
    const mxArray* firstLayers = field(model, "firstBlock");
    const mxArray* secondLayers = field(model, "secondBlock");
    const mwSize layers = mxGetNumberOfElements(firstLayers);
    require(mxIsCell(firstLayers) && mxIsCell(secondLayers) && layers >= 1 &&
            layers == mxGetNumberOfElements(secondLayers), "Invalid attention-layer cells.");
    for (mwSize index = 0; index < layers; ++index) {
        hidden = attentionLayer(hidden, layerAt(firstLayers, index), heads, headDimension, false);
    }
    const Matrix first = linear(hidden, field(model, "reduceFirst"));
    require(first.rows == steps && first.columns == features, "Stage-one reduction shape changed.");
    for (mwSize column = 0; column < features; ++column) {
        for (mwSize row = 0; row < steps; ++row) {
            const mwSize index = row + column * steps;
            embeddedInput.at(row, column) = mask.values[index] * filled.values[index] +
                                           (1.0f - mask.values[index]) * first.values[index];
        }
    }
    hidden = linear(embeddedInput, field(model, "embeddingSecond"));
    addPosition(hidden, field(model, "positionEncoding"));
    std::vector<float> attention;
    for (mwSize index = 0; index < layers; ++index) {
        const bool finalLayer = index + 1 == layers;
        hidden = attentionLayer(hidden, layerAt(secondLayers, index), heads, headDimension,
                                finalLayer, finalLayer ? &attention : nullptr);
    }
    Matrix second = linear(hidden, field(model, "reduceSecondBeta"));
    reluInPlace(second);
    second = linear(second, field(model, "reduceSecondGamma"));
    require(second.rows == 1 && second.columns == features, "Stage-two reduction shape changed.");
    Matrix gateInput(1, features + steps);
    for (mwSize column = 0; column < features; ++column) {
        gateInput.at(0, column) = mask.at(steps - 1, column);
    }
    for (mwSize column = 0; column < steps; ++column) {
        gateInput.at(0, features + column) = attention[column];
    }
    const Matrix logits = linear(gateInput, field(model, "combinationGate"));
    require(logits.rows == 1 && logits.columns == features, "Combination gate shape changed.");
    std::vector<float> prediction(targets);
    for (mwSize column = 0; column < targets; ++column) {
        const float mixing = 1.0f / (1.0f + std::exp(-logits.at(0, column)));
        prediction[column] = (1.0f - mixing) * second.at(0, column) +
                             mixing * first.at(steps - 1, column);
        require(std::isfinite(prediction[column]), "The SAITS prediction is nonfinite.");
    }
    return prediction;
}

} // namespace

void mexFunction(int outputs, mxArray* output[], int inputs, const mxArray* input[]) {
    try {
        require(inputs == 3 && outputs <= 2,
                "Use [physical, normalized] = saits_endpoint_mex(model, rawSingle, logicalMask).");
        const std::vector<float> normalized = infer(input[0], input[1], input[2]);
        const View scale = singleView(field(input[0], "targetScale"), 1, 14);
        if (outputs >= 1) {
            output[0] = mxCreateNumericMatrix(1, 14, mxSINGLE_CLASS, mxREAL);
            float* values = mxGetSingles(output[0]);
            for (mwSize index = 0; index < 14; ++index) {
                values[index] = normalized[index] * scale.values[index];
                require(std::isfinite(values[index]), "Physical SAITS output is nonfinite.");
            }
        }
        if (outputs >= 2) {
            output[1] = mxCreateNumericMatrix(1, 14, mxSINGLE_CLASS, mxREAL);
            std::copy(normalized.begin(), normalized.end(), mxGetSingles(output[1]));
        }
    } catch (const std::exception& error) {
        mexErrMsgIdAndTxt("ARES:SaitsMexInferenceFailure", "%s", error.what());
    } catch (...) {
        mexErrMsgIdAndTxt("ARES:SaitsMexInferenceFailure", "Unknown native SAITS inference failure.");
    }
}
