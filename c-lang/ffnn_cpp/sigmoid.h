#include <iostream>
#include <vector>
#include <assert.h>
#include <math.h>

class Sigmoid
{
private:
    std::vector<double> output_vals;

public:
    Sigmoid(){};
    std::vector<double> feedForward(const std::vector<double> &input);
    std::vector<double> backPropagate(std::vector<double> &grad);
};

std::vector<double> Sigmoid::feedForward(const std::vector<double> &input)
{
    this->output_vals = std::vector<double>();

    for (int in = 0; in < input.size(); in++)
    {
        output_vals.push_back(1.0 / (1.0 + exp(-input[in])));
    }

    return output_vals;
};

std::vector<double> Sigmoid::backPropagate(std::vector<double> &grad)
{
    assert(grad.size() == output_vals.size());

    for (int out = 0; out < output_vals.size(); out++)
    {
        grad[out] += output_vals[out] * (1.0 - output_vals[out]);
    }

    return grad;
};