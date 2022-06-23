#include <iostream>
#include <vector>
#include <assert.h>

class LinearLayer
{
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> output_vals;
    std::vector<double> input_vals;
    int input_size;
    int output_size;
    double lr;

public:
    LinearLayer(){};
    LinearLayer(int input_size, int output_size, double lr);
    std::vector<double> feedForward(const std::vector<double> &input);
    std::vector<double> backPropagate(const std::vector<double> &grad);
};

LinearLayer::LinearLayer(int input_size, int output_size, double lr)
{
    // If condition is true, continue execution
    // Otherwise terminate and throw error
    assert(input_size > 0);
    assert(output_size > 0);

    this->input_size = input_size;
    this->output_size = output_size;
    this->lr = lr;

    // Generate random weights
    for (int out = 0; out < output_size; out++)
    {
        // Add a vector
        weights.push_back(std::vector<double>());
        // input_size + 1 is to add bias
        for (int input = 0; input < input_size + 1; input++)
        {
            // Random value between 0 and 1
            // back() fetches last element in vector
            // In this case, fetches the newly created empty vector
            // Then pushes back a value into it
            weights.back().push_back((double)rand() / RAND_MAX);
        }
    }
}

// & means pass by reference (address)
// https://stackoverflow.com/questions/57483/what-are-the-differences-between-a-pointer-variable-and-a-reference-variable-in/57492#57492

std::vector<double> LinearLayer::feedForward(const std::vector<double> &input)
{
    assert(input.size() == input_size);
    // Initialise output vector
    output_vals = std::vector<double>();
    // Store input vector
    input_vals = input;

    // Perform matrix multiplication
    for (int out = 0; out < output_size; out++)
    {
        double sum = 0.0;
        for (int w = 0; w < input_size; w++)
        {
            sum += weights[out][w] * input_vals[w];
        }
        // Account for the bias, which is last in the vector
        sum += weights[out].back();
        output_vals.push_back(sum);
    }

    return output_vals;
};

std::vector<double> LinearLayer::backPropagate(const std::vector<double> &grad)
{
    assert(grad.size() == output_size);
    std::vector<double> prev_layer_grad;

    // Calculate partial derivatives with respect to input values
    for (int input = 0; input < input_size; input++)
    {
        double g = 0.0;
        for (int out = 0; out < output_size; out++)
        {
            g += grad[out] * weights[out][input];
        }
        prev_layer_grad.push_back(g);
    }

    // Change weights using gradient
    for (int out = 0; out < output_size; out++)
    {
        for (int input = 0; input < input_size; input++)
        {
            weights[out][input] -= lr * grad[out] * input_vals[input];
        }
        // Update bias
        weights[out].back() -= lr * grad[out];
    }

    return prev_layer_grad;
};