#include <iostream>
#include <vector>
#include <assert.h>
#include "LinearLayer.h"
#include "Sigmoid.h"

class Model
{
private:
    LinearLayer fc;
    Sigmoid sig;
    std::vector<double> output_vals;

public:
    Model();
    std::vector<double> feedForward(std::vector<double> input);
    void backPropagate(std::vector<double> exp_vals);
};

Model::Model()
{
    // Linear layer with input size 2, output size 1 and learning rate 0.15
    fc = LinearLayer(2, 1, 0.15);
    sig = Sigmoid();
}

std::vector<double> Model::feedForward(std::vector<double> input)
{
    input = fc.feedForward(input);
    output_vals = sig.feedForward(input);

    return output_vals;
}

void Model::backPropagate(std::vector<double> exp_vals)
{
    std::vector<double> grad;
    assert(exp_vals.size() == output_vals.size());

    // Compute derivative of error with respect to network's output
    for (int out = 0; out < output_vals.size(); out++)
    {
        grad.push_back(output_vals[out] - exp_vals[out]);
    }

    grad = sig.backPropagate(grad);
    fc.backPropagate(grad);
}

int main()
{
    // Training the model to separate data based on the equation 5x_1 + 7x_2 = 0
    Model model;

    // Generate two points above the line
    std::vector<std::vector<double>> train_data;

    for (int i = 0; i < 50; i++)
    {
        double x_1 = (double)rand() / RAND_MAX;
        double x_2 = (-(5.0 / 7.0) * x_1) + ((double)rand() / RAND_MAX) + 0.0001;
        train_data.push_back({x_1, x_2, 1.0});
    }

    // Generate two points below the line
    for (int i = 0; i < 50; i++)
    {
        double x_1 = (double)rand() / RAND_MAX;
        double x_2 = (-(5.0 / 7.0) * x_1) - ((double)rand() / RAND_MAX) - 0.0001;
        train_data.push_back({x_1, x_2, 0.0});
    }

    // Print the points
    for (int i = 0; i < train_data.size(); i++)
    {
        std::cout << train_data[i][0] << ", " << train_data[i][1] << ", " << train_data[i][2] << std::endl;
    }

    // Train
    for (int i = 0; i < 10000; i++)
    {
        int ind = rand() % (train_data.size());
        std::vector<double> output = model.feedForward({train_data[ind][0], train_data[ind][1]});
        model.backPropagate({train_data[ind][2]});
    }

    // Test
    double acc = 0.0;
    for (int i = 0; i < train_data.size(); i++)
    {
        std::vector<double> output = model.feedForward({train_data[i][0], train_data[i][1]});

        if (output[0] > 0.5 && train_data[i][2] == 1.0)
        {
            acc++;
        }
        else if (output[0] < 0.5 && train_data[i][2] == 0.0)
        {
            acc++;
        }
    }

    std::cout << "Accuracy: " << acc / train_data.size() << std::endl;
}