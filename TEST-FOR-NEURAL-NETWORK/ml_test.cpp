#include "ML.h"

int main() {
    Neuron<double> A (2, &sigmoid);
    A.zero();
    A.weight[0] = 5;
    // A.weight[1] = 2;
    // A.bias = - 2.5;

    Neuron<double> B (2, &sigmoid);
    B.zero();
    // B.weight[0] = 5;
    B.weight[1] = 5;
    // B.bias = - 2.5;

    vector< Neuron<double> > Neurons;
    Neurons.push_back(A);
    Neurons.push_back(B);

    NeuralLayer<double> L(Neurons);

    int N = 20;
    double w = 1. / N;

    vector<double> input;
    input.push_back(0);
    input.push_back(0);

    ofstream fout("test.txt");


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            input[0] = w * (.5 + i); // Creiamo una griglia di punti per capire qual è il comportamento del neurone
            input[1] = w * (.5 + j);
            fout << input[0] << " "
                 << input[1] << " "
                 << L.evaluate(input, 0) << " "
                 << L.evaluate(input, 1)
                 << endl;
        }
        fout << endl;
    }

    return 0;
}
