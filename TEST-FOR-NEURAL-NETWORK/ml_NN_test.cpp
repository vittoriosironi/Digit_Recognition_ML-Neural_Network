#include "ML.h"

double F(double x) { return sigmoid(10. * (x - .5)); }

bool circle(double x, double y, double radius)
{ return ((x - .5) * (x - .5) + (y - .5) * (y - .5) < radius * radius); }

bool torus(double x, double y, double innerRadius, double outerRadius)
{ return circle(x, y, outerRadius) && !circle(x, y, innerRadius); }

int main() {
    srand(time(NULL));
    int N = 20;
    double w = 1. / N;

    // Neuroni
    vector<int> NeuronsPerLayer;
    NeuronsPerLayer.push_back(6);
//    NeuronsPerLayer.push_back(6);
    NeuronsPerLayer.push_back(2);

    // Input Data
    int Inputs = 2;

    // Rete neurale
    NeuralNetwork<double> Classifier(Inputs, NeuronsPerLayer, &sigmoid);

    /*
    Classifier.Layer[0].neuron[0].zero();
    Classifier.Layer[0].neuron[1].zero();
    Classifier.Layer[1].neuron[0].zero();
    Classifier.Layer[1].neuron[1].zero();

    Classifier.Layer[0].neuron[0].weight[0] = 5.;
    Classifier.Layer[1].neuron[0].weight[0] = 5.;
    Classifier.Layer[1].neuron[0].bias = -5.;
    */

    int TRAINING_SAMPLE_SIZE = 1000;

    ofstream fout("train_data.txt");
    for (int i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        double x = (double) rand() / RAND_MAX;
        double y = (double) rand() / RAND_MAX;
        fout << x << " " << y << " " << (torus(x, y, .2, .4) ? 1 : 0) << endl;
    }
    fout.close();


    vector< vector<double> > TrainX;
    vector< vector<double> > TrainC;

    vector<double> input;
    input.push_back(0.);
    input.push_back(0.);
    int cat;

    ifstream fin("train_data.txt");
    while (!fin.eof()) {
        fin >> input[0];
        fin >> input[1];
        TrainX.push_back(input);
        fin >> cat;

        if (cat == 0) {
            input[0] = 1;
            input[1] = 0;
        } else {
            input[0] = 0;
            input[1] = 1;
        }
        TrainC.push_back(input);
    }
    // Toglie l'ultimo elemento perchè vengono 101 gli elementi test, infatti legge anche l'ultima linea senza nulla
    TrainX.pop_back(); 
    TrainC.pop_back();
    // cout << TrainX.size() << endl;
    fin.close();

    fout.open("before.txt");
    for (int i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        fout << TrainX[i][0] << " "
             << TrainX[i][1] << " "
             << Classifier.output_evaluate(TrainX[i], 0) << " "
             << Classifier.output_evaluate(TrainX[i], 1) << " "
             << Classifier.erF(TrainX[i], TrainC[i]);
        fout << endl;
    }
    fout.close();

    // EVO
    int REP = 100;
    double t = .1;
    for (int it = 0; it < REP; it++) {
        cout << "\rrep = " << it + 1 << " / " << REP << flush;
        for (int i = 0; i < TRAINING_SAMPLE_SIZE; i++)
            Classifier.evo(TrainX[i], TrainC[i], t);
    }


    fout.open("after.txt");
    for (int i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        fout << TrainX[i][0] << " "
             << TrainX[i][1] << " "
             << Classifier.output_evaluate(TrainX[i], 0) << " "
             << Classifier.output_evaluate(TrainX[i], 1) << " "
             << Classifier.erF(TrainX[i], TrainC[i]);
        fout << endl;
    }
    fout.close();

    return 0;
}
