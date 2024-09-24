#include "ML.h"

using namespace std;

void read_data(const char *filename, vector< vector<double> > &TrainX, vector< vector<double> > &TrainC, int RES) {
    vector<double> pixel;
    for(int i = 0; i < RES * RES; i++)
        pixel.push_back(0.);

    vector<double> expected_output;
    for(int i = 0; i < 10; i++)
        expected_output.push_back(0.);

    ifstream fin(filename);
    int cat; // Legge il numero sotto
    char read; // Legge il singolo carattere del numero

    while(!fin.eof()) {
        for (int i = 0; i < RES * RES; i++) {
            fin >> read;
            pixel[i] = (double) read - '0';  // senza '0' sputerebbe fuori 48,49, quindi togliendo '0' ovvero il carattere ASCII associato a '0'
        }
        TrainX.push_back(pixel);
        fin >> cat;
        expected_output[cat] = 1.;
        TrainC.push_back(expected_output);
        expected_output[cat] = 0.;
    }
    fin.close();

    TrainX.pop_back();
    TrainC.pop_back();
}

void display_dig(vector<double> pixel) {
    int res = sqrt(pixel.size());
    for (int I = 0; I < res; I++) {
        for (int J = 0; J < res; J++) {
            double cell = pixel[J + res * I]; // Con questa operazione si converte da tabellina a una sequenza lineare
            cout << (cell < .25 ? "░" : (cell < .5 ? "▒" : (cell < .75 ? "▓" : "█")));
        }
        cout << endl;
    }
    cout << endl;
}

int max_index (vector<double> V) {
    int index = 0;
    for (int i = 1; i < V.size(); i++)
        if (V[i] > V[index])
            index = i;
    return index;
}
int recog(vector<double> X, NeuralNetwork<double> digits_recog) { return max_index(digits_recog.output_evaluate(X)); }

int second_max_index(vector<double> V) {
    int max = max_index(V);
    int index = (max == 0 ? 1 : 0);
    for (int i = 0; i < V.size(); i++)
        if (i != max)
            if (V[i] > V[index])
                index = i;
    return index;
}


void test(int NT, vector< vector<double> > TrainX, vector< vector<double> > TrainC, NeuralNetwork<double> digits_recog, int TRAINING_SAMPLE_SIZE, int TEST_SAMPLE_SIZE) {
    double Err = 0.;
    for (int i = 0; i < TEST_SAMPLE_SIZE; i++) {
        int j = TRAINING_SAMPLE_SIZE + i;
        Err += digits_recog.erF(TrainX[j], TrainC[j]);
    }
    cout << "Error = " << Err << endl;

    srand(123); // Mettendo questo seme prende sempre gli stessi
    for (int i = 0; i < NT; i++) { // NT sono N numeri casuali dai dati test
        int j = TRAINING_SAMPLE_SIZE + rand() % TEST_SAMPLE_SIZE;
        display_dig(TrainX[j]);
        for (int c = 0; c < 10; c++) {
            cout << (TrainC[j][c] == 1 ? "▓" : "░") << " | ";
            for (int bar = 0; bar < 100 * digits_recog.output_evaluate(TrainX[j], c); bar++)
                cout << "▓";
            for (int bar = 100 * digits_recog.output_evaluate(TrainX[j], c); bar < 100; bar++)
                cout << "░";
            cout << endl;
        }
        double total = 0;
        for (int k = 0; k < 10; k++)
            total += digits_recog.output_evaluate(TrainX[j], k);

        cout << "DIGIT> " << recog(TrainX[j], digits_recog) << " (" << digits_recog.output_evaluate(TrainX[j], recog(TrainX[j], digits_recog)) / total * 100 << "%)" << endl;
        cout << "II D.> " << second_max_index(digits_recog.output_evaluate(TrainX[j])) << " (" << digits_recog.output_evaluate(TrainX[j], second_max_index(digits_recog.output_evaluate(TrainX[j]))) / total * 100 << "%)" << endl << endl;
    }
    cout << endl;
}

vector<double> conv(vector<double> X, int RES, vector<double> kernel, int KRES) { // KRES è la dimensione del KERNEL
    vector<double> out;
    double temp; // Lettura temporanea

    for (int I = 0; I + KRES - 1 < RES; I++)
        for (int J = 0; J + KRES - 1 < RES; J++) {
            temp = 0;
            for (int i = 0; i < KRES; i++)
                for (int j = 0; j < KRES; j++)
                    temp += X[(J + j) + RES * (I + i)] * kernel[j + KRES * i];
            out.push_back(temp);
        }

    return out;
}

int main() {
    system("clear");

    int RES = 32; // Risoluzione con cui sono scritte le immagini. Input = 32 x 32
    int KRES = 1;
    int Inputs = (RES - KRES + 1) * (RES - KRES + 1);
    vector<int> NeuronsPerLayer;
    NeuronsPerLayer.push_back(30);
    NeuronsPerLayer.push_back(10);

    NeuralNetwork<double> digits_recog(Inputs, NeuronsPerLayer, &sigmoid);

    digits_recog.fprint("starting_parameters.txt");

    vector< vector<double> > TrainX;
    vector< vector<double> > TrainC;

    read_data("digits_dataSet.txt", TrainX, TrainC, RES);

    // Possiamo diminuire i parametri della rete, utilizziamo delle tabelline (= KERNEL) che vanno a prendere delle box dell'immagine e fa la media dei numeri contneuti all'interno. Questa si chiama CONVULAZIONE
    vector<double> kernel;
    for (int i = 0; i < KRES; i++)
        for (int j = 0; j < KRES; j++)
            kernel.push_back(1. / KRES / KRES);
/*
    display_dig(TrainX[100]);
    display_dig(conv(TrainX[100], RES, kernel, KRES));
*/

    int TRAINING_SAMPLE_SIZE = TrainX.size() * .8;
    int TEST_SAMPLE_SIZE = TrainX.size() - TRAINING_SAMPLE_SIZE;
    // Dividiamo i dati in 80% e 20%

    test(5, TrainX, TrainC, digits_recog, TRAINING_SAMPLE_SIZE, TEST_SAMPLE_SIZE);

    // TRAINING
    int REP = 10;
    double t = .1;
    for (int it = 0; it < REP; it++) {
        if (it % 8 == 0) t *= .5;
        for (int i = 0; i < TRAINING_SAMPLE_SIZE; i++)
            digits_recog.evo(conv(TrainX[i], RES, kernel, KRES), TrainC[i], t);
        cout << endl;
        test(5, TrainX, TrainC, digits_recog, TRAINING_SAMPLE_SIZE, TEST_SAMPLE_SIZE);
    }

    // RICONOSCIMENTO
    srand(time(NULL));
    int n = TRAINING_SAMPLE_SIZE + rand() % TEST_SAMPLE_SIZE;
    display_dig(TrainX[n]);
    cout << recog(TrainX[n], digits_recog) << endl;

    digits_recog.fprint("final_parameters.txt");

    return 0;
}