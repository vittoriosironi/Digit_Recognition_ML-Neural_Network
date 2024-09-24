#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

using namespace std;

#include "activation-functions.h"

template <class T>
    class Neuron {
    public:
        int N;
        T (*activationFunction) (T);
        vector<T> weight;
        T bias;

        Neuron() {
            N = 0;
            bias = 0;
            activationFunction = &id;
        }
        Neuron(int N_, T (*activationFunction_) (T)) {
            N = N_;
            for (int i = 0; i < N; i++)
                weight.push_back((T) - .5 + 1. * rand() / RAND_MAX);
            bias = (T) -.5 + 1. * rand() / RAND_MAX;
            activationFunction = activationFunction_;
        }

        void zero() {
            for (int i = 0; i < N; i++)
                weight[i] = 0;
            bias = 0;
        }
        T linear_f (vector<T> x) {
            T output = bias;
            for (int i = 0; i < N; i++)
                output += weight[i] * x[i];
            return output;
        }
        T f(vector<T> x) { return activationFunction(linear_f(x));}
        T f_prime(vector<T> x) { return diff(activationFunction, linear_f(x));}

        void fprint(ofstream &fout) {
            fout << bias << " ";
            for (int i = 0; i < N; i++)
                fout << weight[i] << (i < N - 1 ? " " : "");
        }
    };

template <class T>
    class NeuralLayer {
    public:
        int NL, N;
        vector< Neuron<T> > neuron;

        NeuralLayer() { NL = 0; N = 0; }
        NeuralLayer(int NL_, int N_, T (*activationFunction_) (T)) {
            N = N_;
            NL = NL_;
            for (int i = 0; i < NL; i++)
                neuron.push_back(Neuron<T>(N, activationFunction_));
        }
        NeuralLayer(vector< Neuron<T> > neuron_) {
            N = neuron_[0].N;
            NL = neuron_.size();
            neuron = neuron_;
        }

        T evaluate (vector<T> x, int i) { return neuron[i].f(x); }
        vector<T> evaluate (vector<T> x) {
            vector<T> output;
            for (int i = 0; i < NL; i++)
                output.push_back(evaluate(x, i));
            return output;
        }

        T evaluate_prime (vector<T> x, int i) { return neuron[i].f_prime(x); }
        vector<T> evaluate_prime (vector<T> x) {
            vector<T> output;
            for (int i = 0; i < NL; i++)
                output.push_back(evaluate_prime(x, i));
            return output;
        }

        void fprint(ofstream &fout) {
            for (int i = 0; i < NL; i++) {
                neuron[i].fprint(fout);
                if (i < NL - 1)
                    fout << endl;
            }
        }
    };

template<class T>
    class NeuralNetwork {
    public:
        int Inputs;
        vector<int> NeuronsPerLayer; // Nel suo size() è codificato il numero di Layers
        int Layers() { return NeuronsPerLayer.size(); }
        vector<int> InputsPerLayer() {
            vector<int> output;
            output.push_back(Inputs);
            for(int I = 1; I < Layers(); I++)
                output.push_back(NeuronsPerLayer[I - 1]);
            return output;
        }

        vector< NeuralLayer<T> > Layer;

        NeuralNetwork(int Inputs_, vector<int> NeuronsPerLayer_, T (*activationFunction_) (T)) {
            Inputs = Inputs_;
            NeuronsPerLayer = NeuronsPerLayer_;
            for (int I = 0; I < Layers(); I++)
                Layer.push_back(NeuralLayer<T> (NeuronsPerLayer[I], InputsPerLayer()[I], activationFunction_));
        }

        T evaluate(vector<T> x, int I, int i) {
            if (I == 0)
                return Layer[0].evaluate(x, i);

            vector<T> in, mid;
            in = Layer[0].evaluate(x);

            for (int J = 1; J <= I; J++) {
                mid = Layer[J].evaluate(in);
                in = mid;
            }

            return mid[i];
        }
        T evaluate_prime(vector<T> x, int I, int i) {
            if (I == 0)
                return Layer[0].evaluate_prime(x, i);

            vector<T> in, mid;
            in = Layer[0].evaluate(x);

            for (int J = 1; J < I; J++) {
                mid = Layer[J].evaluate(in);
                in = mid;
            }

            return Layer[I].evaluate_prime(in, i);
        }

        vector<T> evaluate(vector<T> x, int I) {
            if (I == 0)
                return Layer[0].evaluate(x);

            vector<T> in, mid;
            in = Layer[0].evaluate(x);

            for (int J = 1; J <= I; J++) {
                mid = Layer[J].evaluate(in);
                in = mid;
            }

            return mid;
        }
        vector<T> evaluate_prime(vector<T> x, int I) {
            if (I == 0)
                return Layer[0].evaluate_prime(x);

            vector<T> in, mid;
            in = Layer[0].evaluate(x);

            for (int J = 1; J < I; J++) {
                mid = Layer[J].evaluate(in);
                in = mid;
            }

            return Layer[I].evaluate_prime(in);
        }

        T output_evaluate(vector<T> x, int i) { return evaluate(x, Layer.size() - 1, i); }
        T output_evaluate_prime(vector<T> x, int i) { return evaluate_prime(x, Layer.size() - 1, i); }

        vector<T> output_evaluate(vector<T> x) { return evaluate(x, Layer.size() - 1); }
        vector<T> output_evaluate_prime(vector<T> x) { return evaluate_prime(x, Layer.size() - 1); }

        // Errore della rete neurale
        T delta(vector<T> x, vector<T> expectedOutput, int i) { return output_evaluate(x, i) - expectedOutput[i]; }
        T erF(vector<T> x, vector<T> expectedOutput) {
            T output = 0.;
            for (int i = 0; i < NeuronsPerLayer.back(); i++) // NeuronsPerLayer.back() è l'ultimo
                output += pow(delta(x, expectedOutput, i), 2);
            return output;
        }

        // EVOLUTORE
        void evo (vector<T> x, vector<T> expectedOutput, T t) // t => Learning Rate
        {
            vector<T> dE_db, dE_db_old;
            NeuralNetwork<T> NN_old = *this;

            vector < vector <T> > eval;
            for (int I = 0; I < Layers (); I++)
                eval.push_back (NN_old.evaluate (x, I));

            vector < vector <T> > eval_prime;
            for (int I = 0; I < Layers (); I++)
                eval_prime.push_back (NN_old.evaluate_prime (x, I));

            vector<T> g;
            for (int i = 0; i < NeuronsPerLayer.back (); i++)
                g.push_back (2. * NN_old.delta (x, expectedOutput, i));

            for (int i = 0; i < NeuronsPerLayer.back (); i++)
            {
                dE_db.push_back (g [i] * eval_prime [Layers () - 1][i]/*NN_old.output_evaluate_prime (x, i)*/);
                Layer.back ().neuron [i].bias += -t * dE_db [i];
                for (int j = 0; j < InputsPerLayer ().back (); j++)
                    Layer.back ().neuron [i].weight [j] += -t * dE_db [i] * eval [Layers () - 2][j]/*NN_old.evaluate (x, Layers () - 2, j)*/;
            }
            
            for (int I = Layers () - 2; I > 0; I--)
            {
                dE_db_old = dE_db;
                dE_db.clear ();
                T temp;
                for (int i = 0; i < NeuronsPerLayer [I]; i++)
                {
                    temp = (T) 0.;
                    for (int k = 0; k < NeuronsPerLayer [I + 1]; k++)
                        temp += dE_db_old [k] * NN_old.Layer [I + 1].neuron [k].weight [i];
                    dE_db.push_back (temp * eval_prime [I][i]/*NN_old.evaluate_prime (x, I, i)*/);
                    Layer [I].neuron [i].bias += -t * dE_db [i];
                    for (int j = 0; j < InputsPerLayer () [I]; j++)
                        Layer [I].neuron [i].weight [j] += - t * dE_db [i] * eval [I - 1][j]/*NN_old.evaluate (x, I - 1, j)*/;
                }
            }

            dE_db_old = dE_db;
            dE_db.clear ();
            T temp;
            for (int i = 0; i < NeuronsPerLayer [0]; i++)
            {
                temp = (T) 0.;
                for (int j = 0; j < NeuronsPerLayer [1]; j++)
                    temp += dE_db_old [j] * NN_old.Layer [1].neuron [j].weight [i];
                dE_db.push_back (temp * eval_prime [0][i]/*NN_old.evaluate_prime (x, 0, i)*/);
                Layer [0].neuron [i].bias += -t * dE_db [i];
                for (int j = 0; j < InputsPerLayer () [0]; j++)
                    Layer [0].neuron [i].weight [j] += - t * dE_db [i] * x [j];
            }
        }

        void fprint(const char* filename) {
            ofstream fout(filename);
            for (int I = 0; I < Layers(); I++) {
                Layer[I].fprint(fout);
                if (I < Layers() - 1)
                    fout << endl;
            }
            fout.close();
        }
        void read_parameters(const char* filename) {
            ifstream fin(filename);
            for (int I = 0; I < Layers(); I++)
                for (int i = 0; i < NeuronsPerLayer[I]; i++) {
                    fin << Layer[i].neuron[i].bias;
                    for (int a = 0; a < InputsPerLayer()[I]; a++)
                        fin << Layer[i].neuron[i].weight[a];
                }
            fin.close();
        }
    };