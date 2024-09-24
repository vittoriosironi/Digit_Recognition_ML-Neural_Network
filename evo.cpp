void evo (vector<T> x, vector<T> expectedOutput, T t)
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