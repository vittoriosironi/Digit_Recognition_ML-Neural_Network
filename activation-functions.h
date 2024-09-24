template <class T>
    T id (T x) { return x; }

template <class T>
    T sigmoid (T x) { return 1. / (1. + exp(-x)); }

template<class T>
    T diff (T (*f) (T), T x) {
        if (f == &id<T>) return 1.;
        else if (f == &sigmoid<T>) return exp(-x) / pow((1 + exp(-x)), 2.);
        return 0.;
    }