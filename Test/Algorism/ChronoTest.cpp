#include <chrono>
#include <ctime>
#include <iostream>

using namespace std;

long fibonacci(unsigned n) {
    if (n < 2) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    auto start = chrono::system_clock::now();
    cout << "f(42) = " << fibonacci(42) << '\n';
    auto end = chrono::system_clock::now();

    chrono::duration<double> elapsed_seconds = end - start;
    time_t end_time = chrono::system_clock::to_time_t(end);

    cout << "finished computation at " << ctime(&end_time)
         << "elapsed time: " << elapsed_seconds.count() << "s\n";
}
