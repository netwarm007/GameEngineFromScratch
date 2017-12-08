#include "Image.hpp"

using namespace std;

namespace My {
    ostream& operator<<(ostream& out, const Image& image)
    {
        out << "Image" << endl;
        out << "-----" << endl;
        out << "Width: " << image.Width << endl;
        out << "Height: " << image.Height << endl;
        out << "Bit Count: " << image.bitcount << endl;
        out << "Pitch: " << image.pitch << endl;
        out << "Data Size: " << image.data_size << endl;

        int byte_count = image.bitcount >> 3;

        for (int i = 0; i < image.Height; i++) {
            for (int j = 0; j < image.Width; j++) {
                for (int k = 0; k < byte_count; k++) {
                    printf("%x ", reinterpret_cast<uint8_t*>(image.data)[image.pitch * i + j * byte_count + k]);
                }
                cout << "\t";
            }
            cout << endl;
        }

        return out;
    }
}

