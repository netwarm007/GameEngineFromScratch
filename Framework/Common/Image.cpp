#include "Image.hpp"

namespace My {
    std::ostream& operator<<(std::ostream& out, const Image& image)
    {
        out << "Image" << std::endl;
        out << "-----" << std::endl;
        out << "Width: 0x" << image.Width << std::endl;
        out << "Height: 0x" << image.Height << std::endl;
        out << "Bit Count: 0x" << image.bitcount << std::endl;
        out << "Pitch: 0x" << image.pitch << std::endl;
        out << "Data Size: 0x" << image.data_size << std::endl;

        return out;
    }
}

