namespace My {
    namespace FillMode {
        int Count = 3;
        enum Enum { 
            Wireframe = 0,
            Solid = 1,
            Point = 2
        };

        static const char* s_value_names[] = {
            "Wireframe",
            "Solid",
            "Point"
        };

        static const char* ToString( Enum e ) {
            return s_value_names[(int)e];
        }
    } // namespace FillMode
} // namespace My
