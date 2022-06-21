namespace My {
    namespace TopologyType {
        int Count = 5;
        enum Enum { 
            Unknown = 0,
            Point = 1,
            Line = 2,
            Triangle = 3,
            Patch = 4
        };

        static const char* s_value_names[] = {
            "Unknown",
            "Point",
            "Line",
            "Triangle",
            "Patch"
        };

        static const char* ToString( Enum e ) {
            return s_value_names[(int)e];
        }
    } // namespace TopologyType
} // namespace My
