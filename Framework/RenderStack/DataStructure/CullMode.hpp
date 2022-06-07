namespace My {
    namespace CullMode {
        int Count = 3;
        enum Enum { 
            None = 0,
            Front = 1,
            Back = 2
        };

        static const char* s_value_names[] = {
            "None",
            "Front",
            "Back"
        };

        static const char* ToString( Enum e ) {
            return s_value_names[(int)e];
        }
    } // namespace CullMode
} // namespace My
