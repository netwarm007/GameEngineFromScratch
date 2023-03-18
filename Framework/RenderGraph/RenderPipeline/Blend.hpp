namespace My::RenderGraph {
    namespace Blend {
        int Count = 17;
        enum Enum { 
            Zero = 0,
            One = 1,
            SrcColor = 2,
            InvSrcColor = 3,
            SrcAlpha = 4,
            InvSrcAlpha = 5,
            DestAlpha = 6,
            InvDestAlpha = 7,
            DestColor = 8,
            InvDestColor = 9,
            SrcAlphaSta = 10,
            BlendFactor = 11,
            InvBlendFactor = 12,
            Src1Color = 13,
            InvSrc1Color = 14,
            Src1Alpha = 15,
            InvSrc1Alpha = 16
        };

        static const char* s_value_names[] = {
            "Zero",
            "One",
            "SrcColor",
            "InvSrcColor",
            "SrcAlpha",
            "InvSrcAlpha",
            "DestAlpha",
            "InvDestAlpha",
            "DestColor",
            "InvDestColor",
            "SrcAlphaSta",
            "BlendFactor",
            "InvBlendFactor",
            "Src1Color",
            "InvSrc1Color",
            "Src1Alpha",
            "InvSrc1Alpha"
        };

        static const char* ToString( Enum e ) {
            return s_value_names[(int)e];
        }
    } // namespace Blend
} // namespace My::RenderGraph
