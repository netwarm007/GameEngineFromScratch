/* recognize tokens in MGEMX file */

%option noyywrap

%%
"("     { printf("OP ");    }
")"     { printf("CP ");    }
":"     { printf("COLON "); }
";"     { printf("SEMIC "); }
","     { printf("COMMA "); }
"*"     { printf("ASTER "); }
"["     { printf("OBK ");   }
"]"     { printf("CBK ");   }
"{"     { printf("OBR ");   }
"}"     { printf("CBR ");   }
"<"     { printf("OAB ");   }
">"     { printf("CAB ");  }
"\""([^"])*"\""  { printf("STR "); }
"//".*                                  { printf("C++ Style Comment\n"); }
"/*"([^*]|(\*+[^*/]))*\*+\/             { printf("C Style Comment\n");   }
[ \t\v\f]                               { /* ignore whitespace */ }
\n      { printf("\n");   }
\0      { printf("EOS");  }
[a-zA-Z][a-zA-Z0-9_]* { printf("IDN "); }
[0-9]+  { printf("NUM "); }
.       { printf("魔法字符 %c\n", *yytext); }
%%

int main(int argc, char** argv)
{
    yylex();
    return 0;
}
