/* recognize tokens in MGEMX file */
%{
#include <cstdlib>
#include "MGEMX.parser.hpp"
using namespace My;
%}

%option reentrant interactive noyywrap noyylineno nodefault outfile="MGEMX.scanner.cpp" header="MGEMX.scanner.hpp"

dseq        ([[:digit:]]+)
dseq_opt    ({dseq}?)
frac        (({dseq_opt}"."{dseq})|{dseq}".")
exp         ([eE][+-]?{dseq})
exp_opt     ({exp}?)
integer     ({dseq})
float       (({frac}{exp_opt})|({dseq}{exp}))

%%
{integer}   { yylval->emplace<long long>(strtoll(yytext, nullptr, 10)); return MGEMXParser::token::INT; }
{float}     { yylval->emplace<double>(strtod(yytext, nullptr)); return MGEMXParser::token::FLT;         }
"("     { return MGEMXParser::token::OP;    }
")"     { return MGEMXParser::token::CP;    }
":"     { return MGEMXParser::token::COLON; }
";"     { return MGEMXParser::token::SEMIC; }
","     { return MGEMXParser::token::COMMA; }
"*"     { return MGEMXParser::token::ASTER; }
"["     { return MGEMXParser::token::OBK;   }
"]"     { return MGEMXParser::token::CBK;   }
"{"     { return MGEMXParser::token::OBR;   }
"}"     { return MGEMXParser::token::CBR;   }
"<"     { return MGEMXParser::token::OAB;   }
">"     { return MGEMXParser::token::CAB;   }
"struct"                                { return MGEMXParser::token::STRUCT;    }
"table"                                 { return MGEMXParser::token::TABLE;     }
"enum"                                  { return MGEMXParser::token::ENUM;      }
"namespace"                             { return MGEMXParser::token::NAMESPACE; }
"attribute"                             { return MGEMXParser::token::ATTR;      }
"root_type"                             { return MGEMXParser::token::ROOT;      }
[a-zA-Z][a-zA-Z0-9_]*                   { yylval->emplace<std::string>(yytext); return MGEMXParser::token::IDN; }
"\""([^"])*"\""                         { yylval->emplace<std::string>(yytext); return MGEMXParser::token::STR; }
"//".*                                  { /* ignore comments */         }
"/*"([^*]|(\*+[^*/]))*\*+\/             { /* ignore C-style comments */ }
[ \t\v\f]                               { /* ignore whitespace */ }
\n      { /* ignore it */   }
\0      { return MGEMXParser::token::EOS;   }
.       { printf("魔法字符 %c\n", *yytext); }
%%

int main() {
    yyscan_t scanner;
    yylex_init(&scanner);
    My::MGEMXParser parser(scanner);
    std::cout.precision(10);
    parser.parse();
    yylex_destroy(scanner);
}