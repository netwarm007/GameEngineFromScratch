/* recognize tokens in MGEMX file */
%{
#include <cstdlib>
#include "MGEMX.parser.hpp"
using namespace My;
%}

%option reentrant noyywrap yylineno nodefault outfile="MGEMX.scanner.cpp" header="MGEMX.scanner.hpp"
%x COMMENT

dseq        ([[:digit:]]+)
dseq_opt    ({dseq}?)
frac        (({dseq_opt}"."{dseq})|{dseq}".")
exp         ([eE][+-]?{dseq})
exp_opt     ({exp}?)
ilen        ([Uu](L|l|LL|ll)?|(L|l|LL|ll)[Uu]?)
integer     ({dseq}){ilen}?
float       (({frac}{exp_opt}[flFL]?)|({dseq}{exp}[flFL]?))

/* Universal Character Name */
ucn     (\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})

%%
{integer}   { yylval->emplace<long long>(strtoll(yytext, nullptr, 10)); return MGEMXParser::token::INT; }
{float}     { yylval->emplace<double>(strtod(yytext, nullptr)); return MGEMXParser::token::FLT;         }
"(" |
")" |
":" |
";" |
"," |
"*" |
"[" |
"]" |
"{" |
"}" |
"<" |
">"                                     { return yytext[0];   }
"struct"                                { return MGEMXParser::token::STRUCT;    }
"table"                                 { return MGEMXParser::token::TABLE;     }
"enum"                                  { return MGEMXParser::token::ENUM;      }
"namespace"                             { return MGEMXParser::token::NAMESPACE; }
"attribute"                             { return MGEMXParser::token::ATTR;      }
"root_type"                             { return MGEMXParser::token::ROOT;      }
[a-zA-Z][a-zA-Z0-9_]*                   { yylval->emplace<std::string>(yytext); return MGEMXParser::token::IDN; }
L?\"([^"\\]|\\['"?\\abfnrtv]|\\[0-7]{1,3}|\\[Xx][0-9a-fA-F]+|{ucn})*\"    {
                                          yylval->emplace<std::string>(yytext); return MGEMXParser::token::STR; }

 /* C++ comment, a common extension */
"//".*                                  { /* ignore comments */         }

 /* C-style comments */
"/*"                                    { BEGIN(COMMENT); }
<COMMENT>"*/"                           { BEGIN(INITIAL); }
<COMMENT>([^*]|\n)+|.
<COMMENT><<EOF>>                        { printf("%d: Unterminated comment\n",
			                                           yylineno); return 0; }
                                                 
[ \t]                               { /* ignore whitespace */ }
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
